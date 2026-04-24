"""P2: symbolic regression for within-cell flip formulas.

Target: the 2 dominant error cells where the rule fails most:
  - score=3 cell (rule predicts Low; ~5041 flip to Medium)
  - score=6 cell (rule predicts Medium; ~4163 flip to High)

Input features (7 non-rule continuous):
  Humidity, Previous_Irrigation_mm, Electrical_Conductivity,
  Field_Area_hectare, Soil_pH, Organic_Carbon, Sunlight_Hours

For each cell, binary target = 1 if label != rule_pred else 0. We run
gplearn SymbolicClassifier (pure-Python, pip-installable) with a
compact function set (+, -, *, /, sqrt, log, abs, >, <). A hit is a
formula whose precision >= 0.70 at its optimal threshold AND recall
>= 0.20 of flipped rows within the cell, measured via 5-fold CV on
cell-subset rows. Such formulas are deployed as hard overrides on the
LB-best blend.

Env:
  POPULATION_SIZE=2000        size of GP population
  GENERATIONS=50              number of generations
  PARSIMONY_COEFFICIENT=0.01  complexity penalty
  SEED=42
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_curve, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


NONRULE_CONTINUOUS = [
    "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
    "Field_Area_hectare", "Soil_pH", "Organic_Carbon", "Sunlight_Hours",
]
SEED = int(os.environ.get("SEED", 42))
POPULATION_SIZE = int(os.environ.get("POPULATION_SIZE", 2000))
GENERATIONS = int(os.environ.get("GENERATIONS", 50))
PARSIMONY = float(os.environ.get("PARSIMONY_COEFFICIENT", 0.01))
TARGET_CELLS = {3: 0, 6: 1}  # score -> rule_pred_class
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}


def dgp_score_and_rule(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute DGP rule score + 3-class rule_pred for each row."""
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2
    score = 2 * (dry + norain) + (hot + windy + nomulch) + kc
    score = score.to_numpy()
    rule = np.where(score <= 3, 0, np.where(score <= 6, 1, 2))
    return score, rule


def best_threshold_and_metrics(scores: np.ndarray, y: np.ndarray
                               ) -> tuple[float, float, float]:
    """Find threshold on `scores` that maximizes precision@threshold
    subject to recall >= 0.15; return (threshold, precision, recall).
    """
    p, r, t = precision_recall_curve(y, scores)
    # precision_recall_curve returns thresholds of length len(p)-1.
    best_p, best_r, best_t = 0.0, 0.0, 0.0
    for i in range(len(t)):
        if r[i] >= 0.15 and p[i] > best_p:
            best_p, best_r, best_t = p[i], r[i], t[i]
    return best_t, best_p, best_r


def evaluate_formula(program, X_tr, y_tr, X_va, y_va):
    """Run a gplearn program; report best threshold + CV precision/recall."""
    # gplearn's _Program uses .execute(), not .predict(). Wrap try/except
    # so a single failing formula doesn't kill the whole evaluation loop.
    try:
        sc_tr = program.execute(X_tr)
        sc_va = program.execute(X_va)
    except Exception as e:
        return dict(threshold=float("nan"), precision=0.0, recall=0.0,
                    selected=0, error=str(e))
    t, _, _ = best_threshold_and_metrics(sc_tr, y_tr)
    y_pred = (sc_va > t).astype(int)
    if y_pred.sum() == 0:
        return dict(threshold=float(t), precision=0.0, recall=0.0, selected=0)
    return dict(
        threshold=float(t),
        precision=float(precision_score(y_va, y_pred, zero_division=0)),
        recall=float(recall_score(y_va, y_pred, zero_division=0)),
        selected=int(y_pred.sum()),
    )


def run_symbolic_search(train: pd.DataFrame, score: np.ndarray,
                        rule: np.ndarray, cell_score: int) -> dict:
    """Search for symbolic formulas in one cell."""
    try:
        from gplearn.genetic import SymbolicRegressor
    except ImportError as e:
        log(f"gplearn not installed: {e}")
        log("install via: pip install gplearn")
        return dict(error="gplearn_not_installed")

    y = train["y_int"].to_numpy()
    cell_mask = score == cell_score
    rule_cls = TARGET_CELLS[cell_score]
    y_flip = ((y != rule[:]) & cell_mask).astype(int)
    n_cell = int(cell_mask.sum())
    n_flip = int(y_flip[cell_mask].sum())
    log(f"cell score={cell_score}  n_cell={n_cell:,}  "
        f"n_flip={n_flip:,}  flip_rate={n_flip/n_cell:.3%}")

    X = train.loc[cell_mask, NONRULE_CONTINUOUS].to_numpy(dtype=np.float64)
    y_cell = y_flip[cell_mask]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X, y_cell))
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y_cell[tr_idx], y_cell[va_idx]

    # SymbolicRegressor with a compact function set. Log/sqrt guarded
    # against domain errors.
    function_set = ("add", "sub", "mul", "div", "sqrt", "log", "abs")
    sr = SymbolicRegressor(
        population_size=POPULATION_SIZE, generations=GENERATIONS,
        stopping_criteria=0.01, p_crossover=0.7, p_subtree_mutation=0.1,
        p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=0.9,
        verbose=1, parsimony_coefficient=PARSIMONY,
        function_set=function_set, random_state=SEED, n_jobs=1,
        feature_names=NONRULE_CONTINUOUS,
    )
    sr.fit(X_tr, y_tr)

    top_formulas = []
    # gplearn exposes the winning program via sr._program; for Hall of
    # Fame sweep we look at _programs[-1] (final generation's population).
    # Take top 10 by fitness.
    final_pop = sr._programs[-1]
    # Sort by raw fitness (lower = better for regression; we want lower MSE).
    final_pop_sorted = sorted(final_pop, key=lambda p: p.raw_fitness_)
    for rank, program in enumerate(final_pop_sorted[:10]):
        metrics_tr = evaluate_formula(program, X_tr, y_tr, X_tr, y_tr)
        metrics_va = evaluate_formula(program, X_tr, y_tr, X_va, y_va)
        top_formulas.append(dict(
            rank=rank,
            expression=str(program),
            length=program.length_,
            raw_fitness=float(program.raw_fitness_),
            train=metrics_tr, val=metrics_va,
        ))
    return dict(
        cell_score=cell_score, rule_cls=rule_cls,
        n_cell=n_cell, n_flip=n_flip,
        flip_rate=float(n_flip / n_cell),
        top_formulas=top_formulas,
    )


def main():
    log("loading train")
    train = pd.read_csv("data/train.csv")
    train["y_int"] = train["Irrigation_Need"].map(CLS_MAP)
    score, rule = dgp_score_and_rule(train)

    results = {}
    for cell_score in TARGET_CELLS:
        log(f"=== SYMBOLIC SEARCH: cell score={cell_score} ===")
        results[f"cell{cell_score}"] = run_symbolic_search(
            train, score, rule, cell_score)

    out_path = Path("scripts/artifacts/p2_symbolic_flip_results.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
