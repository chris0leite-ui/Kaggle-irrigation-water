"""Shared utilities for HP tuning of the three XGB components.

Design:
- HP search uses a single 80/20 stratified inner split (not 5-fold)
  for speed. Ranking of HPs is stable at this data volume per
  previous experience (LGBM Optuna sweep, 200k subsample).
- Objective: prior-reweight bal_acc on inner val. Proxy for the
  tuned log-bias OOF, but NOT tunable — removes log-bias coord
  ascent from the loop (which is slow AND fits to OOF).
- Pruner: MedianPruner with warmup=10 trials so early losers are
  killed after ~100 rounds.
- Final HPs only accepted if the inner-val lift is > 1 fold-std
  (sigma ~0.0009) vs the current defaults. Anything below is
  considered noise and we retain baseline HPs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART_DIR = Path("scripts/artifacts")
OUT_DIR = Path("submissions")
ART_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values

    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)

    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)

    out["dry"] = dry
    out["norain"] = norain
    out["hot"] = hot
    out["windy"] = windy
    out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)

    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)

    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)

    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)

    return out


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    """Coord-ascent over per-class log-bias, same schedule as production scripts."""
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def reweight_bal_acc(probs: np.ndarray, y: np.ndarray, prior: np.ndarray) -> float:
    """Prior-reweight bal_acc: cheap proxy for tuned log-bias bal_acc."""
    # Clip prior to avoid divide-by-zero when a class is absent; result
    # is not meaningful for absent classes but well-defined elsewhere.
    safe_prior = np.clip(prior, 1e-4, 1.0)
    return float(balanced_accuracy_score(y, (probs / safe_prior).argmax(axis=1)))


def argmax_bal_acc(probs: np.ndarray, y: np.ndarray) -> float:
    """Plain argmax bal_acc — use when a class is absent from y (prior-reweight breaks)."""
    return float(balanced_accuracy_score(y, probs.argmax(axis=1)))


def stratified_subsample(X: pd.DataFrame, y: np.ndarray, n: int, rng: int = SEED):
    """Return (X_sub, y_sub) with ~equal-rate stratified sampling."""
    if len(X) <= n:
        return X, y
    keep_idx, _ = train_test_split(
        np.arange(len(X)), train_size=n, stratify=y, random_state=rng
    )
    return X.iloc[keep_idx].reset_index(drop=True), y[keep_idx]


def get_xgb_fixed_kwargs(seed: int = SEED) -> dict:
    return dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=seed,
    )


def suggest_xgb_params(trial) -> dict:
    """Optuna HP search space for XGB on this problem.

    Space chosen from common tabular-boosting prior:
      - max_depth 4-10 (baseline 7)
      - min_child_weight 1-30 log (baseline 5)
      - lr 0.02-0.15 log (baseline 0.05)
      - subsample, colsample_bytree 0.6-1.0 uniform (baseline 0.9)
      - reg_alpha, reg_lambda 1e-8 to 10 log (baseline 0 defaults)
      - gamma 1e-8 to 5 log (baseline 0 default)
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }


def train_and_score(
    dtr,
    dva,
    y_va: np.ndarray,
    prior: np.ndarray,
    hp: dict,
    seed: int = SEED,
    num_boost_round: int = 4000,
    early_stopping_rounds: int = 100,
    callbacks: list | None = None,
):
    """Train one XGB model with given HPs, return (best_iter, reweight_bal_acc)."""
    import xgboost as xgb

    params = {**get_xgb_fixed_kwargs(seed=seed), **hp}
    kw = dict(
        num_boost_round=num_boost_round,
        evals=[(dva, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=0,
    )
    if callbacks:
        kw["callbacks"] = callbacks
    booster = xgb.train(params, dtr, **kw)
    bi = booster.best_iteration
    probs = booster.predict(dva, iteration_range=(0, bi + 1))
    return bi, reweight_bal_acc(probs, y_va, prior)


def save_hp_result(name: str, best_params: dict, best_value: float, baseline_value: float,
                   extra: dict | None = None) -> Path:
    path = ART_DIR / f"hp_{name}_best.json"
    payload = {
        "name": name,
        "best_params": best_params,
        "best_value_reweight_bal_acc": float(best_value),
        "baseline_reweight_bal_acc": float(baseline_value),
        "delta_vs_baseline": float(best_value - baseline_value),
        "accepted": bool(best_value - baseline_value > 1e-3),  # > 1 fold-std
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))
    return path
