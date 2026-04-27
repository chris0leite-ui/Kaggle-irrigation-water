"""Path B — per-cell MLP on rule-cell partition with continuous features.

Mechanism: for each of the 128 rule-cells (stage × dry × norain × hot × windy ×
nomulch), fit a small sklearn MLPClassifier on the cell's continuous-feature
subset. Routes test rows to their cell's MLP. Cells with too few training
rows fall back to the rule's one-hot prior.

Distinct from prior nulls:
- 2026-04-21 per-cell LR used 7 non-rule features only and linear capacity.
- 2026-04-22 v6/v7 MLPs used 13 non-rule features as a global model.
- This: 15 features (11 raw nums + 4 signed dists), per-cell MLP capacity.

Outputs:
  oof_path_b_cell_mlp.npy
  test_path_b_cell_mlp.npy
  path_b_cell_mlp_results.json

SMOKE=1 → 50k subsample, 1 fold, ~3 min wall.
Production → 5-fold seed=42 on full 630k, ~30-45 min wall on 16-core CPU.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
RUN_FOLD = int(os.environ.get("RUN_FOLD", "-1"))  # -1 = all folds

CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ACTIVE_STAGES = ("Flowering", "Vegetative")
RAW_NUMS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
            "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
            "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare"]
STAGES = ["Vegetative", "Flowering", "Maturity", "Sowing", "Harvesting"]
MIN_CELL_TR = int(os.environ.get("MIN_CELL_TR", "200"))


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cell_id(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"].astype(float).values < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float).values < 300).astype(int)
    hot = (df["Temperature_C"].astype(float).values > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float).values > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(int)
    stage = pd.Categorical(df["Crop_Growth_Stage"].astype(str), categories=STAGES).codes
    stage = np.clip(stage, 0, 4).astype(int)
    return (stage * 32 + dry * 16 + norain * 8 + hot * 4 + windy * 2 + nomulch).astype(int)


def rule_pred(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"].astype(float).values < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float).values < 300).astype(int)
    hot = (df["Temperature_C"].astype(float).values > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float).values > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(int)
    kc = np.where(np.isin(df["Crop_Growth_Stage"].astype(str).values, ACTIVE_STAGES), 2, 0)
    s = 2 * (dry + norain) + (hot + windy + nomulch) + kc
    return np.where(s <= 3, 0, np.where(s <= 6, 1, 2)).astype(int)


def features(df: pd.DataFrame) -> np.ndarray:
    nums = df[RAW_NUMS].astype(np.float32).values
    sm_d = df["Soil_Moisture"].astype(float).values - 25.0
    rf_d = df["Rainfall_mm"].astype(float).values - 300.0
    tc_d = df["Temperature_C"].astype(float).values - 30.0
    ws_d = df["Wind_Speed_kmh"].astype(float).values - 10.0
    return np.hstack([nums, np.c_[sm_d, rf_d, tc_d, ws_d].astype(np.float32)])


def fit_predict_cell(X_tr, y_tr, X_va, X_te, smoke=False):
    if len(X_tr) < MIN_CELL_TR or len(np.unique(y_tr)) < 2:
        # Fall back to empirical class distribution of cell train rows.
        prior = np.bincount(y_tr, minlength=3).astype(np.float32) + 1.0
        prior /= prior.sum()
        p_va = np.tile(prior, (len(X_va), 1))
        p_te = np.tile(prior, (len(X_te), 1))
        return p_va, p_te
    sc = StandardScaler().fit(X_tr)
    Xtr_s = sc.transform(X_tr)
    Xva_s = sc.transform(X_va)
    Xte_s = sc.transform(X_te)
    hidden = (32, 16) if smoke else (64, 32)
    iters = 50 if smoke else 150
    # Disable early-stopping when any class has <2 rows (sklearn's stratified
    # internal split fails). Cells with imbalanced class counts are common.
    counts = np.bincount(y_tr, minlength=3)
    es = bool((counts[counts > 0] >= 2).all() and len(np.unique(y_tr)) >= 2 and len(X_tr) >= 50)
    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=iters,
                        early_stopping=es, validation_fraction=0.1 if es else 0.0,
                        n_iter_no_change=10, random_state=SEED, alpha=1e-3,
                        learning_rate_init=1e-3,
                        batch_size=min(256, max(32, len(X_tr) // 64)))
    mlp.fit(Xtr_s, y_tr)
    classes = mlp.classes_
    p_va = np.zeros((len(X_va), 3), dtype=np.float32)
    p_te = np.zeros((len(X_te), 3), dtype=np.float32)
    pv = mlp.predict_proba(Xva_s)
    pt = mlp.predict_proba(Xte_s)
    for i, c in enumerate(classes):
        p_va[:, c] = pv[:, i]
        p_te[:, c] = pt[:, i]
    return p_va, p_te


def main():
    log(f"SMOKE={SMOKE}  RUN_FOLD={RUN_FOLD}")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if SMOKE:
        rng = np.random.default_rng(SEED)
        train = train.sample(50_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(20_000, random_state=SEED).reset_index(drop=True)

    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)
    log(f"train={len(train)}  test={len(test)}  prior={np.bincount(y) / len(y)}")

    cell_tr = cell_id(train)
    cell_te = cell_id(test)
    X_tr = features(train)
    X_te = features(test)
    log(f"X_tr {X_tr.shape}  X_te {X_te.shape}  unique cells train={len(set(cell_tr))} test={len(set(cell_te))}")

    n_folds = 1 if SMOKE else N_FOLDS
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_acc = np.zeros((len(test), 3), dtype=np.float32)
    test_count = 0

    for fi, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), y)):
        if RUN_FOLD >= 0 and fi != RUN_FOLD:
            continue
        if SMOKE and fi >= 1:
            break
        log(f"=== fold {fi+1}/{n_folds}  tr={len(tr_idx)}  va={len(va_idx)} ===")
        cells = sorted(set(cell_tr[tr_idx]) | set(cell_tr[va_idx]) | set(cell_te))
        n_fb = 0
        for c in cells:
            tr_mask = (cell_tr[tr_idx] == c)
            va_mask = (cell_tr[va_idx] == c)
            te_mask = (cell_te == c)
            if not (tr_mask.any() and (va_mask.any() or te_mask.any())):
                continue
            X_t = X_tr[tr_idx[tr_mask]]
            y_t = y[tr_idx[tr_mask]]
            X_v = X_tr[va_idx[va_mask]] if va_mask.any() else np.zeros((0, X_tr.shape[1]))
            X_e = X_te[te_mask] if te_mask.any() else np.zeros((0, X_te.shape[1]))
            p_v, p_e = fit_predict_cell(X_t, y_t, X_v, X_e, smoke=SMOKE)
            if len(X_t) < MIN_CELL_TR:
                n_fb += 1
            if va_mask.any():
                oof[va_idx[va_mask]] = p_v
            if te_mask.any():
                test_acc[te_mask] += p_e
        test_count += 1
        log(f"  fold done  fallback_cells={n_fb}/{len(cells)}")
        if RUN_FOLD >= 0:
            np.save(ART / f"oof_path_b_cell_mlp_fold{fi}.npy", oof.astype(np.float32))
            np.save(ART / f"test_path_b_cell_mlp_fold{fi}.npy", test_acc.astype(np.float32))

    test_avg = test_acc / max(test_count, 1)
    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_path_b_cell_mlp{suffix}.npy", oof.astype(np.float32))
    np.save(ART / f"test_path_b_cell_mlp{suffix}.npy", test_avg.astype(np.float32))

    # Tune log-bias on the OOF rows actually filled.
    if RUN_FOLD < 0 or RUN_FOLD == 0:
        from sklearn.metrics import balanced_accuracy_score
        prior = np.bincount(y, minlength=3) / len(y)
        filled = oof.sum(1) > 1e-6
        if filled.sum() == 0:
            log("no rows filled — skipping bias tune"); return
        bias, bal = tune_log_bias(oof[filled], y[filled], prior)
        log(f"argmax bal={balanced_accuracy_score(y[filled], oof[filled].argmax(1)):.5f}  tuned={bal:.5f}  bias={bias.tolist()}")
        res = {"smoke": SMOKE, "n_folds_run": int(test_count), "log_bias": bias.tolist(),
               "tuned_log_bias_bal_acc": float(bal),
               "n_filled": int(filled.sum()), "min_cell_tr": MIN_CELL_TR}
        (ART / f"path_b_cell_mlp{suffix}_results.json").write_text(json.dumps(res, indent=2))
    log("done")


if __name__ == "__main__":
    main()
