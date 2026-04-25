"""N2 — ExtraTrees diversity-leg for the meta-stacker bank.

Trains ExtraTreesClassifier on the 43-feature dist set (same as
xgb_nonrule), 5-fold StratifiedKFold(seed=42). Different model class
than every existing bank component (gradient-boosted tree, ordered
boosting, NN family). Random-feature splits give different decision
geometry than greedy-split XGB.

Why N2: wguesdon's published 30-model bank includes `et_ote` as a
diversity weak-learner. We've ruled out ET as a direct blend leg
but NEVER as a meta-stacker bank input.

Output: oof_n2_extratrees.npy + test_n2_extratrees.npy + JSON.
SMOKE=1 → 1 fold, smaller n_estimators.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, tune_log_bias  # noqa: E402

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)

# 11 raw numerics + 24 derived from add_distance_features() = 35
DIST_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
    "dry", "norain", "hot", "windy", "nomulch", "kc_active",
    "dgp_score", "rule_pred",
    "sm_dist", "rf_dist", "tc_dist", "ws_dist",
    "sm_abs", "rf_abs", "tc_abs", "ws_abs",
    "min_boundary_dist", "min_axis_abs",
    "score_dist_low_mid", "score_dist_mid_high",
    "sm_x_kc", "sm_x_rf", "rf_x_kc", "tc_x_ws",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    log(f"N2 ExtraTrees on 43-dist feature set — N_FOLDS={N_FOLDS}, SMOKE={SMOKE}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train[TARGET] = train[TARGET].map(CLS_MAP)
    test_ids = test["id"].values

    if SMOKE:
        log("SMOKE=1 — subsampling to 20k train, 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)

    log("computing dist features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    feat_cols = [c for c in DIST_COLS if c in tr_d.columns]
    log(f"  feat_cols={len(feat_cols)} (expected ~35)")

    X_tr_full = tr_d[feat_cols].to_numpy(dtype=np.float32)
    X_te = te_d[feat_cols].to_numpy(dtype=np.float32)
    y = train[TARGET].to_numpy().astype(np.int32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    n_est = 50 if SMOKE else 500
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_full, y), 1):
        t_fold = time.time()
        log(f"=== fold {fold}/{N_FOLDS} ===")
        et = ExtraTreesClassifier(
            n_estimators=n_est,
            max_depth=None, min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1, random_state=SEED, bootstrap=False,
        )
        et.fit(X_tr_full[tr_idx], y[tr_idx])
        oof[va_idx] = et.predict_proba(X_tr_full[va_idx]).astype(np.float32)
        test_pred += et.predict_proba(X_te).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  wall={time.time()-t_fold:.1f}s")
        # Per-fold checkpoint
        np.save(ART / f"oof_n2_extratrees_fold{fold}.npy", oof)
        np.save(ART / f"test_n2_extratrees_fold{fold}.npy", test_pred)

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / "oof_n2_extratrees.npy", oof)
    np.save(ART / "test_n2_extratrees.npy", test_pred)

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, n_features=len(feat_cols),
        n_estimators=n_est,
        fold_scores_argmax=[float(s) for s in fold_scores],
        overall_argmax_bal_acc=float(overall),
        tuned_log_bias_bal_acc=float(tuned),
        log_bias=bias.tolist(),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "n2_extratrees_results.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote oof_n2_extratrees.npy + test + JSON  total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
