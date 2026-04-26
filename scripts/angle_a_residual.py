"""Angle A — residual-correction model targeting LB-best primary's OOF errors.

Mechanism:
  primary = lb_best_3stack ⊗ xgb_metastack_iso α=0.30  (LB 0.98094 / OOF 0.98084)
  target  = one_hot(y) - primary_softprob              (sum-to-zero per row)
  model   = XGBRegressor (one head per class) on dist-features + primary probs
  apply   = primary_softprob + α · residual_pred       (α small, e.g. 0.05-0.15)

Why this evades the magnitude trap:
  - residual_pred bounded by α; can't add wrong-direction errors at scale
  - learns *where* primary is wrong as a function of features (different
    objective than every meta-stacker, all of which predict y)

5-fold StratifiedKFold(seed=42) aligned with every saved OOF.
SMOKE=1 → 50k subsample, 2 folds, fewer rounds.

Outputs:
  scripts/artifacts/oof_angle_a_residual.npy        (post-correction probs)
  scripts/artifacts/test_angle_a_residual.npy
  scripts/artifacts/oof_angle_a_residual_raw.npy    (raw residual model output)
  scripts/artifacts/test_angle_a_residual_raw.npy
  scripts/artifacts/angle_a_residual_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, tune_log_bias  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, SEED, N_FOLDS, build_lbbest_stack, iso_cal, load_y,
    log, normed,
)

SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS_LOCAL = 2 if SMOKE else N_FOLDS

DIST_FEATS = [
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "sm_dist", "rf_dist", "tc_dist", "ws_dist",
    "sm_abs", "rf_abs", "tc_abs", "ws_abs",
    "dry", "norain", "hot", "windy", "nomulch", "kc_active",
    "dgp_score", "rule_pred",
    "score_dist_low_mid", "score_dist_mid_high",
    "min_boundary_dist", "min_axis_abs",
    "sm_x_rf", "tc_x_ws", "sm_x_kc", "rf_x_kc",
]


def build_primary(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LB-best 0.98094 = lb_best_3stack ⊗ xgb_metastack_iso α=0.30 (log-blend)."""
    s_o, s_t = build_lbbest_stack(y)  # 3-stack (lb3 + RealMLP + nonrule_iso)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o, meta_t = iso_cal(meta_o, meta_t, y)
    a = 0.30
    eps = 1e-12
    log_p = (1 - a) * np.log(np.clip(s_o, eps, 1)) + a * np.log(np.clip(meta_o, eps, 1))
    log_t = (1 - a) * np.log(np.clip(s_t, eps, 1)) + a * np.log(np.clip(meta_t, eps, 1))
    p_o = np.exp(log_p - log_p.max(1, keepdims=True))
    p_t = np.exp(log_t - log_t.max(1, keepdims=True))
    return normed(p_o.astype(np.float32)), normed(p_t.astype(np.float32))


def build_features(df: pd.DataFrame, primary: np.ndarray) -> np.ndarray:
    eng = add_distance_features(df)
    X = eng[DIST_FEATS].astype(np.float32).to_numpy()
    extras = np.column_stack([
        primary,
        primary.max(1, keepdims=True).astype(np.float32),
        (-(primary * np.log(np.clip(primary, 1e-9, 1))).sum(1, keepdims=True)).astype(np.float32),
    ])
    return np.concatenate([X, extras], axis=1).astype(np.float32)


def main():
    t0 = time.time()
    log(f"angle A residual-correction. SMOKE={SMOKE}")
    y = load_y()
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_ids = test_df["id"].values

    log("reconstructing LB-best primary (OOF + test)")
    p_o, p_t = build_primary(y)
    base = balanced_accuracy_score(y, (np.log(np.clip(p_o, 1e-12, 1)) + BIAS).argmax(1))
    log(f"  primary OOF tuned bal_acc @ recipe bias = {base:.5f}")

    log("building residual targets + features")
    onehot = np.eye(3, dtype=np.float32)[y]
    resid_target = onehot - p_o  # shape (N, 3)
    X_tr_full = build_features(train_df, p_o)
    X_te = build_features(test_df, p_t)
    log(f"  features: {X_tr_full.shape[1]} cols, train={X_tr_full.shape[0]:,} test={X_te.shape[0]:,}")

    if SMOKE:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(y), size=50_000, replace=False)
        idx.sort()
        y = y[idx]
        p_o = p_o[idx]
        resid_target = resid_target[idx]
        X_tr_full = X_tr_full[idx]
        log(f"  SMOKE subsample: {len(y):,}")

    skf = StratifiedKFold(n_splits=N_FOLDS_LOCAL, shuffle=True, random_state=SEED)
    oof_resid = np.zeros((len(y), 3), dtype=np.float32)
    test_resid = np.zeros((len(test_df), 3), dtype=np.float32)
    fold_scores = []
    xgb_params = dict(
        n_estimators=200 if SMOKE else 1500,
        max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2, reg_lambda=2,
        tree_method="hist", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=30 if SMOKE else 100, verbosity=0,
    )
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_full, y), 1):
        log(f"=== fold {fold}/{N_FOLDS_LOCAL} ===")
        for c in range(3):
            m = xgb.XGBRegressor(**xgb_params, objective="reg:squarederror")
            m.fit(X_tr_full[tr_idx], resid_target[tr_idx, c],
                  eval_set=[(X_tr_full[va_idx], resid_target[va_idx, c])], verbose=False)
            oof_resid[va_idx, c] = m.predict(X_tr_full[va_idx]).astype(np.float32)
            test_resid[:, c] += m.predict(X_te).astype(np.float32) / N_FOLDS_LOCAL
        # quick fold diag: score at α=0.10
        p_corr = normed(np.maximum(p_o + 0.10 * oof_resid, 1e-9))
        bal = balanced_accuracy_score(y[va_idx], (np.log(np.clip(p_corr[va_idx], 1e-12, 1)) + BIAS).argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} α=0.10 corrected bal_acc = {bal:.5f}")

    np.save(ART / "oof_angle_a_residual_raw.npy", oof_resid)
    np.save(ART / "test_angle_a_residual_raw.npy", test_resid)
    log(f"  raw residual saved")

    # α-sweep at fixed BIAS
    sweep = {}
    best_alpha, best_bal = 0.0, base
    for a in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]:
        p_corr = normed(np.maximum(p_o + a * oof_resid, 1e-9))
        bal = balanced_accuracy_score(y, (np.log(np.clip(p_corr, 1e-12, 1)) + BIAS).argmax(1))
        sweep[f"alpha_{a:.3f}"] = float(bal)
        if bal > best_bal:
            best_alpha, best_bal = a, bal
    log(f"  best α={best_alpha:.3f} bal_acc={best_bal:.5f} Δ={best_bal-base:+.5f}")

    # save final corrected probs at best α
    p_corr_test = normed(np.maximum(p_t + best_alpha * test_resid, 1e-9))
    p_corr_oof = normed(np.maximum(p_o + best_alpha * oof_resid, 1e-9))
    np.save(ART / "oof_angle_a_residual.npy", p_corr_oof)
    np.save(ART / "test_angle_a_residual.npy", p_corr_test)

    out = dict(
        smoke=SMOKE, n_folds=N_FOLDS_LOCAL, n_features=X_tr_full.shape[1],
        primary_base_tuned=float(base),
        fold_scores_alpha010=[float(s) for s in fold_scores],
        alpha_sweep_oof=sweep,
        best_alpha=float(best_alpha), best_oof=float(best_bal),
        wall_min=(time.time() - t0) / 60.0,
    )
    out_path = ART / "angle_a_residual_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {out_path}  wall={out['wall_min']:.1f} min")


if __name__ == "__main__":
    main()
