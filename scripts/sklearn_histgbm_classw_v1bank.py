"""Tier B2: sklearn HistGradientBoostingClassifier meta on v1 natural-cal bank.

Different inductive bias than RF (gradient-boosted level-wise vs
bagged). Same v1 7-component bank for clean LB comparison vs RF v1
(LB 0.98129).

HPs mirror rawashishsin's natural-cal regime:
  learning_rate=0.05, max_depth=3, max_iter=2600, l2_regularization=0,
  early stopping on val log loss.

Hypothesis: Hist-GBM on naturally-calibrated inputs may compound
gradient signal differently than RF's averaging. Different error
geometry could open blend opportunities RF v1 misses.

Outputs:
  scripts/artifacts/oof_sklearn_histgbm_classw_v1bank.npy
  scripts/artifacts/test_sklearn_histgbm_classw_v1bank.npy
  scripts/artifacts/sklearn_histgbm_classw_v1bank_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42

SMOKE = os.environ.get("SMOKE") == "1"

# v1 7-component natural-cal bank (LB-validated, produced LB 0.98129)
V1_BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
]

META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    log(f"loading v1 natural-cal bank ({len(V1_BANK)} components)")
    pool = {}
    for name in V1_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  SKIP {name}: missing")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.shape[0] != n_tr or o.ndim != 2 or o.shape[1] != 3:
            log(f"  SKIP {name}: shape {o.shape}")
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    log(f"  loaded {len(pool)}/{len(V1_BANK)} components")

    log("constructing distance / rule meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  feature matrix: train={X_tr.shape}  test={X_te.shape}")

    # No StandardScaler needed for tree-based GBM
    n_iter = 100 if SMOKE else 1000
    max_depth = 3
    n_folds = 2 if SMOKE else 5
    hgb_params = dict(
        learning_rate=0.05,
        max_iter=n_iter,
        max_depth=max_depth,
        l2_regularization=0.0,         # natural-cal regime
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=SEED,
        class_weight="balanced", # B4 variant: explicit class_weight
        verbose=0,
    )
    log(f"HistGBM (B4 classw=balanced): max_iter={n_iter} max_depth={max_depth} "
        f"lr=0.05 l2_reg=0 class_weight=balanced")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr, y))
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"=== fold {fold}/{n_folds}  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        hgb = HistGradientBoostingClassifier(**hgb_params)
        hgb.fit(X_tr[tr_idx], y[tr_idx])
        p_va = hgb.predict_proba(X_tr[va_idx]).astype(np.float32)
        p_te = hgb.predict_proba(X_te).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  n_iter_={hgb.n_iter_}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    drift = (bias - (-np.log(prior))).round(3).tolist()
    log(f"  bias drift from -log(prior): {drift}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    np.save(ART / "oof_sklearn_histgbm_classw_v1bank.npy", oof)
    np.save(ART / "test_sklearn_histgbm_classw_v1bank.npy", test_pred)

    summary = dict(
        n_folds=n_folds, smoke=SMOKE, seed=SEED,
        max_iter=n_iter, max_depth=max_depth,
        bank_version="v1_7components",
        bank=V1_BANK, bank_loaded=sorted(pool.keys()),
        feature_count=X_tr.shape[1],
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        bias_drift=drift,
        per_class_recall=pcr.tolist(),
    )
    with open(ART / "sklearn_histgbm_classw_v1bank_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {ART}/sklearn_histgbm_classw_v1bank_results.json")

    eps = 1e-9
    test_log = safelog(test_pred)
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub_path = SUB / "submission_sklearn_histgbm_classw_v1bank_standalone.csv"
    sub.to_csv(sub_path, index=False)
    log(f"  wrote {sub_path}")


if __name__ == "__main__":
    main()
