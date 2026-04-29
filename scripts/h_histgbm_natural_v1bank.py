"""H_histgbm — HistGradientBoosting with class_weight=None on v1's bank.

Different L2 architecture entirely: gradient-boosted (sequential) vs
RF/ET bagging (parallel). Different decision-boundary geometry.

CLAUDE.md SMOKE 2026-04-29 tested HistGBM with class_weight='balanced'
and confirmed it BREAKS natural calibration (drift |1.4| vs |0.20|).
The `class_weight=None` config has NOT been tested at production.

If this produces orthogonal errors AND magnitude-comparable to v1,
it's a candidate for the combined blend gate.
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
from common import add_distance_features, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"

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
def safelog(p, eps=1e-9): return np.log(np.clip(p, eps, 1.0))
def _normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum() > 0:
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = {}
    for name in V1_BANK:
        oof = _normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        tt = _normed(np.load(ART / f"test_{name}.npy").astype(np.float32))
        pool[name] = (oof, tt)
        log(f"  + {name}")

    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in names]
    log_te = [safelog(pool[n][1]) for n in names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    log(f"feature matrix: train={X_tr_s.shape}  test={X_te_s.shape}")

    n_iter = 200 if SMOKE else 1000
    n_folds = 2 if SMOKE else 5
    hgbm_params = dict(
        max_iter=n_iter, learning_rate=0.05,
        max_depth=8, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, n_iter_no_change=50,
        random_state=SEED, class_weight=None,  # natural-cal
        verbose=0,
    )
    log(f"HistGBM natural: max_iter={n_iter} class_weight=None lr=0.05 max_depth=8")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_s, y), 1):
        t0 = time.time()
        m = HistGradientBoostingClassifier(**hgbm_params)
        m.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = m.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = m.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold}/{n_folds} bal={bal:.5f} wall={time.time()-t0:.1f}s n_iter={m.n_iter_}")

    bal = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(_normed(oof), y, prior)
    pcr = per_class_recall(y, (safelog(_normed(oof)) + bias).argmax(1))
    log(f"=== HistGBM argmax={bal:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    np.save(ART / "oof_h_histgbm_natural.npy", _normed(oof))
    np.save(ART / "test_h_histgbm_natural.npy", _normed(test_pred))

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)

    new_pred = (safelog(_normed(test_pred)) + bias).argmax(1)
    v1_pred = (safelog(v1_test) + v1_bias).argmax(1)
    diff = int((new_pred != v1_pred).sum())
    log(f"test diff vs v1: {diff} / {n_te}")

    summary = dict(smoke=SMOKE, n_iter=n_iter,
                   fold_scores=fold_scores,
                   argmax=float(bal), tuned=float(tuned), bias=bias.tolist(),
                   pcr=pcr.tolist(),
                   v1_tuned=float(v1_tuned),
                   delta_tuned_vs_v1=float(tuned - v1_tuned),
                   test_diff_vs_v1=diff)
    with open(ART / "h_histgbm_natural_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"wrote {ART}/h_histgbm_natural_results.json")

    sub_path = SUB / "submission_h_histgbm_natural_standalone.csv"
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in new_pred]})
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")


if __name__ == "__main__":
    main()
