"""N1 RF natural meta-stacker on v1's EXACT 7-component bank.

Direct port of sklearn_rf_meta_natural.py but with the bank
hardcoded to v1's LB-validated composition (the 7 components loaded
when v1 LB 0.98129 was produced; commit bbddebb).

Usage: rawashishsin_2600 OOF/test on disk are interpreted as the bag
input. Merge script overwrites those files before running this; this
script doesn't know whether single-seed or bag5 is in there.

Output suffix configurable via env META_SUFFIX (default '_v1bank_bag5').
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, tune_log_bias

ART = Path("scripts/artifacts")
SUFFIX = os.environ.get("META_SUFFIX", "_v1bank_bag5")
SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}

# v1's exact 7-component bank (commit bbddebb, LB 0.98129)
BANK = [
    "rawashishsin_2600",
    "realmlp",
    "recipe_full_te",
    "recipe_full_te_catboost",
    "recipe_full_te_catboost_natural",
    "xgb_corn",
    "xgb_dist_digits",
]

META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
    prior = np.bincount(y) / len(y)

    log(f"loading v1's exact 7-component bank")
    pool = {}
    for name in BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  MISSING {name} — abort")
            return
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.shape != (n_tr, 3):
            log(f"  SKIP {name}: shape {o.shape}")
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    log(f"  loaded {len(pool)}/{len(BANK)} components")
    if len(pool) != 7:
        log(f"ERROR: expected 7 components, got {len(pool)}")
        return

    log("building features (META + per-component log-probs)")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]

    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    rf_params = dict(
        n_estimators=500, max_depth=12, min_samples_leaf=20,
        max_features="sqrt", bootstrap=True, n_jobs=-1,
        random_state=SEED, class_weight=None,
        verbose=0,
    )
    log(f"RF natural: n_est=500 max_depth=12 class_weight=None bootstrap=True")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y))
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / N_FOLDS
        bal = float(balanced_accuracy_score(y[va_idx], p_va.argmax(1)))
        fold_scores.append(bal)
        log(f"  fold {fold}: bal={bal:.5f}  wall={time.time()-t0:.1f}s")

    overall = float(balanced_accuracy_score(y, oof.argmax(1)))
    bias, tuned = tune_log_bias(oof, y, prior)
    drift = bias - (-np.log(prior))
    log(f"\n=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}")
    log(f"    bias = {bias.round(4).tolist()}  drift = {drift.round(3).tolist()}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"    PCR = [L={pcr[0]:.5f} M={pcr[1]:.5f} H={pcr[2]:.5f}]")

    np.save(ART / f"oof_sklearn_rf_meta_natural{SUFFIX}.npy", oof)
    np.save(ART / f"test_sklearn_rf_meta_natural{SUFFIX}.npy", test_pred)

    summary = dict(
        bank=BANK, bank_loaded=sorted(pool.keys()),
        fold_scores=fold_scores,
        overall_argmax=overall,
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        drift=drift.tolist(),
        drift_max=float(max(abs(d) for d in drift)),
        per_class_recall=pcr.tolist(),
    )
    out = ART / f"sklearn_rf_meta_natural{SUFFIX}_results.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
