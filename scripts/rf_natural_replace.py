"""REPLACE-variant RF natural meta-stacker.

Same architecture as sklearn_rf_meta_natural.py (the LB-best 0.98129 producer)
but parameterized to swap one component for another at constant 7-component
bank size. Tests whether REPLACE (different failure mode from ADD) breaks
the bank-extension regression pattern that closed a1lgbm and v2.

Env vars:
  REPLACE_OLD: component name to remove (e.g. "realmlp")
  REPLACE_NEW: component name to add (e.g. "a2_natural_calib")
  VARIANT_TAG: filename suffix (e.g. "Va", "Vb")

Outputs:
  scripts/artifacts/oof_rf_natural_replace_<VARIANT_TAG>.npy
  scripts/artifacts/test_rf_natural_replace_<VARIANT_TAG>.npy
  scripts/artifacts/rf_natural_replace_<VARIANT_TAG>_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

REPLACE_OLD = os.environ.get("REPLACE_OLD", "realmlp")
REPLACE_NEW = os.environ.get("REPLACE_NEW", "a2_natural_calib")
VARIANT_TAG = os.environ.get("VARIANT_TAG", "Va")
SMOKE = os.environ.get("SMOKE") == "1"

# v1 LB-best 0.98129 bank (7 components). Confirmed from CLAUDE.md.
V1_BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
]

if REPLACE_OLD not in V1_BANK:
    raise SystemExit(f"REPLACE_OLD={REPLACE_OLD} not in V1_BANK")
VARIANT_BANK = [REPLACE_NEW if c == REPLACE_OLD else c for c in V1_BANK]

META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] [{VARIANT_TAG}] {m}", flush=True)


def safelog(p, eps=1e-9): return np.log(np.clip(p, eps, 1.0))


def _normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum(): rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def load_bank(n_tr, n_te):
    log(f"loading variant bank ({len(VARIANT_BANK)} components)")
    log(f"  REPLACE: {REPLACE_OLD} -> {REPLACE_NEW}")
    pool = {}
    for name in VARIANT_BANK:
        op = ART / f"oof_{name}.npy"
        tp = ART / f"test_{name}.npy"
        if not (op.exists() and tp.exists()):
            log(f"  MISSING: {name}")
            continue
        o = np.load(op).astype(np.float32)
        t = np.load(tp).astype(np.float32)
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != n_tr:
            log(f"  SKIP {name}: shape {o.shape}")
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    log(f"  loaded {len(pool)}/{len(VARIANT_BANK)}")
    return pool


def build_features(pool, train, test):
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)
    names = sorted(pool.keys())
    feature_names = list(META_COLS)
    for n in names: feature_names += [f"{n}_logL", f"{n}_logM", f"{n}_logH"]
    log_tr = [safelog(pool[n][0]) for n in names]
    log_te = [safelog(pool[n][1]) for n in names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")
    return X_tr, X_te, feature_names


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = load_bank(n_tr, n_te)
    if len(pool) < len(VARIANT_BANK):
        log("ERROR: not all components loaded — abort to keep apples-apples")
        return

    X_tr, X_te, _ = build_features(pool, train, test)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    n_est = 100 if SMOKE else 500
    max_depth = 8 if SMOKE else 12
    n_folds = 2 if SMOKE else 5
    rf_params = dict(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=SEED,
        class_weight=None, verbose=0,
    )
    log(f"RF: n_est={n_est} max_depth={max_depth} class_weight=None")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_s, y), 1):
        t0 = time.time()
        log(f"=== fold {fold}/{n_folds}  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax={bal:.5f}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"  PCR = [L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # Drift from -log(prior) (natural-cal diagnostic)
    log_prior = -np.log(prior)
    drift = bias - log_prior
    log(f"  drift = {drift.round(3).tolist()}  max_abs = {np.abs(drift).max():.3f}")

    out_oof = ART / f"oof_rf_natural_replace_{VARIANT_TAG}.npy"
    out_test = ART / f"test_rf_natural_replace_{VARIANT_TAG}.npy"
    out_json = ART / f"rf_natural_replace_{VARIANT_TAG}_results.json"
    np.save(out_oof, oof)
    np.save(out_test, test_pred)

    summary = dict(
        variant_tag=VARIANT_TAG,
        replace_old=REPLACE_OLD, replace_new=REPLACE_NEW,
        bank=VARIANT_BANK, n_components=len(VARIANT_BANK),
        n_folds=n_folds, smoke=SMOKE, seed=SEED,
        n_estimators=n_est, max_depth=max_depth,
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        bias_drift=drift.tolist(),
        bias_drift_max_abs=float(np.abs(drift).max()),
        per_class_recall=pcr.tolist(),
    )
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {out_oof.name}, {out_test.name}, {out_json.name}")


if __name__ == "__main__":
    main()
