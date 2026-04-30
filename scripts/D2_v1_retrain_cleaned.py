"""D2 — retrain v1 RF natural on cleanlab-cleaned labels.

Two variants:
  D1 (drop):    remove 1631 cleanlab-flagged rows; retrain v1 on ~628.4k
  D2 (relabel): replace flagged rows' y with DGP rule_pred; retrain on full 630k

v1 = sklearn RF meta-stacker on 11-component natural-cal bank +
distance/rule meta features. Mirror sklearn_rf_meta_natural.py params:
n_est=500, max_depth=12, class_weight=None, bootstrap=True.

Mode via env: D_MODE=drop|relabel. Smoke via SMOKE=1.
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
from common import add_distance_features, tune_log_bias  # noqa: E402
from sklearn_rf_meta_natural import (  # noqa: E402
    NATURAL_BANK,
    META_COLS,
    safelog,
    _normed,
    load_bank,
    build_features,
    log,
)
from dgp_formula import dgp_score  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
SEED = 42

D_MODE = os.environ.get("D_MODE", "drop")  # 'drop' or 'relabel'
SMOKE = os.environ.get("SMOKE") == "1"

if D_MODE not in ("drop", "relabel"):
    raise ValueError(f"D_MODE must be 'drop' or 'relabel', got {D_MODE}")


def main():
    log(f"=== D retrain v1 mode={D_MODE} SMOKE={SMOKE} ===")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y_orig = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    n_tr, n_te = len(train), len(test)

    flag = np.load(ART / "D_v1_label_issues.npy")
    log(f"  flagged: {flag.sum()} / {n_tr}")

    # rule_pred for relabel
    score = dgp_score(train).astype(np.int16)
    rule_pred = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int32)

    if D_MODE == "drop":
        keep_mask = ~flag
        log(f"  drop: keeping {keep_mask.sum()} rows")
    elif D_MODE == "relabel":
        keep_mask = np.ones(n_tr, dtype=bool)
        y_orig[flag] = rule_pred[flag]
        log(f"  relabel: replaced y on {flag.sum()} flagged rows with rule_pred")
        log(f"    new y dist: L={(y_orig==0).sum()} M={(y_orig==1).sum()} H={(y_orig==2).sum()}")

    # Bank load
    pool = load_bank(y_orig, n_tr, n_te)
    if "rawashishsin_2600" not in pool:
        log("ERROR: rawashishsin_2600 missing — abort")
        return

    X_tr, X_te, _ = build_features(pool, train, test, y_orig)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    # Apply keep_mask AFTER feature construction (so component OOF is unchanged)
    X_tr_use = X_tr_s[keep_mask]
    y_use = y_orig[keep_mask]
    log(f"  X_tr_use {X_tr_use.shape}  y_use len {len(y_use)}")

    n_est = 100 if SMOKE else 500
    max_depth = 8 if SMOKE else 12
    n_folds = 2 if SMOKE else 5
    rf_params = dict(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=SEED,
        class_weight=None, verbose=0,
    )
    log(f"  RF: n_est={n_est} max_depth={max_depth} n_folds={n_folds}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_use, y_use))
    oof_kept = np.zeros((len(y_use), 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"  fold {fold}/{n_folds} tr={len(tr_idx):,} va={len(va_idx):,}")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_use[tr_idx], y_use[tr_idx])
        p_va = rf.predict_proba(X_tr_use[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof_kept[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y_use[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"    fold {fold} argmax_bal_acc={bal:.5f}  wall={time.time()-t0:.1f}s")

    # For OOF saved at full length, we need to map back
    oof_full = np.zeros((n_tr, 3), dtype=np.float32)
    if D_MODE == "drop":
        # Flagged rows have no OOF (they were dropped); use v1's OOF as a placeholder
        v1 = _normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
        oof_full[:] = v1
        oof_full[keep_mask] = oof_kept
    else:
        # All rows are kept in relabel mode
        oof_full[keep_mask] = oof_kept

    # Use ORIGINAL labels (not relabeled) for evaluation — what the host scores us against
    y_true = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    overall = balanced_accuracy_score(y_true, oof_full.argmax(1))
    prior = np.bincount(y_true, minlength=3) / len(y_true)
    bias, tuned = tune_log_bias(oof_full, y_true, prior)
    log(f"\nOOF argmax (original y) = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    out_oof = ART / f"oof_v1_D_{D_MODE}.npy"
    out_test = ART / f"test_v1_D_{D_MODE}.npy"
    np.save(out_oof, oof_full)
    np.save(out_test, test_pred)
    log(f"saved {out_oof} {out_test}")

    summary = {
        "mode": D_MODE,
        "smoke": SMOKE,
        "n_kept": int(keep_mask.sum()),
        "n_flagged": int(flag.sum()),
        "fold_scores_argmax": fold_scores,
        "overall_argmax_originaly": float(overall),
        "tuned_macro_originaly": float(tuned),
        "log_bias": bias.tolist(),
    }
    out = ART / f"D_v1_retrain_{D_MODE}_results.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"saved {out}")


if __name__ == "__main__":
    main()
