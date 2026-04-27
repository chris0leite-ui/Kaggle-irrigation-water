"""B': sklearn RandomForest meta-stacker (cuML pivot after Kaggle P100 sm_60 block).

Uses the same 16-component bank that produced cuml_meta_input.npz. Trains
sklearn RandomForestClassifier with bootstrap=True (different from
ExtraTrees which is bagging-without-replacement on random feature subsets).
True bootstrap aggregation as the L2 architecture — UNTESTED as a meta
on this bank.

Mechanism: bagging-based meta vs the gradient-boosted XGB meta that
produced LB-best 0.98094. Different inductive bias:
  - XGB: sequential additive gradient steps, each tree fits residuals
  - RF: independent bootstrap samples, predictions averaged

Per-class bal_acc trade may differ enough to thread the 4-gate filter
that XGB-meta variants couldn't (LR, MLP, v3-v7, n5b, J2 all NULL).

Outputs:
  scripts/artifacts/oof_sklearn_rf_meta.npy
  scripts/artifacts/test_sklearn_rf_meta.npy
  scripts/artifacts/sklearn_rf_meta_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    npz_path = ART / "cuml_meta_input.npz"
    log(f"loading {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X_tr = data["X_tr"].astype(np.float32)
    X_te = data["X_te"].astype(np.float32)
    y = data["y"].astype(np.int32)
    fold_idx = data["fold_idx"].astype(np.int32)
    log(f"X_tr={X_tr.shape}  X_te={X_te.shape}  fold counts: {np.bincount(fold_idx)}")

    if SMOKE:
        # Subsample to first 50k for smoke (time-bounded)
        idx = np.arange(50_000)
        X_tr = X_tr[idx]; y = y[idx]; fold_idx = fold_idx[idx]
        # Re-bin: if fold_idx has only fold 1 in subset, use 2-fold split
        log(f"SMOKE subset: X_tr={X_tr.shape}  fold counts: {np.bincount(fold_idx, minlength=6)}")

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    n_est = 100 if SMOKE else 500
    max_depth = 8 if SMOKE else 14
    n_folds = 2 if SMOKE else 5
    rf_params = dict(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=SEED,
        class_weight="balanced", verbose=0,
    )
    log(f"RF params: n_est={n_est} max_depth={max_depth} class_weight=balanced bootstrap=True")

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_pred = np.zeros((len(X_te), 3), dtype=np.float32)
    fold_scores = []

    for fold in range(1, n_folds + 1):
        if SMOKE:
            # In smoke we have no proper fold assignment; do a simple 2-fold split
            mask_va = (np.arange(len(y)) % 2) == (fold - 1)
            mask_tr = ~mask_va
        else:
            mask_va = fold_idx == fold
            mask_tr = ~mask_va
        log(f"=== fold {fold}/{n_folds}  tr={int(mask_tr.sum()):,} va={int(mask_va.sum()):,} ===")
        t0 = time.time()
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_s[mask_tr], y[mask_tr])
        p_va = rf.predict_proba(X_tr_s[mask_va]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[mask_va] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[mask_va], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  wall={time.time()-t0:.1f}s")

        np.save(ART / f"oof_sklearn_rf_meta_fold{fold}.npy", p_va)
        np.save(ART / f"test_sklearn_rf_meta_fold{fold}.npy", p_te)

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    np.save(ART / "oof_sklearn_rf_meta.npy", oof)
    np.save(ART / "test_sklearn_rf_meta.npy", test_pred)

    summary = dict(
        n_folds=n_folds, smoke=SMOKE, seed=SEED,
        n_estimators=n_est, max_depth=max_depth,
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
    )
    with open(ART / "sklearn_rf_meta_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {ART}/sklearn_rf_meta_results.json")


if __name__ == "__main__":
    main()
