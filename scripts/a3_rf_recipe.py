"""A3 — Naturally-calibrated Random Forest base on the full V10 recipe FE.

Companion to A2 (sklearn TargetEncoder + depth-3 XGB without
class-balanced weights). Tests whether *bagging* — not sklearn TE — is
the structural source of natural calibration.

Hypothesis (per 2026-04-28 calibration analysis): rawashishsin's bias=0
on High came from a training regime that lets raw probs settle at the
macro-recall optimum (no class_weight, low depth, no L1/L2 reg, smoothed
TE). RF naturally averages bootstrap predictions → smoother per-class
probability estimates → potentially calibrated raw output.

Design:
  * Reuse `recipe_full_te.load_and_engineer()` — full V10 FE (443 cols
    incl. OTE + digits + FREQ + ORIG_stats + threshold flags + LR logits).
  * Per-fold OrderedTE fit on tr_idx (matches recipe_full_te.py exactly).
  * RandomForestClassifier WITHOUT class_weight, bootstrap=True,
    max_features='sqrt'. The "natural-calibration test": raw probs
    should center at the macro-recall optimum.
  * Memory-bounded: max_depth=22 + min_samples_leaf=50 + n_jobs=4.
  * Per-fold checkpointing for rehydrate resilience.

Diagnostic gates:
  1. Tuned log-bias on High should be small (|bias_H| < 1.5) if natural
     calibration works. Compare to recipe XGB's bias_H = +3.40.
  2. Tuned OOF should be ≥ 0.96 for blend-gate consideration.
  3. Standalone errs ≤ 1.05 × LB-best 4-stack anchor for blend viability.

SMOKE=1 → 20k train × 2 folds × 50 trees, ~30s wall.
Production → 504k × 5 folds × 200 trees, ~60-90 min wall.

Outputs:
  scripts/artifacts/oof_a3_rf_recipe.npy
  scripts/artifacts/test_a3_rf_recipe.npy
  scripts/artifacts/a3_rf_recipe_results.json
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

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_full_te import load_and_engineer  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
TARGET = "Irrigation_Need"
ART = Path("scripts/artifacts"); ART.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
N_EST = 50 if SMOKE else int(os.environ.get("N_EST", "200"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "22"))
MIN_LEAF = int(os.environ.get("MIN_LEAF", "50"))
N_JOBS = int(os.environ.get("N_JOBS", "4"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    log(f"A3 RF recipe — N_FOLDS={N_FOLDS}  N_EST={N_EST}  "
        f"MAX_DEPTH={MAX_DEPTH}  MIN_LEAF={MIN_LEAF}  SMOKE={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy().astype(np.int32)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    log(f"numeric_feats={len(numeric_feats)}  te_cols={len(info['te_cols'])}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    fold_walls = []
    ck_prefix = "a3_rf_recipe"

    cached = set()
    for f in range(1, N_FOLDS + 1):
        if (ART / f"oof_{ck_prefix}_fold{f}.npy").exists() and \
           (ART / f"test_{ck_prefix}_fold{f}.npy").exists():
            cached.add(f)
    if cached:
        log(f"resume: {len(cached)} cached folds: {sorted(cached)}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        t_fold = time.time()
        if fold in cached:
            vp = np.load(ART / f"oof_{ck_prefix}_fold{fold}.npy")
            tp = np.load(ART / f"test_{ck_prefix}_fold{fold}.npy")
            oof[va_idx] = vp; test_pred += tp / N_FOLDS
            bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
            fold_scores.append(bal)
            log(f"  fold {fold} CACHED bal={bal:.5f}")
            continue

        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log("  fitting OrderedTE")
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr = te.fit(X_tr, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)

        feat_cols = numeric_feats + te.te_col_names()
        Xtr = X_tr[feat_cols].to_numpy(dtype=np.float32)
        Xva = X_va[feat_cols].to_numpy(dtype=np.float32)
        Xte = X_te[feat_cols].to_numpy(dtype=np.float32)
        del X_tr, X_va, X_te

        log(f"  fitting RandomForest n_est={N_EST}  features={len(feat_cols)}  "
            f"rows={len(Xtr):,}")
        rf = RandomForestClassifier(
            n_estimators=N_EST,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_LEAF,
            max_features="sqrt",
            bootstrap=True,
            class_weight=None,           # natural-calibration test (key knob)
            n_jobs=N_JOBS,
            random_state=SEED,
            verbose=0,
        )
        rf.fit(Xtr, y[tr_idx])
        vp = rf.predict_proba(Xva).astype(np.float32)
        tp = rf.predict_proba(Xte).astype(np.float32)
        oof[va_idx] = vp; test_pred += tp / N_FOLDS

        np.save(ART / f"oof_{ck_prefix}_fold{fold}.npy", vp)
        np.save(ART / f"test_{ck_prefix}_fold{fold}.npy", tp)
        bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_scores.append(bal); fold_walls.append(time.time() - t_fold)
        log(f"  fold {fold} bal={bal:.5f}  wall={fold_walls[-1]:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"OOF argmax={overall:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")

    # Calibration diagnostic: rawashishsin v3's bias=[-1.36,-1.19, 0.00] (LB 0.98109).
    # Recipe XGB v1's bias=[+1.43,+1.47,+3.40] (LB-validated PRIMARY 0.98094).
    nat_cal_score = float(abs(bias[2]))  # |bias_H| -- 0 is best
    log(f"natural-calibration diagnostic: |bias_H|={nat_cal_score:.3f}  "
        f"(0=naturally-calibrated, ≥3=miscalibrated like recipe XGB)")

    np.save(ART / "oof_a3_rf_recipe.npy", oof)
    np.save(ART / "test_a3_rf_recipe.npy", test_pred)
    summary = dict(
        seed=SEED, n_folds=N_FOLDS, n_estimators=N_EST,
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, n_jobs=N_JOBS,
        n_features=len(numeric_feats) + len(info["te_cols"]) * 3,
        smoke=SMOKE,
        fold_scores_argmax=[float(s) for s in fold_scores],
        fold_walls_sec=[float(w) for w in fold_walls],
        overall_argmax_bal_acc=float(overall),
        tuned_log_bias_bal_acc=float(tuned),
        log_bias=[float(b) for b in bias],
        natural_calibration_score=nat_cal_score,
        elapsed_sec=float(time.time() - t0),
    )
    out = ART / "a3_rf_recipe_results.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"wrote oof_a3_rf_recipe.npy + test + {out.name}  "
        f"total={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
