"""W3: binary 'is Medium?' head on 443-feature recipe set.

LB-best 3-way per-class recall: Low 0.9949 / Medium 0.9685 / High 0.9774.
Medium is systematically the weakest class and drives macro-recall under
balanced accuracy. 2026-04-21 binhigh lesson: blend the binary head at
FIXED bias (no retune) to avoid OOF-selection overfit.

Pipeline mirrors recipe_full_te.py for FE + OrderedTE, but:
  - binary:logistic target `y == 1` (Medium)
  - fewer XGB rounds (binary converges faster than multi-class)
  - per-fold OTE regenerated on same 3-class target (no new OTE basis)

Artefacts:
  scripts/artifacts/oof_xgb_bin_medium.npy   (630k, P(Medium) per row)
  scripts/artifacts/test_xgb_bin_medium.npy  (270k)
  scripts/artifacts/binary_medium_head_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from recipe_full_te import load_and_engineer, log  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
MEDIUM_IDX = 1
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)


def run_binary_cv(train, test, info, y3, a_ote: float = 1.0) -> dict:
    """5-fold binary 'is Medium?' XGB with per-fold OrderedTE."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_bin = (y3 == MEDIUM_IDX).astype(np.int32)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    # 3-class OTE cols (same basis as recipe_full_te) — binary head consumes
    # them as features; no need to regenerate at class=2 granularity.

    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    aucs, best_iters = [], []

    xgb_params = dict(
        n_estimators=100 if SMOKE else 1500,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="binary:logistic", tree_method="hist",
        eval_metric="auc", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 150, verbosity=0,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y3), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        # OrderedTE (multi-class target — matches recipe_full_te's setup).
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=a_ote)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"  OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        y_tr_bin = y_bin[tr_idx]
        # Class-balanced on the binary target: lifts Medium rate from 0.38 to 0.50
        sw = compute_sample_weight("balanced", y_tr_bin)

        log(f"  training binary XGB on {len(feat_cols)} features, {len(X_tr):,} rows")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr[feat_cols], y_tr_bin, sample_weight=sw,
                  eval_set=[(X_va[feat_cols], y_bin[va_idx])], verbose=500)
        va_p = model.predict_proba(X_va[feat_cols])[:, 1].astype(np.float32)
        oof[va_idx] = va_p
        test_pred += model.predict_proba(X_te[feat_cols])[:, 1].astype(np.float32) / N_FOLDS
        auc = roc_auc_score(y_bin[va_idx], va_p)
        aucs.append(auc); best_iters.append(model.best_iteration)
        log(f"  fold {fold} AUC={auc:.5f}  best_iter={model.best_iteration}")

    overall_auc = roc_auc_score(y_bin, oof)
    log(f"=== binary-head OOF AUC = {overall_auc:.5f}")
    return dict(oof=oof, test=test_pred, aucs=aucs, best_iters=best_iters,
                overall_auc=float(overall_auc), n_features=len(feat_cols))


def main():
    log(f"W3 binary Medium head  (SMOKE={SMOKE})")
    train, test, info, _ = load_and_engineer()
    y3 = train[TARGET].to_numpy()
    medium_rate = (y3 == MEDIUM_IDX).mean()
    log(f"Medium prior: {medium_rate:.4f}  N_train={len(train)}  N_test={len(test)}")

    result = run_binary_cv(train, test, info, y3)
    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_xgb_bin_medium{suffix}.npy", result["oof"])
    np.save(ART / f"test_xgb_bin_medium{suffix}.npy", result["test"])
    with open(ART / f"binary_medium_head{suffix}_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "medium_rate": float(medium_rate),
            "per_fold_auc": [float(x) for x in result["aucs"]],
            "overall_auc": result["overall_auc"],
            "best_iters_per_fold": [int(x) for x in result["best_iters"]],
            "n_features": int(result["n_features"]),
            "smoke": SMOKE,
        }, f, indent=2)
    log(f"saved oof_xgb_bin_medium{suffix}.npy + test + json")


if __name__ == "__main__":
    main()
