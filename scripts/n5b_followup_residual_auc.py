"""N5b follow-up angle 3: residual-AUC diagnostic.

Train a binary XGB on the 11 N5b features (3 OOD + 8 kNN10k) targeting
`y != PRIMARY_argmax` (i.e., is this row a primary error?). If the
5-fold OOF AUC > 0.55, the 10k-anchor signal SPECIFICALLY targets the
LB-best primary's errors — definitive proof the family carries
orthogonal-to-primary signal worth deploying via residual override.

If AUC > 0.55:
  - Compute precision-recall curve on residual predictions
  - Identify high-precision threshold (>= primary's break-even)
  - Diagnostic for whether residual override could lift LB

If AUC < 0.55:
  - 10k-anchor signal is REDUNDANT with primary; family is closed
  - All N5b deployments confirmed null

Wall: ~5-10 min CPU (5 folds, simple binary XGB on 11 features).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed

ART = Path("scripts/artifacts")
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
SEED = 42
N_FOLDS = 5


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def build_primary_4stack_oof(y):
    """LB-best 4-stack PRIMARY = 3-stack + xgb_metastack__iso @ alpha=0.30."""
    s3_o, _ = build_lbbest_stack(y)
    ms_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_o_iso, _ = iso_cal(ms_o, ms_t, y)
    p4_o = log_blend([s3_o, ms_o_iso], np.array([0.70, 0.30]))
    return p4_o


def main() -> None:
    print("[1] Loading data...")
    y = load_y()
    ood = np.load(ART / "oof_ood3_train.npy").astype(np.float32)  # (N, 3)
    knn = np.load(ART / "oof_knn10k_train.npy").astype(np.float32)  # (N, 8)
    X = np.concatenate([ood, knn], axis=1)
    feature_names = ["ood_gmm", "ood_iso", "ood_knn",
                     "k10_pL", "k10_pM", "k10_pH", "k10_nbr0",
                     "k10_dL", "k10_dM", "k10_dH", "k10_margin"]
    print(f"    X shape={X.shape}, y shape={y.shape}")

    print("[2] Building PRIMARY argmax to compute residual target...")
    p_primary = build_primary_4stack_oof(y)
    pred_primary = (np.log(np.clip(p_primary, 1e-12, 1)) + BIAS).argmax(1)
    residual = (y != pred_primary).astype(np.int32)
    print(f"    PRIMARY error count = {residual.sum()} ({residual.mean()*100:.2f}%)")

    print("[3] 5-fold binary XGB on 11 N5b features, target=is_primary_error...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(X), dtype=np.float32)
    fold_aucs = []
    fold_pr = []
    xgb_params = dict(
        objective="binary:logistic", eval_metric="auc",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=1.0, reg_lambda=1.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, residual)):
        t0 = time.time()
        dtr = xgb.DMatrix(X[tr_idx], label=residual[tr_idx])
        dva = xgb.DMatrix(X[va_idx], label=residual[va_idx])
        booster = xgb.train(xgb_params, dtr, num_boost_round=2000,
                             evals=[(dva, "val")], early_stopping_rounds=100,
                             verbose_eval=0)
        bi = booster.best_iteration
        vp = booster.predict(dva, iteration_range=(0, bi + 1))
        oof_pred[va_idx] = vp
        auc = roc_auc_score(residual[va_idx], vp)
        ap = average_precision_score(residual[va_idx], vp)
        fold_aucs.append(auc); fold_pr.append(ap)
        print(f"  fold {fold+1}/5 best_it={bi} AUC={auc:.4f} AP={ap:.4f} "
              f"wall={time.time()-t0:.1f}s")

    overall_auc = roc_auc_score(residual, oof_pred)
    overall_ap = average_precision_score(residual, oof_pred)
    print(f"\n[4] Overall OOF AUC = {overall_auc:.4f}  AP = {overall_ap:.4f}")
    print(f"    fold AUC mean={np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")

    # Precision-recall analysis at top-K thresholds
    print("\n[5] Top-K precision (where the model 'flags' a primary error):")
    sorted_idx = np.argsort(oof_pred)[::-1]
    for K in [100, 200, 500, 1000, 2000, 5000, 10000]:
        top = sorted_idx[:K]
        prec = residual[top].mean()
        recall = residual[top].sum() / residual.sum()
        print(f"  top-{K:5d}: precision={prec:.4f}  recall_of_errors={recall:.4f}")

    # Verdict
    print("\n[6] VERDICT")
    if overall_auc > 0.55:
        print(f"  AUC {overall_auc:.4f} > 0.55: 10k-anchor signal SPECIFICALLY")
        print(f"  targets PRIMARY's errors — invest in further deployment")
        print(f"  (more 10k-anchor features, residual override, etc.)")
        verdict = "PROCEED"
    else:
        print(f"  AUC {overall_auc:.4f} ≤ 0.55: 10k-anchor signal does NOT")
        print(f"  specifically target PRIMARY's errors.")
        print(f"  Combined with bank-add NULL + direct blend NULL,")
        print(f"  the 10k-anchor lever family is structurally CLOSED.")
        verdict = "KILL"

    out = {"overall_auc": float(overall_auc), "overall_ap": float(overall_ap),
           "fold_aucs": fold_aucs, "fold_aps": fold_pr,
           "n_residual": int(residual.sum()), "verdict": verdict,
           "feature_names": feature_names}
    np.save(ART / "oof_n5b_residual_auc.npy", oof_pred)
    out_path = ART / "n5b_followup_residual_auc_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}, oof_n5b_residual_auc.npy")


if __name__ == "__main__":
    main()
