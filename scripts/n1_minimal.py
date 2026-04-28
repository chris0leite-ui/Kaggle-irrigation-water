"""Minimal macrorec meta: only LB-3-stack + macrorec_base as components.

Eliminates the meta-of-metas circularity AND minimizes the surface for
cross-fold stacking leak. With only 2 components × 3 classes = 6 input
features (+ 14 dist features = 20 total), the meta has limited capacity
to exploit per-row leak in the input OOFs.

If standalone OOF comes back materially above LB-best 4-stack (0.98084),
the lift is REAL signal (macrorec_base contributes orthogonal H direction
over LB-3-stack). If at/below, macrorec adds nothing the existing 4-stack
doesn't already capture.

Theoretical hyperparameters only:
  - lam_ce = 0.3 (LB-validated SMOKE choice)
  - α = 0.30 (LB-validated architecture)
  - depth=4, reg_alpha=5, reg_lambda=5

Output suffix `_minimal`. ~22 min wall.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX
from recipe_macrorecall import make_macrorec_obj, macrorec_eval_metric
from tier1b_xgb_metastack import _normed, build_lbbest_stack, iso_cal, BIAS

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
DATA = Path("data")
ART = Path("scripts/artifacts")
SUFFIX = "_metamacrorec_minimal"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def main():
    log("MINIMAL macrorec meta-stacker  lam_ce=0.3  T=1.0")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best 3-stack OOF = {bal(lb_oof, y):.5f}")

    log("loading macrorec base OOFs")
    macro_oof = _normed(np.load(ART / "oof_recipe_full_te_macrorec_T1_lam03.npy"))
    macro_test = _normed(np.load(ART / "test_recipe_full_te_macrorec_T1_lam03.npy"))
    log(f"  macrorec base OOF tuned = {bal(macro_oof, y):.5f}")

    log("constructing minimal meta features (2 components × 3 cls + 14 dist)")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))
    macro_log_tr = np.log(np.clip(macro_oof, 1e-9, 1.0))
    macro_log_te = np.log(np.clip(macro_test, 1e-9, 1.0))
    X_tr = np.concatenate([lb_log_tr, macro_log_tr, meta_tr], axis=1).astype(np.float32)
    X_te = np.concatenate([lb_log_te, macro_log_te, meta_te], axis=1).astype(np.float32)
    log(f"  meta-feature shape: {X_tr.shape}")

    xgb_params = dict(
        num_class=3, tree_method="hist",
        learning_rate=0.05, max_depth=4,
        min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        verbosity=0, seed=SEED, nthread=-1,
        disable_default_eval_metric=1,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((len(train), 3), dtype=np.float32)
    test_meta_folds = []
    best_iters = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y), 1):
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        obj = make_macrorec_obj(y[tr_idx], n_classes=3, temperature=1.0, lam_ce=0.3)
        feval = macrorec_eval_metric(y[va_idx])
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=3000,
            obj=obj, custom_metric=feval, maximize=False,
            evals=[(dva, "val")], early_stopping_rounds=200,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        z_va = booster.predict(dva, iteration_range=(0, bi + 1),
                               output_margin=True).reshape(-1, 3)
        z_te = booster.predict(dte, iteration_range=(0, bi + 1),
                               output_margin=True).reshape(-1, 3)
        def softmax(z):
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        vp = softmax(z_va).astype(np.float32)
        tp = softmax(z_te).astype(np.float32)
        oof_meta[va_idx] = vp
        test_meta_folds.append(tp)
        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_scores.append(argmax_bal)
        log(f"  fold {fold}/{N_FOLDS} it={bi} val_argmax_bal={argmax_bal:.5f} "
            f"wall={time.time()-t1:.1f}s")

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    np.save(ART / f"oof_xgb_metastack{SUFFIX}.npy", oof_meta)
    np.save(ART / f"test_xgb_metastack{SUFFIX}.npy", test_meta)

    overall_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
    overall_tuned = bal(oof_meta, y)
    log(f"\nOOF argmax = {overall_argmax:.5f}  @recipe-bias = {overall_tuned:.5f}")
    log(f"  best_iters = {best_iters}")

    oof_iso, test_iso = iso_cal(oof_meta, test_meta, y)
    np.save(ART / f"oof_xgb_metastack{SUFFIX}_iso.npy", oof_iso)
    np.save(ART / f"test_xgb_metastack{SUFFIX}_iso.npy", test_iso)
    iso_argmax = balanced_accuracy_score(y, oof_iso.argmax(1))
    iso_tuned = bal(oof_iso, y)
    log(f"  iso  argmax = {iso_argmax:.5f}  @recipe-bias = {iso_tuned:.5f}")

    summary = dict(
        n_folds=N_FOLDS, lam_ce=0.3, temperature=1.0,
        n_components=2,
        meta_feature_shape=list(X_tr.shape),
        fold_scores_argmax=[float(s) for s in fold_scores],
        best_iters=[int(b) for b in best_iters],
        overall_argmax_bal_acc=float(overall_argmax),
        overall_tuned_bal_acc=float(overall_tuned),
        iso_argmax_bal_acc=float(iso_argmax),
        iso_tuned_bal_acc=float(iso_tuned),
        components=["lb_3stack", "macrorec_base"],
    )
    with open(ART / f"xgb_metastack{SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote results JSON")


if __name__ == "__main__":
    main()
