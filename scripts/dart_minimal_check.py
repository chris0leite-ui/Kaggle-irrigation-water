"""Minimal-input meta defense check for DART recipe variant.

Train a 9-feature XGB stacker on (LB-3-stack log-probs + DART log-probs +
3 dist features). If 2-component result lands BELOW LB-best primary at
recipe-bias, the apparent contribution of DART is cross-component
memorization vs orthogonal signal.

Per CLAUDE.md leakage rule.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (ART, BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                             load_y, bal_at_bias)

SEED = 42
N_FOLDS = 5
SUFFIX = "_dart"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack")
    lb3_oof, lb3_test = build_lbbest_stack(y)

    log(f"loading DART recipe: oof_recipe_full_te{SUFFIX}.npy")
    dart_o = np.load(ART / f"oof_recipe_full_te{SUFFIX}.npy")
    dart_t = np.load(ART / f"test_recipe_full_te{SUFFIX}.npy")

    v1_o = np.load(ART / "oof_xgb_metastack.npy")
    v1_t = np.load(ART / "test_xgb_metastack.npy")
    v1_iso_o, v1_iso_t = iso_cal(v1_o, v1_t, y)
    primary_o = log_blend([lb3_oof, v1_iso_o], np.array([0.7, 0.3]))
    primary_t = log_blend([lb3_test, v1_iso_t], np.array([0.7, 0.3]))
    log(f"  LB-best PRIMARY OOF = {bal_at_bias(primary_o, y):.5f}")

    tr_d = add_distance_features(train)[["dgp_score", "sm_dist", "min_axis_abs"]]
    te_d = add_distance_features(test)[["dgp_score", "sm_dist", "min_axis_abs"]]
    X_tr = np.concatenate([
        np.log(np.clip(lb3_oof, 1e-9, 1)),
        np.log(np.clip(dart_o, 1e-9, 1)),
        tr_d.to_numpy(np.float32),
    ], axis=1).astype(np.float32)
    X_te = np.concatenate([
        np.log(np.clip(lb3_test, 1e-9, 1)),
        np.log(np.clip(dart_t, 1e-9, 1)),
        te_d.to_numpy(np.float32),
    ], axis=1).astype(np.float32)
    log(f"  minimal-input dim = {X_tr.shape[1]} (3 lb3 + 3 dart + 3 dist)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_min = np.zeros((len(y), 3), dtype=np.float32)
    test_min_folds = []
    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(xgb_params, dtr, num_boost_round=3000,
                            evals=[(dva, "val")], early_stopping_rounds=200,
                            verbose_eval=0)
        bi = booster.best_iteration
        oof_min[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1)).astype(np.float32)
        test_min_folds.append(
            booster.predict(dte, iteration_range=(0, bi + 1)).astype(np.float32))
        log(f"  fold {fold+1} it={bi} val_argmax={balanced_accuracy_score(y[va_idx], oof_min[va_idx].argmax(1)):.5f} "
            f"wall={time.time()-t1:.1f}s")

    test_min = np.mean(test_min_folds, axis=0).astype(np.float32)
    np.save(ART / f"oof_xgb_metastack_minimal{SUFFIX}.npy", oof_min)
    np.save(ART / f"test_xgb_metastack_minimal{SUFFIX}.npy", test_min)

    min_iso_o, _ = iso_cal(oof_min, test_min, y)
    standalone = bal_at_bias(min_iso_o, y)
    primary_bal = bal_at_bias(primary_o, y)
    log(f"\n=== minimal-input DART meta ===")
    log(f"  iso standalone OOF = {standalone:.5f}")
    log(f"  LB-best PRIMARY    = {primary_bal:.5f}")
    log(f"  Δ                  = {standalone - primary_bal:+.5f}")
    if standalone < primary_bal:
        log(f"  VERDICT: minimal < primary → DART's contribution is "
            f"likely cross-component memorization (CLAUDE.md leakage rule).")
    else:
        log(f"  VERDICT: minimal ≥ primary → marginal DART signal looks real.")
    log(f"\nwall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
