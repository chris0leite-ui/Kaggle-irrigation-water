"""Train one meta-stacker variant with configurable HPs.

Used by tier1b_metastack_ensemble.py to produce 2 additional metas (variants
B and C) with different (depth, xgb_seed, colsample, rounds) than v3.

Variants share the same StratifiedKFold(seed=42) FOLD split — only the
XGB seed and HPs vary so the 3 metas blend usefully on the same OOF rows.

Env vars:
  VARIANT       — output suffix tag (e.g. "B", "C"); saves
                  oof/test_xgb_metastack_var{VARIANT}.npy
  DEPTH         — XGB max_depth (default 4)
  XGB_SEED      — XGB seed (default 42; FOLD seed pinned at 42)
  COLSAMPLE     — colsample_bytree (default 0.9)
  MAX_ROUNDS    — early-stop cap (default 3000)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, DATA, N_FOLDS, SEED, bal_at_bias, build_lbbest_stack, load_pool,
    load_y, log,
)

VARIANT = os.environ.get("VARIANT", "B")
DEPTH = int(os.environ.get("DEPTH", "3"))
XGB_SEED = int(os.environ.get("XGB_SEED", "7"))
COLSAMPLE = float(os.environ.get("COLSAMPLE", "0.7"))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "4000"))


def main():
    t0 = time.time()
    log(f"variant={VARIANT}  depth={DEPTH}  xgb_seed={XGB_SEED}  "
        f"colsample={COLSAMPLE}  max_rounds={MAX_ROUNDS}")
    y = load_y()
    lb_oof, lb_test = build_lbbest_stack(y)
    lb_bal = bal_at_bias(lb_oof, y)
    log(f"  LB-best anchor OOF = {lb_bal:.5f}")

    pool = load_pool()
    log(f"  pool size = {len(pool)}")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)
    component_names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))
    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1).astype(np.float32)
    log(f"  feature dim = {X_tr.shape[1]}")

    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=DEPTH, min_child_weight=5,
        subsample=0.9, colsample_bytree=COLSAMPLE,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=XGB_SEED, nthread=-1,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((len(train), 3), dtype=np.float32)
    test_folds = []
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(xgb_params, dtr, num_boost_round=MAX_ROUNDS,
                            evals=[(dva, "val")], early_stopping_rounds=200,
                            verbose_eval=0)
        bi = booster.best_iteration
        best_iters.append(bi)
        oof_meta[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1)).astype(np.float32)
        test_folds.append(booster.predict(dte, iteration_range=(0, bi + 1)))
        bal = balanced_accuracy_score(y[va_idx], oof_meta[va_idx].argmax(1))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax={bal:.5f} wall={time.time()-t1:.1f}s")

    test_meta = np.mean(test_folds, axis=0).astype(np.float32)
    np.save(ART / f"oof_xgb_metastack_var{VARIANT}.npy", oof_meta)
    np.save(ART / f"test_xgb_metastack_var{VARIANT}.npy", test_meta)
    log(f"saved oof/test_xgb_metastack_var{VARIANT}.npy")

    meta_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
    meta_tuned = bal_at_bias(oof_meta, y)
    log(f"\n=== variant {VARIANT} standalone ===")
    log(f"  argmax OOF = {meta_argmax:.5f}")
    log(f"  @recipe-bias OOF = {meta_tuned:.5f}")
    rows = []
    for a in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        b = bal_at_bias(log_blend([lb_oof, oof_meta], np.array([1 - a, a])), y)
        rows.append({"alpha": a, "oof": float(b), "delta": float(b - lb_bal)})
        log(f"  α={a:.2f}  OOF={b:.5f}  Δ={b-lb_bal:+.5f}")

    out = dict(variant=VARIANT, depth=DEPTH, xgb_seed=XGB_SEED, colsample=COLSAMPLE,
               max_rounds=MAX_ROUNDS, n_components=len(component_names),
               feature_dim=X_tr.shape[1], best_iters=[int(b) for b in best_iters],
               meta_argmax=float(meta_argmax), meta_tuned=float(meta_tuned),
               lb_oof=float(lb_bal), sweep=rows,
               elapsed_sec=float(time.time() - t0))
    (ART / f"tier1b_metastack_var{VARIANT}_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote tier1b_metastack_var{VARIANT}_results.json")


if __name__ == "__main__":
    main()
