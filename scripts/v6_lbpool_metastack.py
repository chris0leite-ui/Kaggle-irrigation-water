"""v6 variant restricted to the EXACT 62-component LB-best v1 pool.

Isolates the aggregate-feature lever (P1 mechanism) from any bank-extension
effect. Same pipeline as v6_metastack.py but loads the pool from
tier1b_xgb_metastack_results.json's "components" list, NOT from disk-scan.

Outputs:
  scripts/artifacts/oof_xgb_metastack_v6lb.npy + test_xgb_metastack_v6lb.npy
  scripts/artifacts/v6_lbpool_results.json
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
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from v6_aggregates import compute_aggregates  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, DATA, BIAS, build_lbbest_stack, log, bal_at_bias,
)

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
SEED = 42
N_FOLDS = 1 if SMOKE else 5
MAX_ROUNDS = 200 if SMOKE else 3000
ES_ROUNDS = 50 if SMOKE else 200
LB_RESULTS = ART / "tier1b_xgb_metastack_results.json"


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def load_lb_pool():
    """Load the 62 components used by LB-best meta-stacker v1."""
    d = json.loads(LB_RESULTS.read_text())
    names = d["components"]
    pool = {}
    missing = []
    for name in names:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not (oof_p.exists() and test_p.exists()):
            missing.append(name)
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.ndim != 2 or o.shape[1] != 3:
            missing.append(name)
            continue
        pool[name] = (_normed(o), _normed(t))
    if missing:
        log(f"WARN: {len(missing)} components missing: {missing}")
    return pool


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best 3-stack OOF = {bal_at_bias(lb_oof, y):.5f}")

    log("loading 62-component LB-best v1 pool")
    pool = load_lb_pool()
    log(f"  loaded {len(pool)} components (target 62)")
    component_names = sorted(pool.keys())

    log("constructing meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    log("computing 22 aggregate features over 62-component pool")
    stack_tr = np.stack([pool[n][0] for n in component_names], axis=0)
    stack_te = np.stack([pool[n][1] for n in component_names], axis=0)
    agg_tr, agg_names = compute_aggregates(stack_tr)
    agg_te, _ = compute_aggregates(stack_te)

    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))

    X_tr = np.concatenate([lb_log_tr, meta_tr, agg_tr] + comp_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([lb_log_te, meta_te, agg_te] + comp_te, axis=1).astype(np.float32)
    log(f"  feature dim: {X_tr.shape[1]}  "
        f"(tier1b v1 baseline = 3 + 14 + 3*{len(pool)} = {3 + 14 + 3*len(pool)})")

    if SMOKE:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(y), 50_000, replace=False); keep.sort()
        X_tr = X_tr[keep]; y_smoke = y[keep]
    else:
        y_smoke = y

    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )

    oof_meta = np.zeros((len(y_smoke), 3), dtype=np.float32)
    test_meta_folds = []
    best_iters = []
    skf = StratifiedKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=SEED)
    n_done = 0
    suffix = "_smoke" if SMOKE else ""
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_smoke)):
        if fold >= N_FOLDS:
            break
        ck_oof = ART / f"oof_xgb_metastack_v6lb{suffix}_fold{fold}.npy"
        ck_test = ART / f"test_xgb_metastack_v6lb{suffix}_fold{fold}.npy"
        ck_meta = ART / f"oof_xgb_metastack_v6lb{suffix}_fold{fold}_meta.json"
        if ck_oof.exists() and ck_test.exists() and ck_meta.exists():
            vp = np.load(ck_oof); tp = np.load(ck_test)
            mm = json.loads(ck_meta.read_text())
            bi = int(mm.get("best_iter", 0))
            oof_meta[va_idx] = vp
            test_meta_folds.append(tp)
            best_iters.append(bi)
            log(f"  fold {fold+1}/{N_FOLDS} CACHED (it={bi} bal={mm.get('argmax_bal',0):.5f})")
            n_done += 1
            continue
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y_smoke[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y_smoke[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(xgb_params, dtr, num_boost_round=MAX_ROUNDS,
                            evals=[(dva, "val")],
                            early_stopping_rounds=ES_ROUNDS, verbose_eval=0)
        bi = booster.best_iteration
        best_iters.append(bi)
        vp = booster.predict(dva, iteration_range=(0, bi + 1))
        oof_meta[va_idx] = vp.astype(np.float32)
        tp = booster.predict(dte, iteration_range=(0, bi + 1))
        test_meta_folds.append(tp)
        np.save(ck_oof, vp.astype(np.float32))
        np.save(ck_test, tp.astype(np.float32))
        argmax_bal = balanced_accuracy_score(y_smoke[va_idx], vp.argmax(1))
        ck_meta.write_text(json.dumps(dict(best_iter=int(bi),
                                           argmax_bal=float(argmax_bal),
                                           wall=float(time.time() - t1))))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax_bal={argmax_bal:.5f} "
            f"wall={time.time()-t1:.1f}s")
        n_done += 1

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    np.save(ART / f"oof_xgb_metastack_v6lb{suffix}.npy", oof_meta)
    np.save(ART / f"test_xgb_metastack_v6lb{suffix}.npy", test_meta)
    log(f"saved oof_xgb_metastack_v6lb{suffix}.npy + test")

    if not SMOKE:
        meta_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
        meta_tuned = bal_at_bias(oof_meta, y)
        lb_bal = bal_at_bias(lb_oof, y)
        log(f"\n=== v6lb META standalone ===")
        log(f"  argmax OOF      = {meta_argmax:.5f}")
        log(f"  @recipe-bias    = {meta_tuned:.5f}")
        log(f"  LB-best 3-stack = {lb_bal:.5f}  Δ={meta_tuned - lb_bal:+.5f}")
        out = dict(n_components=len(component_names), components=component_names,
                   feature_dim=int(X_tr.shape[1]), agg_names=agg_names,
                   best_iters=[int(b) for b in best_iters],
                   meta_argmax_oof=float(meta_argmax),
                   meta_tuned_oof=float(meta_tuned),
                   lb_best_oof=float(lb_bal),
                   elapsed_sec=float(time.time() - t0))
        (ART / "v6_lbpool_results.json").write_text(json.dumps(out, indent=2))
        log(f"wrote {ART / 'v6_lbpool_results.json'}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
