"""Tier 1c v6 — aggregate-stats meta-stacker.

Mirror of `tier1b_xgb_metastack.py` PLUS 22 bank-aggregate features per row
(see `v6_aggregates.py`). Same EXCLUDE pool, same heavy-reg XGB HPs, same
5-fold StratifiedKFold(seed=42) split as every other meta-stacker on disk.

Mechanism (proven by 2026-04-26 EDA):
  agg ⊥ tier1b meta = AUC 0.6714 on missed-H detection in pred=Med override
  domain. tier1b meta has only per-component per-class log-probs as inputs;
  the bank-aggregate uncertainty geometry (std / max / min / entropy /
  disagreement / class-margin variance) is residual signal it cannot extract.

SMOKE=1 runs 1 fold × 50k stratified rows for end-to-end validation.

Outputs:
  scripts/artifacts/oof_xgb_metastack_v6.npy + test_xgb_metastack_v6.npy
  scripts/artifacts/v6_metastack_results.json
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
    ART, DATA, BIAS, build_lbbest_stack, load_pool, log, bal_at_bias,
)

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
SEED = 42
N_FOLDS = 1 if SMOKE else 5
MAX_ROUNDS = 200 if SMOKE else 3000
ES_ROUNDS = 50 if SMOKE else 200


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best stack OOF = {bal_at_bias(lb_oof, y):.5f}")

    log("loading pool (excludes via tier1b_helpers.EXCLUDE)")
    pool = load_pool()
    log(f"  pool: {len(pool)} 3-class components")
    component_names = sorted(pool.keys())

    # Build meta features
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

    # Build (N, R, 3) stacks for aggregate stats
    log("computing 22 bank-aggregate features")
    stack_tr = np.stack([pool[n][0] for n in component_names], axis=0)
    stack_te = np.stack([pool[n][1] for n in component_names], axis=0)
    agg_tr, agg_names = compute_aggregates(stack_tr)
    agg_te, _ = compute_aggregates(stack_te)
    log(f"  aggregates shape: tr={agg_tr.shape} te={agg_te.shape}")

    # Per-component log probs (mirror tier1b)
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))

    X_tr = np.concatenate([lb_log_tr, meta_tr, agg_tr] + comp_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([lb_log_te, meta_te, agg_te] + comp_te, axis=1).astype(np.float32)
    log(f"  feature dim: {X_tr.shape[1]}  (tier1b had {3 + 14 + 3*len(component_names)}; "
        f"v6 adds {agg_tr.shape[1]} aggregate cols)")

    if SMOKE:
        # subsample for speed
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(y), 50_000, replace=False)
        keep.sort()
        X_tr = X_tr[keep]
        y_smoke = y[keep]
        log(f"SMOKE: subsampled to {len(keep)} rows")
    else:
        y_smoke = y

    xgb_params = dict(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
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
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_smoke)):
        if fold >= N_FOLDS:
            break
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
        argmax_bal = balanced_accuracy_score(y_smoke[va_idx], vp.argmax(1))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax_bal={argmax_bal:.5f} "
            f"wall={time.time()-t1:.1f}s")
        n_done += 1

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_xgb_metastack_v6{suffix}.npy", oof_meta)
    np.save(ART / f"test_xgb_metastack_v6{suffix}.npy", test_meta)
    log(f"saved oof_xgb_metastack_v6{suffix}.npy + test_xgb_metastack_v6{suffix}.npy")

    if not SMOKE:
        meta_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
        meta_tuned = bal_at_bias(oof_meta, y)
        lb_bal = bal_at_bias(lb_oof, y)
        log(f"\n=== v6 META standalone ===")
        log(f"  argmax OOF      = {meta_argmax:.5f}")
        log(f"  @recipe-bias    = {meta_tuned:.5f}")
        log(f"  LB-best 3-stack = {lb_bal:.5f}  Δ={meta_tuned - lb_bal:+.5f}")

        out = dict(
            n_components=len(component_names),
            feature_dim=int(X_tr.shape[1]),
            agg_names=agg_names,
            best_iters=[int(b) for b in best_iters],
            meta_argmax_oof=float(meta_argmax),
            meta_tuned_oof=float(meta_tuned),
            lb_best_oof=float(lb_bal),
            elapsed_sec=float(time.time() - t0),
        )
        (ART / "v6_metastack_results.json").write_text(json.dumps(out, indent=2))
        log(f"wrote {ART / 'v6_metastack_results.json'}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
