"""v8 XGB-meta-stacker — adds 12 per-row cross-component diagnostic
features to the Tier-1b 210-dim input.

Diagnostic features (computed across all components in the pool):
  1.  arg_agree_count_with_lb     — # components whose argmax = LB-best argmax
  2.  argmax_mode_pct             — fraction of components voting for the mode class
  3.  mean_pL / mean_pM / mean_pH — bank-mean per-class probability
  4.  std_pL / std_pM / std_pH    — bank-stdev per-class probability
  5.  range_pL / range_pM / range_pH — bank max-min per-class probability
  6.  lb_top1_top2_margin         — LB-best confidence margin

Hypothesis: these aggregates encode disagreement / consensus information
that the depth-4 XGB-meta with reg_alpha=5 cannot fully reconstruct from
210 raw component log-prob features. ~10-min FE + ~5-min meta retrain.

Run:  python3 -m scripts.dist_shift.optD_meta_v8
Output: oof_xgb_metastack_v8.npy + test + results JSON.
Then:   CAND=xgb_metastack_v8 python3 -m scripts.dist_shift.optAB_blend_gate
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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_distance_features, tune_log_bias  # noqa: E402
from tier1b_xgb_metastack import (  # noqa: E402
    EXCLUDE, build_lbbest_stack, iso_cal, load_pool, _normed,
)

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
ART = Path("scripts/artifacts")
DATA = Path("data")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_diagnostic_features(pool, lb_oof, n: int) -> tuple[np.ndarray, list[str]]:
    """Per-row diagnostics across the bank's components.

    Args:
        pool: dict {name: (oof, test)} of 3-class probs (already normed).
        lb_oof: (n, 3) LB-best 3-stack probs.

    Returns:
        feats (n, 12) float32, and feature names.
    """
    names = sorted(pool.keys())
    K = len(names)
    # Stack into (K, n, 3) for easier vectorisation
    stack = np.stack([pool[name][0] for name in names], axis=0).astype(np.float32)
    argmaxes = stack.argmax(axis=2)            # (K, n)
    lb_arg = lb_oof.argmax(axis=1)             # (n,)

    # 1. argmax-agreement with LB-best
    arg_agree = (argmaxes == lb_arg[None, :]).sum(axis=0).astype(np.float32) / K  # (n,)

    # 2. argmax-mode fraction (most-voted class share)
    counts = np.zeros((n, 3), dtype=np.int32)
    for c in range(3):
        counts[:, c] = (argmaxes == c).sum(axis=0)
    mode_pct = counts.max(axis=1).astype(np.float32) / K

    # 3, 4. bank mean / stdev per class
    mean_p = stack.mean(axis=0)               # (n, 3)
    std_p = stack.std(axis=0)                 # (n, 3)

    # 5. bank range per class (max - min)
    max_p = stack.max(axis=0)
    min_p = stack.min(axis=0)
    range_p = max_p - min_p                   # (n, 3)

    # 6. LB-best top1-top2 margin
    sorted_lb = np.sort(lb_oof, axis=1)
    margin = (sorted_lb[:, -1] - sorted_lb[:, -2]).astype(np.float32)  # (n,)

    feats = np.concatenate([
        arg_agree[:, None],
        mode_pct[:, None],
        mean_p, std_p, range_p,
        margin[:, None],
    ], axis=1).astype(np.float32)

    fnames = [
        "arg_agree_count_with_lb", "argmax_mode_pct",
        "mean_pL", "mean_pM", "mean_pH",
        "std_pL", "std_pM", "std_pH",
        "range_pL", "range_pM", "range_pH",
        "lb_top1_top2_margin",
    ]
    assert feats.shape == (n, 12)
    assert len(fnames) == 12
    return feats, fnames


def build_diagnostic_features_test(pool, lb_test, n_test: int) -> np.ndarray:
    names = sorted(pool.keys())
    stack = np.stack([pool[name][1] for name in names], axis=0).astype(np.float32)
    argmaxes = stack.argmax(axis=2)
    lb_arg = lb_test.argmax(axis=1)
    K = len(names)
    arg_agree = (argmaxes == lb_arg[None, :]).sum(axis=0).astype(np.float32) / K
    counts = np.zeros((n_test, 3), dtype=np.int32)
    for c in range(3):
        counts[:, c] = (argmaxes == c).sum(axis=0)
    mode_pct = counts.max(axis=1).astype(np.float32) / K
    mean_p = stack.mean(axis=0)
    std_p = stack.std(axis=0)
    range_p = stack.max(axis=0) - stack.min(axis=0)
    sorted_lb = np.sort(lb_test, axis=1)
    margin = (sorted_lb[:, -1] - sorted_lb[:, -2]).astype(np.float32)
    feats = np.concatenate([
        arg_agree[:, None], mode_pct[:, None],
        mean_p, std_p, range_p, margin[:, None]
    ], axis=1).astype(np.float32)
    assert feats.shape == (n_test, 12)
    return feats


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)

    log("loading pool")
    pool = load_pool(y)
    log(f"  {len(pool)} components loaded")

    log("building 12 diagnostic features (train+test)")
    diag_tr, diag_names = build_diagnostic_features(pool, lb_oof, len(train))
    diag_te = build_diagnostic_features_test(pool, lb_test, len(test))
    log(f"  diagnostic shape: {diag_tr.shape}, names: {diag_names}")
    log(f"  arg_agree mean={diag_tr[:,0].mean():.3f}, "
        f"mode_pct mean={diag_tr[:,1].mean():.3f}, "
        f"std_pH mean={diag_tr[:,7].mean():.3f}")

    # Reuse Tier-1b feature build
    log("constructing meta features (Tier-1b structure + 12 diagnostics)")
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

    # Concat: lb (3) + meta (14) + components (3K) + diagnostics (12)
    X_tr = np.concatenate([lb_log_tr, meta_tr, *comp_tr, diag_tr], axis=1)
    X_te = np.concatenate([lb_log_te, meta_te, *comp_te, diag_te], axis=1)
    log(f"  X_tr shape: {X_tr.shape}  (vs Tier-1b's {X_tr.shape[1]-12})")

    xgb_params = dict(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    log(f"  XGB HPs: {xgb_params}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((len(train), 3), dtype=np.float32)
    test_meta_folds = []
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=3000,
            evals=[(dva, "val")], early_stopping_rounds=200, verbose_eval=0,
        )
        bi = booster.best_iteration
        oof_meta[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_meta_folds.append(booster.predict(dte, iteration_range=(0, bi + 1)))
        best_iters.append(int(bi))
        log(f"  fold {fold+1}/5  best_iter={bi}  wall={time.time()-t1:.1f}s")

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    oof_meta = _normed(oof_meta)
    test_meta = _normed(test_meta)

    BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
    EPS = 1e-12
    pred = (np.log(np.clip(oof_meta, EPS, 1.0)) + BIAS).argmax(1)
    cm_diag = np.array([
        ((pred == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in range(3)
    ])
    bal_at_anchor = float(cm_diag.mean())

    prior = np.bincount(y, minlength=3) / len(y)
    own_bias, tuned = tune_log_bias(oof_meta, y, prior)

    log(f"\nv8 standalone:")
    log(f"  @ recipe-bias [1.43, 1.47, 3.40] = {bal_at_anchor:.5f}")
    log(f"  own-tuned                          = {tuned:.5f}  bias={own_bias}")
    log(f"  best_iters per fold = {best_iters}")
    log(f"  total wall = {time.time()-t0:.1f}s")

    # Save
    np.save(ART / "oof_xgb_metastack_v8.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_v8.npy", test_meta)
    out = {
        "n_components": len(component_names),
        "n_diagnostic_features": 12,
        "diagnostic_names": diag_names,
        "X_tr_shape": list(X_tr.shape),
        "best_iters": best_iters,
        "bal_at_recipe_bias": bal_at_anchor,
        "tuned_bal_acc": float(tuned),
        "own_bias": list(map(float, own_bias)),
        "wall_seconds": float(time.time() - t0),
    }
    (ART / "xgb_metastack_v8_results.json").write_text(json.dumps(out, indent=2))
    log(f"\nWrote oof_xgb_metastack_v8.npy + test + xgb_metastack_v8_results.json")
    log(f"\nNext: CAND=xgb_metastack_v8 python3 -m scripts.dist_shift.optAB_blend_gate")


if __name__ == "__main__":
    main()
