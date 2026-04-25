"""Cross-pollinate XGB meta-stacker (Tier-1b v3).

Adds 5 new components to the v1 pool that weren't present at v1's run-time:
recipe_focal_g2_aH1, recipe_focal_g2_invfreq, soft_distill_small,
soft_distill_tiny, realmlp_ens4.

Same XGB heavy-reg config as v1 (depth=4, alpha=lambda=5, lr=0.05, 3000-cap
+ 200 early stop), same 5-fold StratifiedKFold(seed=42), same recipe-bias
[1.4324, 1.4689, 3.4008] blend gate. Saves oof/test_xgb_metastack_v3.npy.

Decision rule: emit submission iff fixed-bias Δ vs LB-best 3-stack ≥ +2e-4
AND iso-blend in tier1b_final_blend.py confirms.
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
from common import add_distance_features  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, DATA, N_FOLDS, SEED, bal_at_bias, build_lbbest_stack, load_pool,
    load_y, log,
)


def main():
    t0 = time.time()
    log("loading y + LB-best 3-stack anchor")
    y = load_y()
    lb_oof, lb_test = build_lbbest_stack(y)
    lb_bal = bal_at_bias(lb_oof, y)
    log(f"  LB-best 3-stack OOF = {lb_bal:.5f}")

    log("loading expanded pool")
    pool = load_pool()
    new_5 = ["recipe_focal_g2_aH1", "recipe_focal_g2_invfreq",
             "soft_distill_small", "soft_distill_tiny", "realmlp_ens4"]
    for n in new_5:
        present = n in pool
        log(f"  {n}: {'OK' if present else 'MISSING'}")
    log(f"  total {len(pool)} 3-class components")

    log("constructing meta features")
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
    log(f"  meta-feature shape: {X_tr.shape}")

    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
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
        booster = xgb.train(xgb_params, dtr, num_boost_round=3000,
                            evals=[(dva, "val")], early_stopping_rounds=200,
                            verbose_eval=0)
        bi = booster.best_iteration
        best_iters.append(bi)
        oof_meta[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1)).astype(np.float32)
        test_folds.append(booster.predict(dte, iteration_range=(0, bi + 1)))
        bal = balanced_accuracy_score(y[va_idx], oof_meta[va_idx].argmax(1))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax={bal:.5f} wall={time.time()-t1:.1f}s")

    test_meta = np.mean(test_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_xgb_metastack_v3.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_v3.npy", test_meta)
    log("saved oof/test_xgb_metastack_v3.npy")

    # Standalone + blend sweep at fixed recipe bias (binhigh-rule compliant).
    meta_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
    meta_tuned = bal_at_bias(oof_meta, y)
    log(f"\n=== standalone v3 ===")
    log(f"  argmax OOF = {meta_argmax:.5f}")
    log(f"  @recipe-bias OOF = {meta_tuned:.5f}")

    from common import log_blend
    log(f"\n=== fixed-bias α-sweep vs LB-best 3-stack (anchor = {lb_bal:.5f}) ===")
    rows = []
    for a in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        b = bal_at_bias(log_blend([lb_oof, oof_meta], np.array([1 - a, a])), y)
        d = b - lb_bal
        rows.append({"alpha": a, "oof": float(b), "delta": float(d)})
        log(f"  α={a:.3f}  OOF={b:.5f}  Δ={d:+.5f}")
    best = max(rows, key=lambda r: r["delta"])

    pred_lb = (np.log(np.clip(lb_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_meta = (np.log(np.clip(oof_meta, 1e-12, 1)) + BIAS).argmax(1)
    errs_lb = int((pred_lb != y).sum())
    errs_meta = int((pred_meta != y).sum())
    inter = int(((pred_lb != y) & (pred_meta != y)).sum())
    union = int(((pred_lb != y) | (pred_meta != y)).sum())
    jacc = inter / max(union, 1)
    log(f"\nerrs LB={errs_lb}  v3={errs_meta}  Jaccard={jacc:.4f}")

    out = dict(components=component_names, n_components=len(component_names),
               feature_dim=X_tr.shape[1], best_iters=[int(b) for b in best_iters],
               meta_argmax=float(meta_argmax), meta_tuned=float(meta_tuned),
               lb_oof=float(lb_bal), sweep=rows, best=best,
               err_lb=errs_lb, err_meta=errs_meta, jaccard_vs_lb=float(jacc),
               elapsed_sec=float(time.time() - t0))
    (ART / "tier1b_metastack_v3_results.json").write_text(json.dumps(out, indent=2))
    log("wrote tier1b_metastack_v3_results.json")


if __name__ == "__main__":
    main()
