"""Meta-stacker v4 — XGB metastack on bank + N2 (ET + kNN) [+ N3 if present].

Mirrors `tier1b_xgb_metastack.py` but with EXTRA_INCLUDE explicitly listing
the new N2 / N3 components so we know whether they entered the bank. Output
saved as `oof_xgb_metastack_v4.npy` etc., NOT clobbering the LB-best
metastacker.

Auto-includes any of these if their .npy files exist:
  - oof_n2_extratrees.npy
  - oof_n2_knn.npy
  - oof_recipe_5shuffle.npy   (N3, when finished)

If recipe_5shuffle is present, this is effectively v5; if only N2, v4.
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
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, N_FOLDS, SEED, SUB, TARGET,
    bal_at_bias as bal, build_lbbest_stack, iso_cal, load_pool, log,
)

NEW_COMPONENTS = ["n2_extratrees", "n2_knn", "recipe_5shuffle"]


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    lb_bal = bal(lb_oof, y)
    log(f"  LB-best 3-stack OOF = {lb_bal:.5f}")

    log("loading pool (incl. any new N2/N3 components)")
    pool = load_pool()
    new_present = [c for c in NEW_COMPONENTS if c in pool]
    log(f"  total pool: {len(pool)} components")
    log(f"  new N2/N3 components present: {new_present}")

    # Meta features
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

    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1)
    log(f"  meta-feature shape: {X_tr.shape}  (n_components={len(component_names)})")

    xgb_params = dict(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
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
        best_iters.append(bi)
        oof_meta[va_idx] = booster.predict(dva, iteration_range=(0, bi+1)).astype(np.float32)
        test_meta_folds.append(booster.predict(dte, iteration_range=(0, bi+1)))
        argmax_bal = balanced_accuracy_score(y[va_idx], oof_meta[va_idx].argmax(1))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax_bal={argmax_bal:.5f} "
            f"wall={time.time()-t1:.1f}s")

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)

    suffix = "_v4" if "recipe_5shuffle" not in new_present else "_v5"
    np.save(ART / f"oof_xgb_metastack{suffix}.npy", oof_meta)
    np.save(ART / f"test_xgb_metastack{suffix}.npy", test_meta)
    log(f"saved oof_xgb_metastack{suffix}.npy (suffix reflects whether N3 included)")

    # Standalone diagnostic
    raw_argmax = (oof_meta.argmax(1) == y).mean()
    raw_tuned = bal(oof_meta, y)
    iso_o, iso_t = iso_cal(oof_meta, test_meta, y)
    iso_argmax = (iso_o.argmax(1) == y).mean()
    iso_tuned = bal(iso_o, y)
    log(f"\n=== meta_v_{suffix[1:]} standalone ===")
    log(f"  raw  argmax {raw_argmax:.5f}  tuned {raw_tuned:.5f}")
    log(f"  iso  argmax {iso_argmax:.5f}  tuned {iso_tuned:.5f}")

    # Compare vs prior XGB metastack iso (LB-best 4-stack uses this @ α=0.30)
    prior_oof = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    prior_test = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    prior_iso_o, prior_iso_t = iso_cal(prior_oof, prior_test, y)
    prior_iso_tuned = bal(prior_iso_o, y)
    log(f"\nprior XGB metastack iso OOF tuned = {prior_iso_tuned:.5f}")
    log(f"new   XGB metastack iso OOF tuned = {iso_tuned:.5f}  "
        f"Δ = {iso_tuned - prior_iso_tuned:+.5f}")

    # Errors / Jaccard vs LB-best 4-stack (rebuild here — same as helpers
    # build_lbbest_stack + xgb_metastack_iso @ α=0.30)
    lb4_o = log_blend([lb_oof, prior_iso_o], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb_test, prior_iso_t], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_o, y)
    log(f"\nLB-best 4-stack OOF = {lb4_bal:.5f}")

    # Try replacing the prior metastack iso with new one in the 4-stack
    new4_o = log_blend([lb_oof, iso_o], np.array([0.7, 0.3]))
    new4_t = log_blend([lb_test, iso_t], np.array([0.7, 0.3]))
    new4_bal = bal(new4_o, y)
    log(f"new 4-stack (replace metastack with v_{suffix[1:]}) OOF = {new4_bal:.5f}  "
        f"Δ vs LB-best 4-stack = {new4_bal - lb4_bal:+.5f}")

    # Or blend new metastack on top of LB-best 4-stack
    log(f"\n=== blend new_iso × LB-best 4-stack (fixed bias) ===")
    log(f"{'α':>6} {'OOF':>9} {'Δ':>9}")
    rows = []
    for a in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        blend = log_blend([lb4_o, iso_o], np.array([1 - a, a]))
        b = bal(blend, y)
        d = b - lb4_bal
        rows.append({"alpha": float(a), "oof": float(b), "delta": float(d)})
        tag = " *" if d > 1e-4 else ""
        log(f"{a:>6.2f} {b:>9.5f} {d:>+9.5f}{tag}")
    best = max(rows, key=lambda r: r["delta"])

    # Emit submission if Δ ≥ +2e-4
    if best["delta"] >= 2e-4:
        a = best["alpha"]
        test_blend = log_blend([lb4_t, iso_t], np.array([1 - a, a]))
        pred_t = (np.log(np.clip(test_blend, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_tier1c_meta{suffix}_a{int(a*100):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"\n→ wrote {path} (Δ {best['delta']:+.5f} ≥ +2e-4 gate)")
    else:
        log(f"\nbest blend Δ {best['delta']:+.5f} below +2e-4 gate; no submission")

    out = dict(
        suffix=suffix,
        n_components=len(component_names),
        new_components=new_present,
        feature_dim=int(X_tr.shape[1]),
        best_iters=[int(b) for b in best_iters],
        raw_argmax=float(raw_argmax), raw_tuned=float(raw_tuned),
        iso_argmax=float(iso_argmax), iso_tuned=float(iso_tuned),
        prior_iso_tuned=float(prior_iso_tuned),
        lb4_bal=float(lb4_bal),
        new4_replace_bal=float(new4_bal),
        blend_sweep=rows, best=best,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / f"tier1c_metastack{suffix}_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote scripts/artifacts/tier1c_metastack{suffix}_results.json")


if __name__ == "__main__":
    main()
