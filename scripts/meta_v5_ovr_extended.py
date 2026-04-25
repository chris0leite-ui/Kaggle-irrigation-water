"""Meta v5: XGB meta-stacker with OvR + recipe_focal_effnum + other genuinely-
new components added since v1's training. Tests two attack vectors:

  R) REPLACE-v1: v5_iso × α onto LB-best 3-stack (gate at +2e-4 vs lb_best_3stack
     0.98061 OOF, with per-class recall guardrail).
  S) STACK-ON-TOP: v5_iso × α onto LB-best 4-stack (gate at +2e-4 vs 0.98084).

Saves to oof_xgb_metastack_v5.npy + test_..._v5.npy. Does NOT clobber v1.
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
from common import add_distance_features, log_blend  # noqa: E402
from tier1b_helpers import (BIAS, ART, SUB, DATA, SEED, N_FOLDS, CLS2IDX,
                            CLASSES, TARGET, normed, iso_cal,
                            build_lbbest_stack, load_y, bal_at_bias)  # noqa: E402

# Stricter EXCLUDE: all circular metas + confirmed LB regressors +
# bias-mismatched + submission-derived + binary specialists.
EXTRA_EXCLUDE = {
    # All XGB meta outputs (circular for v5).
    "xgb_metastack", "xgb_metastack_v2", "xgb_metastack_v3",
    "xgb_metastack_v3_iso", "xgb_metastack_v4",
    "xgb_metastack_varB", "xgb_metastack_varC",
    "xgb_metastack_bag3", "xgb_metastack_j2bag", "xgb_metastack_narrow",
    "lr_metastack", "lr_metastack_v2",
    # LB regressors (gap >+0.001 confirmed).
    "soft_distill", "soft_distill_small", "soft_distill_tiny",
    # Bias-mismatched (post-SMOTE class prior).
    "recipe_smote_v3",
    # Submission-derived (circular vs primary).
    "primary_sub_tau095", "primary_sub_tau097", "primary_sub_tau099",
    # Derived blends.
    "j6_qp_blend", "greedy_blend", "ovo_boundary_blend",
    "ovo_nonrule_blend", "bagged_greedy_nonrule", "greedy_full_bank_6way",
    "hedge_avg_lb_bests",
    # Tau-sweep pseudos (circular w.r.t. pseudo_s1).
    "recipe_pseudolabel_tau092", "recipe_pseudolabel_tau095",
    "recipe_pseudolabel_tau097", "recipe_pseudolabel_tau099",
    # MLP/NN partial-fold artefacts (might be 0-filled outside fold).
    "tta_recipe_baseline", "tta_recipe_s001", "tta_recipe_s005",
    "tta_recipe_s010", "tta_recipe_s020", "tta_recipe_s030",
    # P3 + leaf_ote: Jaccard 0.65 with 1500+ extra errs — kept v1, drop here
    # to test if leaner pool helps. Will rerun with them in if v5 is null.
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_pool_strict():
    pool = {}
    for oof_p in sorted(ART.glob("oof_*.npy")):
        name = oof_p.stem.replace("oof_", "", 1)
        if name in EXTRA_EXCLUDE:
            continue
        test_p = ART / f"test_{name}.npy"
        if not test_p.exists():
            continue
        try:
            o = np.load(oof_p).astype(np.float32)
            t = np.load(test_p).astype(np.float32)
        except Exception:
            continue
        if o.ndim != 2 or o.shape[1] != 3:
            continue
        # Reject partial-fold (any row sums < 1e-3 → 0-filled).
        if (o.sum(1) < 1e-3).any():
            continue
        pool[name] = (normed(o), normed(t))
    # Apply tier1b_helpers.EXCLUDE too.
    from tier1b_helpers import EXCLUDE as TIER1B_EXCL
    pool = {k: v for k, v in pool.items() if k not in TIER1B_EXCL}
    return pool


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb3_o, lb3_t = build_lbbest_stack(y)
    lb3_bal = bal_at_bias(lb3_o, y)
    log(f"  LB-best 3-stack OOF = {lb3_bal:.5f}")

    log("loading v1_iso (existing meta) for the LB-best 4-stack anchor")
    v1 = (normed(np.load(ART / "oof_xgb_metastack.npy")),
          normed(np.load(ART / "test_xgb_metastack.npy")))
    v1_iso_o, v1_iso_t = iso_cal(v1[0], v1[1], y)
    lb4_o = log_blend([lb3_o, v1_iso_o], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, v1_iso_t], np.array([0.7, 0.3]))
    lb4_bal = bal_at_bias(lb4_o, y)
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f} (target = 0.98084)")

    log("loading pool (strict EXCLUDE)")
    pool = load_pool_strict()
    log(f"  {len(pool)} 3-class components loaded")
    has_ovr = any('ovr' in k for k in pool)
    has_focal = any('focal' in k for k in pool)
    log(f"  OvR present: {has_ovr}   focal present: {has_focal}")

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

    component_names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb3_o, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb3_t, 1e-9, 1.0))

    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1)
    log(f"  meta-feature shape: {X_tr.shape}")

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
        booster = xgb.train(xgb_params, dtr, num_boost_round=3000,
                            evals=[(dva, "val")], early_stopping_rounds=200,
                            verbose_eval=0)
        bi = booster.best_iteration
        best_iters.append(bi)
        vp = booster.predict(dva, iteration_range=(0, bi + 1))
        oof_meta[va_idx] = vp.astype(np.float32)
        tp = booster.predict(dte, iteration_range=(0, bi + 1))
        test_meta_folds.append(tp)
        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax_bal={argmax_bal:.5f} "
            f"wall={time.time()-t1:.1f}s")

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_xgb_metastack_v5.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_v5.npy", test_meta)
    log("saved oof_xgb_metastack_v5.npy + test_xgb_metastack_v5.npy")

    v5_iso_o, v5_iso_t = iso_cal(oof_meta, test_meta, y)
    np.save(ART / "oof_xgb_metastack_v5_iso.npy", v5_iso_o)
    np.save(ART / "test_xgb_metastack_v5_iso.npy", v5_iso_t)

    v5_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
    v5_tuned = bal_at_bias(oof_meta, y)
    v5_iso_tuned = bal_at_bias(v5_iso_o, y)
    log(f"\n=== v5 standalone ===")
    log(f"  argmax = {v5_argmax:.5f}")
    log(f"  raw @recipe-bias = {v5_tuned:.5f}  (v1 was 0.98041)")
    log(f"  iso @recipe-bias = {v5_iso_tuned:.5f}  (v1_iso was 0.98059)")

    def sweep(anchor_o, anchor_label, anchor_target):
        log(f"\n=== blend sweep: v5_iso onto {anchor_label} (target {anchor_target}) ===")
        anchor_bal = bal_at_bias(anchor_o, y)
        log(f"  anchor OOF = {anchor_bal:.5f}")
        anchor_pred = (np.log(np.clip(anchor_o, 1e-12, 1)) + BIAS).argmax(1)
        anchor_recall = [
            ((anchor_pred == c) & (y == c)).sum() / max((y == c).sum(), 1)
            for c in range(3)
        ]
        log(f"  anchor recall: L={anchor_recall[0]:.5f} M={anchor_recall[1]:.5f} "
            f"H={anchor_recall[2]:.5f}")
        rows = []
        alphas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2,
                  0.25, 0.3, 0.35, 0.4, 0.5]
        for a in alphas:
            blend = log_blend([anchor_o, v5_iso_o], np.array([1 - a, a]))
            b = bal_at_bias(blend, y)
            d = b - anchor_bal
            blend_pred = (np.log(np.clip(blend, 1e-12, 1)) + BIAS).argmax(1)
            r = [((blend_pred == c) & (y == c)).sum() / max((y == c).sum(), 1)
                 for c in range(3)]
            rd = [r[c] - anchor_recall[c] for c in range(3)]
            guardrail = all(d_ >= -5e-4 for d_ in rd)
            rows.append({"alpha": a, "oof": float(b), "delta": float(d),
                         "recL": float(r[0]), "recM": float(r[1]),
                         "recH": float(r[2]), "guardrail": guardrail})
            tag = " ← GATE PASS" if d >= 2e-4 and guardrail else ""
            log(f"  α={a:.3f} OOF={b:.5f} Δ={d:+.5f} "
                f"L={r[0]:.4f} M={r[1]:.4f} H={r[2]:.4f}{tag}")
        return rows

    rows_3stack = sweep(lb3_o, "LB-best 3-stack", "0.98061")
    rows_4stack = sweep(lb4_o, "LB-best 4-stack", "0.98084")

    out = dict(
        components=component_names,
        n_components=len(component_names),
        feature_dim=X_tr.shape[1],
        best_iters=[int(b) for b in best_iters],
        v5_standalone_argmax=float(v5_argmax),
        v5_standalone_raw_tuned=float(v5_tuned),
        v5_iso_tuned=float(v5_iso_tuned),
        sweep_vs_3stack=rows_3stack,
        sweep_vs_4stack=rows_4stack,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "meta_v5_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote meta_v5_results.json (elapsed {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
