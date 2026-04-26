"""Combined eval: rebuild meta-stacker with new #3+#4+#6 components, blend gate.

Adds to the LB-validated tier1b meta-stacker setup:
  - oof_poly_fe (#4): new (N,3) component, drops into the pool directly.
  - oof_aux_flipped_from_rule, oof_aux_missed_high, oof_aux_missed_medium (#3):
    binary OOFs (N,) — appended as extra log-prob columns to the meta
    feature matrix.
  - oof_masked_resid (#6): 14-col self-supervised residuals — appended as
    extra raw columns to the meta feature matrix.

Decision gate vs LB-best 4-stack (which is the actual primary at LB 0.98094):
  iso(meta) standalone @ recipe bias ≥ 0.98061 + Jaccard < 0.97 + errs < 9572
  AND blend Δ ≥ +2e-4 OOF
  AND per-class recall guardrail (≥ anchor − 5e-4 each)

If pass: build a candidate submission, ASK user before LB probe.
If fail: 11th saturation confirmation.
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
from common import CLS2IDX, add_distance_features, log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    BIAS, ART, DATA, SEED, N_FOLDS,
    bal_at_bias, build_lbbest_stack, iso_cal, load_pool, normed,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor (= primary's pre-meta state)")
    lb_oof, lb_test = build_lbbest_stack(y)
    lb_bal = bal_at_bias(lb_oof, y)
    log(f"  LB-best 3-stack OOF = {lb_bal:.5f}")

    # Reconstruct primary 4-stack (= 0.7 × 3-stack + 0.3 × xgb_metastack_iso).
    # This is the actual LB 0.98094 submission target.
    log("reconstructing LB-best 4-stack PRIMARY")
    meta_v1 = (np.load(ART / "oof_xgb_metastack.npy"),
                 np.load(ART / "test_xgb_metastack.npy"))
    meta_v1_iso_o, meta_v1_iso_t = iso_cal(meta_v1[0], meta_v1[1], y)
    primary_oof = log_blend([lb_oof, meta_v1_iso_o], np.array([0.70, 0.30]))
    primary_test = log_blend([lb_test, meta_v1_iso_t], np.array([0.70, 0.30]))
    primary_bal = bal_at_bias(primary_oof, y)
    log(f"  primary 4-stack OOF = {primary_bal:.5f}  (target 0.98084)")

    # Standard pool for meta-stacker
    extra_excl = {
        "soft_distill", "xgb_spec_678", "xgb_spec_36",
        "recipe_pseudolabel_stage2",
        "spec_mh_v3_score5", "spec_mh_v3_score6", "spec6_mh", "spec6_mh_v2",
        "xgb_bin_medium", "xgb_bin_high", "binhigh",
        "p_flip", "pflip", "missed_high", "flip_correction",
        "selective_router", "disagree_meta",
        "c0_safe_lb_best_2way", "c0_safe_recipe_full_te",
        "c0_v2_lb_best_2way", "c0_v2_lb_best_3way", "c0_v2_recipe_full_te",
        "c0_v3_lb_best_3way", "c0_v3_recipe_full_te",
        "b2_groupkfold_region", "step1_greedy_lbbest",
        "hybrid_binhigh", "meta_v3", "eb_cell",
        # Prior meta outputs
        "xgb_metastack", "xgb_metastack_v2", "xgb_metastack_v3", "xgb_metastack_v4",
        "xgb_metastack_v5", "xgb_metastack_v3_iso", "xgb_metastack_bag3",
        "xgb_metastack_j2bag", "xgb_metastack_narrow",
        "xgb_nonrule_bag3",
        # Other branch derived
        "trompt_probe", "kan_probe",
        "soft_distill_small", "soft_distill_tiny", "soft_distill_recipeonly",
        "lr_metastack", "lr_metastack_v2", "lr_metastack_v2_isoafter",
        "perturbed_v1", "perturbed_v2", "perturbed_62_v1",
        "p_flip", "p3_embed_propagate", "j6_qp_blend",
        "greedy_blend", "ovo_boundary_blend",
        "primary_sub_tau095", "primary_sub_tau097", "primary_sub_tau099",
        "tta_recipe_baseline", "tta_recipe_s001", "tta_recipe_s005", "tta_recipe_s010",
        # The new poly_fe is what we're testing — keep IT in the pool
    }
    log("loading pool")
    pool = load_pool(extra_excl)
    log(f"  base pool: {len(pool)} components")

    # Mark whether poly_fe is present
    poly_present = "poly_fe" in pool
    log(f"  poly_fe in pool: {poly_present}")

    # Build the meta-feature matrix: [primary_log_probs, dist_features,
    #                                  *per_component_log_probs, *aux_columns,
    #                                  *masked_residual_columns]
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
    primary_log_tr = np.log(np.clip(primary_oof, 1e-9, 1.0))
    primary_log_te = np.log(np.clip(primary_test, 1e-9, 1.0))

    component_names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]

    # Aux columns (logits of binary probs)
    aux_tr_cols = []
    aux_te_cols = []
    aux_loaded = []
    for aux_name in ["aux_flipped_from_rule", "aux_missed_high", "aux_missed_medium"]:
        oof_p = ART / f"oof_{aux_name}.npy"
        test_p = ART / f"test_{aux_name}.npy"
        if oof_p.exists() and test_p.exists():
            o = np.load(oof_p).astype(np.float32)
            t = np.load(test_p).astype(np.float32)
            if o.ndim == 1:
                o = o[:, None]
                t = t[:, None]
            # Add prob, logit, abs(logit)
            for x in [o, t]:
                pass
            logit_o = np.log(np.clip(o, 1e-6, 1 - 1e-6) /
                              np.clip(1 - o, 1e-6, 1 - 1e-6))
            logit_t = np.log(np.clip(t, 1e-6, 1 - 1e-6) /
                              np.clip(1 - t, 1e-6, 1 - 1e-6))
            aux_tr_cols.append(np.concatenate([o, logit_o], axis=1))
            aux_te_cols.append(np.concatenate([t, logit_t], axis=1))
            aux_loaded.append(aux_name)
            log(f"  loaded aux: {aux_name}  shape {o.shape}")
        else:
            log(f"  aux NOT FOUND: {aux_name}")

    # Masked-resid columns (raw)
    masked_present = (ART / "oof_masked_resid.npy").exists()
    if masked_present:
        mt = np.load(ART / "oof_masked_resid.npy").astype(np.float32)
        mtt = np.load(ART / "test_masked_resid.npy").astype(np.float32)
        log(f"  loaded masked-resid  shape {mt.shape}")
    else:
        mt = np.zeros((len(train), 0), dtype=np.float32)
        mtt = np.zeros((len(test), 0), dtype=np.float32)
        log("  masked-resid NOT FOUND")

    # Stack everything into X
    parts_tr = [primary_log_tr, meta_tr] + comp_tr + aux_tr_cols + [mt]
    parts_te = [primary_log_te, meta_te] + comp_te + aux_te_cols + [mtt]
    X_tr = np.concatenate(parts_tr, axis=1).astype(np.float32)
    X_te = np.concatenate(parts_te, axis=1).astype(np.float32)
    log(f"  meta-feature shape: train {X_tr.shape}  test {X_te.shape}")
    log(f"  pool n={len(pool)}  aux={len(aux_loaded)}  masked={mt.shape[1]} cols")

    # Train meta-stacker (same heavy-reg HPs as LB-validated v1)
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
    test_folds = []
    log("training meta-stacker")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(xgb_params, dtr, num_boost_round=3000,
                              evals=[(dva, "v")], early_stopping_rounds=200,
                              verbose_eval=0)
        bi = booster.best_iteration
        oof_meta[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_folds.append(booster.predict(dte, iteration_range=(0, bi + 1)))
        b = balanced_accuracy_score(y[va_idx], oof_meta[va_idx].argmax(1))
        log(f"  fold {fold+1} it={bi}  argmax_bal={b:.5f}  wall={time.time()-t1:.1f}s")
    test_meta = np.mean(test_folds, axis=0).astype(np.float32)

    np.save(ART / "oof_xgb_metastack_v6_combined.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_v6_combined.npy", test_meta)

    meta_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
    meta_at_bias = bal_at_bias(oof_meta, y)
    iso_o, iso_t = iso_cal(oof_meta, test_meta, y)
    iso_at_bias = bal_at_bias(iso_o, y)
    log(f"\n=== META-STACKER V6 (combined) ===")
    log(f"  argmax OOF       = {meta_argmax:.5f}")
    log(f"  @recipe-bias     = {meta_at_bias:.5f}")
    log(f"  iso @recipe-bias = {iso_at_bias:.5f}")
    log(f"  v1 baseline @rb  = {bal_at_bias(meta_v1[0], y):.5f}  iso = {bal_at_bias(meta_v1_iso_o, y):.5f}")

    # Blend sweep vs LB-best 4-stack PRIMARY (the actual 0.98094 anchor)
    log(f"\n=== blend gate vs PRIMARY 4-stack (target 0.98084) ===")
    log(f"  primary OOF        = {primary_bal:.5f}")
    rows = []
    for a in [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        b = log_blend([primary_oof, iso_o], np.array([1 - a, a]))
        bb = bal_at_bias(b, y)
        d = bb - primary_bal
        rows.append({"alpha": a, "oof": float(bb), "delta": float(d)})
        log(f"  α={a:>5.3f} OOF={bb:.5f}  Δ={d:+.5f}")
    best = max(rows, key=lambda r: r["delta"])
    log(f"  PEAK: α={best['alpha']:.3f}  Δ={best['delta']:+.5f}")

    # Per-class recall guardrail
    pred_pri = (np.log(np.clip(primary_oof, 1e-12, 1)) + BIAS).argmax(1)
    a_peak = best["alpha"]
    blend_peak = log_blend([primary_oof, iso_o], np.array([1 - a_peak, a_peak]))
    pred_peak = (np.log(np.clip(blend_peak, 1e-12, 1)) + BIAS).argmax(1)
    rec_peak = []
    rec_pri = []
    for c in range(3):
        m = (y == c)
        rec_peak.append((pred_peak[m] == c).mean())
        rec_pri.append((pred_pri[m] == c).mean())
    log(f"  per-class recall: primary={rec_pri}  blend={rec_peak}")
    pcr_pass = all(rec_peak[c] >= rec_pri[c] - 5e-4 for c in range(3))
    log(f"  per-class guardrail (Δ ≥ -5e-4 each): {'PASS' if pcr_pass else 'FAIL'}")

    # Jaccard + errs
    pred_meta_for_diag = (np.log(np.clip(iso_o, 1e-12, 1)) + BIAS).argmax(1)
    errs_meta = (pred_meta_for_diag != y).sum()
    errs_pri = (pred_pri != y).sum()
    inter = ((pred_meta_for_diag != y) & (pred_pri != y)).sum()
    union = ((pred_meta_for_diag != y) | (pred_pri != y)).sum()
    jacc = inter / max(union, 1)
    log(f"  errs: primary={errs_pri}  meta_iso={errs_meta}  Jaccard={jacc:.4f}")

    summary = {
        "n_components": len(pool),
        "n_aux_loaded": len(aux_loaded),
        "n_masked_cols": int(mt.shape[1]),
        "X_shape": list(X_tr.shape),
        "primary_oof": float(primary_bal),
        "meta_argmax": float(meta_argmax),
        "meta_at_bias": float(meta_at_bias),
        "iso_at_bias": float(iso_at_bias),
        "blend_sweep": rows,
        "best_alpha": best["alpha"],
        "best_delta": best["delta"],
        "errs_primary": int(errs_pri),
        "errs_meta_iso": int(errs_meta),
        "jaccard": float(jacc),
        "per_class_recall_primary": rec_pri,
        "per_class_recall_peak": rec_peak,
        "pcr_guardrail_pass": bool(pcr_pass),
        "wall_seconds": time.time() - t0,
    }
    with open(ART / "combined_meta_eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nwrote combined_meta_eval_results.json  total wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
