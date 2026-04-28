"""Macro-recall surrogate meta on BASE-ONLY pool (no derived metas).

Eliminates the circularity discovered in Stage 1: N1 was meta-of-metas
(80% of gain from other meta outputs). This re-trains with ALL prior
metas / LB-regressors / derived components dropped.

Hyperparameter choices: theoretical only (no OOF grid search):
  - lam_ce = 0.3 (LB-validated SMOKE choice from prior macrorec session)
  - α = 0.30 (LB-validated architecture from primary 4-stack)
  - depth=4, reg_alpha=5, reg_lambda=5 (heavy-reg, same as v1 metastack)

Output suffix `_baseonly` so it doesn't clobber prior outputs.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX
from recipe_macrorecall import make_macrorec_obj, macrorec_eval_metric
from tier1b_xgb_metastack import (
    EXCLUDE, _normed, build_lbbest_stack, iso_cal, load_pool, BIAS,
)

# Comprehensive base-only exclusion set
EXTRA_EXCLUDE_BASEONLY = {
    # Macrorec self-references (circular)
    "recipe_full_te_macrorec_T1_lam03",
    "xgb_metastack_metamacrorec_lam03",
    "xgb_metastack_metamacrorec_lam03_iso",
    "xgb_metastack_metamacrorec_lam0",
    "xgb_metastack_metamacrorec_lam0_iso",
    "xgb_metastack_metamacrorec_lam03_curated",
    "xgb_metastack_metamacrorec_lam03_curated_iso",
    # All prior CE meta-stackers
    "xgb_metastack_v3", "xgb_metastack_v4", "xgb_metastack_v5", "xgb_metastack_v5_iso",
    "xgb_metastack_v6", "xgb_metastack_v6_combined", "xgb_metastack_v6lb",
    "xgb_metastack_varB", "xgb_metastack_varC",
    "xgb_metastack_classw", "xgb_metastack_n5b_both", "xgb_metastack_3wnn",
    "xgb_metastack_bag3", "xgb_metastack_j2bag", "xgb_metastack_narrow",
    "lr_metastack", "lr_metastack_v2", "mlp_metastack",
    "meta_l3_xgb_mlp", "three_meta_l3",
    "meta_perturbed_v1_noise03_csb09_k3", "meta_perturbed_v2_noise05_csb05_k3",
    "meta_perturbed_62_v1",
    # Branch-NULL components
    "recipe_full_te_residte", "recipe_full_te_basemargin_K2",
    "recipe_full_te_dropdet", "tier1b_greedy_meta_l1override",
    # Derived ensemble outputs
    "hillclimb_negweights", "hedge_avg_lb_bests", "greedy_blend",
    "soft_distill_recipeonly", "soft_distill_small", "soft_distill_tiny",
    "distill_no_rule", "ovo_nonrule_blend", "em_uniform",
    "primary_sub_tau095", "primary_sub_tau097", "primary_sub_tau099",
    "recipe_pseudolabel_tau092", "recipe_pseudolabel_tau095",
    "recipe_pseudolabel_tau097", "recipe_pseudolabel_tau099",
    "angle_c3a_mixup", "angle_c3b_mixup", "angle_c2_mixup", "angle_c_mixup",
    "angle_a_residual", "angle_a_residual_raw",
    "recipe_full_te_anchw20", "recipe_full_te_3way", "recipe_full_te_ood_knn10k_ood9",
    "leaf_ote_meta_v2", "leaf_ote_meta",
    "bagged_greedy_nonrule",
    "n5b_residual_head", "n5b_residual_3class", "n5b_residual_auc",
    # own_S* are own-CSV ensembles — derivative
    "own_S1_equal_log", "own_S2_lb_weighted_tau100", "own_S2_lb_weighted_tau200",
    "own_S2_lb_weighted_tau500", "own_S2_lb_weighted_tau1000",
    "own_S3_hard_vote", "own_S4_soft_vote", "own_S5_greedy_forward",
    "own_greedy_fine", "own_3view",
    # Other derivative blends
    "joint_blend", "moe_gated", "mech_d", "j6_qp_blend", "c0_greedy",
    "p3_embed_propagate", "per_bin_blend", "per_cell_meta",
    "cmaes_blend", "ovo_boundary_blend",
    # Multi-seed pseudo (already in core LB-best, redundant as separate inputs)
    "lb_best_fs7", "lb_best_fs123",
    # Path B/C derived
    "path_b_cell_mlp", "path_c_primary_labeler",
    # Multi-task (variant-style)
    "multitask_xgb",
    # Greedy bank derived
    "greedy_full_bank_6way",
    # Adversarial
    "recipe_adv_s050",
    # Focal variants — these are direct base predictors but adversarial
    "recipe_focal_effnum", "recipe_focal_g2_aH1", "recipe_focal_g2_invfreq",
    "recipe_focal_g2h3",
    # Hybrid blend
    "hybrid_lgbmxgb_blend",
    # CatBoost variants — keeping recipe_catboost as base
    "catboost_optuna", "catboost_recipe_gpu",
    # KAN, ExtraTrees — kept as base for diversity
    # Blends from kernels
    "lgbm_competitor",
    # 5shuffle / 2shuffle / poly_fe / 171pair / allpairs — keep as base
}


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
DATA = Path("data")
ART = Path("scripts/artifacts")
SUFFIX = "_metamacrorec_baseonly"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def main():
    log(f"BASE-ONLY macrorec meta-stacker  lam_ce=0.3  T=1.0")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best stack OOF = {bal(lb_oof, y):.5f}")

    log("loading BASE-ONLY pool")
    EXCLUDE.update(EXTRA_EXCLUDE_BASEONLY)
    pool = load_pool(y)
    log(f"  {len(pool)} base components loaded (vs N1's 170)")
    for n in sorted(pool.keys()):
        log(f"    {n}")

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

    names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))
    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1).astype(np.float32)
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
        n_components=len(pool),
        meta_feature_shape=list(X_tr.shape),
        fold_scores_argmax=[float(s) for s in fold_scores],
        best_iters=[int(b) for b in best_iters],
        overall_argmax_bal_acc=float(overall_argmax),
        overall_tuned_bal_acc=float(overall_tuned),
        iso_argmax_bal_acc=float(iso_argmax),
        iso_tuned_bal_acc=float(iso_tuned),
        component_names=names,
    )
    with open(ART / f"xgb_metastack{SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote results JSON")


if __name__ == "__main__":
    main()
