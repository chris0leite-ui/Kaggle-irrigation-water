"""SVGP meta-stacker on the same 63-component bank that produced LB 0.98094.

Mirrors tier1b_xgb_metastack.py architecture EXACTLY (same EXCLUDE list,
same fold split, same fixed bias) — only the meta model class changes
from XGB to SVGP. This isolates the kernel/non-parametric inductive
bias as the lever; differences in OOF/Jaccard cleanly attribute to GP.

Per-fold checkpointing via RUN_FOLD env (1..5). SMOKE=1 for the
20k×2-fold smoke test before production launch.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (ART, BIAS, build_lbbest_stack,  # noqa: E402
                             load_y, bal_at_bias)
import tier1b_xgb_metastack as t1b  # noqa: E402
from svgp_helpers import (DEFAULT_M, DEFAULT_BATCH, DEFAULT_EPOCHS,  # noqa: E402
                           DEFAULT_LR, fit_svgp, predict_proba)

# Curated pool: v1 LB-validated EXCLUDE + drop all derived metas / LB-regressors
# added since 2026-04-25. Mirrors v8 clean-pool exactly so this experiment
# tests "SVGP meta on the LB-best v1 architecture" (only meta class swaps).
META_EXCLUDE_EXTRA = {
    "xgb_metastack", "xgb_metastack_v2", "xgb_metastack_v3", "xgb_metastack_v3_iso",
    "xgb_metastack_v4", "xgb_metastack_v5", "xgb_metastack_v5_iso",
    "xgb_metastack_v6", "xgb_metastack_v6_combined", "xgb_metastack_v6lb",
    "xgb_metastack_v7", "xgb_metastack_v7b",
    "xgb_metastack_varB", "xgb_metastack_varC",
    "xgb_metastack_3wnn", "xgb_metastack_b2clean",
    "xgb_metastack_bag3", "xgb_metastack_classw",
    "xgb_metastack_heavy", "xgb_metastack_j2bag",
    "xgb_metastack_n5b_both", "xgb_metastack_narrow",
    "xgb_metastack_perfoldiso_inputs",
    "xgb_metastack_v1_cleanpool", "xgb_metastack_v1_groupkfold",
    "xgb_metastack_v1_plus_newfe",
    "xgb_metastack_metamacrorec_baseonly",
    "xgb_metastack_metamacrorec_baseonly_iso",
    "xgb_metastack_metamacrorec_lam0", "xgb_metastack_metamacrorec_lam03",
    "xgb_metastack_metamacrorec_lam03_curated",
    "xgb_metastack_metamacrorec_lam03_curated_iso",
    "xgb_metastack_metamacrorec_lam03_iso",
    "xgb_metastack_metamacrorec_lam0_iso",
    "xgb_metastack_metamacrorec_minimal",
    "xgb_metastack_metamacrorec_minimal_iso",
    "lr_metastack", "lr_metastack_v2",
    "mlp_metastack", "sklearn_rf_meta",
    "three_meta_l3", "meta_l3_xgb_mlp",
    "per_cell_meta", "leaf_ote_meta", "leaf_ote_meta_v2",
    "bagged_greedy_nonrule", "c0_greedy",
    "greedy_blend", "greedy_full_bank_6way",
    "hybrid_lgbmxgb_blend",
    "own_S5_greedy_forward", "own_greedy_fine",
    "distill_no_rule",
    "soft_distill_recipeonly", "soft_distill_small", "soft_distill_tiny",
    "xgb_ovr_recipe", "xgb_ovr_recipe_raw",
}
t1b.EXCLUDE = t1b.EXCLUDE | META_EXCLUDE_EXTRA

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
SMOKE = os.environ.get("SMOKE", "") == "1"
RUN_FOLD = int(os.environ.get("RUN_FOLD", "0"))  # 0 = run all, else 1..N_FOLDS
SUFFIX = os.environ.get("META_OUT_SUFFIX", "_svgp")
# Optional per-fold PCA reduction: PCA_DIM=50 cuts D=201 -> 50, ~4x faster
# kernel eval. Fit on tr_idx only (no leakage).
PCA_DIM = int(os.environ.get("PCA_DIM", "0")) or None

# LB-validated 62-component pool that produced LB 0.98094 (v1 meta).
# Hardcoded so this experiment is "v1 architecture with SVGP meta" exactly.
V1_COMPONENTS = [
    "bagged_greedy_nonrule", "c0_greedy", "catboost_optuna", "catboost_recipe_gpu",
    "em_uniform", "extratrees_dist_digits", "extratrees_dist_digits_v2",
    "greedy_blend", "greedy_full_bank_6way", "hybrid_lgbmxgb_blend",
    "lb_best_fs123", "lb_best_fs7", "lgbm_competitor", "lgbm_dist_digits",
    "lgbm_dist_digits_ote", "lgbm_te_orig", "ovo_boundary_blend",
    "ovo_nonrule_blend", "p3_embed_propagate", "realmlp",
    "recipe_171pair", "recipe_allpairs", "recipe_catboost", "recipe_focal_g2h3",
    "recipe_full_te", "recipe_full_te_a01", "recipe_full_te_a10",
    "recipe_full_te_catboost", "recipe_full_te_cldrop", "recipe_full_te_dae",
    "recipe_full_te_fexboth", "recipe_full_te_gby", "recipe_full_te_lgbm",
    "recipe_full_te_seed123", "recipe_full_te_seed7", "recipe_lgbm",
    "recipe_no_combos", "recipe_no_digits", "recipe_no_orig", "recipe_no_ote",
    "recipe_pseudolabel", "recipe_pseudolabel_seed123labeler",
    "recipe_pseudolabel_seed7labeler", "recipe_pseudolabel_tau092",
    "tabpfn", "tta_recipe_baseline", "tta_recipe_s001", "tta_recipe_s005",
    "tta_recipe_s010", "tta_recipe_s020", "tta_recipe_s030",
    "xgb_corn", "xgb_dist_digits", "xgb_dist_digits_ote",
    "xgb_dist_digits_ote_digits", "xgb_dist_digits_ote_digits_light",
    "xgb_dist_digits_ote_digits_pairs", "xgb_dist_digits_ote_light",
    "xgb_dist_routed_v3", "xgb_nonrule", "xgb_spec_36", "xgb_vanilla_dist",
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    if SMOKE:
        log("SMOKE: subsampling 20k stratified train rows + 5k test")
        rng = np.random.default_rng(SEED)
        sm_idx = []
        for c in range(3):
            ci = np.where(y == c)[0]
            sm_idx.extend(rng.choice(ci, min(len(ci), 20_000 // 3 + 1),
                                     replace=False))
        sm_idx = np.array(sorted(sm_idx))
        # We need to keep the FULL fold split aligned, so smoke trains
        # only on a subset of tr_idx but predicts on the full va_idx.
        # Simplest smoke: pretend the subset IS the full data.
        train = train.iloc[sm_idx].reset_index(drop=True)
        y = y[sm_idx]
        test = test.head(5000).reset_index(drop=True)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y) if not SMOKE else (
        # Smoke: skip anchor (gates not meaningful at 20k)
        np.full((len(y), 3), 1/3, dtype=np.float32),
        np.full((len(test), 3), 1/3, dtype=np.float32))
    if not SMOKE:
        log(f"  LB-best stack OOF = {bal_at_bias(lb_oof, y):.5f}")

    log("loading pool (LB-validated v1 62-component bank)")
    if SMOKE:
        pool = {}
    else:
        pool = {}
        for name in V1_COMPONENTS:
            o_p = ART / f"oof_{name}.npy"
            t_p = ART / f"test_{name}.npy"
            if not (o_p.exists() and t_p.exists()):
                log(f"  WARNING: missing {name}, skipping")
                continue
            o = np.load(o_p).astype(np.float32)
            tt = np.load(t_p).astype(np.float32)
            o = o / np.clip(o.sum(1, keepdims=True), 1e-9, None)
            tt = tt / np.clip(tt.sum(1, keepdims=True), 1e-9, None)
            pool[name] = (o, tt)
    if SMOKE:
        # Build a tiny synthetic pool for smoke: just lb_oof itself + zero
        # noise variants so we can exercise the SVGP plumbing on real shapes.
        pool = {f"smoke_{i}": (
            np.clip(lb_oof + np.random.randn(*lb_oof.shape) * 0.01, 1e-6, 1),
            np.clip(lb_test + np.random.randn(*lb_test.shape) * 0.01, 1e-6, 1))
            for i in range(5)}
    log(f"  {len(pool)} components loaded")

    log("constructing meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))
    X_tr_full = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1).astype(np.float32)
    X_te_full = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1).astype(np.float32)
    log(f"  meta-feature shape: {X_tr_full.shape}")

    M = 64 if SMOKE else DEFAULT_M
    epochs = 3 if SMOKE else DEFAULT_EPOCHS
    batch = 1024 if SMOKE else DEFAULT_BATCH

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((len(y), 3), dtype=np.float32)
    test_meta_folds = []

    folds = list(enumerate(skf.split(X_tr_full, y)))
    if RUN_FOLD > 0:
        folds = [folds[RUN_FOLD - 1]]

    for fold, (tr_idx, va_idx) in folds:
        t1 = time.time()
        log(f"=== fold {fold+1}/{N_FOLDS} (n_tr={len(tr_idx)} n_va={len(va_idx)}) ===")
        # Standardize: fit on tr_idx only, apply to va_idx + test.
        mu = X_tr_full[tr_idx].mean(0); sd = X_tr_full[tr_idx].std(0) + 1e-6
        X_tr = (X_tr_full[tr_idx] - mu) / sd
        X_va = (X_tr_full[va_idx] - mu) / sd
        X_te = (X_te_full - mu) / sd
        if PCA_DIM:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr).astype(np.float32)
            X_va = pca.transform(X_va).astype(np.float32)
            X_te = pca.transform(X_te).astype(np.float32)
            log(f"  PCA: D {X_tr_full.shape[1]} -> {PCA_DIM}, "
                f"explained_variance_ratio_sum={pca.explained_variance_ratio_.sum():.4f}")

        model, lik = fit_svgp(X_tr, y[tr_idx], M=M, epochs=epochs,
                              batch_size=batch, lr=DEFAULT_LR, seed=SEED + fold,
                              log=log)
        vp = predict_proba(model, lik, X_va.astype(np.float32))
        oof_meta[va_idx] = vp.astype(np.float32)
        tp = predict_proba(model, lik, X_te.astype(np.float32))
        test_meta_folds.append(tp)
        # Per-fold checkpoint
        np.save(ART / f"oof_xgb_metastack{SUFFIX}_fold{fold+1}.npy", vp)
        np.save(ART / f"test_xgb_metastack{SUFFIX}_fold{fold+1}.npy", tp)
        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        log(f"  fold {fold+1} val_argmax={argmax_bal:.5f} wall={time.time()-t1:.1f}s")
        del model, lik

    if RUN_FOLD == 0:
        test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
        np.save(ART / f"oof_xgb_metastack{SUFFIX}.npy", oof_meta)
        np.save(ART / f"test_xgb_metastack{SUFFIX}.npy", test_meta)
        argmax_oof = balanced_accuracy_score(y, oof_meta.argmax(1))
        tuned_oof = bal_at_bias(oof_meta, y) if not SMOKE else float("nan")
        log(f"\n=== SVGP META standalone ===")
        log(f"  argmax OOF = {argmax_oof:.5f}")
        log(f"  tuned OOF  = {tuned_oof:.5f}")
        out = dict(components=component_names, n_components=len(component_names),
                   feature_dim=X_tr_full.shape[1],
                   argmax_oof=float(argmax_oof), tuned_oof=float(tuned_oof),
                   M=M, epochs=epochs, batch=batch, lr=DEFAULT_LR,
                   elapsed_sec=float(time.time() - t0))
        (ART / f"svgp_metastack{SUFFIX}_results.json").write_text(json.dumps(out, indent=2))
        log(f"wall total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
