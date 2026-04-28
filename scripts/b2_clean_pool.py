"""B2-clean: retrain xgb_metastack with PER-FOLD ISO inputs but RESTRICTED
to v1's exact 62-component pool. Isolates iso-cal effect from bank size.

Comparison:
  v1 (raw inputs, 62 components)         → standalone OOF 0.98041, iso 0.98059
  B2 (per-fold-iso inputs, 179)          → standalone OOF 0.98052, iso 0.98156
  B2-clean (per-fold-iso inputs, 62)     → ?

If B2-clean ≈ v1 (within 1-2 bp): per-fold iso of inputs adds nothing;
B2's gain was bank-size.

If B2-clean > v1 by ≥ +0.0001: per-fold iso IS doing real work; combined
with the bigger bank, B2 captured +0.0001 of clean iso lift + ~+0.0009 of
bank-size lift.

Per-fold checkpointing for rehydrate resilience.

Outputs:
  scripts/artifacts/oof_xgb_metastack_b2clean.npy
  scripts/artifacts/test_xgb_metastack_b2clean.npy
  scripts/artifacts/b2_clean_pool_results.json
"""
from __future__ import annotations

import json
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
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]

# v1's exact 62-component pool (from tier1b_xgb_metastack_results.json)
V1_POOL = [
    "bagged_greedy_nonrule", "c0_greedy", "catboost_optuna",
    "catboost_recipe_gpu", "em_uniform", "extratrees_dist_digits",
    "extratrees_dist_digits_v2", "greedy_blend", "greedy_full_bank_6way",
    "hybrid_lgbmxgb_blend", "lb_best_fs123", "lb_best_fs7",
    "lgbm_competitor", "lgbm_dist_digits", "lgbm_dist_digits_ote",
    "lgbm_te_orig", "ovo_boundary_blend", "ovo_nonrule_blend",
    "p3_embed_propagate", "realmlp", "recipe_171pair",
    "recipe_allpairs", "recipe_catboost", "recipe_focal_g2h3",
    "recipe_full_te", "recipe_full_te_a01", "recipe_full_te_a10",
    "recipe_full_te_catboost", "recipe_full_te_cldrop",
    "recipe_full_te_dae", "recipe_full_te_fexboth",
    "recipe_full_te_gby", "recipe_full_te_lgbm", "recipe_full_te_seed123",
    "recipe_full_te_seed7", "recipe_lgbm", "recipe_no_combos",
    "recipe_no_digits", "recipe_no_orig", "recipe_no_ote",
    "recipe_pseudolabel", "recipe_pseudolabel_seed123labeler",
    "recipe_pseudolabel_seed7labeler", "recipe_pseudolabel_tau092",
    "tabpfn", "tta_recipe_baseline", "tta_recipe_s001",
    "tta_recipe_s005", "tta_recipe_s010", "tta_recipe_s020",
    "tta_recipe_s030", "xgb_corn", "xgb_dist_digits",
    "xgb_dist_digits_ote", "xgb_dist_digits_ote_digits",
    "xgb_dist_digits_ote_digits_light", "xgb_dist_digits_ote_digits_pairs",
    "xgb_dist_digits_ote_light", "xgb_dist_routed_v3", "xgb_nonrule",
    "xgb_spec_36", "xgb_vanilla_dist",
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] B2c: {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_perfold(oof, test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oo = np.zeros_like(oof, dtype=np.float32)
    for tr_idx, va_idx in skf.split(oof, y):
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof[tr_idx, c], (y[tr_idx] == c).astype(np.float32))
            oo[va_idx, c] = ir.predict(oof[va_idx, c])
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        tt[:, c] = ir.predict(test[:, c])
    return normed(oo), normed(tt)


def bal(p, y):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def build_lbbest_3stack(y):
    """3-stack + RealMLP + nonrule_iso(full) — matches v1's anchor."""
    r  = normed(np.load(ART / "oof_recipe_full_te.npy"))
    rt = normed(np.load(ART / "test_recipe_full_te.npy"))
    s1 = normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1t= normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7 = normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7t= normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm = normed(np.load(ART / "oof_realmlp.npy"))
    rmt= normed(np.load(ART / "test_realmlp.npy"))
    nr = normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nrt= normed(np.load(ART / "test_xgb_nonrule.npy"))
    # Match v1: nonrule iso = FULL OOF
    nr_iso = np.zeros_like(nr, dtype=np.float32)
    nrt_iso = np.zeros_like(nrt, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(nr[:, c], (y == c).astype(np.float32))
        nr_iso[:, c]  = ir.predict(nr[:, c])
        nrt_iso[:, c] = ir.predict(nrt[:, c])
    nr_iso = normed(nr_iso); nrt_iso = normed(nrt_iso)

    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    s1_o = log_blend([lb3_o, rm], np.array([0.8, 0.2]))
    s1_t = log_blend([lb3_t, rmt], np.array([0.8, 0.2]))
    s2_o = log_blend([s1_o, nr_iso], np.array([0.925, 0.075]))
    s2_t = log_blend([s1_t, nrt_iso], np.array([0.925, 0.075]))
    return s2_o, s2_t


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building 4-stack base (matches v1)")
    lb_oof, lb_test = build_lbbest_3stack(y)
    log(f"  4-stack OOF = {bal(lb_oof, y):.5f}")

    log(f"loading {len(V1_POOL)} components from v1's exact pool")
    pool = {}
    for name in V1_POOL:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not (oof_p.exists() and test_p.exists()):
            log(f"  WARNING: missing {name}, skipping")
            continue
        try:
            o = np.load(oof_p).astype(np.float32)
            t = np.load(test_p).astype(np.float32)
        except Exception as e:
            log(f"  WARNING: failed to load {name}: {e}")
            continue
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != len(y):
            log(f"  WARNING: bad shape {name}: {o.shape}")
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  WARNING: zero rows in {name}")
            continue
        pool[name] = (normed(o), normed(t))
    log(f"  loaded {len(pool)} components")

    log("applying PER-FOLD iso to each component")
    pool_iso = {}
    for i, (name, (o, t)) in enumerate(pool.items(), 1):
        oo, tt = iso_perfold(o, t, y)
        pool_iso[name] = (oo, tt)
        if i % 10 == 0:
            log(f"  iso'd {i}/{len(pool)}")

    log("building meta-feature matrix")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    component_names = sorted(pool_iso.keys())
    comp_tr = [np.log(np.clip(pool_iso[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool_iso[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))

    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1)
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
    test_meta_folds = []
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        ckpt_oof  = ART / f"_b2c_fold{fold}_oof.npy"
        ckpt_test = ART / f"_b2c_fold{fold}_test.npy"
        ckpt_meta = ART / f"_b2c_fold{fold}_meta.json"
        if ckpt_oof.exists() and ckpt_test.exists() and ckpt_meta.exists():
            log(f"  fold {fold+1}/{N_FOLDS} resuming from checkpoint")
            vp = np.load(ckpt_oof); tp = np.load(ckpt_test)
            mi = json.loads(ckpt_meta.read_text())
            oof_meta[va_idx] = vp.astype(np.float32)
            test_meta_folds.append(tp)
            best_iters.append(mi["best_iter"])
            log(f"    val_argmax={mi['argmax_bal']:.5f} (cached)")
            continue
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=3000,
            evals=[(dva, "val")], early_stopping_rounds=200,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        vp = booster.predict(dva, iteration_range=(0, bi + 1))
        oof_meta[va_idx] = vp.astype(np.float32)
        tp = booster.predict(dte, iteration_range=(0, bi + 1))
        test_meta_folds.append(tp)
        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        np.save(ckpt_oof, vp.astype(np.float32))
        np.save(ckpt_test, tp.astype(np.float32))
        ckpt_meta.write_text(json.dumps({"best_iter": int(bi), "argmax_bal": float(argmax_bal)}))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax={argmax_bal:.5f} wall={time.time()-t1:.1f}s [ckpt saved]")

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_xgb_metastack_b2clean.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_b2clean.npy", test_meta)

    meta_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
    meta_tuned  = bal(oof_meta, y)
    log(f"\n=== B2-clean META standalone ===")
    log(f"  argmax = {meta_argmax:.5f}  (v1: 0.97365, B2: 0.97395)")
    log(f"  @recipe-bias = {meta_tuned:.5f}  (v1: 0.98041, B2: 0.98052)")

    # Iso variants
    def iso_full_local(oof, test, y):
        oo = np.zeros_like(oof, dtype=np.float32); tt = np.zeros_like(test, dtype=np.float32)
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof[:, c], (y == c).astype(np.float32))
            oo[:, c] = ir.predict(oof[:, c])
            tt[:, c] = ir.predict(test[:, c])
        return normed(oo), normed(tt)

    full_iso_o, full_iso_t = iso_full_local(oof_meta, test_meta, y)
    pf_iso_o,   pf_iso_t   = iso_perfold(oof_meta, test_meta, y)
    log(f"  full-iso @recipe-bias = {bal(full_iso_o, y):.5f}  (v1: 0.98059, B2: 0.98156)")
    log(f"  per-f iso @recipe-bias = {bal(pf_iso_o, y):.5f}  (v1: 0.98048, B2: 0.98113)")

    # Architecture-matched primary (4-stack + B2-clean_meta_iso @ α=0.30)
    prim_full_o = log_blend([lb_oof,  full_iso_o], np.array([0.70, 0.30]))
    prim_full_t = log_blend([lb_test, full_iso_t], np.array([0.70, 0.30]))
    prim_pf_o   = log_blend([lb_oof,  pf_iso_o],   np.array([0.70, 0.30]))
    prim_pf_t   = log_blend([lb_test, pf_iso_t],   np.array([0.70, 0.30]))
    log(f"\n  primary (4-stack + B2c_meta full-iso @α=0.30) OOF = {bal(prim_full_o, y):.5f}")
    log(f"  primary (4-stack + B2c_meta per-f iso @α=0.30) OOF = {bal(prim_pf_o, y):.5f}")
    log(f"  reference: current LB-best PRIMARY OOF = 0.98084 (v1 full-iso @α=0.30)")

    out = dict(
        components=component_names,
        n_components=len(component_names),
        feature_dim=X_tr.shape[1],
        best_iters=[int(b) for b in best_iters],
        meta_standalone_argmax=float(meta_argmax),
        meta_standalone_tuned_recipe_bias=float(meta_tuned),
        meta_full_iso_recipe_bias=float(bal(full_iso_o, y)),
        meta_perfold_iso_recipe_bias=float(bal(pf_iso_o, y)),
        primary_full_iso_a030=float(bal(prim_full_o, y)),
        primary_perfold_iso_a030=float(bal(prim_pf_o, y)),
        # Comparison anchors from CLAUDE.md
        v1_meta_full_iso=0.98059,
        v1_primary_full_iso_a030=0.98084,
        b2_meta_full_iso=0.98156,
        b2_primary_full_iso_a030=0.98111,
        elapsed_sec=float(time.time() - t0),
    )
    json_path = ART / "b2_clean_pool_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"\nwrote {json_path}")
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
