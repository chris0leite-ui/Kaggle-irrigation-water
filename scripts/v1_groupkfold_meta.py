"""#1 — Train xgb_metastack on v1's 62-component pool but with GroupKFold
(grouping by dgp_score) instead of StratifiedKFold(y).

Hypothesis: v1's negative OOF→LB gap (PRIMARY at 0.98084 → 0.98094)
may be partly fold-alignment — base components and meta both use
StratifiedKFold(seed=42), so the meta's training rows have features
(component OOFs) that share fold structure with the meta's val rows.
Different fold structure for the meta should reveal whether the gap
is fold-specific.

Concretely: use GroupKFold with groups=dgp_score (10 levels). This
puts entire score-buckets in each fold, completely orthogonal to
component-OOF fold structure.

Trade-off: GroupKFold is harder than StratifiedKFold (each fold has
imbalanced y distribution because score correlates with class), so
the resulting meta will likely have lower argmax bal_acc. But the
question is the GAP, not the absolute number.

Outputs:
  scripts/artifacts/oof_xgb_metastack_v1_groupkfold.npy
  scripts/artifacts/test_xgb_metastack_v1_groupkfold.npy
  scripts/artifacts/v1_groupkfold_meta_results.json
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
from sklearn.model_selection import GroupKFold

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

# v1's 62-component pool (verified from tier1b_xgb_metastack_results.json)
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


def log(m): print(f"[{time.strftime('%H:%M:%S')}] GK: {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def bal(p, y):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("computing dgp_score for GroupKFold")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    groups = tr_d["dgp_score"].to_numpy()
    log(f"  dgp_score levels: {sorted(set(groups))}, n_groups={len(set(groups))}")

    log(f"loading {len(V1_POOL)} components from v1's pool")
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
        except Exception:
            continue
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != len(y):
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  WARNING: zero rows in {name}, skipping")
            continue
        pool[name] = (normed(o), normed(t))
    log(f"  loaded {len(pool)} components")

    # Build meta-feature matrix (matches v1's structure)
    log("constructing meta features (matches v1 structure)")
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    # Build LB-best 3-stack OOF/test (matches v1 anchor)
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
    # Match v1: full-OOF iso on nonrule
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
    s1o = log_blend([lb3_o, rm], np.array([0.8, 0.2]))
    s1t_t = log_blend([lb3_t, rmt], np.array([0.8, 0.2]))
    base_o = log_blend([s1o, nr_iso], np.array([0.925, 0.075]))
    base_t = log_blend([s1t_t, nrt_iso], np.array([0.925, 0.075]))
    log(f"  4-stack base OOF = {bal(base_o, y):.5f}")

    component_names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(base_o, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(base_t, 1e-9, 1.0))

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

    # ─── KEY DIFFERENCE FROM v1: GroupKFold(groups=dgp_score) ───
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_meta = np.zeros((len(train), 3), dtype=np.float32)
    test_meta_folds = []
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y, groups=groups)):
        ckpt_oof  = ART / f"_v1gk_fold{fold}_oof.npy"
        ckpt_test = ART / f"_v1gk_fold{fold}_test.npy"
        ckpt_meta = ART / f"_v1gk_fold{fold}_meta.json"
        if ckpt_oof.exists() and ckpt_test.exists() and ckpt_meta.exists():
            log(f"  fold {fold+1}/{N_FOLDS} resuming from checkpoint")
            vp = np.load(ckpt_oof); tp = np.load(ckpt_test)
            mi = json.loads(ckpt_meta.read_text())
            oof_meta[va_idx] = vp.astype(np.float32)
            test_meta_folds.append(tp); best_iters.append(mi["best_iter"])
            log(f"    val_argmax={mi['argmax_bal']:.5f} (cached)")
            continue
        t1 = time.time()
        # GroupKFold val rows have a DIFFERENT y distribution than tr rows
        log(f"  fold {fold+1}/{N_FOLDS} groups in val: {sorted(set(groups[va_idx]))}")
        log(f"    tr y dist: {np.bincount(y[tr_idx], minlength=3)}")
        log(f"    va y dist: {np.bincount(y[va_idx], minlength=3)}")
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
        log(f"    val_argmax={argmax_bal:.5f} it={bi} wall={time.time()-t1:.1f}s [ckpt saved]")

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_xgb_metastack_v1_groupkfold.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_v1_groupkfold.npy", test_meta)

    meta_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
    meta_tuned = bal(oof_meta, y)
    log(f"\n=== GK META standalone ===")
    log(f"  argmax = {meta_argmax:.5f}  (v1 SKF: 0.97365)")
    log(f"  @recipe-bias = {meta_tuned:.5f}  (v1 SKF: 0.98041)")

    # Apply iso variants
    def iso_full(oof, test):
        oo = np.zeros_like(oof, dtype=np.float32); tt = np.zeros_like(test, dtype=np.float32)
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof[:, c], (y == c).astype(np.float32))
            oo[:, c] = ir.predict(oof[:, c])
            tt[:, c] = ir.predict(test[:, c])
        return normed(oo), normed(tt)

    full_iso_o, full_iso_t = iso_full(oof_meta, test_meta)
    log(f"  full-iso @recipe-bias = {bal(full_iso_o, y):.5f}  (v1 SKF: 0.98059)")

    # Architecture-matched primary
    prim_o = log_blend([base_o, full_iso_o], np.array([0.70, 0.30]))
    prim_t = log_blend([base_t, full_iso_t], np.array([0.70, 0.30]))
    log(f"\n  primary (4-stack + GK_meta_iso @α=0.30) OOF = {bal(prim_o, y):.5f}")
    log(f"  reference: v1 SKF primary OOF = 0.98084 (LB-validated 0.98094, gap -0.00010)")

    # Test diff vs current PRIMARY
    primary_csv = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
    primary_pred = primary_csv[TARGET].map(CLS2IDX).to_numpy()
    pred = (np.log(np.clip(prim_t, 1e-12, 1)) + BIAS).argmax(1)
    diff = int((pred != primary_pred).sum())
    log(f"  test diff vs current PRIMARY: {diff}")

    out = dict(
        components=component_names,
        n_components=len(component_names),
        feature_dim=X_tr.shape[1],
        best_iters=[int(b) for b in best_iters],
        meta_argmax=float(meta_argmax),
        meta_tuned_recipe_bias=float(meta_tuned),
        meta_full_iso_recipe_bias=float(bal(full_iso_o, y)),
        primary_full_iso_a030=float(bal(prim_o, y)),
        diff_vs_primary=diff,
        v1_skf_meta_argmax_reference=0.97365,
        v1_skf_meta_full_iso_reference=0.98059,
        v1_skf_primary_reference=0.98084,
        elapsed_sec=float(time.time() - t0),
    )
    json_path = ART / "v1_groupkfold_meta_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"\nwrote {json_path}")
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
