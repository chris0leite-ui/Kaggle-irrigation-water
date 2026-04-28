"""B2: retrain xgb_metastack with PER-FOLD ISO applied to every input component.

Hypothesis: the current xgb_metastack was trained on raw (un-iso-cal'd)
component OOFs. Each component's OOF carries its own per-class
miscalibration. The meta-stacker's tree splits may exploit per-component
miscalibration patterns that are themselves leak-shape (since each
component's OOF is fit on (oof, y) where iso pulls per-row probs toward
empirical class distribution).

If we apply PER-FOLD iso to every input BEFORE the meta sees them, the
meta is forced to find pattern that survives leak-free calibration. The
resulting meta should have a smaller OOF→LB gap.

Critical design rules (per top-of-file rule + earlier session lessons):
  - lam_ce / sample_weight: same as v1 (no class weighting)
  - XGB HPs: identical to v1 (depth=4, reg_alpha=5, reg_lambda=5, lr=0.05)
  - Same 63-component pool (EXCLUDE matches v1)
  - Same 5-fold split (StratifiedKFold seed=42)
  - The ONLY difference: inputs are per-fold-iso'd

Output:
  scripts/artifacts/oof_xgb_metastack_perfoldiso_inputs.npy
  scripts/artifacts/test_xgb_metastack_perfoldiso_inputs.npy
  scripts/artifacts/b2_metastack_perfoldiso_inputs_results.json

Then build leak-honest primary using THIS retrained meta + per-fold iso
of its output, and compare gates.
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
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]

# Same EXCLUDE as v1 (matches tier1b_xgb_metastack.py)
EXCLUDE = {
    "soft_distill", "xgb_spec_678", "recipe_pseudolabel_stage2",
    "spec_mh_v3_score5", "spec_mh_v3_score6", "spec6_mh", "spec6_mh_v2",
    "xgb_bin_medium", "xgb_bin_high", "binhigh", "p_flip", "pflip",
    "missed_high", "flip_correction",
    "selective_router", "disagree_meta",
    "c0_safe_lb_best_2way", "c0_safe_recipe_full_te",
    "c0_v2_lb_best_2way", "c0_v2_lb_best_3way", "c0_v2_recipe_full_te",
    "c0_v3_lb_best_3way", "c0_v3_recipe_full_te",
    "b2_groupkfold_region", "b2_groupkfold_crop",
    "step1_greedy_lbbest", "hybrid_binhigh", "meta_v3", "eb_cell",
    "spec_lm_v3_score3", "tta_recipe_baseline",
    # Anything tagged with the rolling per-fold or partial OOF naming
    "leak_honest_primary",  # don't loop on the new artifact
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] B2: {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_perfold(oof, test, y):
    """Per-fold leak-safe isotonic, matching the v1 split."""
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


def iso_full(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return normed(oo), normed(tt)


def bal(p, y):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def build_lbbest_stack(y):
    """Match v1's anchor exactly (full-OOF iso on nonrule)."""
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
    nr_iso, nrt_iso = iso_full(nr, nrt, y)  # match v1 exactly here
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    s1_o = log_blend([lb3_o, rm], np.array([0.8, 0.2]))
    s1_t = log_blend([lb3_t, rmt], np.array([0.8, 0.2]))
    s2_o = log_blend([s1_o, nr_iso], np.array([0.925, 0.075]))
    s2_t = log_blend([s1_t, nrt_iso], np.array([0.925, 0.075]))
    return s2_o, s2_t


def load_pool_perfoldiso(y):
    """Load every saved OOF and apply PER-FOLD iso. Returns dict
    {name: (oof_iso, test_iso)} matching v1's shape."""
    pool = {}
    n_train = len(y)
    for oof_p in sorted(ART.glob("oof_*.npy")):
        name = oof_p.stem.replace("oof_", "", 1)
        if name in EXCLUDE:
            continue
        if name.endswith(tuple(f"_fold{f}" for f in range(1, 11))):
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
        if o.shape[0] != n_train:
            continue
        if (o.sum(1) < 1e-3).any():
            continue
        # Per-fold iso for honest input calibration
        oo, tt = iso_perfold(normed(o), normed(t), y)
        pool[name] = (oo, tt)
    return pool


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack (matching v1 anchor)")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  anchor OOF = {bal(lb_oof, y):.5f}")

    log("loading pool with PER-FOLD iso applied to every input")
    pool = load_pool_perfoldiso(y)
    log(f"  {len(pool)} components iso-cal'd (per-fold)")

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

    # Resume from per-fold checkpoints if any
    ckpt_oof_glob = "_b2_meta_fold{f}_oof.npy"
    ckpt_test_glob = "_b2_meta_fold{f}_test.npy"
    ckpt_meta_glob = "_b2_meta_fold{f}_meta.json"

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        ckpt_oof = ART / ckpt_oof_glob.format(f=fold)
        ckpt_test = ART / ckpt_test_glob.format(f=fold)
        ckpt_meta = ART / ckpt_meta_glob.format(f=fold)
        if ckpt_oof.exists() and ckpt_test.exists() and ckpt_meta.exists():
            log(f"  fold {fold+1}/{N_FOLDS} resuming from checkpoint")
            vp = np.load(ckpt_oof)
            tp = np.load(ckpt_test)
            meta_info = json.loads(ckpt_meta.read_text())
            oof_meta[va_idx] = vp.astype(np.float32)
            test_meta_folds.append(tp)
            best_iters.append(meta_info["best_iter"])
            argmax_bal = meta_info["argmax_bal"]
            log(f"    val_argmax={argmax_bal:.5f} (cached)")
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
        # Atomic checkpoint
        np.save(ckpt_oof, vp.astype(np.float32))
        np.save(ckpt_test, tp.astype(np.float32))
        ckpt_meta.write_text(json.dumps({"best_iter": int(bi), "argmax_bal": float(argmax_bal)}))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax={argmax_bal:.5f} wall={time.time()-t1:.1f}s [ckpt saved]")

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_xgb_metastack_perfoldiso_inputs.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_perfoldiso_inputs.npy", test_meta)
    log(f"saved oof_xgb_metastack_perfoldiso_inputs.npy + test")

    # Standalone metrics
    meta_argmax = balanced_accuracy_score(y, oof_meta.argmax(1))
    meta_tuned  = bal(oof_meta, y)
    log(f"\n=== B2 META-STACKER standalone ===")
    log(f"  argmax OOF      = {meta_argmax:.5f}")
    log(f"  @recipe-bias    = {meta_tuned:.5f}  (v1 was 0.98041)")

    # Compare iso variants of B2 meta vs v1 meta
    log("\nbuilding leak-honest primary with B2 meta (per-fold iso on output)")
    meta_pf_iso, meta_pf_iso_t = iso_perfold(oof_meta, test_meta, y)
    meta_full_iso, meta_full_iso_t = iso_full(oof_meta, test_meta, y)

    # 4-stack base (matches v1: nonrule iso full)
    nr  = normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nrt = normed(np.load(ART / "test_xgb_nonrule.npy"))
    nr_iso, nrt_iso = iso_full(nr, nrt, y)
    rm = normed(np.load(ART / "oof_realmlp.npy"))
    rmt= normed(np.load(ART / "test_realmlp.npy"))
    s1_o = log_blend([lb_oof, rm], np.array([0.8, 0.2]))
    s1_t = log_blend([lb_test, rmt], np.array([0.8, 0.2]))
    # Wait — lb_oof above already has 3-stack only; need to rebuild without that nonrule layer
    # Actually lb_oof IS the 3-stack (recipe + pseudo_s1 + pseudo_s7). nonrule comes after.
    s2_o = log_blend([s1_o, nr_iso], np.array([0.925, 0.075]))
    s2_t = log_blend([s1_t, nrt_iso], np.array([0.925, 0.075]))

    # Architecture-matched: + B2_meta_iso @ α=0.30
    prim_pf_o = log_blend([s2_o, meta_pf_iso], np.array([0.70, 0.30]))
    prim_pf_t = log_blend([s2_t, meta_pf_iso_t], np.array([0.70, 0.30]))
    prim_full_o = log_blend([s2_o, meta_full_iso], np.array([0.70, 0.30]))
    prim_full_t = log_blend([s2_t, meta_full_iso_t], np.array([0.70, 0.30]))

    log(f"\n=== B2 leak-honest primary OOFs ===")
    log(f"  4-stack base (3-stack + RealMLP + nonrule_iso) OOF = {bal(s2_o, y):.5f}")
    log(f"  + B2_meta(per-fold iso) @ α=0.30 = {bal(prim_pf_o, y):.5f}  (leak-honest)")
    log(f"  + B2_meta(full-OOF iso) @ α=0.30 = {bal(prim_full_o, y):.5f}  (full-OOF iso, same shape as current PRIMARY)")

    # Reference: v1 meta same architecture
    ms_v1 = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_v1_full, ms_v1_full_t = iso_full(ms_v1, ms_v1_t, y)
    ms_v1_pf, ms_v1_pf_t = iso_perfold(ms_v1, ms_v1_t, y)
    prim_v1_full_o = log_blend([s2_o, ms_v1_full], np.array([0.70, 0.30]))
    prim_v1_pf_o   = log_blend([s2_o, ms_v1_pf],   np.array([0.70, 0.30]))
    log(f"  + v1_meta(full-OOF iso) @ α=0.30 = {bal(prim_v1_full_o, y):.5f}  (this is the LB-validated PRIMARY)")
    log(f"  + v1_meta(per-fold iso) @ α=0.30 = {bal(prim_v1_pf_o, y):.5f}  (leak-honest v1)")

    # Save B2 leak-honest primary submission (per-fold iso variant)
    pred = (np.log(np.clip(prim_pf_t, 1e-12, 1)) + BIAS).argmax(1)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred]
    sub_path = SUB / "submission_b2_leak_honest_primary.csv"
    sub.to_csv(sub_path, index=False)
    log(f"\nwrote {sub_path}")

    # Diff vs current PRIMARY
    primary_csv = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
    primary_pred = primary_csv[TARGET].map(CLS2IDX).to_numpy()
    diff = int((pred != primary_pred).sum())
    log(f"  test rows differing from current PRIMARY: {diff} ({100*diff/270000:.2f}%)")

    out = dict(
        components=component_names,
        n_components=len(component_names),
        feature_dim=X_tr.shape[1],
        best_iters=[int(b) for b in best_iters],
        meta_standalone_argmax=float(meta_argmax),
        meta_standalone_tuned=float(meta_tuned),
        primary_oofs={
            "4_stack_base": float(bal(s2_o, y)),
            "B2_meta_perfold_iso_a030": float(bal(prim_pf_o, y)),
            "B2_meta_full_iso_a030":     float(bal(prim_full_o, y)),
            "v1_meta_full_iso_a030_LBVALIDATED": float(bal(prim_v1_full_o, y)),
            "v1_meta_perfold_iso_a030":  float(bal(prim_v1_pf_o, y)),
        },
        diff_vs_primary=diff,
        elapsed_sec=float(time.time() - t0),
    )
    json_path = ART / "b2_metastack_perfoldiso_inputs_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {json_path}")
    log(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
