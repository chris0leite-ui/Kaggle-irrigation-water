"""Cross-pollinated meta-stacker v3.

Adds to the prior 62-component bank (commit 205b42f, LB 0.98094):
  + recipe_focal_g2_invfreq    (clean training-time focal variant)
  + xgb_nonrule_bag3           (3-seed bag of nonrule)

EXCLUDES known LB regressors and calibration-broken components:
  - soft_distill / _small / _tiny  (teacher-OOF leak family)
  - recipe_focal_g2_aH1            (broken calibration: 0.948 @ recipe bias)
  - realmlp_ens4                   (Tier-1c null, weaker than realmlp n_ens=1)
  - all per-bin / per-cell / OvO derived OOFs

Anchor: same LB-best 4-stack used in tier1b_xgb_metastack.py
        (lb3way + realmlp@0.2 + nonrule_iso@0.075).

Decision gate: blend OOF Δ ≥ +2e-4 over 0.98084 (current 4-stack OOF) → LB probe.
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

# v3 stricter EXCLUDE: prior list + LB regressors + diagnostic/derived/binary/meta
EXCLUDE = {
    # ----- already in v1 EXCLUDE -----
    "soft_distill", "xgb_spec_678", "recipe_pseudolabel_stage2",
    "spec_mh_v3_score5", "spec_mh_v3_score6", "spec6_mh", "spec6_mh_v2",
    "xgb_bin_medium", "xgb_bin_high", "binhigh", "p_flip", "pflip",
    "missed_high", "flip_correction", "selective_router", "disagree_meta",
    "c0_safe_lb_best_2way", "c0_safe_recipe_full_te",
    "c0_v2_lb_best_2way", "c0_v2_lb_best_3way", "c0_v2_recipe_full_te",
    "c0_v3_lb_best_3way", "c0_v3_recipe_full_te",
    "b2_groupkfold_region", "step1_greedy_lbbest", "hybrid_binhigh",
    "meta_v3", "eb_cell",
    # ----- v3 additions: LB-known regressors, calibration-broken, redundant -----
    "soft_distill_small",        # LB 0.97865 (-0.00133)
    "soft_distill_tiny",         # untested but projected null, teacher-OOF leak
    "recipe_focal_g2_aH1",       # OOF 0.94800 @ recipe bias (calibration-broken)
    "realmlp_ens4",              # Tier-1c null (worse than realmlp n_ens=1)
    "hedge_avg_lb_bests",        # derived from 2-way + 3-way; circular if used
    "per_bin_blend",             # derived OOF, overfit signature
    "xgb_metastack",             # prior meta-stacker (would be circular)
    "xgb_metastack_v2",          # prior meta-v2
    "xgb_metastack_bag3",        # prior meta seed-bag
    "spec_lm_v3_score3",         # binary
    # OvO heads will fail .ndim check anyway; listing for clarity
    "xgb_ovo_lowmed", "xgb_ovo_medhigh",
    "xgb_ovo_lowmed_nonrule", "xgb_ovo_medhigh_nonrule",
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def build_lbbest_stack(y):
    """Same as tier1b_xgb_metastack.build_lbbest_stack — LB-best 4-stack
    (lb3way × realmlp@0.2 × nonrule_iso@0.075). OOF 0.98061, LB 0.98008."""
    r = (_normed(np.load(ART / "oof_recipe_full_te.npy")),
         _normed(np.load(ART / "test_recipe_full_te.npy")))
    s1 = (_normed(np.load(ART / "oof_recipe_pseudolabel.npy")),
          _normed(np.load(ART / "test_recipe_pseudolabel.npy")))
    s7 = (_normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")),
          _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")))
    rm = (_normed(np.load(ART / "oof_realmlp.npy")),
          _normed(np.load(ART / "test_realmlp.npy")))
    nr = (_normed(np.load(ART / "oof_xgb_nonrule.npy")),
          _normed(np.load(ART / "test_xgb_nonrule.npy")))
    nr_iso_o, nr_iso_t = iso_cal(nr[0], nr[1], y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r[0], s1[0], s7[0]], w3)
    lb3_t = log_blend([r[1], s1[1], s7[1]], w3)
    s1_o = log_blend([lb3_o, rm[0]], np.array([0.8, 0.2]))
    s1_t = log_blend([lb3_t, rm[1]], np.array([0.8, 0.2]))
    s2_o = log_blend([s1_o, nr_iso_o], np.array([0.925, 0.075]))
    s2_t = log_blend([s1_t, nr_iso_t], np.array([0.925, 0.075]))
    return s2_o, s2_t


def load_pool(y):
    pool = {}
    for oof_p in sorted(ART.glob("oof_*.npy")):
        name = oof_p.stem.replace("oof_", "", 1)
        if name in EXCLUDE:
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
        pool[name] = (_normed(o), _normed(t))
    return pool


def bal(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 4-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best 4-stack OOF = {bal(lb_oof, y):.5f}")

    log("loading pool")
    pool = load_pool(y)
    log(f"  {len(pool)} 3-class components loaded (v3 EXCLUDE applied)")

    # Identify what's NEW vs prior bank
    prior = json.loads((ART / "tier1b_xgb_metastack_results.json").read_text())
    prior_set = set(prior["components"])
    new_set = set(pool.keys()) - prior_set
    dropped_set = prior_set - set(pool.keys())
    log(f"  NEW vs prior v1 bank ({len(new_set)}): {sorted(new_set)}")
    log(f"  dropped from prior v1 bank ({len(dropped_set)}): {sorted(dropped_set)}")

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
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    max_rounds = 3000
    es_rounds = 200

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
            xgb_params, dtr, num_boost_round=max_rounds,
            evals=[(dva, "val")], early_stopping_rounds=es_rounds,
            verbose_eval=0,
        )
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
    np.save(ART / "oof_xgb_metastack_v3.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_v3.npy", test_meta)
    log("saved oof_xgb_metastack_v3.npy + test")

    # Standalone evaluation
    meta_argmax_bal = balanced_accuracy_score(y, oof_meta.argmax(1))
    meta_tuned_bal = bal(oof_meta, y)
    log("\n=== META v3 standalone ===")
    log(f"  argmax OOF bal_acc  = {meta_argmax_bal:.5f}")
    log(f"  @recipe-bias OOF    = {meta_tuned_bal:.5f}")

    # Iso-calibrate v3 (the breakthrough lever from Tier-1b)
    oof_meta_iso, test_meta_iso = iso_cal(oof_meta, test_meta, y)
    np.save(ART / "oof_xgb_metastack_v3_iso.npy", oof_meta_iso)
    np.save(ART / "test_xgb_metastack_v3_iso.npy", test_meta_iso)
    iso_bal = bal(oof_meta_iso, y)
    log(f"  iso-cal'd @recipe   = {iso_bal:.5f}")

    # Blend sweeps vs LB-best 4-stack
    lb_bal = bal(lb_oof, y)
    log(f"\n  LB-best 4-stack OOF = {lb_bal:.5f}")
    log("\n=== fixed-bias blend sweep (v3 raw vs LB-best 4-stack) ===")
    log(f"{'alpha':>10} {'OOF':>9} {'Δ':>9}")
    alphas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    rows_raw = []
    for a in alphas:
        blend = log_blend([lb_oof, oof_meta], np.array([1 - a, a]))
        b = bal(blend, y)
        d = b - lb_bal
        rows_raw.append({"alpha": a, "oof": float(b), "delta": float(d)})
        log(f"{a:>10.3f} {b:>9.5f} {d:>+9.5f}")
    best_raw = max(rows_raw, key=lambda r: r["delta"])

    log("\n=== fixed-bias blend sweep (v3 ISO vs LB-best 4-stack) ===")
    log(f"{'alpha':>10} {'OOF':>9} {'Δ':>9}")
    rows_iso = []
    for a in alphas:
        blend = log_blend([lb_oof, oof_meta_iso], np.array([1 - a, a]))
        b = bal(blend, y)
        d = b - lb_bal
        rows_iso.append({"alpha": a, "oof": float(b), "delta": float(d)})
        log(f"{a:>10.3f} {b:>9.5f} {d:>+9.5f}")
    best_iso = max(rows_iso, key=lambda r: r["delta"])

    # Error / Jaccard diagnostics
    pred_lb = (np.log(np.clip(lb_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_meta = (np.log(np.clip(oof_meta, 1e-12, 1)) + BIAS).argmax(1)
    pred_meta_iso = (np.log(np.clip(oof_meta_iso, 1e-12, 1)) + BIAS).argmax(1)
    errs_lb = int((pred_lb != y).sum())
    errs_meta = int((pred_meta != y).sum())
    errs_meta_iso = int((pred_meta_iso != y).sum())

    def jacc(a, b):
        inter = ((a != y) & (b != y)).sum()
        union = ((a != y) | (b != y)).sum()
        return float(inter / max(union, 1))
    jacc_meta = jacc(pred_lb, pred_meta)
    jacc_meta_iso = jacc(pred_lb, pred_meta_iso)
    log("\n=== diagnostics ===")
    log(f"  errs LB-best={errs_lb}  meta_v3={errs_meta}  meta_v3_iso={errs_meta_iso}")
    log(f"  Jaccard(meta_v3, LB-best)     = {jacc_meta:.4f}")
    log(f"  Jaccard(meta_v3_iso, LB-best) = {jacc_meta_iso:.4f}")

    # Compare to prior tier1b's iso-blend Δ at α=0.30 (= +0.00023 vs LB-best 3-stack)
    log("\n=== verdict vs Tier-1b reference ===")
    log(f"  Tier-1b prior:    iso-meta α=0.30 Δ vs LB-best 3-stack = +0.00023 → LB +0.00086")
    log(f"  v3 raw  best:     α={best_raw['alpha']:.3f} Δ vs LB-best 4-stack = {best_raw['delta']:+.5f}")
    log(f"  v3 iso  best:     α={best_iso['alpha']:.3f} Δ vs LB-best 4-stack = {best_iso['delta']:+.5f}")

    # Emit if iso blend Δ ≥ +2e-4
    emit_path = None
    if best_iso["delta"] >= 2e-4:
        a = best_iso["alpha"]
        test_blend = log_blend([lb_test, test_meta_iso], np.array([1 - a, a]))
        pred_t = (np.log(np.clip(test_blend, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_t]
        emit_path = SUB / f"submission_tier1b_metastack_v3_iso_a{int(a*1000):03d}.csv"
        sub.to_csv(emit_path, index=False)
        log(f"\n→ wrote {emit_path}")
    else:
        log(f"\nbest iso Δ={best_iso['delta']:+.5f} below +2e-4 gate; no submission")

    out = dict(
        components=component_names,
        n_components=len(component_names),
        new_components=sorted(new_set),
        dropped_components=sorted(dropped_set),
        feature_dim=X_tr.shape[1],
        best_iters=[int(b) for b in best_iters],
        meta_argmax=float(meta_argmax_bal),
        meta_tuned=float(meta_tuned_bal),
        meta_iso_tuned=float(iso_bal),
        lb_best_4stack_oof=float(lb_bal),
        sweep_raw=rows_raw,
        sweep_iso=rows_iso,
        best_raw=best_raw,
        best_iso=best_iso,
        errs_lb=errs_lb,
        errs_meta=errs_meta,
        errs_meta_iso=errs_meta_iso,
        jacc_meta_vs_lb=jacc_meta,
        jacc_meta_iso_vs_lb=jacc_meta_iso,
        submission_emitted=str(emit_path) if emit_path else None,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "tier1b_xgb_metastack_v3_results.json").write_text(json.dumps(out, indent=2))
    log("wrote scripts/artifacts/tier1b_xgb_metastack_v3_results.json")


if __name__ == "__main__":
    main()
