"""J2: bootstrap-bagged meta-stacker.

Mechanism: each bag samples a SUBSET of the 62-component bank (without
replacement, fraction=0.7), trains an iso-cal XGB meta-stacker on that
subset, log-averages outputs across bags. Targets the negative OOF→LB
gap (CV-pessimism) on the LB-best stack via DIFFERENT decorrelation than
Tier-1c's seed-bag (which was near-deterministic).

SMOKE=1 → 2 bags × tight max_rounds for ~5 min validation.
Production: 10 bags × full HPs.

Decision gate at fixed recipe bias [1.4324, 1.4689, 3.4008]:
  - bag_iso standalone vs existing single-bag meta_iso (Jaccard, errs)
  - log-blend (LB-best 4-stack × bag_iso) Δ ≥ +2e-4
  - per-class recall guardrail (≥ anchor − 5e-4 per class)
  Pass → emit submission.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, N_FOLDS, SEED, SUB, TARGET,
    bal_at_bias, build_lbbest_stack, iso_cal, load_pool, load_y, log, normed,
)

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_BAGS = int(os.environ.get("N_BAGS", "2" if SMOKE else "10"))
FRACTION = float(os.environ.get("FRACTION", "0.7"))
RNG_SEED = int(os.environ.get("RNG_SEED", "20260425"))

# J2-specific extras on top of tier1b_helpers.EXCLUDE.
# These are components added to scripts/artifacts/ AFTER the LB-validated v1
# meta-stacker (LB 0.98094). Excluding them keeps J2 a fair test of "bagging
# vs full bank" rather than "bagging + new pool composition".
# Each entry is justified inline.
J2_EXTRA_EXCLUDE = {
    # Prior meta-stacker outputs — using as INPUTS would be circular.
    "lr_metastack",                # LR meta — LB 0.97991 (-0.00103, regressor)
    "xgb_metastack_v3",            # cross-poll meta v3 — LB 0.98060 (-0.00034)
    "xgb_metastack_v3_iso",        # iso of v3
    "xgb_metastack_v4",            # bank+ET+kNN meta — LB 0.97992 (-0.00102)
    "xgb_metastack_varB",          # variant B (depth=3, seed=7)
    "xgb_metastack_varC",          # variant C (depth=5, seed=123)
    # Submission-derived OOFs (subs of the LB-best primary itself = circular).
    "primary_sub_tau095",
    "primary_sub_tau097",
    "primary_sub_tau099",
    # Derived blend outputs (already iso/log-blends of saved OOFs).
    "j6_qp_blend",                 # convex QP solver output, NULL (-0.00029)
    "greedy_blend",                # derived greedy 3-way log-blend
    "ovo_boundary_blend",          # OvO boundary derived blend
    # LB-confirmed regressors (would only inject wrong-direction signal).
    "soft_distill",                # LB 0.97850 (-0.00148)
    "soft_distill_small",          # LB 0.97865 (-0.00143)
    "soft_distill_tiny",           # extreme-capacity null
    # Borderline τ-sweep variants — circular w.r.t. pseudo_s1 in LB-stack.
    "recipe_pseudolabel_tau095",
    "recipe_pseudolabel_tau097",
    "recipe_pseudolabel_tau099",
    # SMOTE — confirmed structural null with bias-mismatch (cannot blend at
    # recipe bias).
    "recipe_smote_v3",
}


def build_meta_features(y, lb_oof, lb_test, train, test, comp_names, pool):
    """Same feature recipe as tier1b_xgb_metastack.py: LB-stack log-probs +
    distance/rule meta + per-component log-probs."""
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in comp_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in comp_names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))

    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1)
    return X_tr.astype(np.float32), X_te.astype(np.float32)


def train_meta_xgb(X_tr, X_te, y, max_rounds, es_rounds):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds = []
    best_iters = []
    params = dict(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    for fold, (tr, va) in enumerate(skf.split(X_tr, y)):
        dtr = xgb.DMatrix(X_tr[tr], label=y[tr])
        dva = xgb.DMatrix(X_tr[va], label=y[va])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(
            params, dtr, num_boost_round=max_rounds,
            evals=[(dva, "val")], early_stopping_rounds=es_rounds,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(int(bi))
        oof[va] = booster.predict(dva, iteration_range=(0, bi + 1)).astype(np.float32)
        test_folds.append(booster.predict(dte, iteration_range=(0, bi + 1)))
    test = np.mean(test_folds, axis=0).astype(np.float32)
    return oof, test, best_iters


def per_class_recall(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    return [float(((pred == c) & (y == c)).sum() / max((y == c).sum(), 1))
            for c in range(3)]


def main():
    t0 = time.time()
    log(f"J2 bootstrap meta-stacker  SMOKE={SMOKE}  N_BAGS={N_BAGS}  fraction={FRACTION}")
    y = load_y()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    log("building LB-best 4-stack anchor (the lift baseline at LB 0.98094 minus the meta-iso step)")
    lb_oof, lb_test = build_lbbest_stack(y)
    lb_bal = bal_at_bias(lb_oof, y)
    log(f"  LB-best 3-stack OOF (anchor base)        = {lb_bal:.5f}")

    # Reproduce LB-best 4-stack OOF (with existing meta_iso at α=0.30) for comparison.
    if (ART / "oof_xgb_metastack.npy").exists():
        existing_meta = normed(np.load(ART / "oof_xgb_metastack.npy"))
        existing_meta_test = normed(np.load(ART / "test_xgb_metastack.npy"))
        emi_o, emi_t = iso_cal(existing_meta, existing_meta_test, y)
        lb4_o = log_blend([lb_oof, emi_o], np.array([0.7, 0.3]))
        lb4_t = log_blend([lb_test, emi_t], np.array([0.7, 0.3]))
        lb4_bal = bal_at_bias(lb4_o, y)
        log(f"  LB-best 4-stack OOF (with existing meta) = {lb4_bal:.5f}")
    else:
        log("  WARNING: oof_xgb_metastack.npy missing; cannot compute 4-stack baseline")
        lb4_o = lb_oof
        lb4_t = lb_test
        lb4_bal = lb_bal

    log("loading component pool (with J2 extra excludes for circular/regressor leak)")
    pool = load_pool(extra_exclude=J2_EXTRA_EXCLUDE)
    comp_names = sorted(pool.keys())
    log(f"  {len(comp_names)} 3-class components loaded")
    log(f"  extra-exclude removed: {sorted(J2_EXTRA_EXCLUDE)}")

    bag_size = max(2, int(round(len(comp_names) * FRACTION)))
    if SMOKE:
        bag_size = min(bag_size, 12)
    log(f"  per-bag subsample size = {bag_size} (fraction={bag_size/len(comp_names):.3f})")

    rng = np.random.default_rng(RNG_SEED)
    bag_oofs = []
    bag_tests = []
    bag_meta = []
    max_rounds = 300 if SMOKE else 1500
    es_rounds = 30 if SMOKE else 100

    for i in range(N_BAGS):
        t1 = time.time()
        idx = rng.choice(len(comp_names), size=bag_size, replace=False)
        names = [comp_names[k] for k in sorted(idx)]
        log(f"  bag {i+1}/{N_BAGS}  bag_size={len(names)}  example: {names[:3]}...")
        X_tr, X_te = build_meta_features(y, lb_oof, lb_test, train, test, names, pool)
        oof, te, bis = train_meta_xgb(X_tr, X_te, y, max_rounds, es_rounds)
        oo, tt = iso_cal(oof, te, y)
        b_oof = bal_at_bias(oo, y)
        bag_oofs.append(oo)
        bag_tests.append(tt)
        bag_meta.append({"bag": i, "names": names, "best_iters": bis,
                         "feature_dim": int(X_tr.shape[1]),
                         "iso_oof_bal": float(b_oof),
                         "wall_sec": float(time.time() - t1)})
        log(f"    bag {i+1} iso OOF = {b_oof:.5f}  wall={time.time()-t1:.1f}s")

    # Aggregate: log-mean across bags, then iso-cal the mean.
    bag_oof_mean = log_blend(bag_oofs, np.full(len(bag_oofs), 1.0 / len(bag_oofs)))
    bag_test_mean = log_blend(bag_tests, np.full(len(bag_tests), 1.0 / len(bag_tests)))
    bag_oof_mean, bag_test_mean = iso_cal(bag_oof_mean, bag_test_mean, y)
    bag_mean_bal = bal_at_bias(bag_oof_mean, y)
    log(f"\n=== bag-mean meta-stacker (N={N_BAGS} log-avg + iso) ===")
    log(f"  standalone OOF @ recipe bias = {bag_mean_bal:.5f}")

    if not SMOKE:
        np.save(ART / "oof_xgb_metastack_j2bag.npy", bag_oof_mean)
        np.save(ART / "test_xgb_metastack_j2bag.npy", bag_test_mean)
        log("  saved oof_xgb_metastack_j2bag.npy + test")

    # Diagnostics: vs LB-best 3-stack and vs LB-best 4-stack
    pred_lb3 = (np.log(np.clip(lb_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_lb4 = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)
    pred_bag = (np.log(np.clip(bag_oof_mean, 1e-12, 1)) + BIAS).argmax(1)
    err_lb3 = int((pred_lb3 != y).sum())
    err_lb4 = int((pred_lb4 != y).sum())
    err_bag = int((pred_bag != y).sum())
    inter34 = int(((pred_lb4 != y) & (pred_bag != y)).sum())
    union34 = int(((pred_lb4 != y) | (pred_bag != y)).sum())
    jacc_vs_lb4 = inter34 / max(union34, 1)
    log(f"  errors: LB3={err_lb3}  LB4={err_lb4}  bag={err_bag}")
    log(f"  Jaccard(bag, LB-best 4-stack) = {jacc_vs_lb4:.4f}")

    # Strategy A: blend bag meta INTO LB-best 4-stack (i.e. add to the
    # existing 4-stack as an additional log-blend component). Stricter test.
    log(f"\n=== Strategy A: log-blend bag_iso onto LB-best 4-stack (anchor {lb4_bal:.5f}) ===")
    log(f"{'alpha':>7} {'OOF':>9} {'Δ':>9} {'errs':>6}  recL    recM    recH")
    pcr_lb4 = per_class_recall(lb4_o, y)
    rows_A = []
    for a in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        blend = log_blend([lb4_o, bag_oof_mean], np.array([1 - a, a]))
        b = bal_at_bias(blend, y)
        d = b - lb4_bal
        pcr = per_class_recall(blend, y)
        errs = int(((np.log(np.clip(blend, 1e-12, 1)) + BIAS).argmax(1) != y).sum())
        rows_A.append({"alpha": a, "oof": float(b), "delta": float(d),
                       "errs": errs, "recL": pcr[0], "recM": pcr[1], "recH": pcr[2]})
        log(f"{a:>7.3f} {b:>9.5f} {d:>+9.5f} {errs:>6}  "
            f"{pcr[0]:.4f}  {pcr[1]:.4f}  {pcr[2]:.4f}")

    # Strategy B: REPLACE the existing meta_iso in the 4-stack with bag_iso.
    # Equivalent: log-blend(LB-3-stack × bag_iso) at α=0.30 (the same weight
    # the LB-validated 4-stack uses for its meta-iso step).
    log(f"\n=== Strategy B: replace meta_iso with bag_iso in 4-stack at α=0.30 ===")
    repl_o = log_blend([lb_oof, bag_oof_mean], np.array([0.7, 0.3]))
    repl_t = log_blend([lb_test, bag_test_mean], np.array([0.7, 0.3]))
    repl_bal = bal_at_bias(repl_o, y)
    repl_pcr = per_class_recall(repl_o, y)
    repl_errs = int(((np.log(np.clip(repl_o, 1e-12, 1)) + BIAS).argmax(1) != y).sum())
    log(f"  repl OOF = {repl_bal:.5f}  Δ vs LB-4 = {repl_bal - lb4_bal:+.5f}  "
        f"errs={repl_errs}")
    log(f"  per-class L={repl_pcr[0]:.4f} M={repl_pcr[1]:.4f} H={repl_pcr[2]:.4f}")
    log(f"  LB-4 baseline pcr   L={pcr_lb4[0]:.4f} M={pcr_lb4[1]:.4f} H={pcr_lb4[2]:.4f}")

    # Emit gate: best Strategy A or B Δ ≥ +2e-4 with per-class guardrail
    candidates = []
    for r in rows_A:
        guard = all((r[f"rec{c}"] >= pcr_lb4[i] - 5e-4) for i, c in enumerate("LMH"))
        candidates.append({"strategy": "A", **r, "guardrail_pass": guard})
    repl_guard = all((repl_pcr[i] >= pcr_lb4[i] - 5e-4) for i in range(3))
    candidates.append({"strategy": "B", "alpha": 0.30, "oof": float(repl_bal),
                       "delta": float(repl_bal - lb4_bal),
                       "errs": repl_errs,
                       "recL": repl_pcr[0], "recM": repl_pcr[1], "recH": repl_pcr[2],
                       "guardrail_pass": repl_guard})

    best = max(candidates, key=lambda r: (r["guardrail_pass"], r["delta"]))
    log(f"\nBest gated candidate: strat={best['strategy']} α={best['alpha']:.3f} "
        f"Δ={best['delta']:+.5f} guardrail={'PASS' if best['guardrail_pass'] else 'FAIL'}")
    emitted = None
    if not SMOKE and best["delta"] >= 2e-4 and best["guardrail_pass"]:
        if best["strategy"] == "A":
            a = best["alpha"]
            test_blend = log_blend([lb4_t, bag_test_mean], np.array([1 - a, a]))
            tag = f"j2bag_A_a{int(a * 1000):03d}"
        else:
            test_blend = log_blend([lb_test, bag_test_mean], np.array([0.7, 0.3]))
            tag = "j2bag_B_repl_a300"
        pred_t = (np.log(np.clip(test_blend, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_{tag}.csv"
        sub.to_csv(path, index=False)
        emitted = str(path)
        log(f"\nGate PASSED → wrote {path}")
    else:
        log("\nGate FAILED (Δ < +2e-4 OR guardrail FAIL OR SMOKE) → no submission")

    out = dict(
        smoke=SMOKE,
        n_bags=N_BAGS,
        fraction=FRACTION,
        rng_seed=RNG_SEED,
        n_components=len(comp_names),
        bag_size=bag_size,
        bag_meta=bag_meta,
        lb3_oof=float(lb_bal),
        lb4_oof=float(lb4_bal),
        bag_mean_oof=float(bag_mean_bal),
        bag_jaccard_vs_lb4=float(jacc_vs_lb4),
        bag_errs=err_bag,
        lb4_errs=err_lb4,
        strategy_A_sweep=rows_A,
        strategy_B_repl=candidates[-1],
        best=best,
        emitted=emitted,
        elapsed_sec=float(time.time() - t0),
    )
    out_p = ART / ("j2_bootstrap_metastack_smoke.json" if SMOKE
                   else "j2_bootstrap_metastack_results.json")
    out_p.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
