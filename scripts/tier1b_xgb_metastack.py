"""Tier 1b #1: XGB meta-stacker on the expanded OOF bank.

Input features per row: 3 probs × N components (~40) = ~120 cols
                      + dgp_score + 4 signed distances to thresholds
                      + rule_pred + distance-based rule indicators
                      + the current LB-best 3-stack's 3 probs (as "base")

Target: y (0=Low, 1=Medium, 2=High), 3-class softprob.

Training: 5-fold StratifiedKFold(seed=42) — fold-aligned with every saved
OOF for leak-free stacking.

HPs: heavy-reg XGB (max_depth=4, reg_alpha=5, reg_lambda=5, lr=0.05) to
avoid overfitting the small-margin component disagreements. 3000 round
cap, early stopping 200.

Post: fixed-bias sweep vs current LB-best 3-stack at (α=0.05..0.50). Emit
submission only if fixed-bias Δ OOF ≥ +2e-4.

This is the 2026-04-21 LR meta-stacker analog but with trees + a bigger
expanded pool. Prior LR null was with 10 components; we now have 40+
including RealMLP (Jaccard 0.62 vs LB-best = first genuinely novel NN).
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

import os
ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
# META_OUT_SUFFIX appended to oof_xgb_metastack[suffix].npy outputs to
# avoid clobbering the LB-validated v1 meta-stacker. Use this when adding
# new components to the bank.
META_OUT_SUFFIX = os.environ.get("META_OUT_SUFFIX", "")

EXCLUDE = {
    "soft_distill",              # LB regressor
    "xgb_spec_678",              # sparse carrier
    "recipe_pseudolabel_stage2", # LB-regressor in blends
    "spec_mh_v3_score5",         # binary prob, not 3-class
    "spec_mh_v3_score6",         # binary prob, not 3-class
    "spec6_mh",                  # binary prob
    "spec6_mh_v2",               # binary prob
    "xgb_bin_medium",            # binary
    "xgb_bin_high",              # binary
    "binhigh",                   # binary
    "p_flip",                    # binary
    "pflip",                     # binary
    "missed_high",               # binary
    "flip_correction",           # binary
    "selective_router",          # meta
    "disagree_meta",             # meta
    "c0_safe_lb_best_2way",      # derived
    "c0_safe_recipe_full_te",    # derived
    "c0_v2_lb_best_2way",        # derived
    "c0_v2_lb_best_3way",        # derived
    "c0_v2_recipe_full_te",      # derived
    "c0_v3_lb_best_3way",        # derived
    "c0_v3_recipe_full_te",      # derived
    "b2_groupkfold_region",      # groupkfold diagnostic
    "step1_greedy_lbbest",       # derived
    "hybrid_binhigh",            # binary-ish
    "meta_v3",                   # old meta
    "eb_cell",                   # rule-equivalent
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
    n_train = len(y)
    for oof_p in sorted(ART.glob("oof_*.npy")):
        name = oof_p.stem.replace("oof_", "", 1)
        if name in EXCLUDE:
            continue
        # Skip per-fold checkpoint files (e.g. *_fold1, *_fold2 ...).
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
        # Detect partial-fold OOFs (zero rows where some folds didn't run).
        if (o.sum(1) < 1e-3).any():
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

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best stack OOF = {bal(lb_oof, y):.5f}")

    log("loading pool")
    pool = load_pool(y)
    log(f"  {len(pool)} 3-class components loaded")
    for name in sorted(pool.keys()):
        print(f"    {name}")

    # Build meta-feature matrix: each component contributes log(P_L, P_M, P_H)
    # Also add current stack's log probs + distance/rule features.
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
    oof_path = ART / f"oof_xgb_metastack{META_OUT_SUFFIX}.npy"
    test_path = ART / f"test_xgb_metastack{META_OUT_SUFFIX}.npy"
    np.save(oof_path, oof_meta)
    np.save(test_path, test_meta)
    log(f"saved {oof_path.name} + test")

    # Evaluate the meta-stacker standalone
    meta_argmax_bal = balanced_accuracy_score(y, oof_meta.argmax(1))
    meta_tuned_bal = bal(oof_meta, y)
    log(f"\n=== META-STACKER standalone ===")
    log(f"  argmax OOF bal_acc  = {meta_argmax_bal:.5f}")
    log(f"  @recipe-bias OOF    = {meta_tuned_bal:.5f}")

    # Blend sweep at fixed recipe bias vs LB-best stack
    lb_bal = bal(lb_oof, y)
    log(f"  LB-best 3-stack OOF = {lb_bal:.5f}")
    log(f"\n=== fixed-bias blend sweep vs LB-best 3-stack ===")
    log(f"{'alpha_meta':>10} {'OOF':>9} {'Δ':>9}")
    alphas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    rows = []
    for a in alphas:
        blend = log_blend([lb_oof, oof_meta], np.array([1 - a, a]))
        b = bal(blend, y)
        d = b - lb_bal
        rows.append({"alpha": a, "oof": float(b), "delta": float(d)})
        tag = " ← best" if len(rows) > 1 and d > max(r["delta"] for r in rows[:-1]) else ""
        log(f"{a:>10.3f} {b:>9.5f} {d:>+9.5f}{tag}")
    best = max(rows, key=lambda r: r["delta"])

    # Error-count and Jaccard vs LB-best
    pred_lb = (np.log(np.clip(lb_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_meta = (np.log(np.clip(oof_meta, 1e-12, 1)) + BIAS).argmax(1)
    errs_lb = (pred_lb != y).sum()
    errs_meta = (pred_meta != y).sum()
    inter = ((pred_lb != y) & (pred_meta != y)).sum()
    union = ((pred_lb != y) | (pred_meta != y)).sum()
    jacc = inter / max(union, 1)
    log(f"\nerrs LB-best={errs_lb}  meta={errs_meta}  "
        f"Jaccard(meta, LB-best) = {jacc:.4f}")

    # Emit if best blend Δ ≥ +2e-4 (LB-transfer threshold)
    if best["delta"] >= 2e-4:
        a = best["alpha"]
        test_blend = log_blend([lb_test, test_meta], np.array([1 - a, a]))
        pred_t = (np.log(np.clip(test_blend, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_t]
        tag = f"meta{META_OUT_SUFFIX}_a{int(a*1000):03d}"
        path = SUB / f"submission_tier1b_metastack_{tag}.csv"
        sub.to_csv(path, index=False)
        log(f"\nΔ={best['delta']:+.5f} ≥ +2e-4 → wrote {path}")
    else:
        log(f"\nbest Δ={best['delta']:+.5f} below +2e-4 gate; no submission")

    out = dict(
        components=component_names,
        n_components=len(component_names),
        feature_dim=X_tr.shape[1],
        best_iters=[int(b) for b in best_iters],
        meta_standalone_argmax=float(meta_argmax_bal),
        meta_standalone_tuned=float(meta_tuned_bal),
        lb_best_oof=float(lb_bal),
        blend_sweep=rows,
        best=best,
        err_lb=int(errs_lb),
        err_meta=int(errs_meta),
        jaccard_vs_lb=float(jacc),
        elapsed_sec=float(time.time() - t0),
    )
    json_path = ART / f"tier1b_xgb_metastack{META_OUT_SUFFIX}_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {json_path}")


if __name__ == "__main__":
    main()
