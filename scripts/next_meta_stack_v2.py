"""Next move 2: meta-stacker v2 with the v1 meta-stacker + spec specialists
+ nonrule_bag3 added as input features.

v1 inputs (203 dim):  3 LB-best logprobs + 14 meta + 63×3=189 component logprobs
v2 inputs (~210 dim): v1 + xgb_metastack(3) + xgb_nonrule_bag3(3) + 3 binary
                      specialist probs (spec_lm_v3_score3, spec_mh_v3_score{5,6})

Hypothesis: v1 captured cross-component disagreement at +0.00086 LB. v2 sees
its own previous predictions as an input feature and can refine where v1
under/over-shoots; binary specialists provide focused boundary info that
3-class probs smear across all classes.

Risk: meta-on-meta is double-stacking risk. Diagnostic: standalone OOF AUC,
Jaccard vs v1, and fixed-bias blend sweep on TWO anchors (LB-best 3-stack
and the new meta-iso 4-stack).
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

# 3-class components for the meta-stacker (excluding binaries handled separately)
EXCLUDE_3CLS = {
    "soft_distill", "soft_distill_small", "soft_distill_tiny",
    "xgb_spec_678",
    "recipe_pseudolabel_stage2",
    "spec_mh_v3_score5", "spec_mh_v3_score6", "spec6_mh", "spec6_mh_v2",
    "xgb_bin_medium", "xgb_bin_high", "binhigh", "p_flip", "pflip",
    "missed_high", "flip_correction", "spec_lm_v3_score3",
    "selective_router", "disagree_meta",
    "c0_safe_lb_best_2way", "c0_safe_recipe_full_te",
    "c0_v2_lb_best_2way", "c0_v2_lb_best_3way", "c0_v2_recipe_full_te",
    "c0_v3_lb_best_3way", "c0_v3_recipe_full_te",
    "b2_groupkfold_region",
    "step1_greedy_lbbest", "step1_greedy_on_lbbest",
    "tier1b_greedy_meta",  # would be circular vs v2's anchor blends
    "next_greedy_meta_stack",
    "hybrid_binhigh", "meta_v3", "eb_cell",
}

BINARY_FEATS = ["spec_lm_v3_score3", "spec_mh_v3_score5", "spec_mh_v3_score6"]


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


def build_lbbest_3stack(y):
    """Reconstruct the prior LB-best 3-stack (OOF 0.98061 / LB 0.98008)."""
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


def build_lbbest_4stack(y, lb3_o, lb3_t):
    """Add the meta-iso step → current LB-best."""
    meta_o = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = _normed(np.load(ART / "test_xgb_metastack.npy"))
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    s_o = log_blend([lb3_o, meta_iso_o], np.array([0.7, 0.3]))
    s_t = log_blend([lb3_t, meta_iso_t], np.array([0.7, 0.3]))
    return s_o, s_t


def load_3cls_pool():
    pool = {}
    for p in sorted(ART.glob("oof_*.npy")):
        name = p.stem.replace("oof_", "", 1)
        if name in EXCLUDE_3CLS or name in BINARY_FEATS:
            continue
        test_p = ART / f"test_{name}.npy"
        if not test_p.exists():
            continue
        try:
            o = np.load(p).astype(np.float32)
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

    log("building anchors")
    lb3_o, lb3_t = build_lbbest_3stack(y)
    lb4_o, lb4_t = build_lbbest_4stack(y, lb3_o, lb3_t)
    log(f"  3-stack OOF = {bal(lb3_o, y):.5f}")
    log(f"  4-stack OOF = {bal(lb4_o, y):.5f}  (LB 0.98094)")

    log("loading 3-class pool")
    pool = load_3cls_pool()
    log(f"  {len(pool)} 3-class components loaded")

    # Meta-features: dgp_score / distances
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

    # NEW for v2: also add 4-stack logprobs + binary specialists
    lb4_log_tr = np.log(np.clip(lb4_o, 1e-9, 1.0))
    lb4_log_te = np.log(np.clip(lb4_t, 1e-9, 1.0))

    bin_tr_cols, bin_te_cols = [], []
    for bn in BINARY_FEATS:
        try:
            bo = np.load(ART / f"oof_{bn}.npy").astype(np.float32).reshape(-1, 1)
            bt = np.load(ART / f"test_{bn}.npy").astype(np.float32).reshape(-1, 1)
            bin_tr_cols.append(bo); bin_te_cols.append(bt)
            log(f"  binary added: {bn}  (mean train={bo.mean():.4f}  test={bt.mean():.4f})")
        except FileNotFoundError:
            log(f"  binary skip: {bn} (not on disk)")

    parts_tr = [lb_log_tr, lb4_log_tr, meta_tr] + comp_tr + bin_tr_cols
    parts_te = [lb_log_te, lb4_log_te, meta_te] + comp_te + bin_te_cols
    X_tr = np.concatenate(parts_tr, axis=1).astype(np.float32)
    X_te = np.concatenate(parts_te, axis=1).astype(np.float32)
    log(f"  v2 meta-feature shape: {X_tr.shape}  (v1 was 203)")

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
    np.save(ART / "oof_xgb_metastack_v2.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_v2.npy", test_meta)
    log(f"saved oof_xgb_metastack_v2.npy + test")

    meta_argmax_bal = balanced_accuracy_score(y, oof_meta.argmax(1))
    meta_tuned_bal = bal(oof_meta, y)
    log(f"\n=== META-STACKER v2 standalone ===")
    log(f"  argmax OOF bal_acc  = {meta_argmax_bal:.5f}")
    log(f"  @recipe-bias OOF    = {meta_tuned_bal:.5f}")

    # Compare to v1
    v1_o = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_o_iso = iso_cal(v1_o, _normed(np.load(ART / "test_xgb_metastack.npy")), y)[0]
    log(f"  v1 standalone tuned  = {bal(v1_o, y):.5f}")

    # Iso v2
    meta_iso_o, meta_iso_t = iso_cal(_normed(oof_meta), _normed(test_meta), y)
    log(f"  v2 iso-cal standalone = {bal(meta_iso_o, y):.5f}")

    # Blend sweeps
    for tag, anchor_oof, anchor_test in [
        ("vs LB-best 3-stack", lb3_o, lb3_t),
        ("vs LB-best 4-stack (NEW LB BEST)", lb4_o, lb4_t),
    ]:
        anchor_bal = bal(anchor_oof, y)
        log(f"\n=== {tag} (anchor OOF {anchor_bal:.5f}) ===")
        log(f"{'tag':>20} {'α':>7} {'OOF':>9} {'Δ':>9}")
        rows = []
        for variant_tag, oof_v, test_v in [
            ("v2_raw", _normed(oof_meta), _normed(test_meta)),
            ("v2_iso", meta_iso_o, meta_iso_t),
        ]:
            for a in [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
                blend = log_blend([anchor_oof, oof_v], np.array([1 - a, a]))
                b = bal(blend, y)
                d = b - anchor_bal
                rows.append({"variant": variant_tag, "alpha": a,
                             "oof": float(b), "delta": float(d)})
                marker = " <- best" if (rows[0]["delta"] is not None and
                                          d == max(r["delta"] for r in rows)) else ""
                print(f"{variant_tag:>20} {a:>7.3f} {b:>9.5f} {d:>+9.5f}{marker}")

        best_v = max(rows, key=lambda r: r["delta"])
        log(f"BEST {tag}: {best_v}")

        if best_v["delta"] >= 2e-4:
            a = best_v["alpha"]
            v_oof = _normed(oof_meta) if best_v["variant"] == "v2_raw" else meta_iso_o
            v_test = _normed(test_meta) if best_v["variant"] == "v2_raw" else meta_iso_t
            blend_t = log_blend([anchor_test, v_test], np.array([1 - a, a]))
            pred = (np.log(np.clip(blend_t, 1e-12, 1)) + BIAS).argmax(1)
            sample = pd.read_csv(DATA / "sample_submission.csv")
            sub = sample.copy()
            sub[TARGET] = [CLASSES[i] for i in pred]
            anchor_tag = "stack3" if "3-stack" in tag else "stack4"
            tag_full = f"{anchor_tag}_metaV2{best_v['variant']}_a{int(a*1000):03d}"
            path = SUB / f"submission_next_metav2_{tag_full}.csv"
            sub.to_csv(path, index=False)
            log(f"wrote {path}  (Δ={best_v['delta']:+.5f} ≥ +2e-4)")

    out = dict(
        v1_standalone_tuned=float(bal(v1_o, y)),
        v2_argmax=float(meta_argmax_bal),
        v2_tuned=float(meta_tuned_bal),
        v2_iso_tuned=float(bal(meta_iso_o, y)),
        n_features=X_tr.shape[1],
        best_iters=[int(b) for b in best_iters],
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "next_meta_stack_v2_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
