"""Next move 3: multi-seed meta-stacker bag (XGB_SEED ∈ {42, 7, 123}).

The meta-stacker has 200+ feature dimensions and trains a 4-depth XGB.
Unlike the deterministic xgb_nonrule which has only 13 features and
near-zero seed variance, the high-dim meta-stacker has more sample-by-
feature variance per fold and should benefit meaningfully from multi-seed
averaging.

Reuses tier1b_xgb_metastack.py's feature construction. Trains seed=7
and seed=123 (seed=42 is already on disk as oof_xgb_metastack.npy).
Averages OOF + test probs across the 3 seeds.

Output: oof_xgb_metastack_bag3.npy + test, plus blend-gate analysis vs
both LB-best 3-stack and LB-best 4-stack (current best). Submits if any
config OOF Δ ≥ +1e-4 vs LB-best 4-stack.
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
FOLD_SEED = 42
N_FOLDS = 5
BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]

# Mirror the EXCLUDE list from tier1b_xgb_metastack.py
EXCLUDE = {
    "soft_distill", "soft_distill_small", "soft_distill_tiny",
    "xgb_spec_678",
    "recipe_pseudolabel_stage2",
    "spec_mh_v3_score5", "spec_mh_v3_score6",
    "spec6_mh", "spec6_mh_v2",
    "xgb_bin_medium", "xgb_bin_high", "binhigh", "p_flip", "pflip",
    "missed_high", "flip_correction", "spec_lm_v3_score3",
    "selective_router", "disagree_meta",
    "c0_safe_lb_best_2way", "c0_safe_recipe_full_te",
    "c0_v2_lb_best_2way", "c0_v2_lb_best_3way", "c0_v2_recipe_full_te",
    "c0_v3_lb_best_3way", "c0_v3_recipe_full_te",
    "b2_groupkfold_region",
    "step1_greedy_lbbest", "step1_greedy_on_lbbest",
    "tier1b_greedy_meta", "next_greedy_meta_stack",
    "hybrid_binhigh", "meta_v3", "eb_cell",
    "xgb_metastack",  # exclude self when bagging
    "xgb_metastack_v2",
    "xgb_metastack_bag3",
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


def build_lbbest_3stack(y):
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


def load_pool():
    pool = {}
    for p in sorted(ART.glob("oof_*.npy")):
        name = p.stem.replace("oof_", "", 1)
        if name in EXCLUDE:
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


def construct_features(y):
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    log("building LB-best 3-stack base + meta-features")
    lb3_o, lb3_t = build_lbbest_3stack(y)

    pool = load_pool()
    component_names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]

    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    lb_log_tr = np.log(np.clip(lb3_o, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb3_t, 1e-9, 1.0))

    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1).astype(np.float32)
    return X_tr, X_te, lb3_o, lb3_t, len(component_names)


def train_seed(X_tr, X_te, y, seed):
    log(f"  XGB_SEED={seed}")
    xgb_params = dict(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=seed, nthread=-1,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test = np.zeros((X_te.shape[0], 3), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=3000,
            evals=[(dva, "val")], early_stopping_rounds=200, verbose_eval=0,
        )
        bi = booster.best_iteration
        vp = booster.predict(dva, iteration_range=(0, bi + 1))
        oof[va_idx] = vp.astype(np.float32)
        tp = booster.predict(dte, iteration_range=(0, bi + 1))
        test += tp / N_FOLDS
        log(f"    fold {fold+1}/{N_FOLDS} it={bi}  wall={time.time()-t0:.1f}s")
    return oof, test.astype(np.float32)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    X_tr, X_te, lb3_o, lb3_t, n_components = construct_features(y)
    log(f"meta-features: {X_tr.shape}  ({n_components} components in pool)")

    # Reuse seed=42 from existing artefact, train 7 + 123
    log("loading existing seed=42 meta-stacker artefact")
    o42 = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    t42 = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    oofs = [o42]
    tests = [t42]

    for s in [7, 123]:
        o_s, t_s = train_seed(X_tr, X_te, y, s)
        oofs.append(o_s); tests.append(t_s)

    # Per-seed standalone OOF
    log("\n=== per-seed standalone OOF (@recipe bias) ===")
    for s, o in zip([42, 7, 123], oofs):
        log(f"  seed={s}: {bal(_normed(o), y):.5f}")

    # Bag (prob-space mean)
    bag_oof = np.mean(oofs, axis=0).astype(np.float32)
    bag_test = np.mean(tests, axis=0).astype(np.float32)
    bag_oof = _normed(bag_oof)
    bag_test = _normed(bag_test)
    np.save(ART / "oof_xgb_metastack_bag3.npy", bag_oof)
    np.save(ART / "test_xgb_metastack_bag3.npy", bag_test)
    log(f"\nbag standalone OOF (@recipe bias) = {bal(bag_oof, y):.5f}")

    # Iso-cal both bag and seed=42 for comparison
    bag_iso_o, bag_iso_t = iso_cal(bag_oof, bag_test, y)
    o42_iso, t42_iso = iso_cal(_normed(o42), _normed(t42), y)
    log(f"bag iso-cal standalone           = {bal(bag_iso_o, y):.5f}")
    log(f"seed=42 iso-cal standalone (ref)  = {bal(o42_iso, y):.5f}")

    # Blend sweeps vs LB-best 4-stack (current best)
    lb4_o = log_blend([lb3_o, o42_iso], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, t42_iso], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_o, y)
    log(f"\nLB-best 4-stack (NEW best) OOF = {lb4_bal:.5f} / LB 0.98094")

    log("\n=== sweep: replace single-seed-iso with bag-iso in 4-stack ===")
    log(f"{'config':>30} {'OOF':>9} {'Δ vs LB-best':>14}")
    rows = []
    # Replace meta_iso step with bag_iso at same α=0.300
    new_o = log_blend([lb3_o, bag_iso_o], np.array([0.7, 0.3]))
    new_t = log_blend([lb3_t, bag_iso_t], np.array([0.7, 0.3]))
    new_bal = bal(new_o, y)
    rows.append({"config": "lb3 + bag_iso α=0.300", "oof": float(new_bal),
                 "delta": float(new_bal - lb4_bal),
                 "oof_arr": new_o, "test_arr": new_t})
    print(f"{'lb3 + bag_iso α=0.300':>30} {new_bal:>9.5f} {new_bal-lb4_bal:>+14.5f}")

    # Sweep α for bag-iso replacement
    for a in [0.20, 0.25, 0.275, 0.300, 0.325, 0.35, 0.40]:
        cand_o = log_blend([lb3_o, bag_iso_o], np.array([1 - a, a]))
        cand_t = log_blend([lb3_t, bag_iso_t], np.array([1 - a, a]))
        cb = bal(cand_o, y)
        rows.append({"config": f"lb3 + bag_iso α={a:.3f}", "oof": float(cb),
                     "delta": float(cb - lb4_bal),
                     "oof_arr": cand_o, "test_arr": cand_t})
        marker = " <- best" if cb == max(r["oof"] for r in rows) else ""
        print(f"{'lb3 + bag_iso α=' + f'{a:.3f}':>30} {cb:>9.5f} {cb-lb4_bal:>+14.5f}{marker}")

    # And: full 4-stack + a bit more bag_iso (stack on top)
    for a in [0.025, 0.05, 0.075, 0.10, 0.15]:
        cand_o = log_blend([lb4_o, bag_iso_o], np.array([1 - a, a]))
        cand_t = log_blend([lb4_t, bag_iso_t], np.array([1 - a, a]))
        cb = bal(cand_o, y)
        rows.append({"config": f"4stack + bag_iso α={a:.3f}", "oof": float(cb),
                     "delta": float(cb - lb4_bal),
                     "oof_arr": cand_o, "test_arr": cand_t})
        marker = " <- best" if cb == max(r["oof"] for r in rows) else ""
        print(f"{'4stack + bag_iso α=' + f'{a:.3f}':>30} {cb:>9.5f} {cb-lb4_bal:>+14.5f}{marker}")

    best = max(rows, key=lambda r: r["oof"])
    print(f"\nBEST: {best['config']}  OOF={best['oof']:.5f}  Δ={best['delta']:+.5f}")

    if best["delta"] >= 1e-4:
        pred = (np.log(np.clip(best["test_arr"], 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred]
        tag = best["config"].replace(" ", "_").replace("=", "").replace(".", "p")
        path = SUB / f"submission_next_metabag3_{tag}.csv"
        sub.to_csv(path, index=False)
        log(f"wrote {path}  (Δ={best['delta']:+.5f} ≥ +1e-4)")
    else:
        log(f"best Δ={best['delta']:+.5f} below +1e-4 gate; no submission")

    out = dict(
        per_seed_standalone=[float(bal(_normed(o), y)) for o in oofs],
        bag_standalone=float(bal(bag_oof, y)),
        bag_iso_standalone=float(bal(bag_iso_o, y)),
        lb4_oof=float(lb4_bal),
        sweep=[{k: v for k, v in r.items() if k not in ("oof_arr", "test_arr")}
               for r in rows],
        best={k: v for k, v in best.items() if k not in ("oof_arr", "test_arr")},
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "next_meta_stack_seedbag_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
