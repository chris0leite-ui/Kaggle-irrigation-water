"""Experiment v7b: strict-EXCLUDE bank rerun + explicit B/LR-v2 adds.

Difference from v7: drops ALL prior meta-output artifacts from the pool,
leaving only the original ~63 base components that produced LB-best v1.
Then explicitly adds 2 family-diverse metas as new inputs:
  - oof_mlp_metastack.npy  (B's MLP-meta — LB 0.98091, gap +0.00027)
  - oof_lr_metastack_v2.npy (LR-meta v2 — LB 0.98052, gap +0.00055)

This is the cleaner cdeotte-style "GBDT-meta + NN-meta as inputs to a
new GBDT-meta" test. v7 was tainted by 80+ auto-included meta variants
that v1 didn't see.

Same 5-fold StratifiedKFold(seed=42), same XGB heavy-reg HPs.
Outputs:
  oof_xgb_metastack_v7b.npy / test_xgb_metastack_v7b.npy
  xgb_metastack_v7b_results.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from tier1b_xgb_metastack import (  # noqa: E402
    BIAS, EXCLUDE as BASE_EXCLUDE, build_lbbest_stack, iso_cal, _normed,
)

ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
EPS = 1e-12


# Strict EXCLUDE: base + all prior meta-output names (so the v7b pool
# is clean of everything that "v1 didn't see" + binary specialists +
# LB-confirmed regressors). The 2 explicit adds (mlp_metastack,
# lr_metastack_v2) override this exclusion below.
STRICT_EXCLUDE = BASE_EXCLUDE | {
    # All XGB metastack variants (would create circular leakage)
    "xgb_metastack", "xgb_metastack_v2", "xgb_metastack_v3",
    "xgb_metastack_v3_iso", "xgb_metastack_v4", "xgb_metastack_v5",
    "xgb_metastack_v5_iso", "xgb_metastack_v6", "xgb_metastack_v6_combined",
    "xgb_metastack_v6lb", "xgb_metastack_v7", "xgb_metastack_varB",
    "xgb_metastack_varC", "xgb_metastack_bag3", "xgb_metastack_j2bag",
    "xgb_metastack_n5b_both", "xgb_metastack_narrow",
    # Other meta-stackers that should NOT be inputs (they're outputs)
    "lr_metastack", "lr_metastack_v2",  # default-excluded; explicit-added below
    "mlp_metastack",                     # default-excluded; explicit-added below
    "meta_l3_xgb_mlp",                   # B's L3 itself — derived
    "leaf_ote_meta_v2",                  # tree-leaf-OTE meta
    # Greedy / forward selection / hill-climb outputs
    "c0_greedy", "c0_v2_lb_best_3way", "c0_v3_lb_best_3way",
    "j2_bag", "step1_greedy_lbbest",
    # Own-CSV ensemble outputs
    "own_3view", "own_S1_equal_log", "own_S2_lb_weighted_tau100",
    "own_S2_lb_weighted_tau200", "own_S2_lb_weighted_tau500",
    "own_S2_lb_weighted_tau1000", "own_S3_hard_vote",
    "own_S4_soft_vote", "own_S5_greedy_forward", "own_greedy_fine",
    # Per-fold partial checkpoints
    "mlp_metastack_fold1", "mlp_metastack_fold2", "mlp_metastack_fold3",
    "mlp_metastack_fold4", "mlp_metastack_fold5",
    # Other suspect derived artefacts
    "hillclimb_negweights",  # D's gate-failed result
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y, bias=BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, EPS, 1.0)) + bias).argmax(1))


def load_pool_strict(y):
    pool = {}
    for oof_p in sorted(ART.glob("oof_*.npy")):
        name = oof_p.stem.replace("oof_", "", 1)
        if name in STRICT_EXCLUDE:
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
        if o.shape[0] != 630_000:
            continue
        if (o.sum(1) < 1e-3).any():  # skip partial-fold artefacts
            continue
        pool[name] = (_normed(o), _normed(t))
    return pool


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best stack OOF = {bal(lb_oof, y):.5f}")

    log("loading STRICT pool (drops all prior meta variants)")
    pool = load_pool_strict(y)
    log(f"  strict pool: {len(pool)} components")

    # EXPLICIT add: B's MLP-meta + LR-v2.
    extras = ["mlp_metastack", "lr_metastack_v2"]
    for name in extras:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        assert oof_p.exists() and test_p.exists(), f"missing {name}"
        o = _normed(np.load(oof_p).astype(np.float32))
        t = _normed(np.load(test_p).astype(np.float32))
        assert name not in pool, f"strict EXCLUDE failed: {name} still in pool"
        pool[name] = (o, t)
        log(f"  added '{name}' (oof.shape={o.shape})")
    log(f"  expanded strict pool: {len(pool)} components")

    # Construct meta-feature matrix.
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
    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1).astype(np.float32)
    log(f"  meta-feature shape: {X_tr.shape}")

    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_v7b = np.zeros((len(train), 3), dtype=np.float32)
    test_v7b_folds = []
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
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
        oof_v7b[va_idx] = vp.astype(np.float32)
        tp = booster.predict(dte, iteration_range=(0, bi + 1))
        test_v7b_folds.append(tp)
        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax_bal={argmax_bal:.5f} "
            f"wall={time.time()-t1:.1f}s")

    test_v7b = np.mean(test_v7b_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_xgb_metastack_v7b.npy", oof_v7b)
    np.save(ART / "test_xgb_metastack_v7b.npy", test_v7b)

    # Standalone metrics + iso-cal.
    v7b_argmax_bal = balanced_accuracy_score(y, oof_v7b.argmax(1))
    v7b_tuned_bal = bal(oof_v7b, y)
    log(f"\n=== v7b META-STACKER standalone ===")
    log(f"  argmax OOF bal_acc  = {v7b_argmax_bal:.5f}")
    log(f"  @recipe-bias OOF    = {v7b_tuned_bal:.5f}")
    v7b_iso_oof, v7b_iso_test = iso_cal(oof_v7b, test_v7b, y)
    v7b_iso_bal = bal(v7b_iso_oof, y)
    log(f"  iso-cal'd @bias OOF = {v7b_iso_bal:.5f}")

    # LB-best 4-stack reference.
    v1_meta_iso_oof, v1_meta_iso_test = iso_cal(
        _normed(np.load(ART / "oof_xgb_metastack.npy")),
        _normed(np.load(ART / "test_xgb_metastack.npy")), y)
    lb4_oof = log_blend([lb_oof, v1_meta_iso_oof], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb_test, v1_meta_iso_test], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_oof, y)
    log(f"  LB-best 4-stack (v1 meta α=0.30) OOF = {lb4_bal:.5f}")

    # Blend gate sweep.
    log(f"\n=== blend gate (v7b_iso into LB-best 3-stack, α-sweep) ===")
    log(f"{'alpha':>8} {'OOF':>9} {'Δ vs 4st':>10} {'PCR L':>8} {'PCR M':>8} {'PCR H':>8}  PCR pass")
    rows = []
    pcr_anchor = np.array([
        ((np.log(np.clip(lb4_oof, EPS, 1)) + BIAS).argmax(1)[y == k] == k).mean()
        for k in range(3)])
    for a in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        blend = log_blend([lb_oof, v7b_iso_oof], np.array([1 - a, a]))
        b = bal(blend, y)
        d = b - lb4_bal
        pred = (np.log(np.clip(blend, EPS, 1)) + BIAS).argmax(1)
        pcr = np.array([(pred[y == k] == k).mean() for k in range(3)])
        pcr_delta = pcr - pcr_anchor
        pcr_pass = bool((pcr_delta >= -5e-4).all())
        rows.append({"alpha": a, "oof": float(b), "delta": float(d),
                     "pcr": pcr.tolist(), "pcr_delta": pcr_delta.tolist(),
                     "pcr_pass": pcr_pass,
                     "errs": int((pred != y).sum())})
        log(f"{a:>8.3f} {b:>9.5f} {d:>+10.5f} "
            f"{pcr_delta[0]:>+8.5f} {pcr_delta[1]:>+8.5f} {pcr_delta[2]:>+8.5f}  "
            f"{'PASS' if pcr_pass else 'FAIL'}")

    best = max(rows, key=lambda r: r["delta"] if r["pcr_pass"] else -1)
    log(f"\nBEST gate-passing: α={best['alpha']:.3f}  Δ vs 4st={best['delta']:+.5f}  "
        f"errs={best['errs']}  PCR={best['pcr']}")
    gate_pass = bool(best["delta"] >= 2e-4 and best["pcr_pass"])
    log(f"GATE: {'PASS' if gate_pass else 'FAIL'} (need Δ ≥ +2e-4 AND PCR ≥ -5e-4)")

    pred_anchor = (np.log(np.clip(lb4_oof, EPS, 1)) + BIAS).argmax(1)
    pred_v7b = (np.log(np.clip(v7b_iso_oof, EPS, 1)) + BIAS).argmax(1)
    errs_anchor = int((pred_anchor != y).sum())
    errs_v7b = int((pred_v7b != y).sum())
    inter = int(((pred_anchor != y) & (pred_v7b != y)).sum())
    union = int(((pred_anchor != y) | (pred_v7b != y)).sum())
    jacc = inter / max(union, 1)
    log(f"\nerrs LB-best 4-stack={errs_anchor}  v7b_iso={errs_v7b}  "
        f"Jaccard(v7b, 4-stack) = {jacc:.4f}")

    out = dict(
        components=component_names,
        n_components=len(component_names),
        feature_dim=int(X_tr.shape[1]),
        extras_added=extras,
        best_iters=[int(b) for b in best_iters],
        v7b_argmax=float(v7b_argmax_bal),
        v7b_tuned=float(v7b_tuned_bal),
        v7b_iso=float(v7b_iso_bal),
        lb_best_4stack=float(lb4_bal),
        blend_sweep=rows,
        best=best,
        gate_pass=gate_pass,
        err_4stack=errs_anchor,
        err_v7b=errs_v7b,
        jaccard_vs_4stack=float(jacc),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "xgb_metastack_v7b_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote oof/test_xgb_metastack_v7b.npy + results.json")


if __name__ == "__main__":
    main()
