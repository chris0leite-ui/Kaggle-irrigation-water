"""Experiment B-followup: bank-extended XGB meta-stacker with B + LR-v2 added.

Extends the Tier-1b XGB meta-stacker bank with two new components:
  - oof_mlp_metastack.npy  (B's raw MLP-meta output, before iso-cal)
  - oof_lr_metastack_v2.npy (LR-meta v2 — C=0.1, no class_weight)

Both have measured tight OOF→LB gaps:
  - B (MLP-meta in L3 blend):  OOF 0.98118 -> LB 0.98091, gap +0.00027
  - LR-meta v2:                 OOF 0.98107 -> LB 0.98052, gap +0.00055
Both are simpler-than-XGB metas that hit the LB-best 4-stack region.
Adding them as inputs gives a fresh XGB meta-stacker access to family-
diversity signal that the LB-best v1 meta-stacker did not have.

Hypothesis: a fresh XGB-meta on this expanded bank can compound the
positive transfer that B and LR-v2 both showed individually. Risk:
bank-extension OOF→LB inflation pattern (documented 11+ times).

5-fold StratifiedKFold(seed=42), aligned with every other OOF.
Outputs:
  oof_xgb_metastack_v7.npy / test_xgb_metastack_v7.npy
  xgb_metastack_v7_results.json (blend gate + per-class delta)
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
    BIAS, EXCLUDE, build_lbbest_stack, iso_cal, load_pool, _normed,
)

ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
EPS = 1e-12


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y, bias=BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, EPS, 1.0)) + bias).argmax(1))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best stack OOF = {bal(lb_oof, y):.5f}")

    log("loading base pool (Tier-1b EXCLUDE list)")
    pool = load_pool(y)
    log(f"  base pool: {len(pool)} components")

    # ADD the two new family-diverse metas.
    extras = ["mlp_metastack", "lr_metastack_v2"]
    for name in extras:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        assert oof_p.exists() and test_p.exists(), f"missing {name}"
        o = _normed(np.load(oof_p).astype(np.float32))
        t = _normed(np.load(test_p).astype(np.float32))
        if name in pool:
            log(f"  WARN: {name} was already in base pool (will be replaced)")
        pool[name] = (o, t)
        log(f"  added '{name}' (oof.shape={o.shape})")
    log(f"  expanded pool: {len(pool)} components")

    # Construct meta-feature matrix (mirrors tier1b_xgb_metastack.py).
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

    # XGB meta — same heavy-reg HPs as tier1b v1.
    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    max_rounds = 3000
    es_rounds = 200

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_v7 = np.zeros((len(train), 3), dtype=np.float32)
    test_v7_folds = []
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
        oof_v7[va_idx] = vp.astype(np.float32)
        tp = booster.predict(dte, iteration_range=(0, bi + 1))
        test_v7_folds.append(tp)
        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        log(f"  fold {fold+1}/{N_FOLDS} it={bi} val_argmax_bal={argmax_bal:.5f} "
            f"wall={time.time()-t1:.1f}s")

    test_v7 = np.mean(test_v7_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_xgb_metastack_v7.npy", oof_v7)
    np.save(ART / "test_xgb_metastack_v7.npy", test_v7)

    # Standalone metrics.
    v7_argmax_bal = balanced_accuracy_score(y, oof_v7.argmax(1))
    v7_tuned_bal = bal(oof_v7, y)
    log(f"\n=== v7 META-STACKER standalone ===")
    log(f"  argmax OOF bal_acc  = {v7_argmax_bal:.5f}")
    log(f"  @recipe-bias OOF    = {v7_tuned_bal:.5f}")

    # Iso-calibrate v7 (LB-validated arch).
    v7_iso_oof, v7_iso_test = iso_cal(oof_v7, test_v7, y)
    v7_iso_bal = bal(v7_iso_oof, y)
    log(f"  iso-cal'd @bias OOF = {v7_iso_bal:.5f}")

    # LB-best 4-stack reference (with v1 meta-stacker iso α=0.30).
    v1_meta_iso_oof, v1_meta_iso_test = iso_cal(
        _normed(np.load(ART / "oof_xgb_metastack.npy")),
        _normed(np.load(ART / "test_xgb_metastack.npy")), y)
    lb4_oof = log_blend([lb_oof, v1_meta_iso_oof], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb_test, v1_meta_iso_test], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_oof, y)
    log(f"  LB-best 4-stack (v1 meta α=0.30) OOF = {lb4_bal:.5f}")

    # Blend sweep: v7_iso into LB-best 3-stack at α (replacing v1 meta).
    log(f"\n=== blend gate (v7_iso into LB-best 3-stack, α-sweep) ===")
    log(f"{'alpha':>8} {'OOF':>9} {'Δ vs 4st':>10} {'PCR L':>8} {'PCR M':>8} {'PCR H':>8}  PCR pass")
    rows = []
    pcr_anchor = np.array([
        ((np.log(np.clip(lb4_oof, EPS, 1)) + BIAS).argmax(1)[y == k] == k).mean()
        for k in range(3)])
    for a in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        blend = log_blend([lb_oof, v7_iso_oof], np.array([1 - a, a]))
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

    # Errs/Jaccard vs LB-best 4-stack.
    pred_anchor = (np.log(np.clip(lb4_oof, EPS, 1)) + BIAS).argmax(1)
    pred_v7 = (np.log(np.clip(v7_iso_oof, EPS, 1)) + BIAS).argmax(1)
    errs_anchor = int((pred_anchor != y).sum())
    errs_v7 = int((pred_v7 != y).sum())
    inter = int(((pred_anchor != y) & (pred_v7 != y)).sum())
    union = int(((pred_anchor != y) | (pred_v7 != y)).sum())
    jacc = inter / max(union, 1)
    log(f"\nerrs LB-best 4-stack={errs_anchor}  v7_iso={errs_v7}  "
        f"Jaccard(v7, 4-stack) = {jacc:.4f}")

    out = dict(
        components=component_names,
        n_components=len(component_names),
        feature_dim=int(X_tr.shape[1]),
        extras_added=extras,
        best_iters=[int(b) for b in best_iters],
        v7_argmax=float(v7_argmax_bal),
        v7_tuned=float(v7_tuned_bal),
        v7_iso=float(v7_iso_bal),
        lb_best_4stack=float(lb4_bal),
        blend_sweep=rows,
        best=best,
        gate_pass=gate_pass,
        err_4stack=errs_anchor,
        err_v7=errs_v7,
        jaccard_vs_4stack=float(jacc),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "xgb_metastack_v7_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote oof/test_xgb_metastack_v7.npy + results.json")


if __name__ == "__main__":
    main()
