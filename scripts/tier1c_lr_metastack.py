"""Tier-1c #N1: Multinomial-LR meta-stacker on the same 63-component bank
that produced our LB-best XGB meta-stacker (LB 0.98094).

Why: wguesdon's `ps6e4-30-model-ensemble-with-stacking` (kernel audit round 4)
explicitly chose LGB-stacker over greedy because "greedy CV > LB". Our prior
LR meta-stacker test (2026-04-21 soft-blend session) used a 12-component bank.
This script tests LR on the SAME 63-component bank that gave us +0.00086 LB
via XGB. Cheaper, simpler, structurally orthogonal failure mode (no tree
depth to overfit) — useful as either a hedge candidate (vs XGB stacker) or
as confirmation that depth=4 was the right capacity.

Pipeline (mirrors `tier1b_xgb_metastack.py`):
  - Build LB-best 3-stack via tier1b_helpers.build_lbbest_stack()
  - Load 63-component pool (same EXCLUDE rules)
  - Meta features: log-probs of every component + LB-best stack log-probs +
    14 distance/rule meta features
  - 5-fold StratifiedKFold(seed=42), multinomial LR with class_weight='balanced'
  - StandardScaler before LR (LR is scale-sensitive, unlike XGB)
  - Standalone iso-cal + fixed-bias blend sweep vs LB-best 3-stack
  - Cross-compare vs the XGB meta-stacker (load saved oof_xgb_metastack.npy)
  - Emit submission only if Δ ≥ +2e-4 OR if the LR-stacker is competitive
    enough to be a hedge candidate (within 0.0010 OOF of XGB stacker AND
    Jaccard < 0.97 vs XGB stacker)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, N_FOLDS, SEED, SUB, TARGET,
    bal_at_bias as bal, build_lbbest_stack, iso_cal, load_pool, log,
)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    lb_bal = bal(lb_oof, y)
    log(f"  LB-best 3-stack OOF = {lb_bal:.5f}")

    log("loading pool (same EXCLUDE as XGB meta-stacker)")
    pool = load_pool()
    component_names = sorted(pool.keys())
    log(f"  {len(component_names)} 3-class components loaded")

    log("constructing meta features (mirrors tier1b_xgb_metastack)")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))

    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1)
    log(f"  meta-feature shape: {X_tr.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((len(train), 3), dtype=np.float32)
    test_meta_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t1 = time.time()
        # Per-fold StandardScaler (fit on train fold only).
        scaler = StandardScaler()
        Xt_tr = scaler.fit_transform(X_tr[tr_idx])
        Xt_va = scaler.transform(X_tr[va_idx])
        Xt_te = scaler.transform(X_te)

        # sklearn ≥1.5 removed `multi_class`; lbfgs solver defaults to
        # multinomial loss automatically.
        lr = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
            n_jobs=-1,
        )
        lr.fit(Xt_tr, y[tr_idx])
        vp = lr.predict_proba(Xt_va).astype(np.float32)
        oof_meta[va_idx] = vp
        tp = lr.predict_proba(Xt_te).astype(np.float32)
        test_meta_folds.append(tp)
        argmax_bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        log(f"  fold {fold+1}/{N_FOLDS} val_argmax_bal={argmax_bal:.5f} "
            f"wall={time.time()-t1:.1f}s")

    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_lr_metastack.npy", oof_meta)
    np.save(ART / "test_lr_metastack.npy", test_meta)
    log("saved oof_lr_metastack.npy + test_lr_metastack.npy")

    # Standalone (raw + iso-cal)
    raw_argmax_bal = balanced_accuracy_score(y, oof_meta.argmax(1))
    raw_tuned_bal = bal(oof_meta, y)
    iso_oof, iso_test = iso_cal(oof_meta, test_meta, y)
    iso_argmax_bal = balanced_accuracy_score(y, iso_oof.argmax(1))
    iso_tuned_bal = bal(iso_oof, y)

    log("\n=== LR META-STACKER standalone ===")
    log(f"  raw  argmax        = {raw_argmax_bal:.5f}")
    log(f"  raw  @recipe-bias  = {raw_tuned_bal:.5f}")
    log(f"  iso  argmax        = {iso_argmax_bal:.5f}")
    log(f"  iso  @recipe-bias  = {iso_tuned_bal:.5f}")

    # Errors + Jaccard vs LB-best
    pred_lb = (np.log(np.clip(lb_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_iso = (np.log(np.clip(iso_oof, 1e-12, 1)) + BIAS).argmax(1)
    errs_lb = (pred_lb != y).sum()
    errs_iso = (pred_iso != y).sum()
    inter = ((pred_lb != y) & (pred_iso != y)).sum()
    union = ((pred_lb != y) | (pred_iso != y)).sum()
    jacc_lb = inter / max(union, 1)
    log(f"\nerrs LB-best={errs_lb}  LR_iso={errs_iso}  "
        f"Jaccard(LR_iso, LB-best) = {jacc_lb:.4f}")

    # Cross-compare vs XGB meta-stacker (if on disk)
    xgb_path = ART / "oof_xgb_metastack.npy"
    xgb_compare = None
    if xgb_path.exists():
        xgb_oof = np.load(xgb_path).astype(np.float32)
        xgb_test = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
        xgb_iso_oof, xgb_iso_test = iso_cal(xgb_oof, xgb_test, y)
        xgb_iso_bal = bal(xgb_iso_oof, y)
        pred_xgb_iso = (np.log(np.clip(xgb_iso_oof, 1e-12, 1)) + BIAS).argmax(1)
        errs_xgb_iso = (pred_xgb_iso != y).sum()
        inter2 = ((pred_xgb_iso != y) & (pred_iso != y)).sum()
        union2 = ((pred_xgb_iso != y) | (pred_iso != y)).sum()
        jacc_xgb = inter2 / max(union2, 1)
        log(f"\nXGB stacker_iso  OOF = {xgb_iso_bal:.5f}  errs = {errs_xgb_iso}")
        log(f"LR  stacker_iso  OOF = {iso_tuned_bal:.5f}  errs = {errs_iso}")
        log(f"Jaccard(LR_iso, XGB_iso) = {jacc_xgb:.4f}")
        xgb_compare = dict(
            xgb_iso_oof=float(xgb_iso_bal),
            xgb_iso_errs=int(errs_xgb_iso),
            jaccard_lr_vs_xgb=float(jacc_xgb),
        )

    # Blend sweep at fixed recipe bias vs LB-best 3-stack (use ISO-cal LR)
    log("\n=== fixed-bias blend sweep: LR_iso × LB-best 3-stack ===")
    log(f"{'alpha_lr':>10} {'OOF':>9} {'Δ':>9}")
    alphas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    rows = []
    for a in alphas:
        blend = log_blend([lb_oof, iso_oof], np.array([1 - a, a]))
        b = bal(blend, y)
        d = b - lb_bal
        rows.append({"alpha": a, "oof": float(b), "delta": float(d)})
        tag = " ← best" if len(rows) > 1 and d > max(r["delta"] for r in rows[:-1]) else ""
        log(f"{a:>10.3f} {b:>9.5f} {d:>+9.5f}{tag}")
    best = max(rows, key=lambda r: r["delta"])

    # Emit submission only if blend Δ ≥ +2e-4 (LB-transfer threshold)
    sub_path = None
    if best["delta"] >= 2e-4:
        a = best["alpha"]
        test_blend = log_blend([lb_test, iso_test], np.array([1 - a, a]))
        pred_t = (np.log(np.clip(test_blend, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_t]
        tag = f"lr_iso_a{int(a*1000):03d}"
        sub_path = SUB / f"submission_tier1c_{tag}.csv"
        sub.to_csv(sub_path, index=False)
        log(f"\nΔ={best['delta']:+.5f} ≥ +2e-4 → wrote {sub_path}")
    else:
        log(f"\nbest blend Δ={best['delta']:+.5f} below +2e-4 gate; no submission emitted")

    out = dict(
        components=component_names,
        n_components=len(component_names),
        feature_dim=int(X_tr.shape[1]),
        raw_argmax_bal=float(raw_argmax_bal),
        raw_tuned_bal=float(raw_tuned_bal),
        iso_argmax_bal=float(iso_argmax_bal),
        iso_tuned_bal=float(iso_tuned_bal),
        lb_best_oof=float(lb_bal),
        errs_lb=int(errs_lb),
        errs_iso=int(errs_iso),
        jaccard_iso_vs_lb=float(jacc_lb),
        xgb_compare=xgb_compare,
        blend_sweep=rows,
        best=best,
        submission=str(sub_path) if sub_path else None,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "tier1c_lr_metastack_results.json").write_text(json.dumps(out, indent=2))
    log("wrote scripts/artifacts/tier1c_lr_metastack_results.json")
    log(f"\nTOTAL elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
