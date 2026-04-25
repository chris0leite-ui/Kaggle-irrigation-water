"""LR meta-stacker v2: retry with class_weight=None + C=0.1.

The 2026-04-25 N1 LR null (LB 0.97991, gap +0.00176) was diagnosed
structurally — `class_weight='balanced'` at C=1.0 over 210 dims
overfit the rare-High class on 5-fold OOF.  Closure note explicitly
flagged `class_weight=None + C ≤ 0.1` as the untested config.

This v2 mirrors `tier1c_lr_metastack.py` exactly except for those two
HPs.  All filenames suffixed `_v2` to keep v1 artefacts intact.

Decision rule (binhigh-rule compliant — fixed bias, no retune):
  - Standalone iso + Jaccard vs LB-best 4-stack reported.
  - Fixed-bias α-sweep vs LB-best 3-stack.
  - Emit submission only if Δ ≥ +2e-4 AND per-class recall guardrail
    PASSes (each class ≥ anchor − 5e-4).
  - Submission NOT auto-uploaded; LB probe requires user confirmation.
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


def per_class_recall(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    rec = []
    for k in range(3):
        m = y == k
        rec.append(float((pred[m] == k).mean()) if m.any() else 0.0)
    return rec


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("v2: building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y)
    lb_bal = bal(lb_oof, y)
    log(f"  LB-best 3-stack OOF = {lb_bal:.5f}")

    log("v2: loading pool")
    pool = load_pool()
    component_names = sorted(pool.keys())
    log(f"  {len(component_names)} components loaded")

    log("v2: constructing meta features")
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
        scaler = StandardScaler()
        Xt_tr = scaler.fit_transform(X_tr[tr_idx])
        Xt_va = scaler.transform(X_tr[va_idx])
        Xt_te = scaler.transform(X_te)

        # v2 key change: class_weight=None + stronger L2 (C=0.1)
        lr = LogisticRegression(
            C=0.1,
            class_weight=None,
            solver="lbfgs",
            max_iter=2000,
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
    np.save(ART / "oof_lr_metastack_v2.npy", oof_meta)
    np.save(ART / "test_lr_metastack_v2.npy", test_meta)
    log("saved oof_lr_metastack_v2.npy + test_lr_metastack_v2.npy")

    raw_argmax_bal = balanced_accuracy_score(y, oof_meta.argmax(1))
    raw_tuned_bal = bal(oof_meta, y)
    iso_oof, iso_test = iso_cal(oof_meta, test_meta, y)
    iso_argmax_bal = balanced_accuracy_score(y, iso_oof.argmax(1))
    iso_tuned_bal = bal(iso_oof, y)

    log("\n=== LR META-STACKER v2 standalone (C=0.1, class_weight=None) ===")
    log(f"  raw  argmax        = {raw_argmax_bal:.5f}")
    log(f"  raw  @recipe-bias  = {raw_tuned_bal:.5f}")
    log(f"  iso  argmax        = {iso_argmax_bal:.5f}")
    log(f"  iso  @recipe-bias  = {iso_tuned_bal:.5f}")

    pred_lb = (np.log(np.clip(lb_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_iso = (np.log(np.clip(iso_oof, 1e-12, 1)) + BIAS).argmax(1)
    errs_lb = int((pred_lb != y).sum())
    errs_iso = int((pred_iso != y).sum())
    inter = ((pred_lb != y) & (pred_iso != y)).sum()
    union = ((pred_lb != y) | (pred_iso != y)).sum()
    jacc_lb = float(inter / max(union, 1))
    log(f"\nerrs LB-best={errs_lb}  LR_v2_iso={errs_iso}  "
        f"Jaccard(LR_v2_iso, LB-best) = {jacc_lb:.4f}")

    pcr_lb = per_class_recall(lb_oof, y)
    pcr_iso = per_class_recall(iso_oof, y)
    log(f"per-class recall LB-best : L={pcr_lb[0]:.4f} M={pcr_lb[1]:.4f} H={pcr_lb[2]:.4f}")
    log(f"per-class recall LR_v2_iso: L={pcr_iso[0]:.4f} M={pcr_iso[1]:.4f} H={pcr_iso[2]:.4f}")

    # Try LB-best 4-stack as a richer anchor too
    primary_oof_path = ART / "oof_xgb_metastack.npy"
    lb4_compare = None
    if primary_oof_path.exists():
        meta_oof = np.load(primary_oof_path).astype(np.float32)
        meta_test = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
        meta_iso_oof, meta_iso_test = iso_cal(meta_oof, meta_test, y)
        # LB-best 4-stack = 0.7 LB3 + 0.3 meta_iso
        lb4_oof = log_blend([lb_oof, meta_iso_oof], np.array([0.7, 0.3]))
        lb4_test = log_blend([lb_test, meta_iso_test], np.array([0.7, 0.3]))
        lb4_bal = bal(lb4_oof, y)
        log(f"\nLB-best 4-stack rebuilt: {lb4_bal:.5f}")
        # Jaccard vs 4-stack
        pred_lb4 = (np.log(np.clip(lb4_oof, 1e-12, 1)) + BIAS).argmax(1)
        i2 = ((pred_lb4 != y) & (pred_iso != y)).sum()
        u2 = ((pred_lb4 != y) | (pred_iso != y)).sum()
        jacc_lb4 = float(i2 / max(u2, 1))
        errs_lb4 = int((pred_lb4 != y).sum())
        log(f"Jaccard(LR_v2_iso, LB-best 4-stack) = {jacc_lb4:.4f}")
        log(f"errs LB-best 4-stack = {errs_lb4}")
        lb4_compare = dict(lb4_oof=float(lb4_bal), jacc_vs_lb4=jacc_lb4,
                           errs_lb4=errs_lb4)

    log("\n=== fixed-bias blend sweep: LR_v2_iso × LB-best 3-stack ===")
    log(f"{'alpha':>8} {'OOF':>9} {'Δ':>9} {'recL':>7} {'recM':>7} {'recH':>7}")
    alphas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    rows = []
    for a in alphas:
        blend = log_blend([lb_oof, iso_oof], np.array([1 - a, a]))
        b = bal(blend, y)
        d = b - lb_bal
        pcr = per_class_recall(blend, y)
        # PASS guard: each class ≥ LB-best − 5e-4
        passes = all(pcr[k] >= pcr_lb[k] - 5e-4 for k in range(3))
        rows.append({"alpha": a, "oof": float(b), "delta": float(d),
                     "pcr": pcr, "guardrail_pass": passes})
        gtag = "PASS" if passes else "FAIL"
        log(f"{a:>8.3f} {b:>9.5f} {d:>+9.5f} {pcr[0]:>7.4f} {pcr[1]:>7.4f} {pcr[2]:>7.4f}  {gtag}")
    best = max(rows, key=lambda r: r["delta"])

    sub_path = None
    # Strict gate: Δ ≥ +2e-4 AND guardrail pass
    if best["delta"] >= 2e-4 and best["guardrail_pass"]:
        a = best["alpha"]
        test_blend = log_blend([lb_test, iso_test], np.array([1 - a, a]))
        pred_t = (np.log(np.clip(test_blend, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_t]
        tag = f"lr_v2_iso_a{int(a*1000):03d}"
        sub_path = SUB / f"submission_tier1c_{tag}.csv"
        sub.to_csv(sub_path, index=False)
        log(f"\n[GATE PASS] Δ={best['delta']:+.5f} ≥ +2e-4 AND guardrail PASS")
        log(f"  → wrote {sub_path}")
        log(f"  → DO NOT submit without explicit user confirmation")
    else:
        log(f"\n[GATE FAIL] best Δ={best['delta']:+.5f} guard={best['guardrail_pass']}")
        log(f"  → no submission emitted")

    out = dict(
        variant="v2_C0.1_classweightnone",
        components=component_names,
        n_components=len(component_names),
        feature_dim=int(X_tr.shape[1]),
        raw_argmax_bal=float(raw_argmax_bal),
        raw_tuned_bal=float(raw_tuned_bal),
        iso_argmax_bal=float(iso_argmax_bal),
        iso_tuned_bal=float(iso_tuned_bal),
        lb_best_3stack_oof=float(lb_bal),
        errs_lb=errs_lb, errs_iso=errs_iso,
        jaccard_iso_vs_lb=jacc_lb,
        per_class_recall_lb=pcr_lb,
        per_class_recall_iso=pcr_iso,
        lb4_compare=lb4_compare,
        blend_sweep=rows,
        best=best,
        submission=str(sub_path) if sub_path else None,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "tier1c_lr_metastack_v2_results.json").write_text(json.dumps(out, indent=2))
    log("wrote scripts/artifacts/tier1c_lr_metastack_v2_results.json")
    log(f"\nTOTAL elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
