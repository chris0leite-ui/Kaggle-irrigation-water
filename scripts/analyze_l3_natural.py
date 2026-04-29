"""Analyze L3 RF natural vs LB-best v1 RF natural (LB 0.98129).

Both are STANDALONE candidates (own-tuned-bias submissions). The
4-gate framework was designed for BLEND candidates; for a head-to-head
standalone comparison, the relevant diagnostics are:

  G1 (OOF):     L3 tuned bal_acc vs v1 tuned bal_acc 0.98063
                project LB at v1's -0.00066 gap → L3_LB_proj
  G2 (PCR):     per-class recall delta vs v1, no class drops > -5e-4
  G3 (calib):   bias drift |max| vs -log(prior); compare to v1's 0.20
  G4 (test):    net rare-class flip count + churn; need ADD-High direction

Output: scripts/artifacts/analyze_l3_natural_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")

CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
TARGET = "Irrigation_Need"


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    n_tr = len(train)
    print(f"loaded train n={n_tr} test n={len(test)}")

    # Load both candidates
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy")
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy")
    l3_oof = np.load(ART / "oof_sklearn_rf_meta_l3natural.npy")
    l3_test = np.load(ART / "test_sklearn_rf_meta_l3natural.npy")
    print(f"v1 oof {v1_oof.shape}  test {v1_test.shape}")
    print(f"l3 oof {l3_oof.shape}  test {l3_test.shape}")

    # Read result JSONs
    with open(ART / "sklearn_rf_meta_natural_results.json") as f:
        v1_res = json.load(f)
    with open(ART / "sklearn_rf_meta_l3natural_results.json") as f:
        l3_res = json.load(f)

    print("\n=== STANDALONE OOF COMPARISON ===")
    print(f"  v1 (LB 0.98129)   OOF tuned = {v1_res['tuned_log_bias']:.5f}  "
          f"bias = {v1_res['log_bias']}")
    print(f"  L3 (T1)           OOF tuned = {l3_res['tuned_log_bias']:.5f}  "
          f"bias = {l3_res['log_bias']}")
    delta = l3_res['tuned_log_bias'] - v1_res['tuned_log_bias']
    print(f"  Δ OOF (L3 - v1)   = {delta:+.5f}")

    # G1 — OOF lift + LB projection at v1's documented gap (-0.00066)
    v1_gap = -0.00066  # documented LB - OOF gap for v1
    l3_lb_proj = l3_res['tuned_log_bias'] + (-v1_gap)  # L3_LB ≈ L3_OOF + 0.00066
    print(f"\nG1 PROJECTION (assumes L3 inherits v1's gap):")
    print(f"  v1: OOF 0.98063 → LB 0.98129 (gap -0.00066)")
    print(f"  L3: OOF {l3_res['tuned_log_bias']:.5f} → LB ~{l3_lb_proj:.5f}")

    # Bias drift
    v1_drift = max(abs(b - p) for b, p in
                    zip(v1_res['log_bias'],
                        [-np.log(c) for c in [0.5872, 0.3795, 0.0333]]))
    l3_drift_max = l3_res.get('drift_max_abs', None)
    print(f"\nG3 CALIBRATION:")
    print(f"  v1 drift |max| ~ 0.20")
    print(f"  L3 drift |max| = {l3_drift_max:.4f}  drift = {l3_res['drift_from_minus_log_prior']}")

    # G2 — per-class recall comparison at OWN tuned bias
    v1_bias = np.array(v1_res['log_bias'])
    l3_bias = np.array(l3_res['log_bias'])
    v1_pred_oof = (safelog(v1_oof) + v1_bias).argmax(1)
    l3_pred_oof = (safelog(l3_oof) + l3_bias).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred_oof)
    l3_pcr = per_class_recall(y, l3_pred_oof)
    pcr_delta = l3_pcr - v1_pcr
    print(f"\nG2 PER-CLASS RECALL (at own-tuned bias each):")
    print(f"  v1 PCR = [L={v1_pcr[0]:.5f} M={v1_pcr[1]:.5f} H={v1_pcr[2]:.5f}]")
    print(f"  L3 PCR = [L={l3_pcr[0]:.5f} M={l3_pcr[1]:.5f} H={l3_pcr[2]:.5f}]")
    print(f"  Δ PCR  = [L={pcr_delta[0]:+.5f} M={pcr_delta[1]:+.5f} H={pcr_delta[2]:+.5f}]")
    g2_pass = all(pcr_delta >= -5e-4)
    print(f"  G2 verdict: {'PASS' if g2_pass else 'FAIL'} (each class within -5e-4)")

    # G4 — test-side prediction comparison
    v1_test_pred = (safelog(v1_test) + v1_bias).argmax(1)
    l3_test_pred = (safelog(l3_test) + l3_bias).argmax(1)
    diff_mask = v1_test_pred != l3_test_pred
    n_diff = int(diff_mask.sum())
    print(f"\nG4 TEST-SIDE DISAGREEMENT (vs v1 LB-best):")
    print(f"  Test rows differ: {n_diff} / {len(v1_test_pred)} ({100*n_diff/len(v1_test_pred):.3f}%)")

    # High-class flip analysis
    add_h = int(((l3_test_pred == 2) & (v1_test_pred != 2)).sum())
    rem_h = int(((v1_test_pred == 2) & (l3_test_pred != 2)).sum())
    net_h = add_h - rem_h
    churn_h = add_h + rem_h
    g4_ratio = abs(net_h) / max(churn_h, 1)
    print(f"  H-flips: +{add_h} ADD-H,  -{rem_h} REMOVE-H,  net = {net_h:+d}")
    print(f"  G4 ratio = {g4_ratio:.3f}  (need ≥ 0.5 AND net_h > 0 for clean ADD-H)")
    g4_direction = "ADD-High" if net_h > 0 else ("REMOVE-High" if net_h < 0 else "neutral")
    g4_pass = (net_h > 0) and (g4_ratio >= 0.5)
    print(f"  G4 direction: {g4_direction}  ({'PASS' if g4_pass else 'FAIL'})")

    # Test class distribution shift
    for cls_idx, cls_name in [(0, "Low"), (1, "Medium"), (2, "High")]:
        v1_n = int((v1_test_pred == cls_idx).sum())
        l3_n = int((l3_test_pred == cls_idx).sum())
        print(f"  test cls={cls_name}: v1={v1_n} l3={l3_n} Δ={l3_n - v1_n:+d}")

    # Summary verdict
    print("\n=== OVERALL VERDICT ===")
    g1_pos = delta > 0
    print(f"  G1 (OOF Δ > 0):     {'PASS' if g1_pos else 'FAIL'} ({delta:+.5f})")
    print(f"  G2 (PCR guardrail): {'PASS' if g2_pass else 'FAIL'}")
    g3_pass = (l3_drift_max < 0.5) if l3_drift_max is not None else False
    print(f"  G3 (drift < 0.5):   {'PASS' if g3_pass else 'FAIL'} ({l3_drift_max:.4f})")
    print(f"  G4 (clean ADD-H):   {'PASS' if g4_pass else 'FAIL'}")

    n_pass = sum([g1_pos, g2_pass, g3_pass, g4_pass])
    print(f"\n  GATES PASSED: {n_pass}/4")
    if n_pass == 4:
        print("  → CANDIDATE for LB probe (request user approval)")
    else:
        print("  → ALL 4 gates required for LB probe; document NULL otherwise")

    summary = dict(
        v1_oof_tuned=v1_res['tuned_log_bias'],
        l3_oof_tuned=l3_res['tuned_log_bias'],
        oof_delta=float(delta),
        l3_lb_projection=float(l3_lb_proj),
        v1_pcr=v1_pcr.tolist(),
        l3_pcr=l3_pcr.tolist(),
        pcr_delta=pcr_delta.tolist(),
        v1_drift_max=float(v1_drift),
        l3_drift_max=float(l3_drift_max) if l3_drift_max is not None else None,
        n_test_diff=n_diff,
        add_high=add_h, remove_high=rem_h, net_high=net_h, churn_high=churn_h,
        g4_ratio=float(g4_ratio),
        g4_direction=g4_direction,
        gates=dict(
            g1_oof_lift=g1_pos,
            g2_pcr_guardrail=g2_pass,
            g3_calibration=g3_pass,
            g4_clean_add_high=g4_pass,
        ),
        gates_passed=n_pass,
    )
    out_p = ART / "analyze_l3_natural_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
