"""Blend gate for experiment C (distill_no_rule).

Computes the standard diagnostic suite for C's OOF/test against the
LB-best 4-stack anchor:
  - Jaccard vs anchor (target: < 0.80 for novel orthogonality)
  - errs vs anchor (target: ≤ anchor × 1.05)
  - per-class recall delta (target: ≥ -5e-4 each class)
  - fixed-bias α-sweep into LB-best 3-stack (target: Δ ≥ +2e-4 OOF)

Reports gate PASS/FAIL and recommends LB probe vs no-go.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX
from tier1b_xgb_metastack import (
    BIAS, build_lbbest_stack, iso_cal, _normed,
)

ART = Path("scripts/artifacts")
DATA = Path("data")
EPS = 1e-12


def bal(p, y, bias=BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, EPS, 1.0)) + bias).argmax(1))


def main():
    train = pd.read_csv(DATA / "train.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy(dtype=np.int32)

    print("loading C OOF + LB-best stacks")
    c_oof = _normed(np.load(ART / "oof_distill_no_rule.npy"))
    c_test = _normed(np.load(ART / "test_distill_no_rule.npy"))

    lb_oof, lb_test = build_lbbest_stack(y)
    v1_meta_iso_oof, v1_meta_iso_test = iso_cal(
        _normed(np.load(ART / "oof_xgb_metastack.npy")),
        _normed(np.load(ART / "test_xgb_metastack.npy")), y)
    lb4_oof = log_blend([lb_oof, v1_meta_iso_oof], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb_test, v1_meta_iso_test], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_oof, y)
    print(f"LB-best 4-stack OOF @ recipe bias = {lb4_bal:.5f}")

    # Standalone C diagnostic.
    c_bal = bal(c_oof, y)
    print(f"C @ recipe bias = {c_bal:.5f}  (Δ vs anchor = {c_bal - lb4_bal:+.5f})")

    pred_lb4 = (np.log(np.clip(lb4_oof, EPS, 1)) + BIAS).argmax(1)
    pred_c = (np.log(np.clip(c_oof, EPS, 1)) + BIAS).argmax(1)
    errs_lb4 = int((pred_lb4 != y).sum())
    errs_c = int((pred_c != y).sum())
    inter = int(((pred_lb4 != y) & (pred_c != y)).sum())
    union = int(((pred_lb4 != y) | (pred_c != y)).sum())
    jacc = inter / max(union, 1)
    print(f"errs LB-best 4-stack={errs_lb4}  C={errs_c}  Jaccard(C, 4-stack) = {jacc:.4f}")

    pcr_anchor = np.array([(pred_lb4[y == k] == k).mean() for k in range(3)])
    pcr_c = np.array([(pred_c[y == k] == k).mean() for k in range(3)])
    print(f"anchor PCR: L={pcr_anchor[0]:.5f} M={pcr_anchor[1]:.5f} H={pcr_anchor[2]:.5f}")
    print(f"     C PCR: L={pcr_c[0]:.5f} M={pcr_c[1]:.5f} H={pcr_c[2]:.5f}")
    print(f"   Δ PCR:  L={pcr_c[0]-pcr_anchor[0]:+.5f} M={pcr_c[1]-pcr_anchor[1]:+.5f} H={pcr_c[2]-pcr_anchor[2]:+.5f}")

    # iso-cal C and try blend sweep.
    c_iso_oof, c_iso_test = iso_cal(c_oof, c_test, y)
    print(f"C iso @ recipe bias = {bal(c_iso_oof, y):.5f}")

    print(f"\n=== blend sweep: LB-best 3-stack ⊗ C-iso ===")
    print(f"{'alpha':>8} {'OOF':>9} {'Δ vs 4st':>10} {'errs':>7} {'PCR L':>8} {'PCR M':>8} {'PCR H':>8}  PCR pass")
    rows = []
    for a in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        blend = log_blend([lb_oof, c_iso_oof], np.array([1 - a, a]))
        b = bal(blend, y)
        d = b - lb4_bal
        pred = (np.log(np.clip(blend, EPS, 1)) + BIAS).argmax(1)
        pcr = np.array([(pred[y == k] == k).mean() for k in range(3)])
        pcr_delta = pcr - pcr_anchor
        pcr_pass = bool((pcr_delta >= -5e-4).all())
        errs_blend = int((pred != y).sum())
        rows.append({"alpha": a, "oof": float(b), "delta": float(d),
                     "pcr_delta": pcr_delta.tolist(), "pcr_pass": pcr_pass,
                     "errs": errs_blend})
        print(f"{a:>8.3f} {b:>9.5f} {d:>+10.5f} {errs_blend:>7} "
              f"{pcr_delta[0]:>+8.5f} {pcr_delta[1]:>+8.5f} {pcr_delta[2]:>+8.5f}  "
              f"{'PASS' if pcr_pass else 'FAIL'}")

    best = max(rows, key=lambda r: r["delta"] if r["pcr_pass"] else -1)
    gate_pass = bool(best["delta"] >= 2e-4 and best["pcr_pass"])
    print(f"\nBEST gate-passing: α={best['alpha']:.3f}  Δ vs 4st={best['delta']:+.5f}")
    print(f"GATE: {'PASS' if gate_pass else 'FAIL'} (need Δ ≥ +2e-4 AND PCR ≥ -5e-4)")

    out = dict(
        c_standalone_at_recipe_bias=float(c_bal),
        c_iso_standalone=float(bal(c_iso_oof, y)),
        lb_best_4stack=float(lb4_bal),
        errs_anchor=errs_lb4,
        errs_c=errs_c,
        jaccard_c_vs_4stack=float(jacc),
        pcr_anchor=pcr_anchor.tolist(),
        pcr_c=pcr_c.tolist(),
        blend_sweep=rows,
        best=best,
        gate_pass=gate_pass,
    )
    (ART / "distill_no_rule_blend_gate_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote distill_no_rule_blend_gate_results.json")


if __name__ == "__main__":
    main()
