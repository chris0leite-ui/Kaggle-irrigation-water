"""A2 diagnostics: how does A2 compare to rawashishsin v3 + prior PRIMARY?

Key questions:
  1. Is A2 genuinely distinct from rawashishsin (Jaccard < 0.85)?
  2. Does A2 have positive corr with rawashishsin's LB-positive direction?
  3. What's A2's per-class recall profile vs rawashishsin v3 + prior PRIMARY?
  4. Is A2 a viable LB-probe candidate, or just noisy replica?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed, ART, BIAS
from common import log_blend


def jaccard(a_pred, b_pred):
    """Jaccard of error sets, given two argmax prediction arrays vs y_true."""
    return float((a_pred == b_pred).mean())


def per_class_recall(y, pred):
    out = []
    for c in range(3):
        m = y == c
        out.append(float((pred[m] == c).mean()) if m.any() else 0.0)
    return out


def main():
    y = load_y()

    # Load A2
    a2_oof = np.load("kaggle_kernel/output_a2_prod/oof_a2_natural_calib.npy").astype(np.float32)
    a2_test = np.load("kaggle_kernel/output_a2_prod/test_a2_natural_calib.npy").astype(np.float32)
    print(f"A2: oof shape={a2_oof.shape} test shape={a2_test.shape}")

    # Load rawashishsin v3
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_test = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)
    print(f"rawashishsin v3: oof shape={raw_oof.shape} test shape={raw_test.shape}")

    # Apply A2's tuned bias
    A2_BIAS = np.array([-1.08, -1.005, 0.0])
    RAW_BIAS = np.array([-1.36, -1.19, 0.00])

    eps = 1e-12

    # A2 argmax at A2's bias
    a2_pred = (np.log(np.clip(a2_oof, eps, 1)) + A2_BIAS).argmax(1)
    raw_pred = (np.log(np.clip(raw_oof, eps, 1)) + RAW_BIAS).argmax(1)

    a2_bal = balanced_accuracy_score(y, a2_pred)
    raw_bal = balanced_accuracy_score(y, raw_pred)
    print(f"\nStandalone @ tuned biases:")
    print(f"  A2          bal_acc = {a2_bal:.5f}")
    print(f"  rawashishsin bal_acc = {raw_bal:.5f}")

    # Reconstruct PRIMARY (LB 0.98094) — recipe-family 4-stack
    print("\nReconstructing PRIMARY (LB 0.98094)...")
    s2_o, s2_t = build_lbbest_stack(y)  # 3-stack
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o, meta_t = iso_cal(meta_o, meta_t, y)
    primary_o = log_blend([s2_o, meta_o], np.array([0.7, 0.3]))
    primary_t = log_blend([s2_t, meta_t], np.array([0.7, 0.3]))
    primary_pred = (np.log(np.clip(primary_o, eps, 1)) + BIAS).argmax(1)
    primary_bal = balanced_accuracy_score(y, primary_pred)
    print(f"  PRIMARY recipe-family bal_acc = {primary_bal:.5f}")

    # Jaccards (= prediction agreement rate)
    print("\nPrediction agreement (OOF, train-side):")
    print(f"  A2 vs rawashishsin v3: agree={jaccard(a2_pred, raw_pred):.4f} disagree={(a2_pred != raw_pred).sum():,}")
    print(f"  A2 vs PRIMARY 4-stack: agree={jaccard(a2_pred, primary_pred):.4f} disagree={(a2_pred != primary_pred).sum():,}")
    print(f"  rawashishsin vs PRIMARY: agree={jaccard(raw_pred, primary_pred):.4f} disagree={(raw_pred != primary_pred).sum():,}")

    # Test-side argmax comparison
    a2_test_pred = (np.log(np.clip(a2_test, eps, 1)) + A2_BIAS).argmax(1)
    raw_test_pred = (np.log(np.clip(raw_test, eps, 1)) + RAW_BIAS).argmax(1)
    primary_test_pred = (np.log(np.clip(primary_t, eps, 1)) + BIAS).argmax(1)
    print("\nTest-side argmax disagreement counts:")
    print(f"  A2 vs rawashishsin (270k): {(a2_test_pred != raw_test_pred).sum():,}")
    print(f"  A2 vs PRIMARY (270k):       {(a2_test_pred != primary_test_pred).sum():,}")
    print(f"  rawashishsin vs PRIMARY:    {(raw_test_pred != primary_test_pred).sum():,}")

    # Per-class recall
    print("\nPer-class recall @ tuned biases:")
    a2_pcr = per_class_recall(y, a2_pred)
    raw_pcr = per_class_recall(y, raw_pred)
    primary_pcr = per_class_recall(y, primary_pred)
    print(f"  A2:           L={a2_pcr[0]:.4f} M={a2_pcr[1]:.4f} H={a2_pcr[2]:.4f}")
    print(f"  rawashishsin: L={raw_pcr[0]:.4f} M={raw_pcr[1]:.4f} H={raw_pcr[2]:.4f}")
    print(f"  PRIMARY:      L={primary_pcr[0]:.4f} M={primary_pcr[1]:.4f} H={primary_pcr[2]:.4f}")
    print(f"  Δ A2 - rawashishsin: L={a2_pcr[0]-raw_pcr[0]:+.4f} M={a2_pcr[1]-raw_pcr[1]:+.4f} H={a2_pcr[2]-raw_pcr[2]:+.4f}")

    # Test class dist
    print("\nTest predicted class distribution:")
    for name, pred in [("A2", a2_test_pred), ("rawashishsin", raw_test_pred), ("PRIMARY", primary_test_pred)]:
        cnts = np.bincount(pred, minlength=3)
        pct = 100 * cnts / cnts.sum()
        print(f"  {name:<14} L={cnts[0]:,} ({pct[0]:.1f}%) M={cnts[1]:,} ({pct[1]:.1f}%) H={cnts[2]:,} ({pct[2]:.1f}%)")

    # Verdict guidance
    print("\n" + "="*60)
    a2_raw_disagree = (a2_test_pred != raw_test_pred).sum()
    if a2_raw_disagree < 100:
        print(f"VERDICT: A2 is essentially a noisy copy of rawashishsin "
              f"({a2_raw_disagree} test rows differ). Skip LB probe.")
    elif a2_raw_disagree < 1000:
        print(f"VERDICT: A2 has SLIGHT distinction ({a2_raw_disagree} test rows differ). "
              f"LB probe might lift ±0.0001-0.0003.")
    elif a2_raw_disagree < 5000:
        print(f"VERDICT: A2 has MEANINGFUL distinction ({a2_raw_disagree} test rows differ). "
              f"LB probe is justified — likely between A2 and rawashishsin's LB.")
    else:
        print(f"VERDICT: A2 is genuinely DIFFERENT ({a2_raw_disagree} test rows differ). "
              f"LB probe definitely justified — probably better OR worse than rawashishsin.")


if __name__ == "__main__":
    main()
