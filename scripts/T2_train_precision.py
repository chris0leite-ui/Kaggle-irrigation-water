"""T2 — measure precision on TRAIN OOF using v1 RF natural as 4b proxy.

For TRAIN rows where:
  - v1's argmax = H (i.e., proxy for 4b says H)
  - 14-bank-majority on TRAIN OOF says M
  - bank-mean P(M) > 0.5 (analog of conformal-set-excludes-H)

What's the precision = P(true label == M | filter)?
If precision ≥ 92%, T2's 691 test-side H->M flips are likely net-positive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import bank_mean_probs, load_bank  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")


def main():
    print("=== T2 TRAIN-OOF precision validation ===\n")
    oof_bank = load_bank("oof")
    oof_mean = bank_mean_probs(oof_bank)
    y = pd.read_csv(DATA / "train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    ).to_numpy(dtype=np.int8)

    # 14-bank majority on TRAIN OOF
    from scipy.stats import mode
    oof_argmax_per_model = oof_bank.argmax(axis=2)  # (14, 630000)
    oof_majority = mode(oof_argmax_per_model, axis=0, keepdims=False).mode

    # Use v1 RF natural as 4b proxy (closest single-model to 4b's class distribution)
    v1 = np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32)
    v1_argmax = v1.argmax(1).astype(np.int8)

    # Filter: v1 says H, bank-majority says M
    # (analog of "4b says H, conformal/majority says M")
    h_to_m_mask = (v1_argmax == 2) & (oof_majority == 1)
    n = int(h_to_m_mask.sum())
    print(f"v1 argmax dist: {np.bincount(v1_argmax, minlength=3).tolist()}")
    print(f"oof_majority dist: {np.bincount(oof_majority, minlength=3).tolist()}")
    print(f"v1=H AND maj=M (raw filter): {n}")

    if n > 0:
        precision_M = float((y[h_to_m_mask] == 1).mean())
        precision_H = float((y[h_to_m_mask] == 2).mean())
        precision_L = float((y[h_to_m_mask] == 0).mean())
        print(f"  P(true=M | filter) = {precision_M:.4f}")
        print(f"  P(true=H | filter) = {precision_H:.4f}")
        print(f"  P(true=L | filter) = {precision_L:.4f}")
        print(f"  break-even = 0.92")
        print(f"  verdict: {'PASS' if precision_M >= 0.92 else 'FAIL'} for H->M override")

    # Now add bank-mean P(M) > threshold (mimicking conformal "in set")
    print("\nAdd bank-mean P(M) threshold:")
    for thresh in [0.5, 0.6, 0.7, 0.75, 0.8]:
        f = h_to_m_mask & (oof_mean[:, 1] > thresh)
        n = int(f.sum())
        if n > 0:
            p_m = float((y[f] == 1).mean())
            p_h = float((y[f] == 2).mean())
            p_l = float((y[f] == 0).mean())
            be = "PASS" if p_m >= 0.92 else "fail"
            print(f"  P(M)>{thresh}: n={n:6d}  P(true=M)={p_m:.4f}  "
                  f"P(true=H)={p_h:.4f}  P(true=L)={p_l:.4f}  [{be}]")
        else:
            print(f"  P(M)>{thresh}: n=0")

    # Also analog of conformal: bank-mean P(H) < threshold
    print("\nAdd bank-mean P(H) UPPER threshold:")
    for thresh in [0.5, 0.4, 0.3, 0.286, 0.20, 0.15]:
        f = h_to_m_mask & (oof_mean[:, 2] < thresh)
        n = int(f.sum())
        if n > 0:
            p_m = float((y[f] == 1).mean())
            p_h = float((y[f] == 2).mean())
            be = "PASS" if p_m >= 0.92 else "fail"
            print(f"  P(H)<{thresh}: n={n:6d}  P(true=M)={p_m:.4f}  "
                  f"P(true=H)={p_h:.4f}  [{be}]")

    # The ACTUAL T2 filter: 1 - P(v1=H) <= q_hat AND set has at least 1 class
    # (i.e., bank-mean P(H) is in conformal set => H is excluded means
    #  P(H) is well below q_hat)
    # Replicate exactly: bank-mean P(H) is OUTSIDE conformal set when
    # 1 - P(H) > q_hat, i.e., P(H) < 1 - q_hat
    # But T2 actually checks "v1's class outside set", which means
    # 1 - P(v1's_class) > q_hat, i.e., P(v1's_class) < 1 - q_hat
    # For v1=H: 1 - P(H) > q_hat => P(H) < 1 - q_hat = 1 - 0.7142 = 0.2858

    # Replicate T2 exactly: P(H) < 0.2858 AND v1 = H AND maj = M
    print("\nT2 exact filter (P(H)<0.2858 AND v1=H AND maj=M):")
    f = h_to_m_mask & (oof_mean[:, 2] < (1 - 0.7142))
    n = int(f.sum())
    if n > 0:
        p_m = float((y[f] == 1).mean())
        p_h = float((y[f] == 2).mean())
        p_l = float((y[f] == 0).mean())
        be = "PASS" if p_m >= 0.92 else "fail"
        print(f"  n={n}  P(M)={p_m:.4f}  P(H)={p_h:.4f}  P(L)={p_l:.4f}  [{be}]")

    # Compose with the {raw, tier1b} unanimous axis (4b's full filter)
    print("\nCompose with {raw, tier1b} unanimous M:")
    raw = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    tier1b = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    raw_arg = raw.argmax(1)
    t1_arg = tier1b.argmax(1)
    f_with = f & (raw_arg == 1) & (t1_arg == 1)
    n_with = int(f_with.sum())
    f_without = f & ~((raw_arg == 1) & (t1_arg == 1))
    n_without = int(f_without.sum())
    print(f"  T2 ∩ {{raw,tier1b}} unanimous M:  n={n_with}, "
          f"P(M)={(y[f_with]==1).mean():.4f}" if n_with > 0 else f"  n=0")
    print(f"  T2 \\ {{raw,tier1b}} unanimous M:  n={n_without}, "
          f"P(M)={(y[f_without]==1).mean():.4f}" if n_without > 0 else f"  n=0")


if __name__ == "__main__":
    main()
