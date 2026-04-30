"""4-gate the idea4d k4-unanimous orthogonal-bank candidate on TRAIN OOF.

The 4d test-side analysis surfaced 163 flips (12 L->M + 151 M->L) where
orthogonal bank k4-unanimous disagrees with 4b argmax. Projection:
  @50% precision: 0.98141 (LB regression)
  @65% precision: 0.98155 (+0.00005)
  @80% precision: 0.98168 (+0.00018)

This script measures actual TRAIN OOF precision on the analogous filter
condition, to decide whether the 65%+ precision threshold is plausible.

Anchor: tier1b_greedy_meta (B's L2-stack that 4b inherits).
For each TRAIN OOF row where:
  - orthogonal_bank k4-unanimous = X (X != tier1b argmax)
  - direction in {M->L, L->M, M->H, L->H}
measure fraction where true label == X.

Also compute the unconditional orthogonal-bank-k4-unanimous accuracy
across all TRAIN OOF rows to establish the joint-distribution baseline.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")

LMH = ["L", "M", "H"]
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}


def main():
    print("=== Idea 4d TRAIN OOF precision check ===\n")

    train = pd.read_csv("data/train.csv")
    y = train["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    n_train = len(y)
    print(f"TRAIN rows: {n_train}")

    bank_names = [
        "xgb_corn",
        "recipe_full_te_macrorec_T1_lam03",
        "recipe_full_te_basemargin_K2",
        "recipe_full_te_residte",
    ]
    bank_argmaxes = []
    for n in bank_names:
        oof = np.load(ART / f"oof_{n}.npy").astype(np.float32)
        oof = oof / np.clip(oof.sum(1, keepdims=True), 1e-9, None)
        am = oof.argmax(1).astype(np.int8)
        acc = float((am == y).mean())
        bank_argmaxes.append(am)
        print(f"  oof_{n:50s} argmax_acc={acc:.5f}")
    bank_arr = np.stack(bank_argmaxes, axis=1)

    counts = np.zeros((n_train, 3), dtype=np.int32)
    for c in range(3):
        counts[:, c] = (bank_arr == c).sum(axis=1)
    bank_maj = counts.argmax(axis=1)
    bank_max = counts.max(axis=1)

    n_unan = int((bank_max == 4).sum())
    print(f"\nOrthogonal-bank k4-unanimous: {n_unan} / {n_train} = "
          f"{n_unan/n_train*100:.1f}%")

    unan_mask = bank_max == 4
    unan_acc = float((bank_maj[unan_mask] == y[unan_mask]).mean())
    print(f"Unconditional k4-unanimous accuracy on TRAIN OOF: "
          f"{unan_acc:.5f} ({int((bank_maj[unan_mask]==y[unan_mask]).sum())}/{n_unan})")

    # Tier1b anchor (B's L2-stack lineage)
    tier1b_oof = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    tier1b_oof = tier1b_oof / np.clip(tier1b_oof.sum(1, keepdims=True), 1e-9, None)
    tier1b_am = tier1b_oof.argmax(1).astype(np.int8)
    print(f"Tier1b argmax accuracy: {float((tier1b_am==y).mean()):.5f}")

    # Per-direction precision: orthogonal-bank k4 != tier1b, k4 unanimous
    print("\n=== Per-direction precision on TRAIN OOF ===")
    print(f"(rows where bank_maj != tier1b_am, bank_max == 4)")
    print()
    print(f"{'direction':<10} {'n':>8} {'true=bank':>10} {'true=t1b':>10} "
          f"{'true=other':>11} {'P(bank)':>9} {'break-even':>10} {'verdict':>10}")
    print("-" * 90)

    BREAK_EVEN = {
        (2, 1): 100367 / (100367 + 10174),    # H->M: 0.908
        (2, 0): 159459 / (159459 + 10174),    # H->L: 0.940
        (1, 2): 10174 / (10174 + 100367),     # M->H: 0.092
        (1, 0): 159459 / (159459 + 100367),   # M->L: 0.614
        (0, 2): 10174 / (10174 + 159459),     # L->H: 0.060
        (0, 1): 100367 / (100367 + 159459),   # L->M: 0.386
    }

    diff_mask = bank_maj != tier1b_am
    candidate_mask_all = diff_mask & unan_mask

    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            mask = candidate_mask_all & (tier1b_am == fr) & (bank_maj == to)
            n = int(mask.sum())
            if n == 0:
                continue
            n_bank_correct = int((y[mask] == to).sum())
            n_t1b_correct = int((y[mask] == fr).sum())
            n_other = n - n_bank_correct - n_t1b_correct
            p_bank = n_bank_correct / n if n > 0 else 0.0
            be = BREAK_EVEN.get((fr, to), 0.5)
            verdict = "PASS" if p_bank >= be else "FAIL"
            print(f"{LMH[fr]+'->'+LMH[to]:<10} {n:>8} {n_bank_correct:>10} "
                  f"{n_t1b_correct:>10} {n_other:>11} {p_bank:>9.3f} "
                  f"{be:>10.3f} {verdict:>10}")

    # Now restrict to the test-side directions only (M->L, L->M, M->H, L->H)
    print("\n=== Test-side ALLOWED directions only (M->L, L->M, M->H, L->H) ===")
    print()
    allowed = {(1, 0), (0, 1), (1, 2), (0, 2)}
    n_total = 0
    n_correct = 0
    for fr, to in allowed:
        mask = candidate_mask_all & (tier1b_am == fr) & (bank_maj == to)
        n = int(mask.sum())
        if n == 0:
            continue
        n_bank_correct = int((y[mask] == to).sum())
        n_total += n
        n_correct += n_bank_correct
    print(f"  Total flips on TRAIN OOF: {n_total}")
    print(f"  Total bank-correct: {n_correct} ({n_correct/max(1,n_total)*100:.1f}%)")
    print(f"  vs test-side: 163 flips (151 M->L + 12 L->M)")

    # Project candidate LB if test-side precision matches TRAIN OOF precision
    print("\n=== Test-side LB projection at TRAIN-OOF-derived precision ===")
    print("(per-direction precision applied to test-side n=163 flips)")

    test_dirs = {(1, 0): 151, (0, 1): 12}
    N_L, N_M, N_H = 159459, 100367, 10174
    Ns = [N_L, N_M, N_H]
    macro_delta = 0.0
    for (fr, to), n_test in test_dirs.items():
        mask = candidate_mask_all & (tier1b_am == fr) & (bank_maj == to)
        n_train = int(mask.sum())
        if n_train < 30:
            print(f"  {LMH[fr]}->{LMH[to]}: only {n_train} TRAIN OOF rows, skip projection")
            continue
        p_train = (y[mask] == to).sum() / n_train
        # macro delta = sum_{dir} n_test * (p / Ns[to] - (1-p) / Ns[fr]) / 3
        d = n_test * (p_train / Ns[to] - (1 - p_train) / Ns[fr]) / 3
        macro_delta += d
        print(f"  {LMH[fr]}->{LMH[to]}: n_train={n_train}, p_train={p_train:.3f}, "
              f"n_test={n_test}, macro_delta={d:+.6f}")
    print(f"  Total macro_delta: {macro_delta:+.6f}")
    print(f"  Projected LB (4b 0.98150 + delta): {0.98150 + macro_delta:.5f}")


if __name__ == "__main__":
    main()
