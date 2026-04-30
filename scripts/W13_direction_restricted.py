"""W13 refinement — direction-restricted override.

The wrongness predictor (AUC 0.91) is strong, but applying override to
14-bank-majority across all directions fails because most overrides are
H->M direction (break-even 90.9% — too high).

Refinement: ONLY override in directions with FAVORABLE break-even:
  - M->L  (61.5% break-even)  ← keep at high P(wrong)
  - M->H  (9.1%  break-even)  ← keep almost always
  - L->M  (38.5% break-even)  ← keep at moderate P(wrong)
  - L->H  (9.1%  break-even)  ← keep almost always
  - H->M  (90.9% break-even)  ← SKIP (W13 fails here)
  - H->L  (90.9% break-even)  ← SKIP

This reuses the wrongness predictor's predictions but filters by direction.
Keep 4b's H predictions intact; only flip M->X or L->X cases.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


# Per-direction break-even precision (using B's class counts as test proxy)
N_L, N_M, N_H = 159460, 100261, 10279

BREAK_EVEN = {
    (2, 1): N_M / (N_M + N_H),    # H->M: 90.9%
    (2, 0): N_L / (N_L + N_H),    # H->L: 94%
    (1, 2): N_H / (N_H + N_M),    # M->H: 9.3%
    (1, 0): N_L / (N_L + N_M),    # M->L: 61.4%
    (0, 2): N_H / (N_H + N_L),    # L->H: 6.1%
    (0, 1): N_M / (N_M + N_L),    # L->M: 38.6%
}


def main():
    print("=== W13 direction-restricted: only flip favorable-break-even directions ===\n")

    # Load wrongness probabilities + 14-bank majority + 4b argmax
    fb = csv_argmax("submission_idea4b_selective_override")  # LB 0.98150
    bank_maj = np.load(ART / "stability_test_majority.npy")

    # Reload p_wrong from W13 (need to re-derive since we didn't save it)
    # Re-run the prediction by loading the trained model state...
    # Easier: use the existing W13 candidates at different thresholds, find which rows differ
    # Actually let me use the exact W13 candidates and just filter direction post-hoc

    candidates = {}
    for tau in [50, 70, 80, 90]:
        cand = csv_argmax(f"submission_W13_wrong_pred_tau{tau}")
        candidates[tau] = cand

    # For each tau, restrict to favorable-break-even directions only
    LMH = ["L", "M", "H"]
    print("Direction-restricted variants (favorable break-even directions only):")
    print()

    for tau in [50, 70, 80, 90]:
        cand = candidates[tau]
        original_flips = (cand != fb)
        n_orig = int(original_flips.sum())

        # Build refined: keep only flips where (4b_class -> cand_class) has break-even ≤ 0.62
        # Skip H->M (0.91), H->L (0.94)
        # Keep M->L (0.61), M->H (0.09), L->M (0.39), L->H (0.06)
        keep_mask = np.zeros(len(fb), dtype=bool)
        for fr in [0, 1]:  # only L and M as origin (don't flip H)
            for to in range(3):
                if fr == to:
                    continue
                be = BREAK_EVEN.get((fr, to), 1.0)
                if be < 0.65:
                    direction_mask = (fb == fr) & (cand == to)
                    keep_mask |= direction_mask

        new_pred = fb.copy()
        new_pred[keep_mask] = cand[keep_mask]
        n_keep = int(keep_mask.sum())

        # Direction breakdown
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to: continue
                k = int(((fb == fr) & (new_pred == to)).sum())
                if k > 0:
                    dirs[f"{LMH[fr]}->{LMH[to]}"] = k

        # Net H
        h_a = int(((fb != 2) & (new_pred == 2)).sum())
        h_r = int(((fb == 2) & (new_pred != 2)).sum())

        # LB projection
        # Use TRAIN τ-precision as proxy (47.6% / 60.7% / 67.1% / 79.1%)
        train_prec = {50: 0.476, 70: 0.607, 80: 0.671, 90: 0.791}[tau]
        macro_delta = 0.0
        for d, n in dirs.items():
            fr_c = LMH.index(d[0])
            to_c = LMH.index(d[3])
            n_corr = n * train_prec
            n_wrong = n * (1 - train_prec)
            N_to = [N_L, N_M, N_H][to_c]
            N_fr = [N_L, N_M, N_H][fr_c]
            macro_delta += (n_corr / N_to - n_wrong / N_fr) / 3

        proj_lb = 0.98150 + macro_delta

        print(f"τ={tau/100:.2f}:  orig flips={n_orig}, dir-restricted={n_keep}")
        print(f"  directions: {dirs}")
        print(f"  net_H = +{h_a} -{h_r} = {h_a-h_r:+d}")
        print(f"  proj LB at TRAIN-prec ({train_prec}): {proj_lb:.5f}  Δ={macro_delta:+.6f}")
        print()

        # Emit candidate
        test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()
        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
        })
        out_csv = SUB / f"submission_W13_dir_restricted_tau{tau}.csv"
        sub.to_csv(out_csv, index=False)

    print(f"=== break-even table (per direction, using B's test class counts) ===")
    for (fr, to), be in BREAK_EVEN.items():
        print(f"  {LMH[fr]}->{LMH[to]}: {be:.3f}")


if __name__ == "__main__":
    main()
