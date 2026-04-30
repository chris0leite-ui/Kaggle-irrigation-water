"""Idea 4b — Selective override of B (LB 0.98140) using triple-consensus filter.

After Idea 4 (bagged_v1' + B's mechanism) produced 151 row diffs vs B
with net REMOVE-H direction, the 14-component stability data shows:
  - All 105 H→M override rows have 14-bank-majority = M
  - All 108 selective flips align with bank consensus

This is a MUCH cleaner candidate than Idea 4 raw:
  - Anchor = B (LB 0.98140, current LB-best)
  - Override fires ONLY where 3 independent consensus axes agree:
    (a) bagged_v1' argmax differs from B (fold-seed-bagging signal)
    (b) {raw, tier1b} unanimously say bagged_v1's class
    (c) 14-component bank majority confirms the class
  - 108 row flips total (105 H→M, 2 L→M, 1 M→L)

Break-even precision math:
  H→M direction: 91.94% precision needed for net positive macro
  At 14-bank-majority + bagged_v1 + raw + tier1b agreement, plausible
  precision: 95-98% (matches the original 95.6% from 0.98134 mechanism)

Expected LB outcome:
  - 95% precision: +0.00005 to +0.00010 LB
  - 92% precision: 0 ± noise
  - 88% precision: -0.0001 to -0.0003 LB

Mechanism is genuinely untested:
  - B's original mechanism only used {raw, tier1b} k=2 unanimous
  - Adding bagged_v1' base + 14-bank majority filter = stricter consensus
  - Targets ROWS B missed (105 H→M not in B's original 88)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def csv_to_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def main():
    print("=== Idea 4b: selective override of B with triple-consensus filter ===\n")

    # Load all components
    b = csv_to_argmax("submission_2other_raw_tier1b_k2")            # anchor (LB 0.98140)
    bp = csv_to_argmax("submission_idea4_foldbag_v1_b_mech")         # B' from Idea 4
    maj = np.load(ART / "stability_test_majority.npy")                # 14-bank majority

    # Selective flip: where B disagrees with B' AND 14-bank majority agrees with B'
    diff_mask = b != bp
    bank_agree_mask = maj == bp
    flip_mask = diff_mask & bank_agree_mask

    new_pred = b.copy()
    new_pred[flip_mask] = bp[flip_mask]

    print(f"diff_mask (B != B'): {int(diff_mask.sum())}")
    print(f"bank-agree mask (maj == B'): {int(bank_agree_mask.sum())}")
    print(f"selective flips: {int(flip_mask.sum())}")
    print(f"new_pred class counts: {np.bincount(new_pred, minlength=3).tolist()}")

    # Direction breakdown
    directions = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((b == fr) & (new_pred == to)).sum())
            if n > 0:
                directions[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n
    print(f"directions: {directions}")

    h_added = int(((b != 2) & (new_pred == 2)).sum())
    h_removed = int(((b == 2) & (new_pred != 2)).sum())
    net_h = h_added - h_removed
    print(f"net_H = +{h_added} -{h_removed} = {net_h:+d}")

    # Stability check on flipped rows
    agr = np.load(ART / "stability_test_agreement.npy")
    flip_agr = agr[flip_mask]
    print(f"\nstability agreement on flipped rows:")
    print(f"  p25={np.percentile(flip_agr, 25):.3f}")
    print(f"  p50={np.percentile(flip_agr, 50):.3f}")
    print(f"  p75={np.percentile(flip_agr, 75):.3f}")

    # Emit submission
    test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()
    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
    })
    out_csv = SUB / "submission_idea4b_selective_override.csv"
    sub.to_csv(out_csv, index=False)
    print(f"\nemitted: {out_csv}")

    # Compare to original 0.98134 mechanism precision (95.6% on H->M)
    # Estimate plausible LB if precision similar:
    print("\n=== expected LB (assuming 95% precision on H->M direction) ===")
    n_hm = directions.get("H->M", 0)
    if n_hm > 0:
        # macro_delta ≈ (correct_M_recall_gain - wrong_H_recall_loss) / 3
        # at 95% precision: 0.95 * n_hm correct
        # correct M recall gain: (0.95 * n_hm) / N_M_true
        # wrong H recall loss: (0.05 * n_hm) / N_H_true
        # Use train priors to estimate N_M_true, N_H_true on test
        n_m_test = 100261  # B's M count, proxy for true M
        n_h_test = 10279   # B's H count, proxy for true H (slight under-estimate)
        for prec in [0.95, 0.92, 0.88, 0.80]:
            corr = prec * n_hm
            wrong = (1 - prec) * n_hm
            macro_delta = (corr / n_m_test - wrong / n_h_test) / 3
            print(f"  precision {prec*100:.0f}%: macro_delta = {macro_delta:+.6f} -> proj LB = {0.98140 + macro_delta:.5f}")

    out_json = ART / "idea4b_selective_override_results.json"
    out_json.write_text(json.dumps({
        "n_flips": int(flip_mask.sum()),
        "directions": directions,
        "net_h": net_h,
        "h_added": h_added,
        "h_removed": h_removed,
        "stability_p50_on_flips": float(np.percentile(flip_agr, 50)),
        "candidate_csv": str(out_csv),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
