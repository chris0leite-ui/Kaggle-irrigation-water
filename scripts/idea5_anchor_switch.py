"""Idea 5 — Anchor switch: rawashishsin + 3-OTHER {v1, tier1b, 4b} k=3 unanimous.

Different anchor (rawashishsin v3, LB 0.98109) instead of v1 RF natural.
OTHERS = {v1_RF_natural, tier1b 4-stack, 4b (LB 0.98150)}. k=3 unanimous
override fires only where ALL THREE OTHERS agree on a class different
from rawashishsin's.

Why this might lift past 4b (LB 0.98150):
  - Rawashishsin has favorable -0.00099 OOF->LB gap (LB much higher than CV)
  - 4b carries 14-bank consensus + bagged-v1 signal as evidence
  - {v1, tier1b, 4b} is a strong consensus pool (all LB > rawashishsin's
    standalone)
  - Override rows differ from B/4b's (rawashishsin's argmax distribution
    is structurally different — single XGB on sklearn TE)
  - 3-OTHER unanimous is stricter than B's 2-OTHER, higher precision

Mechanism-novel: anchor switch + 4b as evidence axis. Untested.
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
    print("=== Idea 5: anchor=rawashishsin, OTHERS={v1, tier1b, 4b} k=3 unan ===\n")

    anchor_name = "submission_rawashishsin_2600_standalone"  # LB 0.98109
    anchor = csv_to_argmax(anchor_name)

    others = {
        "v1_rf":  csv_to_argmax("submission_sklearn_rf_meta_natural_standalone_v1_lb98129"),
        "tier1b": csv_to_argmax("submission_tier1b_greedy_meta"),
        "4b":     csv_to_argmax("submission_idea4b_selective_override"),
    }

    # k=3 unanimous: all 3 OTHERS agree on a class != anchor's
    o_arr = np.stack([others["v1_rf"], others["tier1b"], others["4b"]], axis=1)
    same = (o_arr == o_arr[:, [0]]).all(axis=1)
    diff_anchor = o_arr[:, 0] != anchor
    override_mask = same & diff_anchor
    consensus_class = o_arr[:, 0]

    n_override = int(override_mask.sum())
    print(f"k=3 unanimous override on rawashishsin: {n_override} rows")

    # Apply override
    new_pred = anchor.copy()
    new_pred[override_mask] = consensus_class[override_mask]

    # Direction breakdown
    LMH = ["L", "M", "H"]
    directions = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            n = int(((anchor == fr) & (new_pred == to)).sum())
            if n > 0:
                directions[f"{LMH[fr]}->{LMH[to]}"] = n
    print(f"directions vs rawashishsin anchor: {directions}")

    # Class counts
    print(f"new_pred class counts: {np.bincount(new_pred, minlength=3).tolist()}")

    # Compare to current LB-best 4b (LB 0.98150) and B (LB 0.98140)
    fb = csv_to_argmax("submission_idea4b_selective_override")
    b = csv_to_argmax("submission_2other_raw_tier1b_k2")
    diff_4b = int((new_pred != fb).sum())
    diff_b = int((new_pred != b).sum())
    print(f"\nnew_pred vs 4b (LB 0.98150): {diff_4b} rows differ")
    print(f"new_pred vs B  (LB 0.98140): {diff_b} rows differ")

    # Stability check
    agr = np.load(ART / "stability_test_agreement.npy")
    if n_override > 0:
        ovr_agr = agr[override_mask]
        print(f"\nstability agreement on override rows:")
        print(f"  p25={np.percentile(ovr_agr, 25):.3f}")
        print(f"  p50={np.percentile(ovr_agr, 50):.3f}")
        print(f"  p75={np.percentile(ovr_agr, 75):.3f}")

    # Emit candidate
    test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()
    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
    })
    out_csv = SUB / "submission_idea5_anchor_switch.csv"
    sub.to_csv(out_csv, index=False)
    print(f"\nemitted: {out_csv}")

    # Net High change
    h_added = int(((anchor != 2) & (new_pred == 2)).sum())
    h_removed = int(((anchor == 2) & (new_pred != 2)).sum())
    net_h = h_added - h_removed
    print(f"net_H = +{h_added} -{h_removed} = {net_h:+d}")

    # Estimate LB outcome
    # rawashishsin LB 0.98109; override ~93% precision → LB ~0.98109 + macro_delta
    # macro_delta = (correct_M_gain/N_M - wrong_H_loss/N_H) / 3 for each direction
    # Approximate using B's class counts on test: N_M ≈ 100k, N_H ≈ 10k, N_L ≈ 160k
    n_hm = directions.get("H->M", 0)
    n_mh = directions.get("M->H", 0)
    n_lm = directions.get("L->M", 0)
    n_ml = directions.get("M->L", 0)
    n_hl = directions.get("H->L", 0)
    n_lh = directions.get("L->H", 0)

    print(f"\n=== LB projection (rawashishsin baseline 0.98109) ===")
    for prec in [0.95, 0.92, 0.88]:
        # Per direction: delta = (correct_class_gain - wrong_class_loss) for each row
        # H->M: gain in M recall, loss in H recall
        # M->H: gain in H recall, loss in M recall
        # etc
        macro_delta = 0.0
        if n_hm > 0:
            macro_delta += (n_hm * prec / 100261 - n_hm * (1-prec) / 10279) / 3
        if n_mh > 0:
            macro_delta += (n_mh * prec / 10279 - n_mh * (1-prec) / 100261) / 3
        if n_lm > 0:
            macro_delta += (n_lm * prec / 100261 - n_lm * (1-prec) / 159460) / 3
        if n_ml > 0:
            macro_delta += (n_ml * prec / 159460 - n_ml * (1-prec) / 100261) / 3
        proj_lb = 0.98109 + macro_delta
        print(f"  precision {prec*100:.0f}%: macro_delta = {macro_delta:+.6f} -> proj LB = {proj_lb:.5f}")

    out_json = ART / "idea5_anchor_switch_results.json"
    out_json.write_text(json.dumps({
        "anchor": anchor_name,
        "others": list(others.keys()),
        "n_override": n_override,
        "directions": directions,
        "net_h": net_h,
        "diff_vs_4b": diff_4b,
        "diff_vs_b": diff_b,
        "candidate_csv": str(out_csv),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
