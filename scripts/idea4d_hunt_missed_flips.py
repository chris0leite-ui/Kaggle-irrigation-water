"""Idea 4d — hunt for MISSED flips outside 4b's filter.

Most prior work focused on filtering DOWN 4b's 108 flips. This script
asks the inverse: are there test rows where bagged_v1' STRONGLY says
B is wrong (high confidence), but the {raw, tier1b} unanimous filter
or 14-bank-majority filter BLOCKED the flip — and the rule's
recipe-family disagreement is NOT a meaningful blocker?

The confound-aware analysis (commit ebc59ed) established:
  - bagged_v1 ↔ raw boundary correlation: 81.3%
  - bagged_v1 ↔ tier1b boundary correlation: 80.0%
  - bagged_v1 ↔ 14-bank boundary correlation: 54.1%
  - 4b's permissive design (allow rule disagreement at score 7-8) is a feature

So {raw, tier1b} unanimous and 14-bank are PARTIALLY independent constraints.
Rows blocked by EITHER constraint with very-high bagged_v1 confidence may be
missed-positives.

Mechanism:
  Find test rows where:
    - B argmax != bagged_v1' argmax  (disagreement)
    - bagged_v1' prob_margin >= θ    (high confidence in flip)
    - 14-bank-majority == bagged_v1' (one independence axis confirms)
    - {raw, tier1b} are NOT both ≠ bagged_v1' (i.e., at least one agrees)
    - This row was NOT picked by 4b
  → Candidate add: flip B's prediction to bagged_v1's class.

Sweep θ in {0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95} to find the
margin where the candidate set is small enough (≤ 30 flips) to be
plausibly high-precision yet add new signal.

Risk: dropping the {raw, tier1b} unanimous requirement loosens consensus.
  But it's replaced by bagged_v1 PROB MARGIN which is a quantitative
  axis 4b doesn't use (4b uses argmax-only).
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
    print("=== Idea 4d: hunt missed flips OUTSIDE 4b's filter ===\n")

    # Load
    b = csv_to_argmax("submission_2other_raw_tier1b_k2")
    fb = csv_to_argmax("submission_idea4b_selective_override")
    raw = csv_to_argmax("submission_rawashishsin_2600_standalone")
    # tier1b is the 4-stack
    tier1b = csv_to_argmax("submission_tier1b_greedy_meta")
    maj = np.load(ART / "stability_test_majority.npy")
    agr = np.load(ART / "stability_test_agreement.npy")

    bagged_p = np.load(ART / "_test_bagged_v1_probs.npy")  # (270k, 3)
    # Apply v1's bias [0.43, 0.87, 3.20] to get bagged_v1' argmax
    biased = np.log(np.clip(bagged_p, 1e-9, 1)) + np.array([0.43, 0.87, 3.20])
    bagged_arg = biased.argmax(axis=1).astype(np.int8)

    print(f"bagged_v1' (with bias) argmax dist: {np.bincount(bagged_arg, minlength=3).tolist()}")
    print(f"4b flips (vs B): {int((b != fb).sum())}")
    print(f"4b's flips already include all (B != bagged & bank == bagged & {{raw,tier1b}} unan==bagged) cases\n")

    # Compute bagged prob margin: P(class_argmax) - P(class_2nd)
    sorted_p = np.sort(biased, axis=1)
    margin_log = sorted_p[:, -1] - sorted_p[:, -2]  # log-prob margin, biased
    # Convert to soft P-margin via softmax
    sm = np.exp(biased - biased.max(axis=1, keepdims=True))
    sm = sm / sm.sum(axis=1, keepdims=True)
    sm_top = np.sort(sm, axis=1)
    p_margin = sm_top[:, -1] - sm_top[:, -2]  # post-softmax prob margin

    # Hunt: candidates outside 4b
    diff_b_bagged = b != bagged_arg
    not_in_4b = ~(b != fb)  # NOT already 4b'd
    bank_agrees = maj == bagged_arg

    print(f"=== Margin sweep — candidates outside 4b's filter ===\n")
    print(f"Filter: B != bagged AND bank == bagged AND not in 4b's flip set")
    print(f"        AND ((raw == bagged) OR (tier1b == bagged))   [at least one recipe-axis agrees]\n")

    # AT LEAST ONE of {raw, tier1b} agrees with bagged
    raw_agree = raw == bagged_arg
    tier1b_agree = tier1b == bagged_arg
    at_least_one_recipe = raw_agree | tier1b_agree

    base_cand = diff_b_bagged & not_in_4b & bank_agrees & at_least_one_recipe

    print(f"{'theta':>8} {'n_cand':>8} {'p_margin_cand_p25':>20} {'rec':>10}")
    by_theta = {}
    for theta in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        mask = base_cand & (p_margin >= theta)
        n = int(mask.sum())
        if n > 0:
            p25 = np.percentile(p_margin[mask], 25)
        else:
            p25 = 0.0
        # Direction breakdown
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to: continue
                nn = int(((b == fr) & (bagged_arg == to) & mask).sum())
                if nn > 0:
                    dirs[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = nn
        by_theta[theta] = {"n": n, "directions": dirs}
        print(f"{theta:>8.2f} {n:>8d} {p25:>20.4f}   {dirs}")
    print()

    # Pick the operating point with most stringent yet non-trivial size
    pick_theta = None
    pick_n = None
    for theta in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60]:
        n = by_theta[theta]["n"]
        if n >= 5:
            pick_theta = theta
            pick_n = n
            break

    if pick_theta is None or pick_n == 0:
        print("No candidate operating point with n >= 5. Exiting.")
        return

    print(f"=== Operating point: theta = {pick_theta} ({pick_n} candidate flips) ===\n")
    pick_mask = base_cand & (p_margin >= pick_theta)
    pick_idx = np.where(pick_mask)[0]

    # Inspect each candidate
    print(f"Candidate inspection (first 30):")
    print(f"{'idx':>8} {'B':>3} {'bagged':>6} {'bank':>6} {'agr':>6} {'raw':>4} {'tier1b':>6} {'p_marg':>8}")
    for i in pick_idx[:30]:
        print(f"{i:>8d} {'LMH'[b[i]]:>3} {'LMH'[bagged_arg[i]]:>6} {'LMH'[maj[i]]:>6} {agr[i]:>6.2f} "
              f"{'LMH'[raw[i]]:>4} {'LMH'[tier1b[i]]:>6} {p_margin[i]:>8.4f}")

    # Build candidate
    new_pred = fb.copy()
    new_pred[pick_mask] = bagged_arg[pick_mask]
    n_added = int((new_pred != fb).sum())
    print(f"\nAdded flips on top of 4b: {n_added}")
    new_flip_mask = b != new_pred
    new_dirs = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            nn = int(((b == fr) & (new_pred == to) & new_flip_mask).sum())
            if nn > 0:
                new_dirs[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = nn
    print(f"Total flips vs B (4d'): {int(new_flip_mask.sum())}")
    print(f"Directions: {new_dirs}")

    h_added = int(((b != 2) & (new_pred == 2)).sum())
    h_removed = int(((b == 2) & (new_pred != 2)).sum())
    print(f"net_H = +{h_added} -{h_removed} = {h_added - h_removed:+d}")

    # Project LB
    print("\n=== LB projection (vs 4b base 0.98150) ===")
    n_hm_added = sum(v for k, v in by_theta[pick_theta]["directions"].items() if k.endswith("->M") and k.startswith("H"))
    n_other_added = sum(v for k, v in by_theta[pick_theta]["directions"].items() if not (k.endswith("->M") and k.startswith("H")))
    print(f"H->M added: {n_hm_added}")
    print(f"Other added: {n_other_added}")
    n_m_test = 100261; n_h_test = 10279
    base_lb = 0.98150  # 4b
    for prec in [0.98, 0.95, 0.92, 0.88, 0.80]:
        if n_hm_added > 0:
            corr = prec * n_hm_added
            wrong = (1 - prec) * n_hm_added
            macro = (corr / n_m_test - wrong / n_h_test) / 3
        else:
            macro = 0.0
        print(f"  precision {int(prec*100):d}%: macro = {macro:+.6f} -> proj LB = {base_lb + macro:.5f}")

    # Emit
    test_ids = pd.read_csv(SUB / "submission_idea4b_selective_override.csv", usecols=["id"])["id"].to_numpy()
    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
    })
    out_csv = SUB / f"submission_idea4d_hunt_missed_t{int(pick_theta*100)}.csv"
    sub.to_csv(out_csv, index=False)
    print(f"\nemitted: {out_csv}")
    print(f"  ({n_added} rows added vs 4b)")

    # Save diagnostic
    out_json = ART / "idea4d_hunt_missed_flips_results.json"
    out_json.write_text(json.dumps({
        "by_theta": {str(k): v for k, v in by_theta.items()},
        "operating_theta": pick_theta,
        "n_added": n_added,
        "directions_total": new_dirs,
        "candidate_csv": str(out_csv),
        "h_added": h_added,
        "h_removed": h_removed,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
