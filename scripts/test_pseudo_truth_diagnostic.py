"""Test-side pseudo-truth synthesis diagnostic.

Identifies test rows where multiple independent signals UNANIMOUSLY agree
on a class. Treats these as high-purity pseudo-labels (~99% expected
precision). Compares LB-validated submissions on this pseudo-truth set
to validate which submission is structurally closer to ground truth on
test, WITHOUT burning an LB slot.

Mechanism:
  pseudo_truth[i] = majority_class iff:
    14-bank majority == B's argmax == 4b's argmax (==raw_v3==tier1b for narrowest)
    AND 14-bank agreement >= 0.85

  For each LB-validated submission, compute:
    accuracy(submission, pseudo_truth) on rows where pseudo_truth defined.

  Submissions ranked by pseudo-truth accuracy. If 4b > B by margin
  matching the +0.00010 LB gain, the lift is structural.

This is the AGREEMENT-row inversion of T7 (which analyzed disagreement).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def csv_to_argmax(name: str) -> np.ndarray:
    p = SUB / f"{name}.csv"
    s = pd.read_csv(p)["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def main():
    print("=== Test-side pseudo-truth diagnostic ===\n")

    # Load LB-validated submissions
    cands = {
        "4b_LB_0.98150": csv_to_argmax("submission_idea4b_selective_override"),
        "B_LB_0.98140":  csv_to_argmax("submission_2other_raw_tier1b_k2"),
        "v1_RF_LB_0.98129": csv_to_argmax("submission_sklearn_rf_meta_natural_standalone_v1_lb98129"),
        "rawashishsin_LB_0.98109": csv_to_argmax("submission_rawashishsin_2600_standalone"),
    }
    n_test = len(cands["4b_LB_0.98150"])
    print(f"loaded {len(cands)} LB-validated submissions, n_test={n_test}\n")

    # Load 14-bank stability arrays
    maj = np.load(ART / "stability_test_majority.npy")
    agr = np.load(ART / "stability_test_agreement.npy")
    print(f"14-bank majority dist: {np.bincount(maj, minlength=3).tolist()}")
    print(f"14-bank agreement: p25={np.percentile(agr,25):.3f} p50={np.percentile(agr,50):.3f} p99={np.percentile(agr,99):.3f}\n")

    # Build pseudo-truth set with 4 stringency levels
    levels = {}

    # Level 1: ALL 4 LB-validated submissions agree (most stringent — likely 99%+ precision)
    sub_arr = np.stack(list(cands.values()))  # (4, n_test)
    all_agree = (sub_arr == sub_arr[0]).all(axis=0)
    levels["L1_all_4_LB_subs_agree"] = (all_agree, sub_arr[0].copy())

    # Level 2: All 4 LB-subs agree AND 14-bank majority confirms
    bank_agrees = maj == sub_arr[0]
    l2_mask = all_agree & bank_agrees
    levels["L2_LB4_and_14bank_agree"] = (l2_mask, sub_arr[0].copy())

    # Level 3: + 14-bank agreement >= 0.85
    l3_mask = l2_mask & (agr >= 0.85)
    levels["L3_LB4_14bank_agr85"] = (l3_mask, sub_arr[0].copy())

    # Level 4: + 14-bank agreement >= 0.95 (purest)
    l4_mask = l2_mask & (agr >= 0.95)
    levels["L4_LB4_14bank_agr95"] = (l4_mask, sub_arr[0].copy())

    print("=== Pseudo-truth set sizes ===\n")
    for name, (mask, _) in levels.items():
        n = int(mask.sum())
        pct = 100 * n / n_test
        print(f"  {name}: n={n} ({pct:.1f}% of test)")

    print("\n=== Submission accuracy on each pseudo-truth set ===\n")
    print(f"{'level':<32} {'sub':<32} {'n':>10} {'acc':>10} {'errs':>8}")

    rows = []
    for ln, (mask, pt) in levels.items():
        n = int(mask.sum())
        if n == 0:
            continue
        for sn, sa in cands.items():
            errs = int(((sa != pt) & mask).sum())
            acc = 1 - errs / n
            print(f"{ln:<32} {sn:<32} {n:>10} {acc:>10.6f} {errs:>8}")
            rows.append({"level": ln, "submission": sn, "n_pseudo": n, "accuracy": acc, "errors": errs})
        print()

    # Per-class breakdown on the most stringent level (L4)
    print("=== L4 per-class accuracy (purest pseudo-truth) ===\n")
    mask, pt = levels["L4_LB4_14bank_agr95"]
    print(f"{'sub':<32} {'recL':>8} {'recM':>8} {'recH':>8} {'macro':>8}")
    for sn, sa in cands.items():
        per_class = {}
        for k in [0, 1, 2]:
            tk = mask & (pt == k)
            n_k = int(tk.sum())
            if n_k == 0:
                per_class[k] = 0.0
                continue
            corr_k = int((sa[tk] == k).sum())
            per_class[k] = corr_k / n_k
        macro = (per_class[0] + per_class[1] + per_class[2]) / 3
        print(f"{sn:<32} {per_class[0]:>8.5f} {per_class[1]:>8.5f} {per_class[2]:>8.5f} {macro:>8.5f}")

    # 4b vs B differential analysis on disagreement rows
    print("\n=== 4b vs B on the 145-row override set (where 4b != B) ===\n")
    diff_mask = cands["4b_LB_0.98150"] != cands["B_LB_0.98140"]
    n_diff = int(diff_mask.sum())
    print(f"Total rows where 4b differs from B: {n_diff}")

    # On those rows, what does the 14-bank majority say?
    print(f"\n14-bank majority alignment on diff rows:")
    diff_idx = np.where(diff_mask)[0]
    for sn, sa in [("4b", cands["4b_LB_0.98150"]), ("B", cands["B_LB_0.98140"])]:
        n_bank_agrees = int((maj[diff_mask] == sa[diff_mask]).sum())
        pct = 100 * n_bank_agrees / n_diff
        print(f"  {sn} agrees with 14-bank: {n_bank_agrees}/{n_diff} ({pct:.1f}%)")

    print(f"\nMean 14-bank agreement on diff rows: {agr[diff_mask].mean():.3f}")
    print(f"  (vs full-test mean: {agr.mean():.3f})")

    # Cross-check: does v1 RF natural agree with 4b or B on diff rows?
    v1 = cands["v1_RF_LB_0.98129"]
    raw = cands["rawashishsin_LB_0.98109"]
    n_v1_with_4b = int((v1[diff_mask] == cands["4b_LB_0.98150"][diff_mask]).sum())
    n_v1_with_B = int((v1[diff_mask] == cands["B_LB_0.98140"][diff_mask]).sum())
    n_raw_with_4b = int((raw[diff_mask] == cands["4b_LB_0.98150"][diff_mask]).sum())
    n_raw_with_B = int((raw[diff_mask] == cands["B_LB_0.98140"][diff_mask]).sum())
    print(f"\nIndependent-model agreement on the 145 override rows:")
    print(f"  v1 RF natural with 4b: {n_v1_with_4b}, with B: {n_v1_with_B}")
    print(f"  rawashishsin     with 4b: {n_raw_with_4b}, with B: {n_raw_with_B}")
    print(f"  (rawashishsin is IN B's construction so it WILL agree with B on most diffs)")

    # Estimate true precision of 4b's flips using independent signal v1 + bank
    print("\n=== Estimated precision of 4b's overrides via independent signal ===\n")
    n_v1_and_bank_with_4b = int(((v1[diff_mask] == cands["4b_LB_0.98150"][diff_mask]) & (maj[diff_mask] == cands["4b_LB_0.98150"][diff_mask])).sum())
    n_v1_and_bank_with_B = int(((v1[diff_mask] == cands["B_LB_0.98140"][diff_mask]) & (maj[diff_mask] == cands["B_LB_0.98140"][diff_mask])).sum())
    n_split = n_diff - n_v1_and_bank_with_4b - n_v1_and_bank_with_B
    print(f"  v1 + 14-bank both with 4b: {n_v1_and_bank_with_4b}/{n_diff} = {100*n_v1_and_bank_with_4b/n_diff:.1f}%")
    print(f"  v1 + 14-bank both with B:  {n_v1_and_bank_with_B}/{n_diff} = {100*n_v1_and_bank_with_B/n_diff:.1f}%")
    print(f"  split (v1 and bank disagree): {n_split}/{n_diff} = {100*n_split/n_diff:.1f}%")

    # Save
    out = ART / "test_pseudo_truth_diagnostic.json"
    out.write_text(json.dumps({
        "n_test": int(n_test),
        "level_sizes": {ln: int(m.sum()) for ln, (m, _) in levels.items()},
        "accuracy_table": rows,
        "diff_4b_B": {
            "n_diff": n_diff,
            "v1_with_4b": n_v1_with_4b,
            "v1_with_B": n_v1_with_B,
            "raw_with_4b": n_raw_with_4b,
            "raw_with_B": n_raw_with_B,
            "v1_AND_bank_with_4b": n_v1_and_bank_with_4b,
            "v1_AND_bank_with_B": n_v1_and_bank_with_B,
            "split": n_split,
        },
    }, indent=2, default=str))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
