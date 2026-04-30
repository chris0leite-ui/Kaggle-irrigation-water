"""Final summary table — uses the actual on-disk CSVs for existing candidates
(test-side only) and the reproducible OOF analogs for the new options.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surprise.loaders import load_test_argmax, load_v1_anchor, load_winner_anchor  # noqa: E402
from surprise.eval import direction_breakdown  # noqa: E402


CANDIDATES = [
    # (label, csv_filename, has_oof_eval, mechanism_summary)
    ("opt1a_v1_HMonly", "submission_opt1a_v1_HMonly.csv", True,
     "v1 + 4 OTHERS k=4 unan, H->M ONLY (65 overrides)"),
    ("opt3_winner_5h_k4of5", "submission_opt3_winner_5helpers_k4of5.csv", True,
     "winner + {raw,tier1b,lb3,3way,t4} k=4 of 5 majority"),
    ("recursive_k4_override (existing)", "submission_recursive_k4_override.csv", False,
     "improved-OTHERS k=4 unan; differs from 0.98140 by 68 rows"),
    ("curated_pool_best (existing)", "submission_curated_pool_best.csv", False,
     "k=3 of 4 majority; +0.00021 OOF per commit; 378 rows diff vs 0.98140"),
    ("consensus_override_all_helpers (existing)", "submission_consensus_override_all_helpers.csv", False,
     "all 3 helpers (raw,tier1b,cb) consensus override of v1"),
    ("hardvote_top5 (existing)", "submission_hardvote_top5_lb_validated.csv", False,
     "hard-vote across all 5 LB-validated subs"),
]


def main():
    _, v1_test_a, _, _ = load_v1_anchor()
    winner = load_winner_anchor()
    print(f"v1 (LB 0.98129) class dist: {dict(zip(['L','M','H'], np.bincount(v1_test_a, minlength=3)))}")
    print(f"winner 0.98140 class dist: {dict(zip(['L','M','H'], np.bincount(winner, minlength=3)))}")
    print(f"winner differs from v1 on {(winner != v1_test_a).sum()} rows")
    print()

    print(f"{'candidate':46s}  {'rows_v_v1':>10s}  {'rows_v_98140':>13s}  netL/M/H direction")
    print("-" * 110)
    for label, fname, _, _ in CANDIDATES:
        try:
            cand = load_test_argmax(fname)
        except Exception as e:
            print(f"{label}: ERROR {e}")
            continue
        rv1 = (cand != v1_test_a).sum()
        r98 = (cand != winner).sum()
        b = direction_breakdown(v1_test_a, cand)
        n = b["net_per_class"]
        print(f"{label:46s}  {rv1:>10d}  {r98:>13d}  L{n['Low']:+4d} / M{n['Medium']:+4d} / H{n['High']:+4d}")

    print()
    print("Direction breakdown vs v1 anchor (overrides only):")
    print()
    for label, fname, _, _ in CANDIDATES:
        cand = load_test_argmax(fname)
        b = direction_breakdown(v1_test_a, cand)
        d = b["directions"]
        print(f"  {label}:")
        for k, n in sorted(d.items(), key=lambda kv: -kv[1]):
            print(f"    {k:14s}  {n:5d}")
    print()
    print("=== Risk synopsis (heuristic; not LB) ===")
    print("  net_H > 0 + Mass≤200 rows = ADD-High small footprint = lowest regression risk")
    print("  net_H < 0 + Mass≥300 rows = REMOVE-High large footprint = highest regression risk")
    print()
    print("=== Existing LB-probed reference points ===")
    print("  v1 RF natural standalone:                 LB 0.98129")
    print("  k=4 unan v1+4OTHERS (lbbest_overridden):  LB 0.98134  (+0.00005)")
    print("  k=2 unan {raw,tier1b} 0.98140 winner:    LB 0.98140  (+0.00011)")
    print("  rawashishsin_k4_overridden:              LB 0.98112  (-0.00017 regression)")


if __name__ == "__main__":
    main()
