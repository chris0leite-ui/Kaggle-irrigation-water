"""Idea 4b QUAD-CONSENSUS — strict version of 4b dropping risky flips.

Diagnostic finding (test_pseudo_truth_diagnostic.json):
  - 4b's 108 overrides have 14-bank majority agreement on 108/108 (by construction)
  - v1 RF natural (LB 0.98129) — NOT used in 4b's filter — agrees with 4b's
    flip on 98/108 = 90.7%, with B on 0/108
  - 10 "split" rows: v1 disagrees with both 4b and B's class

Mechanism:
  Drop the 10 flips where v1 RF natural disagrees with 4b's flip class.
  Resulting candidate has:
    - 98 high-confidence flips (4 independent consensus axes agree)
    - All 4b's H→M direction structure preserved
    - Higher expected precision: 95-98% vs 4b's 90.7%

Math (break-even = 91.94% on H→M):
  - 4b at 90.7% measured precision → just below break-even on independent signal
    But 4b LB +0.00010 measured → so true precision is somewhere ~91-93%
  - 4b' at 95-98% precision (with 10 risky drops) → projected LB +0.00018 to +0.00025

The 4 axes (all unanimously agree on the flip class):
  1. bagged_v1' (4 RF natural fold-seed variants) → flip class
  2. {raw, tier1b} unanimous → flip class
  3. 14-bank majority → flip class
  4. v1 RF natural standalone (single LB-validated model) → flip class  [NEW]

This is mechanism-novel: 4b used 3 axes; v1 standalone is now the 4th.
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
    print("=== Idea 4b QUAD-CONSENSUS: drop v1-disagreeing flips from 4b ===\n")

    # Load all relevant submissions and the 14-bank
    b = csv_to_argmax("submission_2other_raw_tier1b_k2")            # anchor (LB 0.98140)
    fb = csv_to_argmax("submission_idea4b_selective_override")      # 4b (LB 0.98150)
    v1 = csv_to_argmax("submission_sklearn_rf_meta_natural_standalone_v1_lb98129")
    raw = csv_to_argmax("submission_rawashishsin_2600_standalone")
    maj = np.load(ART / "stability_test_majority.npy")
    agr = np.load(ART / "stability_test_agreement.npy")

    # 4b's flip set
    flip_mask_4b = b != fb
    flip_idx = np.where(flip_mask_4b)[0]
    print(f"4b flip set: {flip_mask_4b.sum()} rows")

    # Direction breakdown
    def directions(a, c, mask):
        d = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                n = int(((a == fr) & (c == to) & mask).sum())
                if n > 0:
                    d[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n
        return d
    print(f"  4b directions: {directions(b, fb, flip_mask_4b)}")

    # The QUAD filter: keep flip only if v1 ALSO agrees with the flip class
    v1_agrees_with_flip = v1[flip_mask_4b] == fb[flip_mask_4b]
    print(f"\nv1 RF natural agreement with 4b's flip class:")
    print(f"  agrees: {int(v1_agrees_with_flip.sum())}/{len(v1_agrees_with_flip)} ({100*v1_agrees_with_flip.mean():.1f}%)")

    # Build new candidate: original 4b minus risky flips
    new_pred = fb.copy()
    risky_indices = flip_idx[~v1_agrees_with_flip]
    n_risky = len(risky_indices)
    print(f"\nRisky flips to revert (v1 disagrees with 4b's class): {n_risky}")

    # Inspect the risky rows
    print(f"\nRisky-row breakdown (v1's class vs 4b's class):")
    for i in risky_indices:
        b_cls = "LMH"[b[i]]
        fb_cls = "LMH"[fb[i]]
        v1_cls = "LMH"[v1[i]]
        maj_cls = "LMH"[maj[i]]
        raw_cls = "LMH"[raw[i]]
        a = agr[i]
        print(f"  idx={i:6d}  B:{b_cls}  4b:{fb_cls}  v1:{v1_cls}  bank:{maj_cls} (agr={a:.2f})  raw:{raw_cls}")

    # Revert risky flips back to B
    new_pred[risky_indices] = b[risky_indices]

    # Verify: new candidate has fb-flips minus risky, all should match
    final_flip_mask = b != new_pred
    n_final_flips = int(final_flip_mask.sum())
    print(f"\nQuad-consensus candidate:")
    print(f"  total flips vs B: {n_final_flips}")
    print(f"  directions: {directions(b, new_pred, final_flip_mask)}")
    h_added = int(((b != 2) & (new_pred == 2)).sum())
    h_removed = int(((b == 2) & (new_pred != 2)).sum())
    net_h = h_added - h_removed
    print(f"  net_H = +{h_added} -{h_removed} = {net_h:+d}")

    # Compare to 4b (delta = quad keeps 98 of 108 4b flips)
    delta_vs_4b = int((new_pred != fb).sum())
    print(f"\nDiffs vs 4b: {delta_vs_4b}")

    # Precision projection
    # Assume v1 + bank + bagged + {raw,tier1b} unanimous = ~98% precision on H->M
    # B observed at 88 flips (LB 0.98140) and 4b adds 105 H->M (108 total flips)
    # 4b LB +0.00010 -> ~91% measured precision on the 105 H->M
    # quad keeps the 95+ flips with 4-way agreement -> ~95-98% projected
    print("\n=== Expected LB projection ===")
    n_hm = sum(1 for c, n in directions(b, new_pred, final_flip_mask).items() if "H->" in c and "->M" in c)
    n_hm_actual = directions(b, new_pred, final_flip_mask).get("H->M", 0)
    print(f"H->M count in quad: {n_hm_actual}")
    n_m_test = 100261; n_h_test = 10279
    base_lb = 0.98140  # B
    for prec in [0.98, 0.95, 0.92, 0.88]:
        corr = prec * n_hm_actual
        wrong = (1 - prec) * n_hm_actual
        macro = (corr / n_m_test - wrong / n_h_test) / 3
        print(f"  precision {int(prec*100):d}%: macro_delta = {macro:+.6f} -> proj LB = {base_lb + macro:.5f}")

    # Emit (use ids from existing submission CSV)
    test_ids = pd.read_csv(SUB / "submission_idea4b_selective_override.csv", usecols=["id"])["id"].to_numpy()
    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
    })
    out_csv = SUB / "submission_idea4b_quad_consensus.csv"
    sub.to_csv(out_csv, index=False)
    print(f"\nemitted: {out_csv}")
    print(f"  ({delta_vs_4b} rows differ from 4b LB 0.98150)")

    out_json = ART / "idea4b_quad_consensus_results.json"
    out_json.write_text(json.dumps({
        "n_flips_4b": int(flip_mask_4b.sum()),
        "n_flips_quad": n_final_flips,
        "n_dropped": n_risky,
        "diffs_vs_4b": delta_vs_4b,
        "directions_quad": directions(b, new_pred, final_flip_mask),
        "net_h": net_h,
        "v1_agrees_pct": float(v1_agrees_with_flip.mean()),
        "candidate_csv": str(out_csv),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
