"""Tier 1.1: B (LB 0.98140) as anchor, override with k=N consensus of OTHERS.

B was built by overriding v1_rf_natural (LB 0.98129) with k=2 unanimous of
{raw, tier1b}. Tier 1.1 reverses the question: where do OTHERS unanimously
disagree with B? Override B's argmax on those rows.

OTHERS pool (LB-validated only):
  v1_rf_natural   LB 0.98129  (the prior anchor, not used as OTHER in B's build)
  k4_override     LB 0.98134  (4-OTHER k=3 majority, was NOT a B-OTHER)
  rawashishsin    LB 0.98109  (was a B-OTHER)
  tier1b_4stack   LB 0.98094  (was a B-OTHER)
  3way_multiseed  LB 0.98005  (was NOT a B-OTHER, much weaker)

Configurations tested (each emits a candidate CSV):
  TC1: 2-OTHER {v1, k4} k=2 unanimous (smallest, most LB-strong)
  TC2: 3-OTHER {v1, k4, 3way} k=3 unanimous (adds 3way for diversity)
  TC3: 5-OTHER {v1, k4, raw, tier1b, 3way} k=5 unanimous (strictest)
  TC4: 5-OTHER {v1, k4, raw, tier1b, 3way} k=4 majority (looser)
  TC5: 4-OTHER {v1, k4, raw, tier1b} k=4 unanimous (no 3way)
  TC6: 4-OTHER {v1, k4, raw, tier1b} k=3 majority (no 3way, looser)

Each variant reports:
  - n overrides
  - direction breakdown vs B
  - net rare-class delta
  - decision: emit submission for LB-probe consideration

NOTE: Per CLAUDE.md, never auto-submit. Candidates are emitted for
user approval only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SUB = Path("submissions")
ART = Path("scripts/artifacts")

# Anchor: B (LB 0.98140)
ANCHOR_NAME = "submission_2other_raw_tier1b_k2"
ANCHOR_LB = 0.98140

# OTHERS pool (LB-validated only, with their LB)
OTHERS = {
    "v1_rf":       ("submission_sklearn_rf_meta_natural_standalone_v1_lb98129", 0.98129),
    "k4_override": ("submission_lbbest_overridden_by_unanimous_others",          0.98134),
    "raw":         ("submission_rawashishsin_2600_standalone",                   0.98109),
    "tier1b":      ("submission_tier1b_greedy_meta",                             0.98094),
    "3way":        ("submission_3way_recipe025_s1035_s7040",                     0.98005),
}

CONFIGS = {
    "tc1_v1_k4_k2":       (["v1_rf", "k4_override"],                          2),
    "tc2_v1_k4_3w_k3":    (["v1_rf", "k4_override", "3way"],                  3),
    "tc3_5o_k5":          (["v1_rf", "k4_override", "raw", "tier1b", "3way"], 5),
    "tc4_5o_k4":          (["v1_rf", "k4_override", "raw", "tier1b", "3way"], 4),
    "tc5_4o_k4":          (["v1_rf", "k4_override", "raw", "tier1b"],         4),
    "tc6_4o_k3":          (["v1_rf", "k4_override", "raw", "tier1b"],         3),
}


def load(name: str) -> pd.Series:
    return pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]


def cls2int(s: pd.Series) -> np.ndarray:
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def int2cls(arr: np.ndarray) -> np.ndarray:
    return np.array(["Low", "Medium", "High"])[arr]


def main():
    print(f"=== Tier 1.1: anchor = {ANCHOR_NAME} (LB {ANCHOR_LB:.5f}) ===\n")

    ids = pd.read_csv(SUB / f"{ANCHOR_NAME}.csv")["id"].to_numpy()
    anchor = cls2int(load(ANCHOR_NAME))
    print(f"Anchor class counts: L={int((anchor==0).sum())} M={int((anchor==1).sum())} H={int((anchor==2).sum())}")
    print()

    others_arr = {}
    for k, (name, lb) in OTHERS.items():
        others_arr[k] = cls2int(load(name))
        print(f"  {k:<14} (LB {lb:.5f})")
    print()

    summary = {}
    for cfg_name, (pool, k_threshold) in CONFIGS.items():
        N = len(pool)
        # Stack OTHERS into matrix (n_rows, N_others)
        oth_mat = np.stack([others_arr[p] for p in pool], axis=1)
        # For each row + each class c, count how many OTHERS predict c
        votes = np.zeros((len(anchor), 3), dtype=np.int8)
        for c in range(3):
            votes[:, c] = (oth_mat == c).sum(axis=1)

        # Find rows where any class has ≥ k_threshold votes AND that class != anchor
        max_vote = votes.max(axis=1)
        consensus_class = votes.argmax(axis=1)
        # Override mask: max consensus reaches threshold AND differs from anchor
        override_mask = (max_vote >= k_threshold) & (consensus_class != anchor)

        # Apply override
        new_pred = anchor.copy()
        new_pred[override_mask] = consensus_class[override_mask]

        n_override = int(override_mask.sum())
        diff_rows = int((new_pred != anchor).sum())  # = n_override

        # Direction breakdown
        directions = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                n = int(((anchor == fr) & (new_pred == to)).sum())
                if n > 0:
                    directions[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n

        # Net rare-class change (High)
        h_added   = int(((anchor != 2) & (new_pred == 2)).sum())
        h_removed = int(((anchor == 2) & (new_pred != 2)).sum())
        net_h     = h_added - h_removed

        new_counts = {
            "L": int((new_pred == 0).sum()),
            "M": int((new_pred == 1).sum()),
            "H": int((new_pred == 2).sum()),
        }

        print(f"--- {cfg_name}  (pool={pool}, k={k_threshold}) ---")
        print(f"  overrides: {n_override}")
        print(f"  new counts: L={new_counts['L']} M={new_counts['M']} H={new_counts['H']}")
        print(f"  direction breakdown: {directions}")
        print(f"  net_H = +{h_added} -{h_removed} = {net_h:+d}")
        print()

        summary[cfg_name] = {
            "pool": pool,
            "k_threshold": k_threshold,
            "n_override": n_override,
            "directions": directions,
            "net_h": net_h,
            "h_added": h_added,
            "h_removed": h_removed,
            "new_counts": new_counts,
        }

        # Emit submission for review
        out_csv = SUB / f"submission_tier_1_1_{cfg_name}.csv"
        sub_df = pd.DataFrame({"id": ids, "Irrigation_Need": int2cls(new_pred)})
        sub_df.to_csv(out_csv, index=False)
        print(f"  emitted: {out_csv.name}")
        print()

    out_json = ART / "tier_1_1_override_b_results.json"
    out_json.write_text(json.dumps({
        "anchor": ANCHOR_NAME,
        "anchor_lb": ANCHOR_LB,
        "others": {k: {"file": v[0], "lb": v[1]} for k, v in OTHERS.items()},
        "configs": summary,
    }, indent=2))
    print(f"=== summary written to {out_json} ===")


if __name__ == "__main__":
    main()
