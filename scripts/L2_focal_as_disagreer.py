"""L2 — focal as DISAGREER in 4b's triple-consensus mechanism.

L1 tested focal as a consensus AXIS (in the bank-majority filter) and
found 99.84% agreement → no orthogonal signal.

L2 tests the dual: replace bagged_v1' with focal's argmax as the
candidate-flip source. Mechanism:
  flip B where:
    (a) focal_majority != B          (disagreer condition)
    (b) raw == focal_majority        (raw confirms flip)
    (c) tier1b == focal_majority     (tier1b confirms flip)
    (d) bank_majority == focal_majority  (14-bank confirms flip)

If focal_majority points to flips bagged_v1' missed AND those flips
pass the same triple-consensus, it's a candidate to ADD on top of 4b.
If focal_majority points to a SUBSET of bagged_v1's 151 disagreement
candidates, it just narrows 4b's flip set.

For each focal variant individually (g2h3, g2_aH1, g2_invfreq,
effnum) and for the focal-only-majority (4-vote) case.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")
CLS = {"Low": 0, "Medium": 1, "High": 2}

FOCAL_NAMES = ["recipe_focal_g2h3", "recipe_focal_g2_aH1",
               "recipe_focal_g2_invfreq", "recipe_focal_effnum"]


def _norm(p, eps=1e-9):
    return p / np.clip(p.sum(axis=1, keepdims=True), eps, None)


def _argmax_test(name):
    return _norm(np.load(ART / f"test_{name}.npy").astype(np.float32)).argmax(1)


def _csv_argmax(name):
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map(CLS).to_numpy(dtype=np.int8)


def _majority(votes):
    n = votes.shape[1]
    out = np.empty(n, dtype=np.int8)
    for r in range(n):
        out[r] = np.bincount(votes[:, r], minlength=3).argmax()
    return out


def evaluate(disagreer, name, b, raw, t1b, bank_maj, b_4b):
    """For a given disagreer (n_test argmax), compute the 4b-style mask
    and report the directional breakdown vs B and vs 4b's primary."""
    cand = (b != disagreer) & (raw == disagreer) & (t1b == disagreer) \
        & (bank_maj == disagreer)
    new_pred = b.copy()
    new_pred[cand] = disagreer[cand]

    # Compare to current LB-best primary (4b: 0.98150)
    diff_vs_4b = (new_pred != b_4b).sum()
    flips_in_4b_not_here = ((b_4b != b) & ~cand).sum()
    flips_here_not_in_4b = (cand & (b_4b == b)).sum()
    flips_overlap = (cand & (b_4b != b)).sum()

    directions = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((b == fr) & (new_pred == to)).sum())
            if n > 0:
                directions[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n

    return {
        "name": name,
        "n_flips_vs_B": int(cand.sum()),
        "directions": directions,
        "diff_vs_4b": int(diff_vs_4b),
        "flips_in_4b_not_here": int(flips_in_4b_not_here),
        "flips_here_not_in_4b": int(flips_here_not_in_4b),
        "flips_overlap_with_4b": int(flips_overlap),
    }


def main():
    print("=== L2: focal as disagreer in 4b's mechanism ===\n")
    b      = _csv_argmax("submission_2other_raw_tier1b_k2")     # B (LB 0.98140)
    b_4b   = _csv_argmax("submission_idea4b_selective_override") # 4b (LB 0.98150)

    # Reconstruct {raw, tier1b} argmaxes used in 4b's filter
    raw  = _argmax_test("recipe_full_te")
    t1b  = _argmax_test("tier1b_greedy_meta")
    bank_maj = np.load(ART / "stability_test_majority.npy")

    # 4b's actual flip mask (108 rows)
    bagvp = _csv_argmax("submission_idea4_foldbag_v1_b_mech")
    flip_4b = (b != bagvp) & (bank_maj == bagvp)
    print(f"4b flip count: {int(flip_4b.sum())} (sanity: 108)")
    print(f"4b directions: H->M={int(((b==2)&(b_4b==1)).sum())}  "
          f"M->L={int(((b==1)&(b_4b==0)).sum())}  "
          f"L->M={int(((b==0)&(b_4b==1)).sum())}\n")

    # Per-focal-variant
    focals_argmax = {n: _argmax_test(n) for n in FOCAL_NAMES}
    focal_stack = np.stack([focals_argmax[n] for n in FOCAL_NAMES], axis=0)
    focal_maj = _majority(focal_stack)

    rows = []
    for n in FOCAL_NAMES:
        r = evaluate(focals_argmax[n], n, b, raw, t1b, bank_maj, b_4b)
        rows.append(r)
        print(f"{n:30s}  flips_vs_B={r['n_flips_vs_B']:4d}  "
              f"dirs={r['directions']}  "
              f"overlap_w_4b={r['flips_overlap_with_4b']:3d}  "
              f"new_vs_4b={r['flips_here_not_in_4b']:3d}")

    r = evaluate(focal_maj, "focal_majority(4)", b, raw, t1b, bank_maj, b_4b)
    rows.append(r)
    print(f"\n{r['name']:30s}  flips_vs_B={r['n_flips_vs_B']:4d}  "
          f"dirs={r['directions']}  "
          f"overlap_w_4b={r['flips_overlap_with_4b']:3d}  "
          f"new_vs_4b={r['flips_here_not_in_4b']:3d}")

    # Compose: union of 4b's 108 flips + focal-majority new flips
    cand_focal = (b != focal_maj) & (raw == focal_maj) & (t1b == focal_maj) \
        & (bank_maj == focal_maj)
    flip_4b_mask = b != b_4b
    union_flip = flip_4b_mask | cand_focal
    new_pred_union = b.copy()
    new_pred_union[flip_4b_mask] = b_4b[flip_4b_mask]
    new_pred_union[cand_focal & ~flip_4b_mask] = focal_maj[cand_focal & ~flip_4b_mask]
    print(f"\nUNION (4b ∪ focal-maj-disagreer): {int(union_flip.sum())} total flips "
          f"({int(flip_4b_mask.sum())} from 4b + "
          f"{int((cand_focal & ~flip_4b_mask).sum())} new from focal-maj)")

    out = {"per_focal_variant": rows,
           "union_total_flips": int(union_flip.sum()),
           "union_4b_flips": int(flip_4b_mask.sum()),
           "union_focal_only_new_flips": int((cand_focal & ~flip_4b_mask).sum())}
    (ART / "L2_focal_as_disagreer.json").write_text(json.dumps(out, indent=2))
    print(f"\n→ wrote scripts/artifacts/L2_focal_as_disagreer.json")


if __name__ == "__main__":
    main()
