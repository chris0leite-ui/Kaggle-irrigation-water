"""Build all surprise-option candidates and re-evaluate the 2 unprobed
candidates already on disk, against the actual current LB-best 0.98140.

Options:
  1a. v1 anchor + 4 OTHERS k=4 unanimous, H->M ONLY (drop other dirs)
  1b. winner anchor + 4 OTHERS k=4 unanimous, H->M ONLY
  2.  winner anchor + 4 OTHERS k=4 unanimous (all directions)
  3.  winner anchor + {raw, tier1b, lb3, 3way, t4} k=4-of-5 majority
Re-evals:
  - submission_recursive_k4_override.csv vs 0.98140
  - submission_curated_pool_best.csv     vs 0.98140
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surprise.loaders import (  # noqa: E402
    all_helpers, load_test_argmax, load_v1_anchor, load_winner_anchor, load_y,
)
from surprise.eval import emit_csv, evaluate, fmt_summary  # noqa: E402


def k_unanimous_mask(helper_argmaxes: list[np.ndarray], anchor_argmax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All helpers agree on a class != anchor's class. Returns (mask, vote)."""
    P = np.stack(helper_argmaxes, axis=1)
    same = (P == P[:, [0]]).all(1)
    diff = P[:, 0] != anchor_argmax
    return same & diff, P[:, 0]


def k_majority_mask(helper_argmaxes: list[np.ndarray], anchor_argmax: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """At least k of N helpers agree on the same class != anchor's class.
    Returns (mask, vote-class). Tie-break: lowest class index wins (stable).
    """
    P = np.stack(helper_argmaxes, axis=1)
    n = P.shape[0]
    mask = np.zeros(n, dtype=bool)
    vote = anchor_argmax.copy()
    for c in range(3):
        agree = (P == c).sum(1) >= k
        eligible = agree & (anchor_argmax != c) & ~mask  # lowest-c wins on ties
        mask |= eligible
        vote = np.where(eligible, c, vote)
    return mask, vote


def restrict_to_HM(out_argmax: np.ndarray, anchor_argmax: np.ndarray) -> np.ndarray:
    """Keep override only on rows where anchor=High and out=Medium; revert other diffs."""
    out = anchor_argmax.copy()
    hm = (anchor_argmax == 2) & (out_argmax == 1)
    out[hm] = 1
    return out


def main():
    y = load_y()
    v1_oof_a, v1_test_a, v1_oof, v1_bias = load_v1_anchor()
    winner_test_a = load_winner_anchor()
    helpers = all_helpers(y)
    print(f"v1 OOF macro at v1 bias = {(v1_oof_a == y).mean():.5f}  (informative; not balanced)")

    # Reproduce winner OOF analog: apply k=2 unanimous of {raw, tier1b} OOF on top of v1 OOF argmax
    raw_oof_a, tier1b_oof_a = helpers["raw"][2], helpers["tier1b"][2]
    mask, vote = k_unanimous_mask([raw_oof_a, tier1b_oof_a], v1_oof_a)
    winner_oof_a = v1_oof_a.copy()
    winner_oof_a[mask] = vote[mask]
    others_test = [helpers[k][3] for k in ["raw", "tier1b", "lb3", "3way"]]
    others_oof = [helpers[k][2] for k in ["raw", "tier1b", "lb3", "3way"]]

    out_dir = Path("scripts/artifacts")
    diags = {}

    # --- Option 1a: v1 anchor + 4 OTHERS k=4 unanimous, H->M only
    m_test, v_test = k_unanimous_mask(others_test, v1_test_a)
    m_oof, v_oof = k_unanimous_mask(others_oof, v1_oof_a)
    full_test = v1_test_a.copy(); full_test[m_test] = v_test[m_test]
    full_oof = v1_oof_a.copy(); full_oof[m_oof] = v_oof[m_oof]
    out_test = restrict_to_HM(full_test, v1_test_a)
    out_oof = restrict_to_HM(full_oof, v1_oof_a)
    p = emit_csv(out_test, "submission_opt1a_v1_HMonly.csv")
    diags["opt1a_v1_HMonly"] = evaluate(out_test, out_oof, v1_test_a, v1_oof_a, winner_test_a, y)
    diags["opt1a_v1_HMonly"]["csv"] = str(p)

    # --- Option 1b: winner anchor + 4 OTHERS k=4 unanimous, H->M only
    m_test, v_test = k_unanimous_mask(others_test, winner_test_a)
    m_oof, v_oof = k_unanimous_mask(others_oof, winner_oof_a)
    full_test = winner_test_a.copy(); full_test[m_test] = v_test[m_test]
    full_oof = winner_oof_a.copy(); full_oof[m_oof] = v_oof[m_oof]
    out_test = restrict_to_HM(full_test, winner_test_a)
    out_oof = restrict_to_HM(full_oof, winner_oof_a)
    p = emit_csv(out_test, "submission_opt1b_winner_HMonly.csv")
    diags["opt1b_winner_HMonly"] = evaluate(out_test, out_oof, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["opt1b_winner_HMonly"]["csv"] = str(p)

    # --- Option 2: winner anchor + 4 OTHERS k=4 unanimous (all directions)
    m_test, v_test = k_unanimous_mask(others_test, winner_test_a)
    m_oof, v_oof = k_unanimous_mask(others_oof, winner_oof_a)
    out_test = winner_test_a.copy(); out_test[m_test] = v_test[m_test]
    out_oof = winner_oof_a.copy(); out_oof[m_oof] = v_oof[m_oof]
    p = emit_csv(out_test, "submission_opt2_winner_anchored_k4unan.csv")
    diags["opt2_winner_k4unan"] = evaluate(out_test, out_oof, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["opt2_winner_k4unan"]["csv"] = str(p)

    # --- Option 3: winner anchor + {raw, tier1b, lb3, 3way, t4} k=4-of-5 majority
    h5_test = others_test + [helpers["t4"][3]]
    h5_oof = others_oof + [helpers["t4"][2]]
    m_test, v_test = k_majority_mask(h5_test, winner_test_a, k=4)
    m_oof, v_oof = k_majority_mask(h5_oof, winner_oof_a, k=4)
    out_test = winner_test_a.copy(); out_test[m_test] = v_test[m_test]
    out_oof = winner_oof_a.copy(); out_oof[m_oof] = v_oof[m_oof]
    p = emit_csv(out_test, "submission_opt3_winner_5helpers_k4of5.csv")
    diags["opt3_winner_5helpers_k4of5"] = evaluate(out_test, out_oof, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["opt3_winner_5helpers_k4of5"]["csv"] = str(p)

    # --- Re-eval: existing submission_recursive_k4_override.csv vs 0.98140
    rec = load_test_argmax("submission_recursive_k4_override.csv")
    cur = load_test_argmax("submission_curated_pool_best.csv")
    # OOF analog: we don't have these stored, so approximate via test-side delta
    # vs winner. Setting candidate OOF == winner OOF (i.e., assume identity OOF
    # macro is winner's). That underestimates OOF Δ; flag in the report.
    diags["existing_recursive_k4_override"] = evaluate(rec, winner_oof_a, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["existing_recursive_k4_override"]["note"] = (
        "OOF analog for re-evaluation requires the candidate OOF "
        "(not on disk); shown OOF=anchor (lower-bound)."
    )
    diags["existing_curated_pool_best"] = evaluate(cur, winner_oof_a, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["existing_curated_pool_best"]["note"] = (
        "OOF analog requires candidate OOF (not on disk); shown OOF=anchor."
    )

    out_path = out_dir / "surprise_options_results.json"
    out_path.write_text(json.dumps(diags, indent=2, default=str))
    print(f"\nWrote diagnostic JSON: {out_path}\n")
    for label, d in diags.items():
        print(fmt_summary(label, d))
        print()
    return diags


if __name__ == "__main__":
    main()
