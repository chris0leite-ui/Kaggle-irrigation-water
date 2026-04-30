"""Direction-masked variants of W3 (k=3 unan of {lb3, 3way, t4} on winner).

W3 raw: 4 directions; only H→M (prec 92.2%>91.9%) and M→H (11.7%>8.1%) are
above their respective macro-recall break-even precisions. Mask to keep only
those two, drop M→L (57% < 60.7%) and L→M (29% < 39.3%).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surprise.loaders import all_helpers, load_v1_anchor, load_winner_anchor, load_y  # noqa: E402
from surprise.eval import emit_csv, evaluate, fmt_summary  # noqa: E402
from surprise.options import k_unanimous_mask  # noqa: E402


def restrict_directions(out_argmax: np.ndarray, anchor_argmax: np.ndarray,
                         allowed: set[tuple[int, int]]) -> np.ndarray:
    """Keep override only where (anchor, out) ∈ allowed; revert other diffs."""
    keep = anchor_argmax.copy()
    for a, b in allowed:
        mask = (anchor_argmax == a) & (out_argmax == b)
        keep[mask] = b
    return keep


def main():
    y = load_y()
    v1_oof_a, v1_test_a, _, _ = load_v1_anchor()
    winner_test_a = load_winner_anchor()
    h = all_helpers(y)
    raw_oof_a, tier1b_oof_a = h["raw"][2], h["tier1b"][2]
    m, vote = k_unanimous_mask([raw_oof_a, tier1b_oof_a], v1_oof_a)
    winner_oof_a = v1_oof_a.copy(); winner_oof_a[m] = vote[m]

    # W3 helpers
    h_oof = [h["lb3"][2], h["3way"][2], h["t4"][2]]
    h_test = [h["lb3"][3], h["3way"][3], h["t4"][3]]
    m_t, v_t = k_unanimous_mask(h_test, winner_test_a)
    m_o, v_o = k_unanimous_mask(h_oof, winner_oof_a)
    full_t = winner_test_a.copy(); full_t[m_t] = v_t[m_t]
    full_o = winner_oof_a.copy(); full_o[m_o] = v_o[m_o]

    diags = {}

    # H→M only (anchor=High=2, out=Medium=1)
    out_t = restrict_directions(full_t, winner_test_a, {(2, 1)})
    out_o = restrict_directions(full_o, winner_oof_a, {(2, 1)})
    p = emit_csv(out_t, "submission_W3_HMonly.csv")
    diags["W3_HMonly"] = evaluate(out_t, out_o, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["W3_HMonly"]["csv"] = str(p)

    # M→H only (anchor=Medium=1, out=High=2)
    out_t = restrict_directions(full_t, winner_test_a, {(1, 2)})
    out_o = restrict_directions(full_o, winner_oof_a, {(1, 2)})
    p = emit_csv(out_t, "submission_W3_MHonly.csv")
    diags["W3_MHonly"] = evaluate(out_t, out_o, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["W3_MHonly"]["csv"] = str(p)

    # H→M AND M→H (drop M→L and L→M which are below break-even)
    out_t = restrict_directions(full_t, winner_test_a, {(2, 1), (1, 2)})
    out_o = restrict_directions(full_o, winner_oof_a, {(2, 1), (1, 2)})
    p = emit_csv(out_t, "submission_W3_HM_and_MH_only.csv")
    diags["W3_HM_and_MH_only"] = evaluate(out_t, out_o, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["W3_HM_and_MH_only"]["csv"] = str(p)

    Path("scripts/artifacts/surprise_w3_masked_results.json").write_text(
        json.dumps(diags, indent=2, default=str))

    for k, d in diags.items():
        print(fmt_summary(k, d))
        print()


if __name__ == "__main__":
    main()
