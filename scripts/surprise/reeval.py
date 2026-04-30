"""Reconstruct OOF analogs for the existing on-disk candidates and re-eval
against current LB-best 0.98140.

  * curated_pool_best   = k=3-of-4 majority of {raw, tier1b, lb3, 3way} on v1
  * recursive_k4_override = k=4 unanimous of "improved" OTHERS on v1, where
                            each "improved" OTHER is itself overridden by
                            the other 4 LB-validated subs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surprise.loaders import all_helpers, load_test_argmax, load_v1_anchor, load_winner_anchor, load_y  # noqa: E402
from surprise.eval import emit_csv, evaluate, fmt_summary  # noqa: E402
from surprise.options import k_majority_mask, k_unanimous_mask  # noqa: E402


def improved_other(target_argmax: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    """k=4 unanimous override of `target_argmax` using `others` (4 of the 5 LB-validated)."""
    m, v = k_unanimous_mask(others, target_argmax)
    out = target_argmax.copy()
    out[m] = v[m]
    return out


def main():
    y = load_y()
    v1_oof_a, v1_test_a, _, _ = load_v1_anchor()
    winner_test_a = load_winner_anchor()
    helpers = all_helpers(y)

    raw_t, tier1b_t, lb3_t, three_t = (helpers[k][3] for k in ["raw", "tier1b", "lb3", "3way"])
    raw_o, tier1b_o, lb3_o, three_o = (helpers[k][2] for k in ["raw", "tier1b", "lb3", "3way"])
    others_test = [raw_t, tier1b_t, lb3_t, three_t]
    others_oof = [raw_o, tier1b_o, lb3_o, three_o]

    # ---------------- Reconstruct curated_pool_best (k=3 of 4 majority) ----
    m_test, v_test = k_majority_mask(others_test, v1_test_a, k=3)
    m_oof, v_oof = k_majority_mask(others_oof, v1_oof_a, k=3)
    cur_test = v1_test_a.copy(); cur_test[m_test] = v_test[m_test]
    cur_oof = v1_oof_a.copy(); cur_oof[m_oof] = v_oof[m_oof]

    # Sanity check: should match submissions/submission_curated_pool_best.csv
    saved = load_test_argmax("submission_curated_pool_best.csv")
    print(f"reconstructed curated_pool_best matches saved? {np.array_equal(cur_test, saved)} "
          f"(mismatches: {(cur_test != saved).sum()})")

    # ---------------- Reconstruct recursive_k4 ----------------------
    # "improved" OTHER = override the OTHER's predictions using the other 4 subs in the
    # 5-sub pool {v1, raw, tier1b, lb3, 3way}.
    pool_test = {"v1": v1_test_a, "raw": raw_t, "tier1b": tier1b_t, "lb3": lb3_t, "3way": three_t}
    pool_oof = {"v1": v1_oof_a, "raw": raw_o, "tier1b": tier1b_o, "lb3": lb3_o, "3way": three_o}
    improved_test = {}
    improved_oof = {}
    for name in ["raw", "tier1b", "lb3", "3way"]:
        rest_test = [pool_test[n] for n in pool_test if n != name]
        rest_oof = [pool_oof[n] for n in pool_oof if n != name]
        improved_test[name] = improved_other(pool_test[name], rest_test)
        improved_oof[name] = improved_other(pool_oof[name], rest_oof)
    imp_test_list = [improved_test[n] for n in ["raw", "tier1b", "lb3", "3way"]]
    imp_oof_list = [improved_oof[n] for n in ["raw", "tier1b", "lb3", "3way"]]
    m_test, v_test = k_unanimous_mask(imp_test_list, v1_test_a)
    m_oof, v_oof = k_unanimous_mask(imp_oof_list, v1_oof_a)
    rec_test = v1_test_a.copy(); rec_test[m_test] = v_test[m_test]
    rec_oof = v1_oof_a.copy(); rec_oof[m_oof] = v_oof[m_oof]

    saved = load_test_argmax("submission_recursive_k4_override.csv")
    print(f"reconstructed recursive_k4 matches saved? {np.array_equal(rec_test, saved)} "
          f"(mismatches: {(rec_test != saved).sum()})")

    diags = {}
    diags["curated_pool_best (k=3 of 4)"] = evaluate(
        cur_test, cur_oof, v1_test_a, v1_oof_a, winner_test_a, y)
    diags["recursive_k4 (improved-OTHERS k=4)"] = evaluate(
        rec_test, rec_oof, v1_test_a, v1_oof_a, winner_test_a, y)

    Path("scripts/artifacts/surprise_reeval_results.json").write_text(
        json.dumps(diags, indent=2, default=str))
    for k, d in diags.items():
        print(fmt_summary(k, d))
        print()


if __name__ == "__main__":
    main()
