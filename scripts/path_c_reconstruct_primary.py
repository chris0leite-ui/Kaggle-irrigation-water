"""Reconstruct the LB-best 4-stack (LB 0.98094) test posterior for use as
a pseudo-label labeler in path C iterative refinement.

Composition (from CLAUDE.md hypothesis board):
  lb3      = log_blend(recipe_full_te, recipe_pseudolabel, recipe_pseudolabel_seed7labeler;
                       0.25/0.35/0.40)
  stack1   = log_blend(lb3, realmlp;                       0.80/0.20)
  stack2   = log_blend(stack1, xgb_nonrule_iso;            0.925/0.075)
  final    = log_blend(stack2, xgb_metastack_iso;          0.70/0.30)
  bias     = [1.4324, 1.4689, 3.4008]

Outputs:
  scripts/artifacts/test_path_c_primary_labeler.npy
  scripts/artifacts/oof_path_c_primary_labeler.npy   (sanity)
  scripts/artifacts/path_c_primary_labeler_results.json   (log_bias)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, bal_at_bias, build_lbbest_stack, iso_cal, load_y, normed,
)


def main():
    y = load_y()
    print(f"loaded y: shape={y.shape}, prior={np.bincount(y) / len(y)}")

    # Step 1 — LB-best 3-stack (already includes lb3 + realmlp + nonrule_iso).
    s2_oof, s2_test = build_lbbest_stack(y)
    print(f"3-stack OOF tuned bal_acc @recipe-bias = {bal_at_bias(s2_oof, y):.5f}")

    # Step 2 — load + iso-cal xgb_metastack.
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    print(f"meta_iso standalone @recipe-bias = {bal_at_bias(meta_o_iso, y):.5f}")

    # Step 3 — 4-stack: blend in xgb_metastack_iso at α=0.30.
    w4 = np.array([0.70, 0.30])
    final_o = log_blend([s2_oof, meta_o_iso], w4)
    final_t = log_blend([s2_test, meta_t_iso], w4)
    final_o_n = normed(final_o)
    final_t_n = normed(final_t)

    bal = bal_at_bias(final_o_n, y)
    print(f"\nLB-best 4-stack OOF tuned bal_acc @recipe-bias = {bal:.5f}")
    print(f"(expected ~0.98084 from hypothesis board)")

    # Save artefacts in the format recipe_pseudolabel.py consumes.
    oof_path = ART / "oof_path_c_primary_labeler.npy"
    test_path = ART / "test_path_c_primary_labeler.npy"
    json_path = ART / "path_c_primary_labeler_results.json"

    np.save(oof_path, final_o_n.astype(np.float32))
    np.save(test_path, final_t_n.astype(np.float32))

    res = {
        "log_bias": BIAS.tolist(),
        "tuned_log_bias_bal_acc": float(bal),
        "weights": {"3stack": [0.70], "metastack_iso": [0.30]},
        "composition": "lb3 + realmlp + nonrule_iso + metastack_iso",
        "anchor": "LB 0.98094",
    }
    json_path.write_text(json.dumps(res, indent=2))
    print(f"\nsaved → {oof_path}")
    print(f"saved → {test_path}")
    print(f"saved → {json_path}")


if __name__ == "__main__":
    main()
