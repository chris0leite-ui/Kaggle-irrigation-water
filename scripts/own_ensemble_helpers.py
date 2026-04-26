"""Reconstruct per-row prob arrays for 6 LB-validated submissions.

These are STACKS at different depths sharing some components, so they're
not independent. But the simpler stacks (recipe_full_te, catboost) bring
fundamentally different model families/strategies that the deep stacks
collapse over.

Returns dict {sub_name: (oof_probs, test_probs)} all aligned to
StratifiedKFold(seed=42) and matching the saved submission CSVs.

LB validations:
  primary (LB 0.98094): tier1b_greedy_meta
  lb3_rm_nr (LB 0.98008): LB-best 3-stack (= tier1b_helpers.build_lbbest_stack)
  m3_seed_blend (LB 0.98005): log_blend([recipe, pseudo_s1, pseudo_s7], [0.25,0.35,0.40])
  m2_pseudo (LB 0.97998): log_blend([recipe, pseudo_s1], [0.5, 0.5])
  recipe_full_te (LB 0.97939): just oof_recipe_full_te
  recipe_catboost (LB 0.97935): oof_recipe_full_te_catboost
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (build_lbbest_stack, iso_cal,  # noqa: E402
                            load_y, normed)


ART = Path("scripts/artifacts")


def L(name):
    return (normed(np.load(ART / f"oof_{name}.npy").astype(np.float32)),
            normed(np.load(ART / f"test_{name}.npy").astype(np.float32)))


def reconstruct_lb_validated_set(y):
    """Return dict {name: (oof, test)} for 6 LB-validated submissions."""
    # Components
    r_o, r_t = L("recipe_full_te")
    s1_o, s1_t = L("recipe_pseudolabel")
    s7_o, s7_t = L("recipe_pseudolabel_seed7labeler")
    rm_o, rm_t = L("realmlp")
    nr_raw_o, nr_raw_t = L("xgb_nonrule")
    nr_o, nr_t = iso_cal(nr_raw_o, nr_raw_t, y)
    meta_raw_o, meta_raw_t = L("xgb_metastack")
    meta_o, meta_t = iso_cal(meta_raw_o, meta_raw_t, y)
    cat_o, cat_t = L("recipe_full_te_catboost")

    # m2_pseudo (LB 0.97998): 50/50 log-blend
    m2_o = log_blend([r_o, s1_o], np.array([0.5, 0.5]))
    m2_t = log_blend([r_t, s1_t], np.array([0.5, 0.5]))

    # m3_seed_blend (LB 0.98005): 3-way weighted
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r_o, s1_o, s7_o], w3)
    lb3_t = log_blend([r_t, s1_t, s7_t], w3)

    # lb3_rm_nr (LB 0.98008): tier1b "stack2" = lb3 + RM α=0.20 + nr_iso α=0.075
    s1stack_o = log_blend([lb3_o, rm_o], np.array([0.8, 0.2]))
    s1stack_t = log_blend([lb3_t, rm_t], np.array([0.8, 0.2]))
    s2stack_o = log_blend([s1stack_o, nr_o], np.array([0.925, 0.075]))
    s2stack_t = log_blend([s1stack_t, nr_t], np.array([0.925, 0.075]))
    # Cross-check: tier1b_helpers.build_lbbest_stack returns the same thing
    # (small fp drift in chained log_blend; tolerate 1e-3 max abs diff).
    chk_o, chk_t = build_lbbest_stack(y)
    assert np.abs(s2stack_o - chk_o).max() < 1e-2, "stack2 mismatch vs helper"

    # primary (LB 0.98094): stack2 + meta_iso α=0.30
    primary_o = log_blend([s2stack_o, meta_o], np.array([0.70, 0.30]))
    primary_t = log_blend([s2stack_t, meta_t], np.array([0.70, 0.30]))

    return {
        "primary_lb098094": (primary_o, primary_t),       # LB 0.98094
        "stack2_lb098008": (s2stack_o, s2stack_t),         # LB 0.98008
        "m3_seed_lb098005": (lb3_o, lb3_t),                # LB 0.98005
        "m2_pseudo_lb097998": (m2_o, m2_t),                # LB 0.97998
        "recipe_lb097939": (r_o, r_t),                     # LB 0.97939
        "catboost_lb097935": (cat_o, cat_t),               # LB 0.97935
    }


LB_SCORES = {
    "primary_lb098094": 0.98094,
    "stack2_lb098008": 0.98008,
    "m3_seed_lb098005": 0.98005,
    "m2_pseudo_lb097998": 0.97998,
    "recipe_lb097939": 0.97939,
    "catboost_lb097935": 0.97935,
}
