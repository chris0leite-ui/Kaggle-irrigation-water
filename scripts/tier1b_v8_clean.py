#!/usr/bin/env python3
"""xgb_metastack v8: CLEAN-POOL retrain with rawashishsin in bank.

Difference vs v6 retrain (which had circular meta-of-metas leakage):
  - 58 prior meta-stacker / blend / derived components REMOVED from pool
  - Pool reduced from 189 to ~131 base-only components
  - Both rawashishsin (v2) + rawashishsin_2600 (v3) INCLUDED
  - Same XGB heavy-reg HPs (depth=4, alpha=5, lambda=5, lr=0.05, max=3000)
  - Same 5-fold StratifiedKFold(seed=42), aligned with our OOF bank

Output: oof_xgb_metastack_v8.npy + test_xgb_metastack_v8.npy
"""
import os
# Strict EXCLUDE: all meta-stacker outputs + blend outputs + derived stacks +
# distillation children + greedy chains + per-cell metas + RF meta + L3
# Items already excluded by tier1b_xgb_metastack.EXCLUDE remain excluded.
META_EXCLUDE_EXTRA = {
    # CRITICAL: xgb_metastack (v1) is the meta-stacker in our LB-best 4-stack.
    # Including it as an INPUT feature to a new meta-stacker is circular.
    "xgb_metastack",
    # Prior xgb_metastack variants (CIRCULAR if used as input)
    "xgb_metastack_v2", "xgb_metastack_v3", "xgb_metastack_v3_iso",
    "xgb_metastack_v4", "xgb_metastack_v5", "xgb_metastack_v5_iso",
    "xgb_metastack_v6", "xgb_metastack_v6_combined", "xgb_metastack_v6lb",
    "xgb_metastack_v7", "xgb_metastack_v7b",
    "xgb_metastack_varB", "xgb_metastack_varC",
    "xgb_metastack_3wnn", "xgb_metastack_b2clean",
    "xgb_metastack_bag3", "xgb_metastack_classw",
    "xgb_metastack_heavy", "xgb_metastack_j2bag",
    "xgb_metastack_n5b_both", "xgb_metastack_narrow",
    "xgb_metastack_perfoldiso_inputs",
    "xgb_metastack_v1_cleanpool", "xgb_metastack_v1_groupkfold",
    "xgb_metastack_v1_plus_newfe",
    # Macrorec meta variants (also derived)
    "xgb_metastack_metamacrorec_baseonly",
    "xgb_metastack_metamacrorec_baseonly_iso",
    "xgb_metastack_metamacrorec_lam0",
    "xgb_metastack_metamacrorec_lam03",
    "xgb_metastack_metamacrorec_lam03_curated",
    "xgb_metastack_metamacrorec_lam03_curated_iso",
    "xgb_metastack_metamacrorec_lam03_iso",
    "xgb_metastack_metamacrorec_lam0_iso",
    "xgb_metastack_metamacrorec_minimal",
    "xgb_metastack_metamacrorec_minimal_iso",
    # Other meta architectures (LR / RF / MLP / 3-meta L3)
    "lr_metastack", "lr_metastack_v2",
    "mlp_metastack", "sklearn_rf_meta",
    "three_meta_l3", "meta_l3_xgb_mlp",
    "per_cell_meta", "leaf_ote_meta", "leaf_ote_meta_v2",
    # Greedy/blend/hybrid outputs (derived from existing components)
    "bagged_greedy_nonrule", "c0_greedy",
    "greedy_blend", "greedy_full_bank_6way",
    "hybrid_lgbmxgb_blend",
    "own_S5_greedy_forward", "own_greedy_fine",
    # Distillation children (use teacher OOF/test as labels)
    "distill_no_rule",
    "soft_distill_recipeonly", "soft_distill_small", "soft_distill_tiny",
    # OvR XGBs (binary-derived, mostly redundant with multiclass anchor)
    "xgb_ovr_recipe", "xgb_ovr_recipe_raw",
}

# Patch EXCLUDE before import to take effect
import sys
sys.path.insert(0, "scripts")
import tier1b_xgb_metastack as t
t.EXCLUDE = t.EXCLUDE | META_EXCLUDE_EXTRA

# Set output suffix to _v8
os.environ["META_OUT_SUFFIX"] = "_v8"

# Verify pool
import pandas as pd
data = pd.read_csv("data/train.csv")
y = data["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values
pool = t.load_pool(y)
print(f"[clean-pool] {len(pool)} base components (excluded {len(t.EXCLUDE)} meta/derived)")
assert "rawashishsin" in pool, "rawashishsin (v2) missing"
assert "rawashishsin_2600" in pool, "rawashishsin_2600 (v3) missing"
print(f"[clean-pool] rawashishsin present: True")
print(f"[clean-pool] rawashishsin_2600 present: True")

# Run main
t.main()
