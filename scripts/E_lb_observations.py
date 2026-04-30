"""E — Historical LB observations as a structured dataset for GP-on-LB optimization.

Compiles ALL LB-probed submissions documented in CLAUDE.md into a single
CSV with (label, composition_json, OOF_tuned, LB_public, gap) rows.

The CSV is the input substrate for a GP fit over (composition_vector → LB).
Once enough observations exist, GP can suggest unprobed operating points
in blend-weight space with high expected improvement.

Output: scripts/artifacts/E_lb_observations.csv

This script does NOT yet fit the GP — that's a follow-up step. First the
data assembly + structure check.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
OUT_CSV = ART / "E_lb_observations.csv"


# Each LB observation:
#   label, OOF_tuned, LB_public, components (dict: name -> weight ratio in the
#   final stack), notes
#
# Components are documented as "fraction of effective log-prob mass" in the
# final blended posterior. Where the architecture is multi-stage (3-stack →
# 4-stack → meta-blend), I derive the EFFECTIVE per-component weight in the
# final mixture by chaining the stage weights:
#   stack1 = 0.80*lb3 + 0.20*realmlp
#   stack2 = 0.925*stack1 + 0.075*nr_iso = 0.74*lb3 + 0.185*realmlp + 0.075*nr_iso
#   primary = 0.70*stack2 + 0.30*meta_iso
#       = 0.518*lb3 + 0.130*realmlp + 0.0525*nr_iso + 0.30*meta_iso
#   lb3 = 0.25*recipe + 0.35*pseudo_s1 + 0.40*pseudo_s7
#       so primary effective:
#         recipe=0.130, pseudo_s1=0.181, pseudo_s7=0.207, realmlp=0.130,
#         nr_iso=0.0525, meta_iso=0.30
#
# Where the documented composition is incomplete, components left as 0.

OBSERVATIONS = [
    # ---- Final-selection candidates (locked) ----
    {
        "label": "tier1b_greedy_meta",  # PRIMARY (LB-best)
        "OOF_tuned": 0.98084, "LB_public": 0.98094,
        "components": {
            "recipe_full_te": 0.130,
            "recipe_pseudolabel": 0.181,
            "recipe_pseudolabel_seed7labeler": 0.207,
            "realmlp": 0.130,
            "xgb_nonrule_iso": 0.0525,
            "xgb_metastack_iso": 0.30,
        },
        "notes": "LB-best primary 4-stack. Composition: 0.70 × LB-3-stack + 0.30 × meta_iso.",
    },
    {
        "label": "lb3_realmlp_nonruleiso",  # LB-best 3-stack
        "OOF_tuned": 0.98061, "LB_public": 0.98008,
        "components": {
            "recipe_full_te": 0.185,
            "recipe_pseudolabel": 0.259,
            "recipe_pseudolabel_seed7labeler": 0.296,
            "realmlp": 0.185,
            "xgb_nonrule_iso": 0.075,
        },
        "notes": "LB-best 3-stack: lb3 + RealMLP@0.20 + nonrule_iso@0.075",
    },
    {
        "label": "3way_recipe025_s1035_s7040",  # 3-way multi-seed
        "OOF_tuned": 0.98029, "LB_public": 0.98005,
        "components": {
            "recipe_full_te": 0.25,
            "recipe_pseudolabel": 0.35,
            "recipe_pseudolabel_seed7labeler": 0.40,
        },
        "notes": "3-way multi-seed log-blend",
    },
    {
        "label": "recipe_x_pseudolabel_50_50",  # 2-way
        "OOF_tuned": 0.98012, "LB_public": 0.97998,
        "components": {
            "recipe_full_te": 0.50,
            "recipe_pseudolabel": 0.50,
        },
        "notes": "2-way recipe × pseudo_s1 50/50 log-blend",
    },
    {
        "label": "recipe_full_te",  # single-model baseline
        "OOF_tuned": 0.97967, "LB_public": 0.97939,
        "components": {"recipe_full_te": 1.0},
        "notes": "Pure V10 recipe single model",
    },
    {
        "label": "recipe_full_te_catboost",
        "OOF_tuned": 0.97936, "LB_public": 0.97935,
        "components": {"recipe_full_te_catboost": 1.0},
        "notes": "CatBoost on recipe FE, single model. Tightest calibration in ladder (gap +0.00001).",
    },
    # ---- LB regressions: meta-stacker family ----
    {
        "label": "lr_metastack_v1",
        "OOF_tuned": 0.98167, "LB_public": 0.97991,
        "components": {
            "recipe_full_te": 0.0925,
            "recipe_pseudolabel": 0.1295,
            "recipe_pseudolabel_seed7labeler": 0.1480,
            "realmlp": 0.0925,
            "xgb_nonrule_iso": 0.0375,
            "lr_metastack_v1_iso": 0.50,  # placed at α=0.50 onto LB-best 3-stack
        },
        "notes": "LR meta-stacker (C=1.0, balanced) at α=0.50. Gap +0.00176.",
    },
    {
        "label": "lr_metastack_v2",
        "OOF_tuned": 0.98107, "LB_public": 0.98052,
        "components": {
            "recipe_full_te": 0.130,
            "recipe_pseudolabel": 0.181,
            "recipe_pseudolabel_seed7labeler": 0.207,
            "realmlp": 0.130,
            "xgb_nonrule_iso": 0.0525,
            "lr_metastack_v2_iso": 0.30,
        },
        "notes": "LR v2 (C=0.1, no class_weight) at α=0.30. Gap +0.00055 — 3x tighter than v1.",
    },
    {
        "label": "mlp_metastack",
        "OOF_tuned": 0.98118, "LB_public": 0.98091,
        "components": {
            "recipe_full_te": 0.0925,
            "recipe_pseudolabel": 0.1295,
            "recipe_pseudolabel_seed7labeler": 0.1480,
            "realmlp": 0.0925,
            "xgb_nonrule_iso": 0.0375,
            "xgb_metastack_iso": 0.25,  # α=0.50 on lb3, B' MLP-meta blend at 0.50
            "mlp_metastack_iso": 0.25,
        },
        "notes": "MLP-meta + XGB-meta at L3, both iso, weighted 50/50 onto LB-3-stack at α=0.50. Gap +0.00027.",
    },
    {
        "label": "three_meta_l3_mlp090_lr010_a060",
        "OOF_tuned": 0.98152, "LB_public": 0.98060,
        "components": {
            "recipe_full_te": 0.10,
            "recipe_pseudolabel": 0.14,
            "recipe_pseudolabel_seed7labeler": 0.16,
            "realmlp": 0.10,
            "xgb_nonrule_iso": 0.04,
            "mlp_metastack_iso": 0.54,  # 0.90*0.60
            "lr_metastack_v2_iso": 0.06,  # 0.10*0.60
        },
        "notes": "L3 weighted (90% MLP + 10% LR-v2) at α=0.60 onto 3-stack. Gap +0.00092.",
    },
    {
        "label": "n5b_angle1_geo_mean_a030",
        "OOF_tuned": 0.98094, "LB_public": 0.98055,
        "components": {
            "recipe_full_te": 0.0455,
            "recipe_pseudolabel": 0.0635,
            "recipe_pseudolabel_seed7labeler": 0.0725,
            "realmlp": 0.0455,
            "xgb_nonrule_iso": 0.018,
            "xgb_metastack_iso": 0.105,
            "n5b_v1_geo_mean": 0.65,
        },
        "notes": "N5b geo-mean v1+new at α=0.30. Gap +0.00046.",
    },
    {
        "label": "tier1b_greedy_meta_l1override",
        "OOF_tuned": 0.98091, "LB_public": 0.98062,
        "components": {
            "recipe_full_te": 0.130,
            "recipe_pseudolabel": 0.181,
            "recipe_pseudolabel_seed7labeler": 0.207,
            "realmlp": 0.130,
            "xgb_nonrule_iso": 0.0525,
            "xgb_metastack_iso": 0.30,
            "_l1_override": 1.0,  # 36 hard-overrides on top
        },
        "notes": "PRIMARY + L1 override (36 rows). Gap +0.00029.",
    },
    {
        "label": "soft_distill_small",
        "OOF_tuned": 0.98066, "LB_public": 0.97865,
        "components": {"soft_distill_small": 1.0},
        "notes": "Soft distill small (depth=3, rounds=1500). Gap +0.00201 — capacity reduction insufficient.",
    },
    {
        "label": "soft_distill",
        "OOF_tuned": 0.98096, "LB_public": 0.97850,
        "components": {"soft_distill": 1.0},
        "notes": "Soft distill (depth=4, rounds=3000). Gap +0.00246 — student memorizes teacher OOF noise.",
    },
    {
        "label": "v6_full_a350",  # combined v6
        "OOF_tuned": 0.98050, "LB_public": 0.98012,
        "components": {
            "recipe_full_te": 0.0845,
            "recipe_pseudolabel": 0.1183,
            "recipe_pseudolabel_seed7labeler": 0.1351,
            "realmlp": 0.0845,
            "xgb_nonrule_iso": 0.0341,
            "xgb_metastack_iso": 0.195,
            "xgb_metastack_v6": 0.35,
        },
        "notes": "v6 meta-stacker (108-pool + aggregates) at α=0.35. Gap +0.00038.",
    },
    {
        "label": "metastack_v3_iso_a300",
        "OOF_tuned": 0.98099, "LB_public": 0.98060,
        "components": {
            "recipe_full_te": 0.130,
            "recipe_pseudolabel": 0.181,
            "recipe_pseudolabel_seed7labeler": 0.207,
            "realmlp": 0.130,
            "xgb_nonrule_iso": 0.0525,
            "xgb_metastack_v3_iso": 0.30,
        },
        "notes": "Cross-poll metastack v3 at α=0.30. Gap +0.00039.",
    },
    {
        "label": "tier1c_meta_v4_a035",
        "OOF_tuned": 0.98121, "LB_public": 0.97992,
        "components": {
            "recipe_full_te": 0.130,
            "recipe_pseudolabel": 0.181,
            "recipe_pseudolabel_seed7labeler": 0.207,
            "realmlp": 0.130,
            "xgb_nonrule_iso": 0.0525,
            "xgb_metastack_v4": 0.35,
        },
        "notes": "Meta v4 (with ET+kNN bank-add) at α=0.35. Gap +0.00129.",
    },
    {
        "label": "p3_perturbed_v1",
        "OOF_tuned": 0.98122, "LB_public": 0.97955,
        "components": {
            "recipe_full_te": 0.0925,
            "recipe_pseudolabel": 0.1295,
            "recipe_pseudolabel_seed7labeler": 0.1480,
            "realmlp": 0.0925,
            "xgb_nonrule_iso": 0.0375,
            "p3_perturbed_v1_iso": 0.50,
        },
        "notes": "P3 perturbed meta v1 at α=0.50. Gap +0.00177.",
    },
    {
        "label": "n5b_angle2_swap_a350",
        "OOF_tuned": 0.98094, "LB_public": 0.98025,
        "components": {
            "recipe_full_te": 0.0975,
            "recipe_pseudolabel": 0.1365,
            "recipe_pseudolabel_seed7labeler": 0.156,
            "realmlp": 0.0975,
            "xgb_nonrule_iso": 0.039,
            "xgb_metastack_n5b_iso": 0.35,
        },
        "notes": "N5b angle2 swap at α=0.35. Gap +0.00069.",
    },
    {
        "label": "n5b_angle2_swap_a425",
        "OOF_tuned": 0.98101, "LB_public": 0.97988,
        "components": {
            "recipe_full_te": 0.0863,
            "recipe_pseudolabel": 0.1208,
            "recipe_pseudolabel_seed7labeler": 0.138,
            "realmlp": 0.0863,
            "xgb_nonrule_iso": 0.0345,
            "xgb_metastack_n5b_iso": 0.425,
        },
        "notes": "N5b angle2 swap at α=0.425. Gap +0.00113.",
    },
    {
        "label": "tier1c_lr_v2_iso_4stack_a050",
        "OOF_tuned": 0.98167, "LB_public": 0.97991,
        "components": {
            "recipe_full_te": 0.0925,
            "recipe_pseudolabel": 0.1295,
            "recipe_pseudolabel_seed7labeler": 0.1480,
            "realmlp": 0.0925,
            "xgb_nonrule_iso": 0.0375,
            "lr_metastack_v2_iso": 0.50,
        },
        "notes": "LR v2 at α=0.50. Gap +0.00176 (= LR v1's gap; α 0.30→0.50 inflated 3x).",
    },
    {
        "label": "primary_oof_optimal_bias",
        "OOF_tuned": 0.98094, "LB_public": 0.98093,
        "components": {
            "recipe_full_te": 0.130,
            "recipe_pseudolabel": 0.181,
            "recipe_pseudolabel_seed7labeler": 0.207,
            "realmlp": 0.130,
            "xgb_nonrule_iso": 0.0525,
            "xgb_metastack_iso": 0.30,
            "_bias_alt": 1.0,  # OOF-optimal bias [1.09, 1.45, 3.40] vs PRIMARY's [1.43, 1.47, 3.40]
        },
        "notes": "PRIMARY composition at OOF-optimal bias. 185 test rows differ. Gap +0.00001 — tightest ever.",
    },
    {
        "label": "leak_honest_primary_retuned",
        "OOF_tuned": 0.98089, "LB_public": 0.98089,
        "components": {
            "recipe_full_te": 0.130,
            "recipe_pseudolabel": 0.181,
            "recipe_pseudolabel_seed7labeler": 0.207,
            "realmlp": 0.130,
            "xgb_nonrule_iso_perfold": 0.0525,
            "xgb_metastack_iso_perfold": 0.30,
            "_bias_alt2": 1.0,  # leak-honest bias [1.04, 1.45, 3.40]
        },
        "notes": "Per-fold-iso version at retuned bias. Gap 0 — perfectly calibrated, slightly LB-inferior.",
    },
    {
        "label": "sklearn_rf_meta_tuned",
        "OOF_tuned": 0.98069, "LB_public": 0.98059,
        "components": {"sklearn_rf_meta": 1.0},
        "notes": "sklearn RF meta-stacker standalone. Gap +0.00010 — tightest non-XGB-meta calibration.",
    },
    {
        "label": "combined_v6_a030",
        "OOF_tuned": 0.98122, "LB_public": 0.98059,
        "components": {
            "recipe_full_te": 0.0925,
            "recipe_pseudolabel": 0.1295,
            "recipe_pseudolabel_seed7labeler": 0.1480,
            "realmlp": 0.0925,
            "xgb_nonrule_iso": 0.0375,
            "xgb_metastack_combined_v6_iso": 0.50,
        },
        "notes": "Combined v6 (aux + masked + poly) at α=0.50. Gap +0.00063.",
    },
    {
        "label": "R2_hybrid075_a015",
        "OOF_tuned": 0.98140, "LB_public": 0.98048,
        "components": {
            "recipe_full_te": 0.110,
            "recipe_pseudolabel": 0.1538,
            "recipe_pseudolabel_seed7labeler": 0.176,
            "realmlp": 0.110,
            "xgb_nonrule_iso": 0.0444,
            "macrorec_hybrid075_iso": 0.15,
            "xgb_metastack_iso": 0.255,
        },
        "notes": "Macrorec hybrid 0.75 at α=0.15. Gap +0.00092 (selection bias from 24-pt grid).",
    },
]


def main():
    rows = []
    # Determine all unique components.
    component_names = sorted(set(
        c for obs in OBSERVATIONS for c in obs["components"]
    ))
    for obs in OBSERVATIONS:
        d = {
            "label": obs["label"],
            "OOF_tuned": obs["OOF_tuned"],
            "LB_public": obs["LB_public"],
            "gap": round(obs["OOF_tuned"] - obs["LB_public"], 5),
            "Δ_vs_primary": round(obs["LB_public"] - 0.98094, 5),
            "notes": obs["notes"],
            "composition_json": json.dumps(obs["components"]),
        }
        # Also flatten components into per-component columns for easy GP fit.
        for c in component_names:
            d[f"w_{c}"] = obs["components"].get(c, 0.0)
        rows.append(d)

    # Sort by LB descending.
    rows.sort(key=lambda r: -r["LB_public"])

    # Write CSV.
    fields = ["label", "OOF_tuned", "LB_public", "gap", "Δ_vs_primary", "notes",
              "composition_json"] + [f"w_{c}" for c in component_names]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved {OUT_CSV} with {len(rows)} observations")
    print(f"unique components in basis: {len(component_names)}")
    print()
    print("Top-10 by LB:")
    print(f"{'label':<45} {'OOF':>8} {'LB':>8} {'gap':>7} {'vs_prim':>8}")
    for r in rows[:10]:
        print(f"{r['label']:<45} {r['OOF_tuned']:>8.5f} {r['LB_public']:>8.5f} "
              f"{r['gap']:>+7.5f} {r['Δ_vs_primary']:>+8.5f}")
    print()
    print(f"  primary (LB-best) marked at LB 0.98094")
    print(f"  pack frontier   = LB 0.98114  (+0.00020 above primary)")
    print(f"  leader          = LB 0.98219  (+0.00125 above primary)")


if __name__ == "__main__":
    main()
