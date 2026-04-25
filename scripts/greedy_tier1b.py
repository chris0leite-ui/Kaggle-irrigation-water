"""Tier 1b greedy refit — post-merge pool refresh.

After merging main at commit a937da9, the OOF bank gained 3 new
3-class candidates never seen by prior greedy runs on this branch:
  * recipe_focal_g2h3      — focal-loss XGB on recipe features (main's
                              #3 experiment, null standalone but
                              different error geometry)
  * ovo_boundary_blend     — one-vs-one boundary blend (main)
  * ovo_nonrule_blend      — one-vs-one nonrule blend (main)
Plus assorted DAE / GBY / tau092 / fexboth variants that hadn't all
been evaluated against the LB-best 3-way + realmlp anchor.

This script runs greedy forward-selection from both anchors with the
expanded pool, reusing the same log-blend infrastructure. Success =
any new 4-component stack that strictly improves OOF over the current
3-stack (OOF 0.98061) by ≥ +0.0002 (the LB-transfer threshold for
non-NN additions; +0.0005 for NN additions per the 2026-04-24 rule).

No retraining, no LB spend. Emits no submission; just prints the
greedy path + any ceiling-beating configurations.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

ART = Path("scripts/artifacts")
DATA = Path("data")
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])
LB_BEST_3WAY_OOF = 0.98029
CURRENT_3STACK_OOF = 0.98061  # LB-best 3-way + realmlp(α=0.2) + nonrule_iso(α=0.075)
EXCLUDE = {
    # LB-confirmed regressors — never add to greedy.
    "soft_distill", "recipe_pseudolabel_stage2",
    # Specialists (not full 3-class OOFs or sub-domain only).
    "xgb_spec_678", "xgb_spec_36",
    # Already-composed blends (would double-count).
    "greedy_blend", "greedy_full_bank_6way",
    "bagged_greedy_nonrule",
    "hybrid_binhigh", "hybrid_lgbmxgb_blend",
    "lb_best_fs7", "lb_best_fs123",
    "c0_greedy", "c0_safe_lb_best_2way", "c0_safe_recipe_full_te",
    "c0_v2_lb_best_2way", "c0_v2_lb_best_3way", "c0_v2_recipe_full_te",
    "c0_v3_lb_best_3way", "c0_v3_recipe_full_te",
    "disagree_meta", "selective_router",
    "b2_groupkfold_region",
    # TTA variants — confirmed null, structurally redundant with recipe.
    "tta_recipe_baseline", "tta_recipe_s001", "tta_recipe_s005",
    "tta_recipe_s010", "tta_recipe_s020", "tta_recipe_s030",
    # Embedded propagation — closed NULL.
    "p3_embed_propagate",
    # Labelers in 3-way anchor (would double-count).
    "recipe_pseudolabel_seed7labeler",
}

CANDIDATES = [
    # Core recipe family.
    "recipe_full_te", "recipe_pseudolabel",
    "recipe_full_te_seed7", "recipe_full_te_seed123",
    "recipe_pseudolabel_seed123labeler", "recipe_pseudolabel_tau092",
    "recipe_allpairs", "recipe_catboost", "recipe_lgbm", "recipe_171pair",
    "recipe_full_te_a01", "recipe_full_te_a10",
    "recipe_full_te_catboost", "recipe_full_te_lgbm",
    "recipe_full_te_cldrop", "recipe_full_te_dae",
    "recipe_full_te_fexboth", "recipe_full_te_gby",
    "recipe_no_ote", "recipe_no_digits", "recipe_no_combos", "recipe_no_orig",
    # *** NEW FROM MAIN MERGE ***
    "recipe_focal_g2h3",
    "ovo_boundary_blend", "ovo_nonrule_blend",
    # EM / core tree family.
    "em_uniform", "xgb_corn", "xgb_nonrule",
    "xgb_dist_digits", "lgbm_dist_digits",
    "xgb_dist_digits_ote", "xgb_dist_digits_ote_digits",
    "xgb_dist_digits_ote_digits_pairs", "xgb_dist_digits_ote_digits_light",
    "xgb_dist_digits_ote_light", "lgbm_dist_digits_ote",
    "xgb_dist_routed_v3", "xgb_vanilla_dist",
    "catboost_optuna", "catboost_recipe_gpu",
    "extratrees_dist_digits", "extratrees_dist_digits_v2",
    "lgbm_competitor", "lgbm_te_orig", "tabpfn",
    "realmlp",  # Jaccard-0.62 NN leg.
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend(probs, weights):
    eps = 1e-12
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    lp = np.zeros_like(probs[0], dtype=np.float64)
    for wi, p in zip(w, probs):
        lp += wi * np.log(np.clip(p, eps, 1))
    lp -= lp.max(axis=1, keepdims=True)
    ez = np.exp(lp)
    return (ez / ez.sum(axis=1, keepdims=True)).astype(np.float32)


def bal_bias(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + RECIPE_BIAS).argmax(1),
    )


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    oo = oo / np.clip(oo.sum(1, keepdims=True), 1e-9, None)
    tt = tt / np.clip(tt.sum(1, keepdims=True), 1e-9, None)
    return oo, tt


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}).to_numpy()

    candidates = [n for n in CANDIDATES if n not in EXCLUDE]
    log(f"Tier 1b greedy → {len(candidates)} candidates, "
        f"EXCLUDE={len(EXCLUDE)}")

    pool = {}
    for name in candidates:
        oof_p = ART / f"oof_{name}.npy"; test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  SKIP missing: {name}")
            continue
        oof_raw = np.load(oof_p).astype(np.float32)
        if oof_raw.ndim != 2 or oof_raw.shape[1] != 3:
            log(f"  SKIP non-3class: {name} shape={oof_raw.shape}")
            continue
        test_raw = np.load(test_p).astype(np.float32)
        oof = oof_raw / np.clip(oof_raw.sum(1, keepdims=True), 1e-9, None)
        test = test_raw / np.clip(test_raw.sum(1, keepdims=True), 1e-9, None)
        oof_i, test_i = iso_cal(oof, test, y)
        pool[name] = (oof, test)
        pool[f"{name}__iso"] = (oof_i, test_i)
    log(f"  {len(pool)//2} components loaded (raw + iso = {len(pool)})")

    new_arrivals = {"recipe_focal_g2h3", "ovo_boundary_blend",
                    "ovo_nonrule_blend"}
    log(f"  Post-merge new arrivals present: "
        f"{sorted(n for n in new_arrivals if n in pool)}")

    summary = dict(anchors={})
    alphas = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15,
              0.175, 0.2, 0.225, 0.25, 0.3, 0.4, 0.5]

    # Anchor 1: LB-best 3-way (the LB-verified strongest anchor).
    # Anchor 2: current 3-stack (LB 0.98008) — see if any 4th component
    # improves on it directly.
    recipe_oof = pool["recipe_full_te"][0]
    pseudo_s1_oof = pool["recipe_pseudolabel"][0]
    pseudo_s7_oof = np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"
                            ).astype(np.float32)
    pseudo_s7_oof = pseudo_s7_oof / np.clip(
        pseudo_s7_oof.sum(1, keepdims=True), 1e-9, None)
    realmlp_oof = pool["realmlp"][0]
    nonrule_iso_oof = pool["xgb_nonrule__iso"][0]

    lb3_oof = log_blend(
        [recipe_oof, pseudo_s1_oof, pseudo_s7_oof],
        np.array([0.25, 0.35, 0.40]),
    )
    stack3_oof = log_blend(
        [lb3_oof, realmlp_oof], np.array([0.8, 0.2]),
    )
    stack3_oof = log_blend(
        [stack3_oof, nonrule_iso_oof], np.array([0.925, 0.075]),
    )
    already_in_stack3 = {
        "recipe_full_te", "recipe_pseudolabel", "realmlp", "xgb_nonrule",
    }

    anchors = [
        ("lb_best_3way", lb3_oof, {"recipe_full_te",
                                   "recipe_pseudolabel"}),
        ("current_3stack_LB0.98008", stack3_oof, already_in_stack3),
    ]

    for anchor_name, anchor_oof, anchor_components in anchors:
        log("=" * 70)
        bal_cur = bal_bias(anchor_oof, y)
        log(f"Anchor: {anchor_name} — start bal={bal_cur:.5f}")
        cur = anchor_oof.copy()
        picked = set(anchor_components)
        chosen = []
        for step in range(1, 8):
            best = None
            for key, (oof_k, _) in pool.items():
                base = key.replace("__iso", "")
                if base in picked:
                    continue
                for a in alphas:
                    trial = log_blend([cur, oof_k], [1 - a, a])
                    s = bal_bias(trial, y)
                    if best is None or s > best[0]:
                        best = (s, key, base, a, trial)
            s, key, base, a, trial = best
            d = s - bal_cur
            flag = "  NEW-ARRIVAL" if base in new_arrivals else ""
            if "realmlp" in base:
                flag += "  NN"
            log(f"  step{step}: + {key:55s} α={a:.3f}  "
                f"OOF={s:.5f}  Δ={d:+.5f}{flag}")
            if d < 1e-4:
                log("  stop (Δ < 1e-4)")
                break
            chosen.append((key, float(a), float(s), float(d)))
            picked.add(base); cur = trial; bal_cur = s

        log(f"final[{anchor_name}]: OOF {bal_cur:.5f}  "
            f"Δ vs LB3 0.98029 = {bal_cur - LB_BEST_3WAY_OOF:+.5f}  "
            f"Δ vs 3-stack 0.98061 = {bal_cur - CURRENT_3STACK_OOF:+.5f}")
        summary["anchors"][anchor_name] = dict(
            final_oof=float(bal_cur),
            delta_vs_3way=float(bal_cur - LB_BEST_3WAY_OOF),
            delta_vs_3stack=float(bal_cur - CURRENT_3STACK_OOF),
            chosen=chosen,
        )

    summary["elapsed_sec"] = float(time.time() - t0)
    out = ART / "greedy_tier1b_results.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
