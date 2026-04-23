"""Greedy forward-selection starting from recipe_full_te (LB 0.97939).

Same procedure as greedy_full_bank.py but with two critical changes:

  1. Anchor = recipe_full_te (LB-best at OOF 0.97967 tuned). The bias
     used throughout is recipe_full_te's fitted bias, not digit-XGB's.

  2. Adds recipe_full_te to the candidate pool as well as using it as
     the start. The greedy algorithm is free to reweight it against
     any orthogonal additions.

Hypothesis: recipe_full_te was built on a DIFFERENT feature set
(~500 cols incl. OTE on ~117 cats, LR-formula logits, ORIG stats,
FREQ counts) from our digit-XGB stack. Orthogonal components like
xgb_nonrule (13 non-rule cats), xgb_corn (CORN ordinal), greedy_full_bank
(6-way blend of digit-family) could add signal that recipe doesn't
already capture.

Emit gate: Δ ≥ +1e-4 (same as greedy_full_bank.py; pre-computed OOFs
have low selection risk).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)

TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


CANDIDATES = [
    ("recipe_full_te",   "oof_recipe_full_te.npy",               "test_recipe_full_te.npy"),
    ("greedy_full_bank", "oof_greedy_full_bank_6way.npy",        "test_greedy_full_bank_6way.npy"),
    ("digit_xgb",        "oof_xgb_dist_digits.npy",              "test_xgb_dist_digits.npy"),
    ("digits_ote",       "oof_xgb_dist_digits_ote_digits.npy",   "test_xgb_dist_digits_ote_digits.npy"),
    ("digits_pairs",     "oof_xgb_dist_digits_ote_digits_pairs.npy", "test_xgb_dist_digits_ote_digits_pairs.npy"),
    ("digits_light_ote", "oof_xgb_dist_digits_ote_digits_light.npy", "test_xgb_dist_digits_ote_digits_light.npy"),
    ("cat_ote",          "oof_xgb_dist_digits_ote.npy",          "test_xgb_dist_digits_ote.npy"),
    ("cat_ote_light",    "oof_xgb_dist_digits_ote_light.npy",    "test_xgb_dist_digits_ote_light.npy"),
    ("lgbm_digit",       "oof_lgbm_dist_digits.npy",             "test_lgbm_dist_digits.npy"),
    ("lgbm_digit_ote",   "oof_lgbm_dist_digits_ote.npy",         "test_lgbm_dist_digits_ote.npy"),
    ("xgb_nonrule",      "oof_xgb_nonrule.npy",                  "test_xgb_nonrule.npy"),
    ("xgb_vanilla_dist", "oof_xgb_vanilla_dist.npy",             "test_xgb_vanilla_dist.npy"),
    ("xgb_routed_v3",    "oof_xgb_dist_routed_v3.npy",           "test_xgb_dist_routed_v3.npy"),
    ("hybrid_lgbmxgb",   "oof_hybrid_lgbmxgb_blend.npy",         "test_hybrid_lgbmxgb_blend.npy"),
    ("xgb_corn",         "oof_xgb_corn.npy",                     "test_xgb_corn.npy"),
    ("lgbm_te_orig",     "oof_lgbm_te_orig.npy",                 "test_lgbm_te_orig.npy"),
    ("extratrees_v2",    "oof_extratrees_dist_digits_v2.npy",    "test_extratrees_dist_digits_v2.npy"),
    # Recipe-subset XGB variants (N1 from main's next-steps menu).
    ("recipe_no_ote",    "oof_recipe_no_ote.npy",                "test_recipe_no_ote.npy"),
    ("recipe_no_digits", "oof_recipe_no_digits.npy",             "test_recipe_no_digits.npy"),
    ("recipe_no_combos", "oof_recipe_no_combos.npy",             "test_recipe_no_combos.npy"),
    ("recipe_no_orig",   "oof_recipe_no_orig.npy",               "test_recipe_no_orig.npy"),
    # Recipe LGBM + CatBoost (merged from main; tree-family nulls but
    # keep for greedy selection — may complement in multi-component).
    ("recipe_lgbm",      "oof_recipe_full_te_lgbm.npy",          "test_recipe_full_te_lgbm.npy"),
    ("recipe_catboost",  "oof_recipe_full_te_catboost.npy",      "test_recipe_full_te_catboost.npy"),
]


def load_components():
    comps = {}
    for name, oof_name, test_name in CANDIDATES:
        op = ART / oof_name
        tp = ART / test_name
        if not op.exists() or not tp.exists():
            log(f"  skip {name}: missing artifact")
            continue
        comps[name] = {"oof": np.load(op), "test": np.load(tp)}
    return comps


def log_blend_list(probs_list, weights) -> np.ndarray:
    logs = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        logs = logs + w * np.log(np.clip(p, 1e-9, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def fixed_bias_bal_acc(oof, y, bias):
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    return float(balanced_accuracy_score(y, (lp + bias).argmax(axis=1)))


def main() -> None:
    log("loading components")
    comps = load_components()
    log(f"loaded {len(comps)}: {sorted(comps.keys())}")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    # Anchor bias = recipe_full_te's fitted bias (LB-best calibrated).
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias = np.array(recipe_res["log_bias"])
    log(f"anchor = recipe_full_te,  bias = {bias.round(4).tolist()}")

    # Standalone OOF under recipe bias.
    log("--- standalone OOF at recipe_full_te's bias ---")
    standalone = {}
    for n, v in comps.items():
        ba = fixed_bias_bal_acc(v["oof"], y, bias)
        standalone[n] = ba
        log(f"  {n:25s}  OOF = {ba:.5f}")

    # Greedy forward selection starting from recipe_full_te.
    current_names = ["recipe_full_te"]
    current_weights = [1.0]
    current_blend = comps["recipe_full_te"]["oof"]
    current_ba = standalone["recipe_full_te"]
    log(f"\n--- greedy from recipe_full_te ---")
    log(f"anchor OOF = {current_ba:.5f}")

    alpha_grid = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    MIN_IMPROVEMENT = 1e-5

    while True:
        best_add = None
        best_delta = 0.0
        best_alpha = 0.0
        for cand in comps:
            if cand in current_names:
                continue
            best_a_ba = current_ba
            best_a = 0.0
            for a in alpha_grid:
                blend = log_blend_list(
                    [comps[cand]["oof"], current_blend], [a, 1 - a]
                )
                b = fixed_bias_bal_acc(blend, y, bias)
                if b > best_a_ba:
                    best_a_ba = b
                    best_a = a
            delta = best_a_ba - current_ba
            if delta > best_delta:
                best_delta = delta
                best_add = cand
                best_alpha = best_a
        if best_add is None or best_delta < MIN_IMPROVEMENT:
            log(f"  no candidate improves by >= {MIN_IMPROVEMENT}; stop.")
            break
        new_weights = [(1 - best_alpha) * w for w in current_weights] + [best_alpha]
        current_names = current_names + [best_add]
        current_weights = new_weights
        current_blend = log_blend_list(
            [comps[n]["oof"] for n in current_names], current_weights
        )
        current_ba = fixed_bias_bal_acc(current_blend, y, bias)
        log(f"  + {best_add:25s}  α={best_alpha:.3f}  OOF={current_ba:.5f}  "
            f"Δ={best_delta:+.5f}")

    log(f"\n--- final blend anchored on recipe_full_te ---")
    for n, w in zip(current_names, current_weights):
        log(f"  {w:.4f}  {n}")
    log(f"OOF at recipe bias = {current_ba:.5f}")
    log(f"Δ vs standalone recipe = {current_ba - standalone['recipe_full_te']:+.5f}")

    # CM + error Jaccard vs recipe standalone.
    cm = confusion_matrix(
        y, (np.log(np.clip(current_blend, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"\nOOF CM at final blend:\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")
    new_err = (
        np.log(np.clip(current_blend, 1e-9, 1.0)) + bias
    ).argmax(axis=1) != y
    recipe_err = (
        np.log(np.clip(comps["recipe_full_te"]["oof"], 1e-9, 1.0)) + bias
    ).argmax(axis=1) != y
    jacc = (new_err & recipe_err).sum() / max(1, (new_err | recipe_err).sum())
    log(f"error count:  blend={new_err.sum()}  recipe_only={recipe_err.sum()}")
    log(f"Jaccard vs recipe standalone = {jacc:.4f}")

    # Build test blend + emit if Δ >= +1e-4.
    test_blend = log_blend_list(
        [comps[n]["test"] for n in current_names], current_weights
    )
    delta = current_ba - standalone["recipe_full_te"]
    action = "no_submission"
    if delta >= 1e-4:
        preds = (np.log(np.clip(test_blend, 1e-9, 1.0)) + bias).argmax(axis=1)
        sub = SUB / "submission_greedy_from_recipe.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub, index=False
        )
        log(f"\nwrote {sub}  Δ = {delta:+.5f}")
        action = "submission_ready" if delta >= 5e-4 else "submission_borderline"
    else:
        log(f"\nno submission: Δ {delta:+.5f} below +1e-4 gate")

    with open(ART / "greedy_from_recipe_results.json", "w") as f:
        json.dump({
            "anchor_bias": bias.tolist(),
            "standalone_at_recipe_bias": standalone,
            "recipe_standalone_oof": standalone["recipe_full_te"],
            "final_blend": {
                "components": current_names,
                "weights_log_space": current_weights,
                "oof_fixed_bias": current_ba,
                "delta_vs_recipe_standalone": delta,
                "error_count": int(new_err.sum()),
                "jaccard_vs_recipe_standalone": float(jacc),
            },
            "action": action,
        }, f, indent=2)
    log(f"wrote {ART}/greedy_from_recipe_results.json")


if __name__ == "__main__":
    main()
