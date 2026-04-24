"""Evaluate recipe+DAE (A2 / P1) against LB-best and emit candidates.

Run after recipe_full_te.py has completed with DAE_EMBED_PATH set
(produces oof_recipe_full_te_dae.npy + test_recipe_full_te_dae.npy).

Produces:
  (A) Standalone-vs-anchor comparison at recipe's fixed bias
      (argmax, tuned, error count, Jaccard vs each anchor).
  (B) Fixed-bias log-blend α-sweep vs two anchors:
        - recipe_full_te alone    (OOF 0.97967 / LB 0.97939)
        - 2-way LB-best blend     (OOF 0.98012 / LB 0.97998)
  (C) 3-way grid (recipe + pseudolabel + recipe_dae) for an
      asymmetric extension that keeps the LB-best structure.
  (D) Auto-emit submission CSV when fixed-bias Δ ≥ +5e-4 vs LB-best.

Rules of thumb (LEARNINGS.md):
  - Never retune bias when adding a candidate to a tuned stack.
  - +OOF < +0.0002 likely doesn't transfer to LB.
  - Jaccard < 0.80 AND errors ≤ anchor's = blend-lift fingerprint.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
IDX2CLS = {i: c for i, c in enumerate(CLASSES)}
EPS = 1e-9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def blend_log(probs_list, weights):
    logs = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        logs = logs + w * np.log(np.clip(p, EPS, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def fixed_bias_argmax(probs, bias):
    return (np.log(np.clip(probs, EPS, 1.0)) + bias).argmax(1)


def main():
    y = pd.read_csv("data/train.csv")[TARGET].map(
        {c: i for i, c in enumerate(CLASSES)}
    ).to_numpy().astype(np.int32)
    test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()

    # --- components
    oof_dae = np.load(ART / "oof_recipe_full_te_dae.npy")
    test_dae = np.load(ART / "test_recipe_full_te_dae.npy")
    oof_recipe = np.load(ART / "oof_recipe_full_te.npy")
    test_recipe = np.load(ART / "test_recipe_full_te.npy")
    oof_pseudo = np.load(ART / "oof_recipe_pseudolabel.npy")
    test_pseudo = np.load(ART / "test_recipe_pseudolabel.npy")

    # --- anchors
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias_recipe = np.array(recipe_res["log_bias"])
    log(f"recipe bias = {bias_recipe.round(4).tolist()}")

    dae_res = json.loads((ART / "recipe_full_te_dae_results.json").read_text())
    bias_dae = np.array(dae_res["log_bias"])
    tuned_dae = dae_res["tuned_log_bias_bal_acc"]
    argmax_dae = dae_res["overall_argmax_bal_acc"]
    log(f"recipe+DAE: argmax={argmax_dae:.5f}  tuned={tuned_dae:.5f}  "
        f"bias={bias_dae.round(4).tolist()}")

    # LB-best: 0.5 log(recipe) + 0.5 log(pseudo) using recipe's bias.
    oof_lbbest = blend_log([oof_recipe, oof_pseudo], [0.5, 0.5])
    test_lbbest = blend_log([test_recipe, test_pseudo], [0.5, 0.5])

    # --- standalone metrics (at recipe's fixed bias for comparability)
    log("--- standalone at recipe bias ---")
    for name, oof in [("recipe_full_te", oof_recipe),
                      ("recipe_pseudolabel", oof_pseudo),
                      ("LB-best (recipe × pseudo)", oof_lbbest),
                      ("recipe+DAE", oof_dae)]:
        pred = fixed_bias_argmax(oof, bias_recipe)
        ba_fixed = fast_bal_acc(y, pred)
        b, ba_tuned = tune_log_bias(oof, y, np.bincount(y) / len(y))
        errs = int((pred != y).sum())
        log(f"  {name:32s}  fixed={ba_fixed:.5f}  tuned={ba_tuned:.5f}  "
            f"errs@fixed={errs}  tuned_bias={b.round(3).tolist()}")

    # Jaccard between each candidate's error set and the anchors (at fixed bias)
    e_recipe = fixed_bias_argmax(oof_recipe, bias_recipe) != y
    e_lbbest = fixed_bias_argmax(oof_lbbest, bias_recipe) != y
    e_dae = fixed_bias_argmax(oof_dae, bias_recipe) != y
    jaccards = {}
    for label, a, b in [("DAE vs recipe", e_dae, e_recipe),
                        ("DAE vs LB-best", e_dae, e_lbbest),
                        ("LB-best vs recipe", e_lbbest, e_recipe)]:
        jac = (a & b).sum() / max((a | b).sum(), 1)
        jaccards[label] = float(jac)
        log(f"  Jaccard {label:30s} = {jac:.4f}")

    # --- α sweep vs each anchor
    alphas = np.round(np.arange(0.0, 0.55, 0.05), 3)
    results = {"anchors": {}, "sweeps": {}, "jaccards": jaccards,
               "standalone": {"recipe_dae_tuned": float(tuned_dae),
                              "recipe_dae_argmax": float(argmax_dae)}}

    for anchor_name, oof_a, test_a in [
        ("recipe", oof_recipe, test_recipe),
        ("lbbest", oof_lbbest, test_lbbest),
    ]:
        log(f"--- fixed-bias sweep vs {anchor_name} (recipe bias) ---")
        base = fast_bal_acc(y, fixed_bias_argmax(oof_a, bias_recipe))
        results["anchors"][anchor_name] = dict(base_fixed=float(base))
        sweep = {}
        for alpha in alphas:
            blended = blend_log([oof_a, oof_dae], [1.0 - alpha, alpha])
            score = fast_bal_acc(y, fixed_bias_argmax(blended, bias_recipe))
            sweep[float(alpha)] = float(score)
            log(f"  α_dae={alpha:.3f}  OOF={score:.5f}  Δ={score - base:+.5f}")
        results["sweeps"][anchor_name] = sweep
        best_alpha = max(sweep, key=sweep.get)
        best_score = sweep[best_alpha]
        results["sweeps"][anchor_name + "_peak"] = dict(
            alpha=best_alpha, score=best_score, delta=best_score - base
        )
        if anchor_name == "lbbest" and best_score - base >= 5e-4 and best_alpha > 0:
            blended_test = blend_log(
                [test_a, test_dae], [1.0 - best_alpha, best_alpha]
            )
            pred_test = fixed_bias_argmax(blended_test, bias_recipe)
            sub = pd.DataFrame({
                "id": test_ids,
                TARGET: [IDX2CLS[i] for i in pred_test],
            })
            path = SUB / f"submission_lbbest_dae_w{int(best_alpha*100):03d}.csv"
            sub.to_csv(path, index=False)
            log(f"  ^ EMIT SUBMISSION: {path}  α={best_alpha:.2f}  "
                f"expected LB ≈ {best_score - 0.0001:.5f}")

    # --- 3-way grid (recipe + pseudo + dae), keeping LB-best structure
    log("--- 3-way grid (recipe + pseudo + dae) ---")
    best_3way = None
    for wr in np.arange(0.20, 0.70, 0.05):
        for wp in np.arange(0.10, 0.60, 0.05):
            wd = 1.0 - wr - wp
            if wd < 0.025 or wd > 0.6:
                continue
            blended = blend_log([oof_recipe, oof_pseudo, oof_dae],
                                [float(wr), float(wp), float(wd)])
            ba = fast_bal_acc(y, fixed_bias_argmax(blended, bias_recipe))
            if best_3way is None or ba > best_3way[0]:
                best_3way = (ba, float(wr), float(wp), float(wd))
    log(f"  best 3-way: OOF={best_3way[0]:.5f}  "
        f"w=(recipe={best_3way[1]:.2f}, pseudo={best_3way[2]:.2f}, "
        f"dae={best_3way[3]:.2f})")
    results["3way_best"] = dict(
        score=best_3way[0], w_recipe=best_3way[1],
        w_pseudo=best_3way[2], w_dae=best_3way[3],
    )

    base_lbbest = results["anchors"]["lbbest"]["base_fixed"]
    if best_3way[0] - base_lbbest >= 5e-4:
        blended_test = blend_log(
            [test_recipe, test_pseudo, test_dae],
            [best_3way[1], best_3way[2], best_3way[3]]
        )
        pred_test = fixed_bias_argmax(blended_test, bias_recipe)
        sub = pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in pred_test],
        })
        path = (SUB / f"submission_3way_recipe{int(best_3way[1]*100):03d}"
                f"_pseudo{int(best_3way[2]*100):03d}"
                f"_dae{int(best_3way[3]*100):03d}.csv")
        sub.to_csv(path, index=False)
        log(f"  ^ EMIT 3-WAY SUBMISSION: {path}  "
            f"Δ_vs_lbbest={best_3way[0] - base_lbbest:+.5f}")

    with open(ART / "blend_recipe_dae_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART / 'blend_recipe_dae_results.json'}")


if __name__ == "__main__":
    main()
