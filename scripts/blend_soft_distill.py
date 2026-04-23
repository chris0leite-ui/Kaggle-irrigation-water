"""Evaluate soft-distill student against LB-best teacher and emit submission candidates.

Run this after soft_distill_xgb.py finishes. Produces:
  (A) Standalone-vs-anchor comparison (argmax, tuned, error count, Jaccard).
  (B) Fixed-anchor-bias log-blend sweep against 3 anchors:
        - greedy_full_bank-era recipe_full_te           (OOF 0.97967 / LB 0.97939)
        - 2-way blend (LB-best)  recipe x pseudolabel   (OOF 0.98012 / LB 0.97998)
        - teacher blend itself (sanity)
  (C) Greedy-forward add-on from the LB-best anchor (is distillation
      a useful ADDITIONAL blend component, or a replacement?).
  (D) Emit submission if fixed-bias Δ vs LB-best ≥ +5e-4 (competition LB-probe gate).

Rules of thumb from CLAUDE.md:
  - Never retune bias when adding a candidate to a tuned stack.
  - Any +OOF < +0.0002 very likely does not transfer to LB.
  - Jaccard < 0.80 AND errors <= anchor's is the blend-lift fingerprint.
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

    # --- load components
    oof_distill = np.load(ART / "oof_soft_distill.npy")
    test_distill = np.load(ART / "test_soft_distill.npy")
    oof_recipe = np.load(ART / "oof_recipe_full_te.npy")
    test_recipe = np.load(ART / "test_recipe_full_te.npy")
    oof_pseudo = np.load(ART / "oof_recipe_pseudolabel.npy")
    test_pseudo = np.load(ART / "test_recipe_pseudolabel.npy")

    # --- anchors
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias_recipe = np.array(recipe_res["log_bias"])
    log(f"recipe bias = {bias_recipe.round(4).tolist()}")

    # LB-best: 0.5 log(recipe) + 0.5 log(pseudo), using recipe's bias.
    oof_lbbest = blend_log([oof_recipe, oof_pseudo], [0.5, 0.5])
    test_lbbest = blend_log([test_recipe, test_pseudo], [0.5, 0.5])

    # --- standalone metrics
    log("--- standalone at recipe bias ---")
    for name, oof in [("recipe_full_te", oof_recipe),
                      ("recipe_pseudolabel", oof_pseudo),
                      ("LB-best (recipe x pseudo)", oof_lbbest),
                      ("soft_distill", oof_distill)]:
        pred = fixed_bias_argmax(oof, bias_recipe)
        ba_fixed = fast_bal_acc(y, pred)
        b, ba_tuned = tune_log_bias(oof, y, np.bincount(y) / len(y))
        errs = int((pred != y).sum())
        log(f"  {name:32s}  fixed={ba_fixed:.5f}  tuned={ba_tuned:.5f}  "
            f"errs@fixed={errs}  tuned_bias={b.round(3).tolist()}")

    # Jaccard between each pair's error sets (at fixed recipe bias)
    pred_recipe = fixed_bias_argmax(oof_recipe, bias_recipe) != y
    pred_lbbest = fixed_bias_argmax(oof_lbbest, bias_recipe) != y
    pred_distill = fixed_bias_argmax(oof_distill, bias_recipe) != y
    for label, a, b in [("distill vs recipe", pred_distill, pred_recipe),
                        ("distill vs LB-best", pred_distill, pred_lbbest),
                        ("LB-best vs recipe", pred_lbbest, pred_recipe)]:
        jac = (a & b).sum() / max((a | b).sum(), 1)
        log(f"  Jaccard {label:30s} = {jac:.4f}")

    # --- alpha sweep vs each anchor
    alphas = np.round(np.arange(0.0, 0.55, 0.05), 3)
    results = {"anchors": {}, "sweeps": {}}

    for anchor_name, oof_a, test_a in [
        ("recipe", oof_recipe, test_recipe),
        ("lbbest", oof_lbbest, test_lbbest),
    ]:
        log(f"--- fixed-bias sweep vs {anchor_name} (recipe bias) ---")
        base = fast_bal_acc(y, fixed_bias_argmax(oof_a, bias_recipe))
        results["anchors"][anchor_name] = dict(base_fixed=float(base))
        sweep = {}
        for alpha in alphas:
            blended = blend_log([oof_a, oof_distill], [1.0 - alpha, alpha])
            pred = fixed_bias_argmax(blended, bias_recipe)
            score = fast_bal_acc(y, pred)
            sweep[float(alpha)] = float(score)
            log(f"  α_distill={alpha:.3f}  OOF={score:.5f}  Δ={score - base:+.5f}")
        results["sweeps"][anchor_name] = sweep
        best_alpha = max(sweep, key=sweep.get)
        best_score = sweep[best_alpha]
        results["sweeps"][anchor_name + "_peak"] = dict(
            alpha=best_alpha, score=best_score, delta=best_score - base
        )
        # Emit if Δ >= 5e-4 vs LB-best
        if anchor_name == "lbbest" and best_score - base >= 5e-4 and best_alpha > 0:
            blended_test = blend_log(
                [test_a, test_distill], [1.0 - best_alpha, best_alpha]
            )
            pred_test = fixed_bias_argmax(blended_test, bias_recipe)
            sub = pd.DataFrame({
                "id": test_ids,
                TARGET: [IDX2CLS[i] for i in pred_test],
            })
            path = SUB / f"submission_lbbest_soft_distill_w{int(best_alpha*100):03d}.csv"
            sub.to_csv(path, index=False)
            log(f"  ^ EMIT SUBMISSION: {path}  "
                f"(α={best_alpha:.2f}, expected LB ~{best_score - 0.0001:.5f})")

    # --- 3-way grid: recipe + pseudo + distill at various weights
    log("--- 3-way grid (recipe + pseudo + distill) ---")
    best_3way = None
    for wr in np.arange(0.20, 0.65, 0.05):
        for wp in np.arange(0.10, 0.60, 0.05):
            wd = 1.0 - wr - wp
            if wd < 0.025 or wd > 0.6:
                continue
            blended = blend_log([oof_recipe, oof_pseudo, oof_distill],
                                 [wr, wp, wd])
            ba = fast_bal_acc(y, fixed_bias_argmax(blended, bias_recipe))
            if best_3way is None or ba > best_3way[0]:
                best_3way = (ba, float(wr), float(wp), float(wd))
    log(f"  best 3-way: OOF={best_3way[0]:.5f}  "
        f"w=(recipe={best_3way[1]:.2f}, pseudo={best_3way[2]:.2f}, "
        f"distill={best_3way[3]:.2f})")
    results["3way_best"] = dict(
        score=best_3way[0], w_recipe=best_3way[1],
        w_pseudo=best_3way[2], w_distill=best_3way[3],
    )

    base_lbbest = results["anchors"]["lbbest"]["base_fixed"]
    if best_3way[0] - base_lbbest >= 5e-4:
        blended_test = blend_log(
            [test_recipe, test_pseudo, test_distill],
            [best_3way[1], best_3way[2], best_3way[3]]
        )
        pred_test = fixed_bias_argmax(blended_test, bias_recipe)
        sub = pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in pred_test],
        })
        path = (SUB / f"submission_3way_recipe{int(best_3way[1]*100):03d}"
                f"_pseudo{int(best_3way[2]*100):03d}"
                f"_distill{int(best_3way[3]*100):03d}.csv")
        sub.to_csv(path, index=False)
        log(f"  ^ EMIT 3-WAY SUBMISSION: {path}  "
            f"Δ_vs_lbbest={best_3way[0] - base_lbbest:+.5f}")

    with open(ART / "blend_soft_distill_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART / 'blend_soft_distill_results.json'}")


if __name__ == "__main__":
    main()
