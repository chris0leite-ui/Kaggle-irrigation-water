"""Post-TTA analysis: blend-gate and recommendations.

Loads TTA OOFs + LB-best components, computes:
  1. Per-variant argmax/tuned OOF (already in results.json).
  2. Error count + Jaccard(errors) vs baseline (recipe_full_te).
  3. Error count + Jaccard(errors) vs LB-best 2-way
     (recipe_full_te x recipe_pseudolabel 50/50).
  4. Fixed-bias log-blend sweep vs both anchors.

Writes scripts/artifacts/tta_blend_gate_results.json and prints a table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jaccard_err(err_a, err_b):
    inter = (err_a & err_b).sum()
    union = (err_a | err_b).sum()
    return float(inter / union) if union else 0.0


def lb_best_probs(recipe_oof, pseudo_oof):
    """Reproduce LB-best blend: 50/50 log-avg at recipe's fixed bias."""
    eps = 1e-12
    la = np.log(np.clip(recipe_oof, eps, 1))
    lb = np.log(np.clip(pseudo_oof, eps, 1))
    z = 0.5 * la + 0.5 * lb
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def tuned_bal_at_bias(probs, y, bias):
    eps = 1e-12
    logp = np.log(np.clip(probs, eps, 1)) + bias
    return balanced_accuracy_score(y, logp.argmax(1))


def blend_sweep(anchor_probs, anchor_bias, cand_probs, y, name, alphas):
    eps = 1e-12
    la = np.log(np.clip(anchor_probs, eps, 1))
    lc = np.log(np.clip(cand_probs, eps, 1))
    results = []
    for a in alphas:
        z = (1 - a) * la + a * lc
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(1, keepdims=True)
        bal = tuned_bal_at_bias(p, y, anchor_bias)
        results.append((float(a), float(bal)))
    peak_a, peak_bal = max(results, key=lambda r: r[1])
    log(f"  blend vs {name}: peak alpha={peak_a:.3f}  OOF={peak_bal:.5f}  "
        f"delta={peak_bal - results[0][1]:+.5f}")
    return dict(results=results, peak_alpha=peak_a, peak_bal=peak_bal,
                baseline_bal=results[0][1], delta=peak_bal - results[0][1])


def main():
    log("loading")
    y = pd.read_csv("data/train.csv")["Irrigation_Need"].map(CLS_MAP).to_numpy()
    recipe = np.load(ART / "oof_recipe_full_te.npy")
    pseudo = np.load(ART / "oof_recipe_pseudolabel.npy")
    lb_best = lb_best_probs(recipe, pseudo)
    recipe_bias = np.array([1.4324, 1.4689, 3.4008])
    lb_bias, lb_tuned = tune_log_bias(
        lb_best, y, np.bincount(y, minlength=3) / len(y))
    log(f"LB-best 2-way tuned bal_acc = {lb_tuned:.5f}  bias={lb_bias.round(3).tolist()}")

    # TTA variants (from tta_recipe_full results.json).
    variants = {}
    for tag in ["baseline", "s001", "s005", "s010"]:
        oof = np.load(ART / f"oof_tta_recipe_{tag}.npy")
        variants[tag] = oof
        bal_argmax = balanced_accuracy_score(y, oof.argmax(1))
        bias, tuned = tune_log_bias(oof, y, np.bincount(y, minlength=3)/len(y))
        err = (oof.argmax(1) != y)
        log(f"{tag}: argmax={bal_argmax:.5f} tuned={tuned:.5f} errs={err.sum():,}")

    # Error-geometry diagnostics vs recipe_full_te (baseline anchor).
    recipe_err = (recipe.argmax(1) != y)
    lb_err = (lb_best.argmax(1) != y)

    log("error geometry (vs recipe):")
    for tag in ["baseline", "s001", "s005", "s010"]:
        err = (variants[tag].argmax(1) != y)
        j_rec = jaccard_err(err, recipe_err)
        j_lb = jaccard_err(err, lb_err)
        log(f"  {tag}: errs={int(err.sum()):,}  "
            f"jaccard_vs_recipe={j_rec:.4f}  jaccard_vs_lb_best={j_lb:.4f}")

    # Blend gate: any σ lift vs recipe or vs LB-best?
    alphas = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    blend_results = {}
    for tag in ["s001", "s005", "s010"]:
        log(f"--- blend {tag} ---")
        vs_recipe = blend_sweep(recipe, recipe_bias, variants[tag], y,
                                "recipe", alphas)
        vs_lb = blend_sweep(lb_best, lb_bias, variants[tag], y,
                            "lb_best_2way", alphas)
        blend_results[tag] = dict(vs_recipe=vs_recipe, vs_lb_best=vs_lb)

    # Write JSON for posterity.
    summary = dict(
        lb_best_tuned=float(lb_tuned),
        lb_best_bias=lb_bias.tolist(),
        variants={
            tag: dict(
                argmax=float(balanced_accuracy_score(y, v.argmax(1))),
                errors=int((v.argmax(1) != y).sum()),
                jaccard_vs_recipe=jaccard_err(v.argmax(1) != y, recipe_err),
                jaccard_vs_lb_best=jaccard_err(v.argmax(1) != y, lb_err),
            )
            for tag, v in variants.items()
        },
        blend_results=blend_results,
    )
    out_path = ART / "tta_blend_gate_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
