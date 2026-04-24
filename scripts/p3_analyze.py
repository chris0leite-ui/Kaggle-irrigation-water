"""Post-P3 analysis: blend-gate vs recipe and vs LB-best 2-way.

Loads P3 OOF + test, computes Jaccard(errors) and error-magnitude vs
recipe_full_te and vs LB-best 2-way, then fixed-bias log-blend sweep.

Writes scripts/artifacts/p3_blend_gate_results.json.
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


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jaccard_err(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union else 0.0


def tuned_bal_at_bias(probs, y, bias):
    eps = 1e-12
    logp = np.log(np.clip(probs, eps, 1)) + bias
    return balanced_accuracy_score(y, logp.argmax(1))


def lb_best_probs(recipe, pseudo):
    eps = 1e-12
    la = np.log(np.clip(recipe, eps, 1))
    lb = np.log(np.clip(pseudo, eps, 1))
    z = 0.5 * la + 0.5 * lb
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def blend_sweep(anchor, anchor_bias, cand, y, name, alphas):
    eps = 1e-12
    la = np.log(np.clip(anchor, eps, 1))
    lc = np.log(np.clip(cand, eps, 1))
    results = []
    for a in alphas:
        z = (1 - a) * la + a * lc
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(1, keepdims=True)
        bal = tuned_bal_at_bias(p, y, anchor_bias)
        results.append((float(a), float(bal)))
    peak_a, peak_bal = max(results, key=lambda r: r[1])
    baseline_bal = results[0][1]
    log(f"  blend vs {name}: peak alpha={peak_a:.3f}  OOF={peak_bal:.5f}  "
        f"delta={peak_bal - baseline_bal:+.5f}")
    return dict(results=results, peak_alpha=peak_a, peak_bal=peak_bal,
                baseline_bal=baseline_bal, delta=peak_bal - baseline_bal)


def main():
    log("loading")
    y = pd.read_csv("data/train.csv")["Irrigation_Need"].map(CLS_MAP).to_numpy()
    recipe = np.load(ART / "oof_recipe_full_te.npy")
    pseudo = np.load(ART / "oof_recipe_pseudolabel.npy")
    p3 = np.load(ART / "oof_p3_embed_propagate.npy")
    lb_best = lb_best_probs(recipe, pseudo)
    recipe_bias = np.array([1.4324, 1.4689, 3.4008])
    lb_bias, lb_tuned = tune_log_bias(lb_best, y, np.bincount(y, minlength=3)/len(y))
    log(f"LB-best tuned={lb_tuned:.5f}  bias={lb_bias.round(3).tolist()}")

    # Standalone diagnostics.
    p3_bias, p3_tuned = tune_log_bias(p3, y, np.bincount(y, minlength=3)/len(y))
    recipe_err = recipe.argmax(1) != y
    lb_err = lb_best.argmax(1) != y
    p3_err = p3.argmax(1) != y
    j_rec = jaccard_err(p3_err, recipe_err)
    j_lb = jaccard_err(p3_err, lb_err)
    log(f"p3 standalone: argmax={balanced_accuracy_score(y, p3.argmax(1)):.5f}  "
        f"tuned={p3_tuned:.5f}  errs={int(p3_err.sum()):,}")
    log(f"  jaccard_vs_recipe={j_rec:.4f}  jaccard_vs_lb_best={j_lb:.4f}")
    log(f"  recipe_errs={int(recipe_err.sum()):,}  "
        f"lb_best_errs={int(lb_err.sum()):,}  "
        f"p3_extra_vs_recipe={int(p3_err.sum() - recipe_err.sum()):+,}  "
        f"p3_extra_vs_lb={int(p3_err.sum() - lb_err.sum()):+,}")

    # Fixed-bias blend sweep vs both anchors.
    alphas = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    log("--- blend vs recipe ---")
    vs_recipe = blend_sweep(recipe, recipe_bias, p3, y, "recipe", alphas)
    log("--- blend vs LB-best 2-way ---")
    vs_lb = blend_sweep(lb_best, lb_bias, p3, y, "lb_best_2way", alphas)

    summary = dict(
        p3_standalone=dict(
            argmax=float(balanced_accuracy_score(y, p3.argmax(1))),
            tuned=float(p3_tuned),
            errors=int(p3_err.sum()),
            log_bias=p3_bias.tolist(),
            jaccard_vs_recipe=j_rec,
            jaccard_vs_lb_best=j_lb,
            recipe_errors=int(recipe_err.sum()),
            lb_best_errors=int(lb_err.sum()),
        ),
        vs_recipe=vs_recipe,
        vs_lb_best=vs_lb,
        lb_best_tuned=float(lb_tuned),
    )
    out_path = ART / "p3_blend_gate_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
