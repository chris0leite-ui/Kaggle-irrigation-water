"""Quick-fire analysis of a new OOF vs recipe_full_te (LB-best anchor).

Reports:
  - Standalone tuned bal_acc at recipe's fixed bias
  - Standalone tuned bal_acc with own log-bias (for reference)
  - Error count vs recipe's ~10,114
  - Jaccard of error sets (critical — < 0.80 signals novel signal path)
  - Fixed-bias log-blend sweep vs recipe (α ∈ [0, 0.5])

Usage: python scripts/analyze_vs_recipe.py <oof_name>
  e.g. python scripts/analyze_vs_recipe.py recipe_no_ote
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias

ART = Path("scripts/artifacts")
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}


def fixed_bias_bal(oof, y, bias):
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    return float(balanced_accuracy_score(y, (lp + bias).argmax(1)))


def main(tag: str) -> None:
    oof_new = np.load(ART / f"oof_{tag}.npy")
    oof_recipe = np.load(ART / "oof_recipe_full_te.npy")
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    recipe_bias = np.array(recipe_res["log_bias"])
    tr = pd.read_csv("data/train.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)

    # Standalone metrics
    own_bias, own_tuned = tune_log_bias(oof_new, y, prior)
    fixed_tuned = fixed_bias_bal(oof_new, y, recipe_bias)
    recipe_tuned = fixed_bias_bal(oof_recipe, y, recipe_bias)

    print(f"=== {tag} vs recipe_full_te ===")
    print(f"  own-bias tuned   = {own_tuned:.5f}  bias={own_bias.round(3).tolist()}")
    print(f"  @ recipe bias    = {fixed_tuned:.5f}  (recipe@recipe_bias = {recipe_tuned:.5f})")

    # Error geometry at recipe's fixed bias
    err_new = (np.log(np.clip(oof_new, 1e-9, 1.0)) + recipe_bias).argmax(1) != y
    err_rec = (np.log(np.clip(oof_recipe, 1e-9, 1.0)) + recipe_bias).argmax(1) != y
    inter = int((err_new & err_rec).sum())
    union = int((err_new | err_rec).sum()) or 1
    jac = inter / union
    print(f"  err count:       {tag}={err_new.sum():,}  recipe={err_rec.sum():,}")
    print(f"  Jaccard(err):    {jac:.4f}")
    gate = "NOVEL (<0.80)" if jac < 0.80 else ("MARGINAL (0.80-0.85)" if jac < 0.85 else "REDUNDANT (>=0.85)")
    print(f"  verdict:         {gate}")

    # Fixed-bias log-blend sweep
    print(f"  --- fixed-bias log-blend vs recipe ---")
    lp_r = np.log(np.clip(oof_recipe, 1e-9, 1.0))
    lp_n = np.log(np.clip(oof_new, 1e-9, 1.0))
    peak_a, peak_ba = 0.0, recipe_tuned
    for a in [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        blend_lp = (1 - a) * lp_r + a * lp_n
        blend_p = np.exp(blend_lp - blend_lp.max(1, keepdims=True))
        blend_p = blend_p / blend_p.sum(1, keepdims=True)
        ba = fixed_bias_bal(blend_p, y, recipe_bias)
        delta = ba - recipe_tuned
        marker = ""
        if ba > peak_ba:
            peak_a, peak_ba = a, ba
            marker = " ← peak"
        print(f"    a={a:.3f}  OOF={ba:.5f}  Δ={delta:+.5f}{marker}")
    print(f"  best α_new = {peak_a:.3f}  OOF = {peak_ba:.5f}  "
          f"Δ = {peak_ba - recipe_tuned:+.5f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "recipe_no_ote")
