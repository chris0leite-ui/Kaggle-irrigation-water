"""Emit both candidate RealMLP blend submissions.

Variant A (LB3 anchor): α=0.200 log-blend of LB-best 3-way × RealMLP.
  OOF 0.98047, Δ=+0.00019 vs LB3 anchor (0.98029).
  Expected LB ~0.98023 if gap holds at ~+0.00024 (LB3's historical gap).

Variant B (LB2 anchor): α=0.375 log-blend of LB-best 2-way × RealMLP.
  OOF 0.98039, Δ=+0.00027 vs LB2 anchor (0.98012).
  Expected LB ~0.98025 if gap holds at ~+0.00014 (LB2's historical gap).

Uses recipe's fixed tuned bias [1.4324, 1.4689, 3.4008] throughout
(no retune per-candidate) per the binhigh rule.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend

ART = Path("scripts/artifacts")
SUB = Path("submissions")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"

BIAS_RECIPE = np.array([1.4324, 1.4689, 3.4008], dtype=np.float64)


def main() -> None:
    realmlp_test = np.load(ART / "test_realmlp.npy")
    recipe_test = np.load(ART / "test_recipe_full_te.npy")
    pseudo_s1_test = np.load(ART / "test_recipe_pseudolabel.npy")
    pseudo_s7_test = np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")

    lb2_test = log_blend([recipe_test, pseudo_s1_test],
                         np.array([0.5, 0.5]))
    lb3_test = log_blend([recipe_test, pseudo_s1_test, pseudo_s7_test],
                         np.array([0.25, 0.35, 0.40]))

    sample = pd.read_csv("data/sample_submission.csv")

    # Variant A: LB3 × RealMLP @ α=0.200
    alpha_a = 0.200
    blend_a = log_blend([lb3_test, realmlp_test],
                        np.array([1.0 - alpha_a, alpha_a]))
    pred_a = (np.log(np.clip(blend_a, 1e-9, 1.0)) + BIAS_RECIPE).argmax(1)
    sub_a = sample.copy()
    sub_a[TARGET] = [CLASSES[i] for i in pred_a]
    path_a = SUB / "submission_lb3_realmlp_a020.csv"
    sub_a.to_csv(path_a, index=False)
    print(f"variant A  LB3 × RealMLP α={alpha_a:.3f}  -> {path_a}")
    print(f"  class counts: {sub_a[TARGET].value_counts().to_dict()}")

    # Variant B: LB2 × RealMLP @ α=0.375
    alpha_b = 0.375
    blend_b = log_blend([lb2_test, realmlp_test],
                        np.array([1.0 - alpha_b, alpha_b]))
    pred_b = (np.log(np.clip(blend_b, 1e-9, 1.0)) + BIAS_RECIPE).argmax(1)
    sub_b = sample.copy()
    sub_b[TARGET] = [CLASSES[i] for i in pred_b]
    path_b = SUB / "submission_lb2_realmlp_a0375.csv"
    sub_b.to_csv(path_b, index=False)
    print(f"variant B  LB2 × RealMLP α={alpha_b:.3f}  -> {path_b}")
    print(f"  class counts: {sub_b[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
