"""50/50 log-blend of recipe_full_te × recipe_pseudolabel.

Reads:
  scripts/artifacts/test_recipe_full_te.npy
  scripts/artifacts/test_recipe_pseudolabel.npy
  scripts/artifacts/recipe_full_te_results.json   (for the tuned bias)
  submissions/submission_recipe_full_te.csv       (for row ids — already
                                                   aligned with the npy arrays)

Writes:
  submissions/submission_recipe_pseudolabel_blend.csv

Reproduces LB 0.97998 (committed reference: submissions/submission_recipe_greedy_recipe_pseudolabel.csv).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
IDX2CLS = {0: "Low", 1: "Medium", 2: "High"}


def main() -> None:
    test_a = np.load(ART / "test_recipe_full_te.npy")
    test_b = np.load(ART / "test_recipe_pseudolabel.npy")
    assert test_a.shape == test_b.shape, (test_a.shape, test_b.shape)
    bias = np.array(
        json.loads((ART / "recipe_full_te_results.json").read_text())["log_bias"]
    )

    eps = 1e-9
    log_blend = 0.5 * np.log(np.clip(test_a, eps, 1.0)) \
              + 0.5 * np.log(np.clip(test_b, eps, 1.0))
    pred_idx = (log_blend + bias).argmax(1)

    # Pull ids from recipe_full_te's own submission so SMOKE-sized runs work
    # without re-aligning against the full-size data/test.csv.
    test_ids = pd.read_csv(SUB / "submission_recipe_full_te.csv")["id"].values
    assert len(test_ids) == len(pred_idx), (len(test_ids), len(pred_idx))
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in pred_idx],
    })
    sub_path = SUB / "submission_recipe_pseudolabel_blend.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}  shape={sub.shape}  "
          f"dist={dict(sub[TARGET].value_counts())}")


if __name__ == "__main__":
    main()
