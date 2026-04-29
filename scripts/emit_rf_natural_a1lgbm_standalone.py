"""Emit standalone submission CSV for the A1+LGBM RF natural meta.

Mirrors the LB-validated path that produced
`submission_sklearn_rf_meta_natural_standalone.csv` (LB 0.98129) but
loads the META_SUFFIX="_a1lgbm" artifacts so we have an LB-probe
candidate without overwriting the LB-validated meta.

Usage:
  META_SUFFIX=_a1lgbm python3 scripts/sklearn_rf_meta_natural.py
  python3 scripts/emit_rf_natural_a1lgbm_standalone.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

SUFFIX = "_a1lgbm"


def main():
    test = pd.read_csv("data/test.csv")
    test_ids = test["id"].values

    train = pd.read_csv("data/train.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)

    oof = np.load(ART / f"oof_sklearn_rf_meta_natural{SUFFIX}.npy").astype(np.float32)
    test_pred = np.load(ART / f"test_sklearn_rf_meta_natural{SUFFIX}.npy").astype(np.float32)

    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    print(f"OOF tuned bal_acc = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    eps = 1e-9
    test_log = np.log(np.clip(test_pred, eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)

    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    out = SUB / f"submission_sklearn_rf_meta_natural{SUFFIX}_standalone.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out}")

    # Diff vs LB 0.98129 standalone (no suffix)
    prev = SUB / "submission_sklearn_rf_meta_natural_standalone.csv"
    if prev.exists():
        prev_sub = pd.read_csv(prev)
        diff = (sub[TARGET].values != prev_sub[TARGET].values).sum()
        print(f"diff vs LB 0.98129 standalone: {diff}/{len(sub)} rows "
              f"({100*diff/len(sub):.3f}%)")


if __name__ == "__main__":
    main()
