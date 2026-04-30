"""Emit L2 SupCon-NCM standalone submission CSV.

NCM has NO post-hoc log-bias retune (this is the load-bearing structural
property). Submission predictions are argmax of the macro-recall-Bayes-
optimal posterior produced by predict_proba_macro_recall.

Usage:
  python scripts/emit_l2_submission.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)
IDX2CLS = {0: "Low", 1: "Medium", 2: "High"}


def main() -> None:
    test_probs = np.load(ART / "test_l2_supcon_ncm.npy")
    sample = pd.read_csv("data/sample_submission.csv")
    pred = test_probs.argmax(1)
    sample["Irrigation_Need"] = [IDX2CLS[i] for i in pred]
    out_path = SUB / "submission_l2_supcon_ncm_standalone.csv"
    sample.to_csv(out_path, index=False)
    print(f"wrote {out_path}  rows={len(sample)}")
    print(f"class dist: {sample['Irrigation_Need'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
