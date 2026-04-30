"""T2 alpha sweep — find useful conformal operating point.

The single-alpha=0.05 run produced 0 overrides because bank-mean is nearly
unanimous. Sweep alpha to identify the regime where conformal sets DISAGREE
with 4b's argmax. Goal: characterise where the lever is non-trivial.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import (  # noqa: E402
    bank_mean_probs,
    conformal_threshold,
    in_prediction_set,
    load_bank,
    nonconformity,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def main():
    print("=== T2 alpha sweep ===\n")
    oof_bank = load_bank("oof")
    test_bank = load_bank("test")
    oof_mean = bank_mean_probs(oof_bank)
    test_mean = bank_mean_probs(test_bank)
    y_train = pd.read_csv(DATA / "train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    ).to_numpy(dtype=np.int8)
    cal_scores = nonconformity(oof_mean, y_train)

    fb = csv_argmax("submission_idea4b_selective_override")
    n_test = len(fb)

    print(f"{'alpha':<8} {'q_hat':<8} {'cov':<8} "
          f"{'1cl':<8} {'2cl':<8} {'3cl':<8} "
          f"{'fb_out':<8} {'override':<10}")
    for alpha in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        q = conformal_threshold(cal_scores, alpha)
        in_oof = in_prediction_set(oof_mean, q)
        cov = float(in_oof[np.arange(len(y_train)), y_train].mean())

        in_test = in_prediction_set(test_mean, q)
        sz = in_test.sum(1)
        sz1 = int((sz == 1).sum())
        sz2 = int((sz == 2).sum())
        sz3 = int((sz == 3).sum())

        # 4b is OUTSIDE conformal set
        fb_out = int((~in_test[np.arange(n_test), fb]).sum())

        # candidates: fb outside set AND set has at least 1 class
        cand = (~in_test[np.arange(n_test), fb]) & (sz >= 1) & (sz < 3)
        n_cand = int(cand.sum())

        print(f"{alpha:<8.2f} {q:<8.4f} {cov:<8.4f} "
              f"{sz1:<8} {sz2:<8} {sz3:<8} "
              f"{fb_out:<8} {n_cand:<10}")


if __name__ == "__main__":
    main()
