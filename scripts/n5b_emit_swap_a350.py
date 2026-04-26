"""Build the angle2_swap_a350 submission CSV (variance test for angle1).

Different mechanism from angle1_geo_mean_a030:
  - angle1: 0.7 * lb3 + 0.30 * geo_mean(v1_iso, new_iso)  -> mean-blend
  - angle2: 0.65 * lb3 + 0.35 * new_iso                   -> pure swap

Different OOF lift: angle1 +0.00017, angle2_a350 +0.00026.

If angle1 LB regression (-0.00039) was unlucky public split, this
candidate (different mechanism + larger OOF Δ) should land closer to
its OOF expectation. If both regress similarly, the OOF→LB carryover
is structural.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed

ART = Path("scripts/artifacts")
SUB = Path("submissions")
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
LABELS = ["Low", "Medium", "High"]


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)

    new_o = normed(np.load(ART / "oof_xgb_metastack_n5b_both.npy"))
    new_t = normed(np.load(ART / "test_xgb_metastack_n5b_both.npy"))
    new_iso_o, new_iso_t = iso_cal(new_o, new_t, y)

    alpha = 0.350
    p_t = log_blend([s3_t, new_iso_t], np.array([1 - alpha, alpha]))
    pred_t = (np.log(np.clip(p_t, 1e-12, 1)) + BIAS).argmax(1)

    # v1 PRIMARY for diff count
    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    _, v1_iso_t = iso_cal(v1_o, v1_t, y)
    p_v1_t = log_blend([s3_t, v1_iso_t], np.array([0.70, 0.30]))
    pred_v1_t = (np.log(np.clip(p_v1_t, 1e-12, 1)) + BIAS).argmax(1)
    n_diff = int((pred_t != pred_v1_t).sum())

    # angle1 for diff count too
    geo_t = log_blend([v1_iso_t, new_iso_t], np.array([0.5, 0.5]))
    p_a1_t = log_blend([s3_t, geo_t], np.array([0.70, 0.30]))
    pred_a1_t = (np.log(np.clip(p_a1_t, 1e-12, 1)) + BIAS).argmax(1)
    n_diff_a1 = int((pred_t != pred_a1_t).sum())

    test = pd.read_csv("data/test.csv")
    sub = pd.DataFrame({"id": test["id"].values,
                        "Irrigation_Need": [LABELS[i] for i in pred_t]})
    fname = "submission_n5b_followup_angle2_swap_a350.csv"
    sub.to_csv(SUB / fname, index=False)
    print(f"Saved {fname}")
    print(f"  predict dist: {pd.Series(sub['Irrigation_Need']).value_counts().to_dict()}")
    print(f"  test diff vs v1 PRIMARY: {n_diff} rows ({n_diff/270000*100:.3f}%)")
    print(f"  test diff vs angle1: {n_diff_a1} rows ({n_diff_a1/270000*100:.3f}%)")


if __name__ == "__main__":
    main()
