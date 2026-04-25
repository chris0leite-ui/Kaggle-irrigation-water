"""Item 4: submission-level log-mean of the two LB-verified bests.

Pure post-hoc combiner — no retraining. Reconstructs both LB-best blends
from their committed components, log-averages them at 50/50 on test,
applies 3-way blend bias, and emits a submission.

2-way LB-best (LB 0.97998, OOF 0.98012):
    recipe_full_te (0.50) + recipe_pseudolabel (0.50)
    bias = recipe_full_te_results.json tuned bias

3-way LB-best (LB 0.98005, OOF 0.98029):
    recipe_full_te (0.25) + recipe_pseudolabel (0.35) + recipe_pseudolabel_seed7labeler (0.40)
    bias = recipe_full_te_results.json tuned bias (same anchor)

Rationale: both LB-verified. Averaging their TEST probs at 50/50 in log
space produces a submission whose per-row predictions are a geometric
midpoint. Goal is variance protection on private LB, not a public-LB
peak. Argmax disagreements between the two submissions are broken by
the component that agrees with the geo-mid, which itself reflects the
balance of evidence across the three underlying OOF components.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    CLASSES, IDX2CLS, fast_bal_acc, log_blend, load_oof_pair, tune_log_bias,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions"); SUB.mkdir(exist_ok=True)


def build_two_way_testprob():
    """2-way LB-best: recipe + pseudo_s1 at 0.5/0.5 log-blend."""
    _, t_recipe = load_oof_pair("recipe_full_te")
    _, t_pseudo = load_oof_pair("recipe_pseudolabel")
    return log_blend([t_recipe, t_pseudo], np.array([0.5, 0.5]))


def build_three_way_testprob():
    """3-way LB-best: recipe (0.25) + pseudo_s1 (0.35) + pseudo_s7 (0.40)."""
    _, t_recipe = load_oof_pair("recipe_full_te")
    _, t_pseudo = load_oof_pair("recipe_pseudolabel")
    _, t_pseudo_s7 = load_oof_pair("recipe_pseudolabel_seed7labeler")
    return log_blend(
        [t_recipe, t_pseudo, t_pseudo_s7],
        np.array([0.25, 0.35, 0.40]),
    )


def build_two_way_oofprob():
    o_recipe, _ = load_oof_pair("recipe_full_te")
    o_pseudo, _ = load_oof_pair("recipe_pseudolabel")
    return log_blend([o_recipe, o_pseudo], np.array([0.5, 0.5]))


def build_three_way_oofprob():
    o_recipe, _ = load_oof_pair("recipe_full_te")
    o_pseudo, _ = load_oof_pair("recipe_pseudolabel")
    o_pseudo_s7, _ = load_oof_pair("recipe_pseudolabel_seed7labeler")
    return log_blend(
        [o_recipe, o_pseudo, o_pseudo_s7],
        np.array([0.25, 0.35, 0.40]),
    )


def class_str_to_int(s: pd.Series) -> np.ndarray:
    m = {c: i for i, c in enumerate(CLASSES)}
    return s.map(m).to_numpy(dtype=np.int64)


def main():
    print("[hedge] loading components + recipe tuned bias")
    with open(ART / "recipe_full_te_results.json") as f:
        recipe_meta = json.load(f)
    bias = np.array(recipe_meta["log_bias"], dtype=np.float64)
    print(f"[hedge] anchor bias = {bias.tolist()}")

    y = class_str_to_int(pd.read_csv("data/train.csv", usecols=["Irrigation_Need"])["Irrigation_Need"])

    # OOF diagnostic (same anchor bias, no retune)
    oof2 = build_two_way_oofprob()
    oof3 = build_three_way_oofprob()
    hedge_oof = log_blend([oof2, oof3], np.array([0.5, 0.5]))

    log2 = np.log(np.clip(oof2, 1e-9, 1.0))
    log3 = np.log(np.clip(oof3, 1e-9, 1.0))
    logh = np.log(np.clip(hedge_oof, 1e-9, 1.0))

    cc = np.bincount(y, minlength=3)
    bal2 = fast_bal_acc(y, (log2 + bias).argmax(1), class_counts=cc)
    bal3 = fast_bal_acc(y, (log3 + bias).argmax(1), class_counts=cc)
    balh = fast_bal_acc(y, (logh + bias).argmax(1), class_counts=cc)
    print(f"[hedge] OOF bal_acc @ recipe bias")
    print(f"        2-way           = {bal2:.5f}")
    print(f"        3-way           = {bal3:.5f}")
    print(f"        50/50 hedge     = {balh:.5f}")

    # Test probs
    test2 = build_two_way_testprob()
    test3 = build_three_way_testprob()
    hedge_test = log_blend([test2, test3], np.array([0.5, 0.5]))
    test_scaled = np.log(np.clip(hedge_test, 1e-9, 1.0)) + bias
    pred_idx = test_scaled.argmax(axis=1)

    # Report argmax disagreement between 2way/3way/hedge on test
    a2 = (np.log(np.clip(test2, 1e-9, 1.0)) + bias).argmax(1)
    a3 = (np.log(np.clip(test3, 1e-9, 1.0)) + bias).argmax(1)
    ah = pred_idx
    print(f"[hedge] test-argmax disagreement: 2w vs 3w = "
          f"{int((a2 != a3).sum()):,} rows")
    print(f"        hedge vs 2w = {int((ah != a2).sum()):,} rows")
    print(f"        hedge vs 3w = {int((ah != a3).sum()):,} rows")

    # Save OOF/test artefacts + submission
    np.save(ART / "oof_hedge_avg_lb_bests.npy", hedge_oof.astype(np.float32))
    np.save(ART / "test_hedge_avg_lb_bests.npy", hedge_test.astype(np.float32))
    with open(ART / "hedge_avg_lb_bests_results.json", "w") as f:
        json.dump({
            "bias": bias.tolist(),
            "oof_tuned_bal_acc_2way": bal2,
            "oof_tuned_bal_acc_3way": bal3,
            "oof_tuned_bal_acc_hedge": balh,
            "test_argmax_disagree_2w_3w": int((a2 != a3).sum()),
            "test_argmax_disagree_hedge_2w": int((ah != a2).sum()),
            "test_argmax_disagree_hedge_3w": int((ah != a3).sum()),
            "components": {
                "2way_LB_0.97998": {
                    "recipe_full_te": 0.50,
                    "recipe_pseudolabel": 0.50,
                },
                "3way_LB_0.98005": {
                    "recipe_full_te": 0.25,
                    "recipe_pseudolabel": 0.35,
                    "recipe_pseudolabel_seed7labeler": 0.40,
                },
                "hedge_weights": {"2way": 0.5, "3way": 0.5},
            },
        }, f, indent=2)

    sample = pd.read_csv("data/sample_submission.csv")
    sub = pd.DataFrame({
        "id": sample["id"],
        "Irrigation_Need": [IDX2CLS[int(i)] for i in pred_idx],
    })
    out_path = SUB / "submission_hedge_avg_2way_3way.csv"
    sub.to_csv(out_path, index=False)
    print(f"[hedge] wrote {out_path}")
    print(f"[hedge] class dist: {sub['Irrigation_Need'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
