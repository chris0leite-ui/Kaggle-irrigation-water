"""T2 — verify alignment with 14-bank-majority (4b's proven gate axis)."""
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
    test_bank = load_bank("test")
    test_mean = bank_mean_probs(test_bank)

    # 14-bank majority used by 4b: argmax across each component, then majority vote
    test_argmax_per_model = test_bank.argmax(axis=2)  # (14, 270000)
    # majority via mode
    from scipy.stats import mode
    test_majority = mode(test_argmax_per_model, axis=0, keepdims=False).mode

    # Sanity check vs the existing stability_test_majority.npy
    stab_maj = np.load(ART / "stability_test_majority.npy")
    print(f"stability_test_majority equality with our recompute: "
          f"{int((test_majority == stab_maj).sum())}/{len(test_majority)}")

    # Use the precomputed one (already validated by 4b)
    maj = stab_maj

    # Reconstruct conformal sets
    oof_bank = load_bank("oof")
    oof_mean = bank_mean_probs(oof_bank)
    y_train = pd.read_csv(DATA / "train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    ).to_numpy(dtype=np.int8)
    cal_scores = nonconformity(oof_mean, y_train)
    q_hat = conformal_threshold(cal_scores, 0.01)
    in_test = in_prediction_set(test_mean, q_hat)
    sz = in_test.sum(1)

    fb = csv_argmax("submission_idea4b_selective_override")
    fb_out = ~in_test[np.arange(len(fb)), fb]
    cand_mask = fb_out & (sz >= 1) & (sz < 3)

    # For each candidate, what's the override class (top in conformal set)?
    override_class = np.full(len(fb), -1, dtype=np.int8)
    for i in np.where(cand_mask)[0]:
        cand = np.where(in_test[i])[0]
        cand = cand[np.argsort(-test_mean[i, cand])]
        override_class[i] = cand[0]

    # Q1: agreement between T2's override_class and 14-bank-majority
    cand_idx = np.where(cand_mask)[0]
    t2_class = override_class[cand_idx]
    bank_maj = maj[cand_idx]
    agree = (t2_class == bank_maj).sum()
    print(f"\nT2 override class agrees with 14-bank-majority: "
          f"{int(agree)}/{len(cand_idx)} ({100*agree/len(cand_idx):.1f}%)")

    # Q2: where T2 disagrees with bank-majority, what's bank-majority's vote?
    disagree = t2_class != bank_maj
    if disagree.any():
        for c in range(3):
            n = int(((bank_maj == c) & disagree).sum())
            print(f"  bank-majority={['L','M','H'][c]}, T2-disagree count: {n}")

    # Q3: For STRICT subset (T2 ∩ bank-majority agreement), what's the size and direction?
    strict_mask = cand_mask & (override_class == maj)
    print(f"\nStrict subset (T2 ∩ 14-bank-maj agreement): {int(strict_mask.sum())}")

    directions = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((fb == fr) & (override_class == to) & strict_mask).sum())
            if n > 0:
                directions[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n
    print(f"  directions: {directions}")

    # Q4: Apply the strict subset and see TEST-side metrics
    new_pred = fb.copy()
    new_pred[strict_mask] = override_class[strict_mask]
    h_added = int(((fb != 2) & (new_pred == 2)).sum())
    h_removed = int(((fb == 2) & (new_pred != 2)).sum())
    net_h = h_added - h_removed
    print(f"  net_H: +{h_added} -{h_removed} = {net_h:+d}")
    print(f"  asymmetry ratio: |net_H|/(add+remove) = "
          f"{abs(net_h)/(h_added+h_removed) if (h_added+h_removed) else 0:.3f}")
    new_dist = np.bincount(new_pred, minlength=3).tolist()
    print(f"  new pred dist: {new_dist}")

    # Test sub vs 4b (compute test rows different)
    diff_count = int((fb != new_pred).sum())
    print(f"  diff vs 4b: {diff_count}")

    # Q5: bank-mean confidence on the strict subset
    s_idx = np.where(strict_mask)[0]
    p_alt = test_mean[s_idx, override_class[s_idx]]
    p_fb = test_mean[s_idx, fb[s_idx]]
    print(f"\n  bank-mean P(override) p10/p50/p90: "
          f"{np.percentile(p_alt, [10,50,90])}")
    print(f"  bank-mean P(4b class) p10/p50/p90: "
          f"{np.percentile(p_fb, [10,50,90])}")


if __name__ == "__main__":
    main()
