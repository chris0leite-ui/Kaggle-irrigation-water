"""T2 diagnostic — verify mechanism alignment with 4b.

Key questions:
  Q1. Do T2's 692 H->M candidates SUBSUME 4b's 105 H->M flips?
      If yes, T2 is a strict expansion; we can trust 4b's precision (93%+)
      transfers to the overlap, and only need to validate the ~587 extra.
  Q2. Of T2's "extra" candidates beyond 4b, what's the bank-majority
      direction? If 14-bank-majority agrees with override class on most
      extras, they're high-confidence flips like 4b's.
  Q3. What's the 99th-percentile bank-mean P(H) on T2's 692 candidates?
      High P(H) = T2 is overriding rows where bank IS confident on H,
      which is risky.
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

CLASSES = ["L", "M", "H"]


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def main():
    print("=== T2 diagnostic — alignment with 4b ===\n")
    oof_bank = load_bank("oof")
    test_bank = load_bank("test")
    oof_mean = bank_mean_probs(oof_bank)
    test_mean = bank_mean_probs(test_bank)
    y_train = pd.read_csv(DATA / "train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    ).to_numpy(dtype=np.int8)
    cal_scores = nonconformity(oof_mean, y_train)
    q_hat = conformal_threshold(cal_scores, 0.01)
    in_test = in_prediction_set(test_mean, q_hat)
    sz = in_test.sum(1)

    fb = csv_argmax("submission_idea4b_selective_override")
    b = csv_argmax("submission_2other_raw_tier1b_k2")

    # T2 candidates: 4b's class outside set, set non-empty and non-full.
    fb_out = ~in_test[np.arange(len(fb)), fb]
    cand_mask = fb_out & (sz >= 1) & (sz < 3)

    # 4b's flips vs B (the anchor): 4b changes B's argmax on these rows.
    fb_flip = b != fb
    print(f"Total rows: {len(fb)}")
    print(f"4b vs B: {int(fb_flip.sum())} flipped rows")

    fb_hm_flip = (b == 2) & (fb == 1)  # 4b's H->M flips relative to B
    print(f"4b H->M flips (vs B): {int(fb_hm_flip.sum())}")

    # T2 overlap with 4b's flips
    overlap_4b = cand_mask & fb_flip
    overlap_4b_hm = cand_mask & fb_hm_flip
    print(f"\nQ1: T2 candidate overlap with 4b's flips:")
    print(f"  T2 candidates: {int(cand_mask.sum())}")
    print(f"  T2 ∩ 4b-flips: {int(overlap_4b.sum())}")
    print(f"  T2 ∩ 4b-H->M-flips: {int(overlap_4b_hm.sum())}")

    # T2 NEW candidates (not in 4b)
    t2_new = cand_mask & ~fb_flip
    print(f"  T2 candidates NEW (not 4b's flips): {int(t2_new.sum())}")

    # Q2: For the NEW candidates, where does T2 want to flip them?
    # cand_mask + fb_out means override class is in test_mean's most-likely
    # class within the conformal set.
    print(f"\nQ2: For T2-new candidates, what override classes & bank-mean confidences?")
    new_idx = np.where(t2_new)[0]
    if len(new_idx) > 0:
        # for each, override class = highest-prob class in conformal set
        override_class = np.full(len(fb), -1, dtype=np.int8)
        for i in new_idx:
            cand = np.where(in_test[i])[0]
            cand = cand[np.argsort(-test_mean[i, cand])]
            override_class[i] = cand[0]

        # Direction breakdown for NEW candidates only
        directions_new = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                m = (fb == fr) & (override_class == to) & t2_new
                if m.sum() > 0:
                    directions_new[f"{CLASSES[fr]}->{CLASSES[to]}"] = int(m.sum())
        print(f"  T2-new directions: {directions_new}")

        # Q3: bank-mean P(4b's class) on these rows — since they're outside the set,
        # P(4b's class) should be small, but we want to see how small.
        p_fb = test_mean[new_idx, fb[new_idx]]
        p_alt = test_mean[new_idx, override_class[new_idx]]
        print(f"\nQ3: bank-mean confidences on T2-new (n={len(new_idx)}):")
        print(f"  P(4b's class) p10/p50/p90: {np.percentile(p_fb, [10,50,90])}")
        print(f"  P(alternative)  p10/p50/p90: {np.percentile(p_alt, [10,50,90])}")
        print(f"  margin = P(alt) - P(fb) p10/p50/p90: "
              f"{np.percentile(p_alt - p_fb, [10,50,90])}")

    # Q4: For 4b's existing 105 H->M flips, are they ALL in T2's candidate set?
    fb_hm_idx = np.where(fb_hm_flip)[0]
    if len(fb_hm_idx) > 0:
        in_t2 = cand_mask[fb_hm_idx].sum()
        print(f"\nQ4: 4b's H->M flips inside T2 candidate set:")
        print(f"  {int(in_t2)} / {len(fb_hm_idx)} ({100*in_t2/len(fb_hm_idx):.1f}%)")

    # Q5: At STRICTER alpha (e.g., 0.005), does T2 narrow to a high-precision subset?
    print("\nQ5: Stricter alpha sweep:")
    for alpha in [0.001, 0.002, 0.005, 0.01]:
        q = conformal_threshold(cal_scores, alpha)
        in_t = in_prediction_set(test_mean, q)
        sz_t = in_t.sum(1)
        cm = (~in_t[np.arange(len(fb)), fb]) & (sz_t >= 1) & (sz_t < 3)
        # overlap with 4b's H->M
        ov_hm = int((cm & fb_hm_flip).sum())
        # T2-new H->M direction
        new_count = 0
        for i in np.where(cm & ~fb_flip)[0]:
            cand = np.where(in_t[i])[0]
            cand = cand[np.argsort(-test_mean[i, cand])]
            if fb[i] == 2 and cand[0] == 1:
                new_count += 1
        print(f"  alpha={alpha}: cand={int(cm.sum())}, "
              f"4b-HM-overlap={ov_hm}/{int(fb_hm_flip.sum())}, "
              f"T2-new H->M={new_count}")


if __name__ == "__main__":
    main()
