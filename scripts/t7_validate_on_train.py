"""T7 validation: replicate the high-minority disagreement pattern on TRAIN OOF
and measure precision. If train precision ≥ 50% on Medium→High consensus rows,
then U1 hardcoded override has positive expected LB.

If precision < 9% (break-even under macro-recall), U1 is dead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, build_lbbest_stack, iso_cal, load_y, normed,
)

DATA = Path("data")


def dgp_score(df):
    dry = (df["Soil_Moisture"].astype(float) < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float) < 300).astype(int)
    hot = (df["Temperature_C"].astype(float) > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float) > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str) == "No").astype(int)
    kc = np.where(df["Crop_Growth_Stage"].astype(str).isin(("Flowering", "Vegetative")), 2, 0)
    return 2 * (dry + norain) + (hot + windy + nomulch) + kc


def argmax_at_bias(p, bias=BIAS):
    return (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)


def main():
    y = load_y()
    train = pd.read_csv(DATA / "train.csv")
    score = dgp_score(train).values

    # Reconstruct the 6 OOF candidates that mirror the LB-verified test subs.
    # primary  = LB-best 4-stack (lb3 + meta_iso α=0.30)
    # realmlp  = LB-best 3-stack with realmlp+nonrule_iso (= lb3 itself)
    # 3way     = recipe × pseudo_s1 × pseudo_s7 at (0.25, 0.35, 0.40)
    # pseudo   = recipe × pseudo_s1 2-way (50/50)
    # recipe   = recipe_full_te standalone
    # catboost = recipe_full_te_catboost standalone

    lb3_o, _ = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, _ = iso_cal(meta_o, np.zeros((1, 3)), y)
    primary_oof = normed(log_blend([lb3_o, meta_o_iso], np.array([0.70, 0.30])))

    recipe_o = normed(np.load(ART / "oof_recipe_full_te.npy").astype(np.float32))
    pseudo_o = normed(np.load(ART / "oof_recipe_pseudolabel.npy").astype(np.float32))
    pseudo7_o = normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy").astype(np.float32))
    cat_o = normed(np.load(ART / "oof_recipe_full_te_catboost.npy").astype(np.float32))

    realmlp_oof = lb3_o  # the LB-best 3-stack already includes realmlp+nonrule_iso
    way3_oof = normed(log_blend([recipe_o, pseudo_o, pseudo7_o],
                                 np.array([0.25, 0.35, 0.40])))
    pseudo2way_oof = normed(log_blend([recipe_o, pseudo_o], np.array([0.50, 0.50])))
    catboost_oof = cat_o

    candidates = {
        "primary": primary_oof,
        "realmlp": realmlp_oof,
        "3way": way3_oof,
        "pseudo": pseudo2way_oof,
        "recipe": recipe_o,
        "catboost": catboost_oof,
    }

    preds = {k: argmax_at_bias(v) for k, v in candidates.items()}
    print(f"Candidate OOF tuned bal_acc:")
    from sklearn.metrics import balanced_accuracy_score
    for k, p in preds.items():
        print(f"  {k:<10} {balanced_accuracy_score(y, p):.5f}")
    print()

    primary = preds["primary"]
    others = [preds[k] for k in candidates if k != "primary"]
    diff_count = np.zeros(len(y), dtype=np.int32)
    for o in others:
        diff_count += (o != primary).astype(np.int32)

    high_minority = (diff_count >= 3)
    print(f"OOF rows where ≥3/5 non-primary disagree with primary: "
          f"{high_minority.sum():,} ({high_minority.mean():.3%})\n")

    # Distribution by primary class.
    print(f"--- on rows where ≥3/5 disagree with primary (TRAIN OOF) ---")
    print(f"{'primary_class':<14} {'count':>8} {'truly_class':>30} {'majority_non_primary':>25}")
    from collections import Counter
    for c in range(3):
        m = high_minority & (primary == c)
        if not m.any():
            continue
        true_dist = np.bincount(y[m], minlength=3)
        votes = []
        for o in others:
            votes.append(o[m])
        votes_arr = np.stack(votes)
        maj_per_row = []
        for j in range(m.sum()):
            cnt = Counter(votes_arr[:, j])
            maj_per_row.append(cnt.most_common(1)[0][0])
        maj_dist = np.bincount(maj_per_row, minlength=3)
        print(f"{['Low','Medium','High'][c]:<14} {int(m.sum()):>8} "
              f"truth={true_dist.tolist()}  maj={maj_dist.tolist()}")
    print()

    # KEY CHECK: on rows where primary=Medium AND consensus=High, what is the
    # actual precision (truth==High)?
    cons_high = np.zeros(len(y), dtype=bool)
    for o in others:
        # majority of others is High requires careful tally
        pass
    # Compute majority-non-primary class explicitly
    others_arr = np.stack(others, axis=0)  # (5, n)

    def majority(arr_col):
        cnt = np.bincount(arr_col, minlength=3)
        return cnt.argmax()

    print(f"--- KEY: precision on (primary=Medium, consensus=High) rows ---")
    primary_med_mask = (primary == 1) & high_minority
    n_med = int(primary_med_mask.sum())
    print(f"primary=Medium & high_minority: {n_med} OOF rows")
    if n_med > 0:
        # for each such row, get majority of others
        idxs = np.where(primary_med_mask)[0]
        majs = np.array([majority(others_arr[:, i]) for i in idxs])
        consensus_high = (majs == 2)
        print(f"  of which majority votes High: {int(consensus_high.sum())}")
        if consensus_high.any():
            ch_idxs = idxs[consensus_high]
            true_high = (y[ch_idxs] == 2).sum()
            true_med = (y[ch_idxs] == 1).sum()
            true_low = (y[ch_idxs] == 0).sum()
            n_ch = len(ch_idxs)
            print(f"  TRUE labels of those: Low={true_low}  Med={true_med}  High={true_high}")
            print(f"  PRECISION (override would be correct): {true_high}/{n_ch} = {true_high/n_ch:.3f}")
            print(f"  break-even threshold under macro-recall: 1/(1 + N_M/N_H) "
                  f"= 1/(1 + {(y==1).sum()}/{(y==2).sum()}) = "
                  f"{(y==2).sum() / ((y==1).sum() + (y==2).sum()):.3f}")

    # Same for primary=Low → consensus=Medium
    print(f"\n--- (primary=Low, consensus=Medium) rows ---")
    primary_low_mask = (primary == 0) & high_minority
    n_low = int(primary_low_mask.sum())
    print(f"primary=Low & high_minority: {n_low} OOF rows")
    if n_low > 0:
        idxs = np.where(primary_low_mask)[0]
        majs = np.array([majority(others_arr[:, i]) for i in idxs])
        consensus_med = (majs == 1)
        if consensus_med.any():
            cm_idxs = idxs[consensus_med]
            tl = (y[cm_idxs] == 0).sum()
            tm = (y[cm_idxs] == 1).sum()
            th = (y[cm_idxs] == 2).sum()
            n_cm = len(cm_idxs)
            print(f"  TRUE labels: Low={tl}  Med={tm}  High={th}")
            print(f"  PRECISION (override correct): {tm}/{n_cm} = {tm/n_cm:.3f}")

    # Same for primary=High → consensus=Medium
    print(f"\n--- (primary=High, consensus=Medium) rows ---")
    primary_high_mask = (primary == 2) & high_minority
    if primary_high_mask.any():
        idxs = np.where(primary_high_mask)[0]
        majs = np.array([majority(others_arr[:, i]) for i in idxs])
        consensus_med = (majs == 1)
        if consensus_med.any():
            cm_idxs = idxs[consensus_med]
            tl = (y[cm_idxs] == 0).sum()
            tm = (y[cm_idxs] == 1).sum()
            th = (y[cm_idxs] == 2).sum()
            n_cm = len(cm_idxs)
            print(f"  primary=High & majority votes Med: {n_cm}")
            print(f"  TRUE labels: Low={tl}  Med={tm}  High={th}")
            print(f"  PRECISION (override correct): {tm}/{n_cm} = {tm/n_cm:.3f}")


if __name__ == "__main__":
    main()
