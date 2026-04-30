"""Strictest subset of the deep-dive's 32 H->M counter-flip candidates.

Deep-dive characterized 32 test rows where 4b=H, natural-cal axis (bagged + 14-bank
+ RFnat) all say M, but recipe-family (raw + tier1b) says H. Deep-dive's verdict:
80-90% TRAIN-OOF precision, expected LB -0.00009 to +0.00004, 'not worth probing'.

This script tests the STRICTEST joint condition: score=6 (rule's M domain) AND
14-bank-unanimous M AND bagged_v1 raw P(M) > 0.95 AND RFnat=M. If this surfaces
3-5 rows with TRAIN-OOF precision approaching 95%+, the test-side asymmetry
(natural-cal stronger on full-data) might push it past break-even.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}


def csv_to_argmax(path: Path) -> np.ndarray:
    s = pd.read_csv(path)["Irrigation_Need"]
    return s.map(LMH_REV).to_numpy(dtype=np.int8)


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).to_numpy()


def main():
    print("=== Strictest 32-counter-flip subset analysis ===\n")

    # Load test data
    test = pd.read_csv("data/test.csv")
    score_test = compute_dgp_score(test)
    test_ids = test["id"].to_numpy()
    n_test = len(test)

    # Load 4b argmax + components
    fb = csv_to_argmax(SUB / "submission_idea4b_selective_override.csv")
    raw = csv_to_argmax(SUB / "submission_rawashishsin_2600_standalone.csv")
    tier1b = csv_to_argmax(SUB / "submission_tier1b_greedy_meta.csv")

    # Test-side bagged_v1 prob array (3-seed mean of natural-cal RF)
    s7 = np.load(ART / "test_rf_natural_v1_n500_fs7.npy").astype(np.float32)
    s42 = np.load(ART / "test_rf_natural_v1_n1000_fs42.npy").astype(np.float32)
    s123 = np.load(ART / "test_rf_natural_v1_n500_fs123.npy").astype(np.float32)
    bagged_test = (s7 + s42 + s123) / 3
    bagged_test = bagged_test / np.clip(bagged_test.sum(1, keepdims=True), 1e-9, None)
    bagged_pm_test = bagged_test[:, 1]
    bagged_am_test = bagged_test.argmax(1).astype(np.int8)

    # 14-bank stability artifact
    bank_maj_test = np.load(ART / "stability_test_majority.npy")
    bank_agr_test = np.load(ART / "stability_test_agreement.npy")
    print(f"bank stability shapes: maj={bank_maj_test.shape}, agr={bank_agr_test.shape}")
    print(f"bank_maj distribution L/M/H: "
          f"{int((bank_maj_test==0).sum())}/{int((bank_maj_test==1).sum())}/"
          f"{int((bank_maj_test==2).sum())}")
    print(f"bank_agr range: [{bank_agr_test.min():.3f}, {bank_agr_test.max():.3f}]")

    # The deep-dive 32 counter-flip candidates: 4b=H, raw=H, tier1b=H, bagged=M, bank_maj=M
    candidates = (fb == 2) & (raw == 2) & (tier1b == 2) & (bagged_am_test == 1) & (bank_maj_test == 1)
    n_cand = int(candidates.sum())
    print(f"\nDeep-dive's 32 counter-flip candidates (re-derived): {n_cand}")
    if n_cand == 0:
        print("Re-derivation off; abort")
        return

    # Apply strict filters
    print("\n=== Strict subsets ===")
    print(f"{'subset':<60} {'n':>6}")
    print("-" * 70)

    subsets = [
        ("base (4b=H, raw=H, tier1b=H, bagged=M, bank_maj=M)", candidates),
        ("+ score = 6", candidates & (score_test == 6)),
        ("+ score in {5,6}", candidates & np.isin(score_test, [5, 6])),
        ("+ bagged P(M) >= 0.90", candidates & (bagged_pm_test >= 0.90)),
        ("+ bagged P(M) >= 0.93", candidates & (bagged_pm_test >= 0.93)),
        ("+ bagged P(M) >= 0.95", candidates & (bagged_pm_test >= 0.95)),
        ("+ bank_agreement >= 0.95", candidates & (bank_agr_test >= 0.95)),
        ("+ bank_agreement >= 0.99", candidates & (bank_agr_test >= 0.99)),
        ("+ bank_agreement == 1.00 (unanimous)", candidates & (bank_agr_test >= 0.999)),
        ("STRICTEST: score=6, bagged>=0.95, bank=unanimous",
         candidates & (score_test == 6) & (bagged_pm_test >= 0.95) & (bank_agr_test >= 0.999)),
        ("STRICTEST: score in {5,6}, bagged>=0.95, bank>=0.99",
         candidates & np.isin(score_test, [5, 6]) & (bagged_pm_test >= 0.95) & (bank_agr_test >= 0.99)),
    ]

    for label, mask in subsets:
        n = int(mask.sum())
        print(f"{label:<60} {n:>6}")

    # TRAIN-OOF analog precision for each subset
    print("\n\n=== TRAIN-OOF analog precision (find proxy for test precision) ===")
    train = pd.read_csv("data/train.csv")
    y = train["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    score_tr = compute_dgp_score(train)
    n_train = len(y)

    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_oof = raw_oof / np.clip(raw_oof.sum(1, keepdims=True), 1e-9, None)
    raw_am_tr = raw_oof.argmax(1).astype(np.int8)

    tier1b_oof = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    tier1b_oof = tier1b_oof / np.clip(tier1b_oof.sum(1, keepdims=True), 1e-9, None)
    tier1b_am_tr = tier1b_oof.argmax(1).astype(np.int8)

    s7t = np.load(ART / "oof_rf_natural_v1_n500_fs7.npy").astype(np.float32)
    s42t = np.load(ART / "oof_rf_natural_v1_n1000_fs42.npy").astype(np.float32)
    s123t = np.load(ART / "oof_rf_natural_v1_n500_fs123.npy").astype(np.float32)
    bagged_tr = (s7t + s42t + s123t) / 3
    bagged_tr = bagged_tr / np.clip(bagged_tr.sum(1, keepdims=True), 1e-9, None)
    bagged_pm_tr = bagged_tr[:, 1]
    bagged_am_tr = bagged_tr.argmax(1).astype(np.int8)

    # Reconstruct 14-bank-majority on TRAIN OOF (use the 9 we have as proxy)
    bank_oofs_tr = []
    for fn in ["oof_rawashishsin_2600.npy", "oof_tier1b_greedy_meta.npy",
               "oof_rf_natural_v1_n500_fs7.npy", "oof_rf_natural_v1_n1000_fs42.npy",
               "oof_rf_natural_v1_n500_fs123.npy", "oof_xgb_corn.npy",
               "oof_recipe_full_te_macrorec_T1_lam03.npy",
               "oof_recipe_full_te_basemargin_K2.npy",
               "oof_recipe_full_te_residte.npy"]:
        oof = np.load(ART / fn).astype(np.float32)
        oof = oof / np.clip(oof.sum(1, keepdims=True), 1e-9, None)
        bank_oofs_tr.append(oof.argmax(1).astype(np.int8))
    bank_arr_tr = np.stack(bank_oofs_tr, axis=1)
    bank_says_m = (bank_arr_tr == 1).sum(axis=1)
    bank_says_h = (bank_arr_tr == 2).sum(axis=1)

    # 4b-analog candidates on TRAIN OOF
    train_cand = ((raw_am_tr == 2) & (tier1b_am_tr == 2) & (bagged_am_tr == 1) &
                  (bank_says_m >= 7))
    n_train_cand = int(train_cand.sum())
    print(f"\nTrain analog candidates (raw=H, t1b=H, bagged=M, 7+/9-bank=M): {n_train_cand}")
    if n_train_cand > 0:
        p_overall = (y[train_cand] == 1).mean()
        print(f"  Overall P(true=M): {p_overall:.3f}  (break-even 0.908)")

    # Strictest TRAIN OOF analog
    strict_train_cand = ((raw_am_tr == 2) & (tier1b_am_tr == 2) & (bagged_am_tr == 1) &
                         (bank_says_m >= 7) & (np.isin(score_tr, [5, 6])) &
                         (bagged_pm_tr >= 0.93))
    n_strict_tr = int(strict_train_cand.sum())
    print(f"\nStrictest train analog (score in {{5,6}}, bagged_pm>=0.93): {n_strict_tr}")
    if n_strict_tr > 0:
        p_strict = (y[strict_train_cand] == 1).mean()
        print(f"  Strictest P(true=M): {p_strict:.3f}")
    elif n_strict_tr == 0:
        # Loosen a bit
        for spm in [0.90, 0.85, 0.80]:
            tc = ((raw_am_tr == 2) & (tier1b_am_tr == 2) & (bagged_am_tr == 1) &
                  (bank_says_m >= 7) & (np.isin(score_tr, [5, 6])) &
                  (bagged_pm_tr >= spm))
            n = int(tc.sum())
            if n > 0:
                p = (y[tc] == 1).mean()
                print(f"  Looser bagged_pm>={spm}: n={n}, P(true=M)={p:.3f}  "
                      f"({'PASS' if p >= 0.908 else 'FAIL'})")

    # Brute force: scan all bagged_pm thresholds
    print("\n=== TRAIN OOF precision sweep (any score, any bagged_pm threshold) ===")
    base = (raw_am_tr == 2) & (tier1b_am_tr == 2) & (bagged_am_tr == 1) & (bank_says_m >= 7)
    for spm in [0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95]:
        m = base & (bagged_pm_tr >= spm)
        n = int(m.sum())
        if n == 0:
            continue
        p = (y[m] == 1).mean()
        verdict = "PASS" if p >= 0.908 else "FAIL"
        print(f"  bagged_pm>={spm}: n={n:>4}  P(true=M)={p:.3f}  {verdict}")


if __name__ == "__main__":
    main()
