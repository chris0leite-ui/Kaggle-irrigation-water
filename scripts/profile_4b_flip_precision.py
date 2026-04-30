"""TRAIN-OOF precision profile of analogous '4b H->M' decisions, by stratum.

Goal: characterize when bagged_v1 H->M disagreement with tier1b on TRAIN OOF
yields the correct M class. Identify high-precision and low-precision strata
to understand if 4b's filter could be tightened.

Strata:
  - DGP rule score (continuous 0-8)
  - bagged_v1 P(M) magnitude
  - raw + tier1b individual M-confidence
  - 14-bank agreement on M (strict count)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")

LMH = ["L", "M", "H"]
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).to_numpy()


def main():
    print("=== TRAIN-OOF precision profile of '4b H->M' analog decisions ===\n")

    train = pd.read_csv("data/train.csv")
    print(f"train cols: {train.columns.tolist()}")
    y = train["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    n_train = len(y)
    print(f"TRAIN rows: {n_train}")

    score = compute_dgp_score(train)
    print(f"DGP score distribution: {pd.Series(score).value_counts().sort_index().to_dict()}")

    # Load OOF arrays for the key axes
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_oof = raw_oof / np.clip(raw_oof.sum(1, keepdims=True), 1e-9, None)
    raw_am = raw_oof.argmax(1).astype(np.int8)
    raw_pm = raw_oof[:, 1]

    tier1b_oof = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    tier1b_oof = tier1b_oof / np.clip(tier1b_oof.sum(1, keepdims=True), 1e-9, None)
    tier1b_am = tier1b_oof.argmax(1).astype(np.int8)
    tier1b_pm = tier1b_oof[:, 1]

    # bagged_v1: 3-seed mean (proxy for 5-seed bagged)
    s7 = np.load(ART / "oof_rf_natural_v1_n500_fs7.npy").astype(np.float32)
    s42 = np.load(ART / "oof_rf_natural_v1_n1000_fs42.npy").astype(np.float32)
    s123 = np.load(ART / "oof_rf_natural_v1_n500_fs123.npy").astype(np.float32)
    bagged_oof = (s7 + s42 + s123) / 3
    bagged_oof = bagged_oof / np.clip(bagged_oof.sum(1, keepdims=True), 1e-9, None)
    bagged_am = bagged_oof.argmax(1).astype(np.int8)
    bagged_pm = bagged_oof[:, 1]

    # H->M decision filter on TRAIN OOF: tier1b says H, bagged says M
    h2m_mask = (tier1b_am == 2) & (bagged_am == 1)
    n_h2m = int(h2m_mask.sum())
    print(f"\nTRAIN OOF rows with tier1b=H, bagged=M: {n_h2m}")
    if n_h2m == 0:
        print("No analogous rows — abort")
        return
    p_correct_overall = (y[h2m_mask] == 1).mean()
    print(f"  Overall P(true=M): {p_correct_overall:.3f}  (break-even 0.908)")

    # Stratify by raw P(M)
    print("\n=== Stratify by raw_am agreement ===")
    for raw_eq in [True, False]:
        m = h2m_mask & ((raw_am == 1) == raw_eq)
        n = int(m.sum())
        if n == 0: continue
        p = (y[m] == 1).mean()
        print(f"  raw_am=={'M' if raw_eq else 'NOT M':<8}: n={n:>5}  P(true=M)={p:.3f}")

    # Strata: raw_am == M AND bagged_pm thresholds
    print("\n=== Strata: raw=M AND bagged_pm ranges ===")
    rm = h2m_mask & (raw_am == 1)
    for low, high in [(0.50, 0.70), (0.70, 0.85), (0.85, 0.95), (0.95, 1.01)]:
        m = rm & (bagged_pm >= low) & (bagged_pm < high)
        n = int(m.sum())
        if n == 0: continue
        p = (y[m] == 1).mean()
        verdict = "PASS" if p >= 0.908 else "FAIL"
        print(f"  bagged_pm in [{low:.2f}, {high:.2f}): n={n:>5}  P(true=M)={p:.3f}  {verdict}")

    # Strata: raw=M AND tier1b_pm (tier1b's confidence in M, even though tier1b argmaxes H)
    print("\n=== Strata: raw=M AND tier1b_pm ranges (tier1b's M-confidence) ===")
    for low, high in [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.01)]:
        m = rm & (tier1b_pm >= low) & (tier1b_pm < high)
        n = int(m.sum())
        if n == 0: continue
        p = (y[m] == 1).mean()
        verdict = "PASS" if p >= 0.908 else "FAIL"
        print(f"  tier1b_pm in [{low:.2f}, {high:.2f}): n={n:>5}  P(true=M)={p:.3f}  {verdict}")

    # Strata: by DGP rule score
    print("\n=== Strata: raw=M AND DGP rule score ===")
    for s_val in sorted(set(score)):
        m = rm & (score == s_val)
        n = int(m.sum())
        if n == 0: continue
        p = (y[m] == 1).mean()
        verdict = "PASS" if p >= 0.908 else "FAIL"
        print(f"  score={s_val}: n={n:>5}  P(true=M)={p:.3f}  {verdict}")

    # Joint strata: raw=M, score in {5,6}, bagged_pm >= 0.85
    print("\n=== Joint strata: raw=M AND score in {5,6} AND bagged_pm >= 0.85 ===")
    m = rm & np.isin(score, [5, 6]) & (bagged_pm >= 0.85)
    n = int(m.sum())
    if n > 0:
        p = (y[m] == 1).mean()
        verdict = "PASS" if p >= 0.908 else "FAIL"
        print(f"  n={n}  P(true=M)={p:.3f}  {verdict}")

    # 14-bank-style: count how many of {raw, tier1b, bagged, xgb_corn, recipe_*} say M on h2m_mask
    extra_bank = []
    for fn in ["oof_xgb_corn.npy", "oof_recipe_full_te_macrorec_T1_lam03.npy",
               "oof_recipe_full_te_basemargin_K2.npy", "oof_recipe_full_te_residte.npy"]:
        oof = np.load(ART / fn).astype(np.float32)
        oof = oof / np.clip(oof.sum(1, keepdims=True), 1e-9, None)
        extra_bank.append(oof.argmax(1).astype(np.int8))
    extra_bank_arr = np.stack(extra_bank, axis=1)
    extra_says_m = (extra_bank_arr == 1).sum(axis=1)

    print("\n=== Strata: raw=M AND 4-extra-bank agreement on M ===")
    for k in [0, 1, 2, 3, 4]:
        m = rm & (extra_says_m == k)
        n = int(m.sum())
        if n == 0: continue
        p = (y[m] == 1).mean()
        verdict = "PASS" if p >= 0.908 else "FAIL"
        print(f"  extra-bank-says-M = {k}/4: n={n:>5}  P(true=M)={p:.3f}  {verdict}")


if __name__ == "__main__":
    main()
