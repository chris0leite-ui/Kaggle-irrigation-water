"""C7-LM — Restricted-direction variant: L->M reverts only.

C7 in full was catastrophic at scale (-0.0148 projected). But the L->M
sub-direction passed TRAIN-OOF break-even at 56.5% (need 38.6%). With only
17 test flips, the absolute LB delta is small but possibly positive.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}
LMH_NAMES = {0: "Low", 1: "Medium", 2: "High"}


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


def dgp_rule(score):
    pred = np.full_like(score, 1, dtype=np.int8)
    pred[score <= 3] = 0
    pred[score >= 7] = 2
    return pred


def main():
    print("=== C7-LM: L->M-only restricted variant ===\n")
    fb = csv_to_argmax(SUB / "submission_idea4b_selective_override.csv")
    test = pd.read_csv("data/test.csv")
    test_ids = test["id"].to_numpy()
    score_test = compute_dgp_score(test)
    rule_test = dgp_rule(score_test)
    bank_maj_test = np.load(ART / "stability_test_majority.npy")

    # L->M: 4b=L, bank-maj=M, AND 4b ≠ rule
    mask = (fb == 0) & (bank_maj_test == 1) & (fb != rule_test)
    n = int(mask.sum())
    print(f"L->M-only candidate flips: {n}")
    if n == 0:
        return

    # Score breakdown
    print(f"Score distribution: "
          f"{pd.Series(score_test[mask]).value_counts().sort_index().to_dict()}")

    # Rule prediction on these rows
    rule_dist = pd.Series(rule_test[mask]).value_counts().sort_index().to_dict()
    print(f"Rule pred dist: {rule_dist}")

    # Build candidate
    new_pred = fb.copy()
    new_pred[mask] = bank_maj_test[mask]

    # Sanity: exact 17-row diff vs 4b
    diff = int((new_pred != fb).sum())
    print(f"\nDiff vs 4b: {diff} rows (should match {n})")

    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": pd.Series(new_pred).map(LMH_NAMES),
    })
    out = SUB / "submission_C7_LM_only.csv"
    sub.to_csv(out, index=False)
    print(f"\nEmitted: {out}")

    # Projected LB envelope
    print(f"\n=== Projected LB envelope ===")
    print(f"At TRAIN-OOF precision 56.5% (no asymmetry):")
    p = 0.565
    md = n * (p / 100367 - (1 - p) / 159459) / 3
    print(f"  macro_delta = {md:+.6f}, projected LB = {0.98150 + md:.5f}")
    print(f"At test precision 70% (modest asymmetry):")
    p = 0.70
    md = n * (p / 100367 - (1 - p) / 159459) / 3
    print(f"  macro_delta = {md:+.6f}, projected LB = {0.98150 + md:.5f}")
    print(f"At test precision 86% (full 30pp asymmetry from natural-cal H->M):")
    p = 0.86
    md = n * (p / 100367 - (1 - p) / 159459) / 3
    print(f"  macro_delta = {md:+.6f}, projected LB = {0.98150 + md:.5f}")
    print(f"Worst case (test precision = 30%, asymmetry inverts):")
    p = 0.30
    md = n * (p / 100367 - (1 - p) / 159459) / 3
    print(f"  macro_delta = {md:+.6f}, projected LB = {0.98150 + md:.5f}")


if __name__ == "__main__":
    main()
