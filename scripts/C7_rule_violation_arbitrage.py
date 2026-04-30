"""C7 — Rule-violation arbitrage subtract-flip on 4b.

Brainstormed by Lens C, kept by harsh critique as the only clear survivor:
purely subtractive, two-witness disagreement (DGP rule + 14-bank-majority
both disagree with 4b's argmax), no OOF parameter selection so the 30pp
TRAIN-OOF→TEST asymmetry doesn't kill it.

Mechanism:
  Set S = test rows where 4b's argmax differs from BOTH (a) DGP rule prediction
  AND (b) 14-bank majority. On these rows, revert 4b's argmax to bank-maj.
  Note: 4b's own filter requires bank-maj to AGREE on flips, so on the 108
  already-flipped rows bank-maj equals 4b. The non-empty S is therefore on
  the rows where B's (and 4b's, on non-flipped rows) argmax differs from
  rule AND bank-maj — a strictly subtractive intersection beyond 4b's filter.

Validation:
  - Replicate S construction on TRAIN OOF (tier1b argmax as B-on-OOF analog)
  - Measure precision per direction on TRAIN OOF flips
  - Apply break-even gates (H->M 0.908, M->L 0.614, L->M 0.386, M->H 0.092)
  - If |S| ∈ [10, 40] AND OOF precision passes break-even, surface to user
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")
LMH = ["L", "M", "H"]
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


def dgp_rule(score: np.ndarray) -> np.ndarray:
    pred = np.full_like(score, 1, dtype=np.int8)  # default M
    pred[score <= 3] = 0  # L
    pred[score >= 7] = 2  # H
    return pred


BREAK_EVEN = {
    "H->M": 100367 / (100367 + 10174),    # 0.908
    "H->L": 159459 / (159459 + 10174),    # 0.940
    "M->H": 10174 / (10174 + 100367),     # 0.092
    "M->L": 159459 / (159459 + 100367),   # 0.614
    "L->H": 10174 / (10174 + 159459),     # 0.060
    "L->M": 100367 / (100367 + 159459),   # 0.386
}


def main():
    print("=== C7: Rule-violation arbitrage subtract-flip on 4b ===\n")

    # === TEST SIDE: identify set S ===
    fb = csv_to_argmax(SUB / "submission_idea4b_selective_override.csv")
    test = pd.read_csv("data/test.csv")
    test_ids = test["id"].to_numpy()
    score_test = compute_dgp_score(test)
    rule_test = dgp_rule(score_test)
    bank_maj_test = np.load(ART / "stability_test_majority.npy")

    n_test = len(fb)
    print(f"TEST rows: {n_test}")

    # 4b argmax distribution
    print(f"4b argmax dist: L={int((fb==0).sum())} M={int((fb==1).sum())} H={int((fb==2).sum())}")
    print(f"DGP rule dist: L={int((rule_test==0).sum())} M={int((rule_test==1).sum())} H={int((rule_test==2).sum())}")
    print(f"Bank-maj dist: L={int((bank_maj_test==0).sum())} M={int((bank_maj_test==1).sum())} H={int((bank_maj_test==2).sum())}")

    # Set S: 4b ≠ rule AND 4b ≠ bank-maj
    s_mask = (fb != rule_test) & (fb != bank_maj_test)
    n_s = int(s_mask.sum())
    print(f"\nSet S: |S| = {n_s} (rows where 4b disagrees with both rule and bank-maj)")

    # Direction: 4b argmax → bank-maj (the proposed revert direction)
    dirs = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            n = int(((fb == fr) & (bank_maj_test == to) & s_mask).sum())
            if n > 0:
                dirs[f"{LMH[fr]}->{LMH[to]}"] = n
    print(f"Revert directions (4b->bank-maj): {dirs}")

    # Sanity check: any of S overlap with 4b's 108 flips?
    b = csv_to_argmax(SUB / "submission_2other_raw_tier1b_k2.csv")
    flip_mask_4b = b != fb
    overlap = int((s_mask & flip_mask_4b).sum())
    print(f"\nSanity: overlap with 4b's 108 flips: {overlap}")
    print("(Should be 0 since 4b's flips have bank-maj == 4b argmax by construction)")

    # Also tabulate by score
    print(f"\nDGP score distribution within S:")
    for sc in sorted(set(score_test[s_mask])):
        n = int((s_mask & (score_test == sc)).sum())
        print(f"  score={sc}: n={n}")

    # === TRAIN OOF: replicate S construction ===
    train = pd.read_csv("data/train.csv")
    y = train["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    score_tr = compute_dgp_score(train)
    rule_tr = dgp_rule(score_tr)
    n_train = len(y)
    print(f"\n\nTRAIN rows: {n_train}")

    # B-on-OOF analog: tier1b argmax (same as 4d analysis)
    tier1b_oof = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    tier1b_oof = tier1b_oof / np.clip(tier1b_oof.sum(1, keepdims=True), 1e-9, None)
    tier1b_am = tier1b_oof.argmax(1).astype(np.int8)
    print(f"Tier1b argmax (B-on-OOF analog) accuracy: {float((tier1b_am==y).mean()):.5f}")

    # Reconstruct an approximate 14-bank-majority on TRAIN OOF (use 9 components we have)
    bank_oofs = [
        "oof_rawashishsin_2600.npy",
        "oof_tier1b_greedy_meta.npy",
        "oof_rf_natural_v1_n500_fs7.npy",
        "oof_rf_natural_v1_n1000_fs42.npy",
        "oof_rf_natural_v1_n500_fs123.npy",
        "oof_xgb_corn.npy",
        "oof_recipe_full_te_macrorec_T1_lam03.npy",
        "oof_recipe_full_te_basemargin_K2.npy",
        "oof_recipe_full_te_residte.npy",
    ]
    bank_argmaxes_tr = []
    for fn in bank_oofs:
        oof = np.load(ART / fn).astype(np.float32)
        oof = oof / np.clip(oof.sum(1, keepdims=True), 1e-9, None)
        bank_argmaxes_tr.append(oof.argmax(1).astype(np.int8))
    bank_arr_tr = np.stack(bank_argmaxes_tr, axis=1)
    bank_counts_tr = np.zeros((n_train, 3), dtype=np.int32)
    for c in range(3):
        bank_counts_tr[:, c] = (bank_arr_tr == c).sum(axis=1)
    bank_maj_tr = bank_counts_tr.argmax(axis=1)

    # Set S analog on TRAIN OOF: tier1b ≠ rule AND tier1b ≠ bank-maj
    s_mask_tr = (tier1b_am != rule_tr) & (tier1b_am != bank_maj_tr)
    n_s_tr = int(s_mask_tr.sum())
    print(f"\nSet S on TRAIN OOF: |S_tr| = {n_s_tr}")

    # Direction breakdown + precision
    print(f"\n=== TRAIN OOF: direction precision (revert tier1b → bank-maj) ===")
    print(f"{'direction':<10} {'n':>8} {'P(true=bank)':>14} {'P(true=t1b)':>13} "
          f"{'break-even':>10} {'verdict':>10}")
    print("-" * 80)
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            mask = s_mask_tr & (tier1b_am == fr) & (bank_maj_tr == to)
            n = int(mask.sum())
            if n == 0: continue
            p_bank = (y[mask] == to).mean()
            p_t1b = (y[mask] == fr).mean()
            d_label = f"{LMH[fr]}->{LMH[to]}"
            be = BREAK_EVEN.get(d_label, 0.5)
            verdict = "PASS" if p_bank >= be else "FAIL"
            print(f"{d_label:<10} {n:>8} {p_bank:>14.3f} {p_t1b:>13.3f} "
                  f"{be:>10.3f} {verdict:>10}")

    # Apply test-side directions only
    print(f"\n=== Test-side projection at TRAIN-OOF-derived precision ===")
    N_L, N_M, N_H = 159459, 100367, 10174
    Ns = [N_L, N_M, N_H]
    macro_delta = 0.0
    for d_label, n_test_dir in dirs.items():
        fr, to = d_label.split("->")
        fr_i, to_i = LMH.index(fr), LMH.index(to)
        mask = s_mask_tr & (tier1b_am == fr_i) & (bank_maj_tr == to_i)
        n_train_dir = int(mask.sum())
        if n_train_dir < 30:
            print(f"  {d_label}: n_train_dir={n_train_dir} (too small for projection)")
            continue
        p_train = (y[mask] == to_i).mean()
        d = n_test_dir * (p_train / Ns[to_i] - (1 - p_train) / Ns[fr_i]) / 3
        macro_delta += d
        print(f"  {d_label}: n_train={n_train_dir}, p_train={p_train:.3f}, "
              f"n_test={n_test_dir}, macro_delta={d:+.6f}")
    print(f"  Total macro_delta: {macro_delta:+.6f}")
    print(f"  Projected LB (4b 0.98150 + delta): {0.98150 + macro_delta:.5f}")

    # Apply 30pp asymmetry haircut: how much would test precision need to be ABOVE
    # train precision to be net-positive?
    print(f"\n=== Sensitivity to TRAIN-OOF→TEST precision asymmetry ===")
    print(f"For each direction, the test precision threshold for net-positive:")
    for d_label, n_test_dir in dirs.items():
        fr, to = d_label.split("->")
        fr_i, to_i = LMH.index(fr), LMH.index(to)
        be = BREAK_EVEN[d_label]
        print(f"  {d_label}: needs test precision >= {be:.3f}")

    # Build the candidate CSV (regardless of TRAIN-OOF verdict — for inspection)
    if n_s > 0:
        new_pred = fb.copy()
        new_pred[s_mask] = bank_maj_test[s_mask]
        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map(LMH_NAMES),
        })
        out_csv = SUB / "submission_C7_rule_violation_arbitrage.csv"
        sub.to_csv(out_csv, index=False)
        print(f"\n=== Candidate emitted ===")
        print(f"  {out_csv}")
        print(f"  flips applied: {n_s}")


if __name__ == "__main__":
    main()
