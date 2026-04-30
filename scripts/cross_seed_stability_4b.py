"""Cross-seed-stability filter on 4b's 108 flips.

Mechanism: 4b's filter uses bagged_v1' (5-seed mean) as one axis. But within
the bagged mean, individual seeds may disagree. Some of 4b's 108 flips may
be 'seed-stable' (all 3 individual seeds we have agree on the new class) vs
'seed-unstable' (only the bagged mean argmax flips, individual seeds split).

Hypothesis: seed-stable subset has higher precision than seed-unstable subset.
If true, dropping seed-unstable from 4b's flip set lifts LB.

Sanity check: measure on TRAIN OOF using analogous filter logic.
- Seed-stable: all 3 seeds individually argmax = bagged mean argmax (= flip class)
- Seed-unstable: bagged mean argmax = flip class but at least 1 seed disagrees
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")

LMH = ["L", "M", "H"]
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}


def csv_to_argmax(path: Path) -> np.ndarray:
    s = pd.read_csv(path)["Irrigation_Need"]
    return s.map(LMH_REV).to_numpy(dtype=np.int8)


def main():
    print("=== Cross-seed-stability filter on 4b's 108 flips ===\n")

    # TEST-side: identify 4b's 108 flip rows
    b_am = csv_to_argmax(SUB / "submission_2other_raw_tier1b_k2.csv")  # B (LB 0.98140)
    fb_am = csv_to_argmax(SUB / "submission_idea4b_selective_override.csv")  # 4b (LB 0.98150)
    flip_mask_test = b_am != fb_am
    n_flips = int(flip_mask_test.sum())
    print(f"4b flips vs B on TEST: {n_flips}")

    dirs = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            n = int(((b_am == fr) & (fb_am == to) & flip_mask_test).sum())
            if n > 0:
                dirs[f"{LMH[fr]}->{LMH[to]}"] = n
    print(f"  directions: {dirs}")

    # TEST-side seed argmaxes
    seed_test = []
    for s in ["fs7", "fs42", "fs123"]:
        suffix = f"_n1000_{s}" if s == "fs42" else f"_n500_{s}"
        arr = np.load(ART / f"test_rf_natural_v1{suffix}.npy").astype(np.float32)
        arr = arr / np.clip(arr.sum(1, keepdims=True), 1e-9, None)
        seed_test.append(arr.argmax(1).astype(np.int8))
    seed_test_arr = np.stack(seed_test, axis=1)  # (n_test, 3)

    # Cross-seed stability on flip rows: count seeds agreeing with 4b's flip class
    fb_class_test = fb_am  # the flipped-to class
    seeds_agree_test = (seed_test_arr == fb_class_test[:, None]).sum(axis=1)
    flip_3_of_3 = flip_mask_test & (seeds_agree_test == 3)
    flip_2_of_3 = flip_mask_test & (seeds_agree_test == 2)
    flip_1_of_3 = flip_mask_test & (seeds_agree_test == 1)
    flip_0_of_3 = flip_mask_test & (seeds_agree_test == 0)

    print(f"\nTest-side seed-stability breakdown of 4b's {n_flips} flips:")
    print(f"  3/3 seeds agree (seed-stable):  {int(flip_3_of_3.sum())}")
    print(f"  2/3 seeds agree (seed-loose):   {int(flip_2_of_3.sum())}")
    print(f"  1/3 seeds agree (seed-fragile): {int(flip_1_of_3.sum())}")
    print(f"  0/3 seeds agree (seed-against): {int(flip_0_of_3.sum())}")

    # TRAIN OOF: replicate the 4b filter logic to find analogous flip rows
    train = pd.read_csv("data/train.csv")
    y = train["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    n_train = len(y)
    print(f"\nTRAIN rows: {n_train}")

    # B-on-TRAIN-OOF analog: tier1b + raw unanimous override on tier1b argmax
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_oof = raw_oof / np.clip(raw_oof.sum(1, keepdims=True), 1e-9, None)
    raw_am = raw_oof.argmax(1).astype(np.int8)

    tier1b_oof = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    tier1b_oof = tier1b_oof / np.clip(tier1b_oof.sum(1, keepdims=True), 1e-9, None)
    tier1b_am = tier1b_oof.argmax(1).astype(np.int8)

    # B = tier1b's argmax overridden where raw + tier1b unanimously disagree.
    # Since here raw + tier1b are the only two "OTHERS", k=2 unanimous means raw == tier1b.
    # On rows where raw != tier1b, no override fires. On rows where raw == tier1b but
    # both differ from tier1b's stack-argmax, the override fires. But tier1b argmax IS
    # tier1b... so the unanimous rule fires when raw == tier1b but both differ from
    # an underlying anchor we don't have. Simplification: use tier1b argmax as B-on-OOF.
    b_oof_am = tier1b_am.copy()
    print(f"B-on-OOF analog (tier1b argmax) accuracy: {float((b_oof_am==y).mean()):.5f}")

    # bagged_v1' on OOF: 3-seed mean of natural-cal RF
    s7 = np.load(ART / "oof_rf_natural_v1_n500_fs7.npy").astype(np.float32)
    s42 = np.load(ART / "oof_rf_natural_v1_n1000_fs42.npy").astype(np.float32)
    s123 = np.load(ART / "oof_rf_natural_v1_n500_fs123.npy").astype(np.float32)
    bagged_oof = (s7 + s42 + s123) / 3
    bagged_oof = bagged_oof / np.clip(bagged_oof.sum(1, keepdims=True), 1e-9, None)
    bagged_am = bagged_oof.argmax(1).astype(np.int8)
    print(f"Bagged 3-seed mean OOF accuracy: {float((bagged_am==y).mean()):.5f}")

    # 4b filter on TRAIN OOF: bagged != B AND raw == tier1b == bagged
    diff_mask_train = bagged_am != b_oof_am
    rt_unan_train = (raw_am == tier1b_am) & (raw_am == bagged_am)

    # Need 14-bank majority too. Use a quick proxy: load any large multi-component
    # bank if available, otherwise approximate with raw + tier1b + bagged (3-bank-maj).
    # Actually we can construct an approximate 14-bank majority from existing OOFs.
    bank_oofs = [
        ("oof_rawashishsin_2600.npy", "raw"),
        ("oof_tier1b_greedy_meta.npy", "t1b"),
        ("oof_rf_natural_v1_n500_fs7.npy", "rf_s7"),
        ("oof_rf_natural_v1_n1000_fs42.npy", "rf_s42"),
        ("oof_rf_natural_v1_n500_fs123.npy", "rf_s123"),
        ("oof_xgb_corn.npy", "xgb_corn"),
        ("oof_recipe_full_te_macrorec_T1_lam03.npy", "macrorec"),
        ("oof_recipe_full_te_basemargin_K2.npy", "basemargin"),
        ("oof_recipe_full_te_residte.npy", "residte"),
    ]
    bank_argmaxes = []
    for fname, _ in bank_oofs:
        oof = np.load(ART / fname).astype(np.float32)
        oof = oof / np.clip(oof.sum(1, keepdims=True), 1e-9, None)
        bank_argmaxes.append(oof.argmax(1).astype(np.int8))
    bank_arr = np.stack(bank_argmaxes, axis=1)
    bank_counts = np.zeros((n_train, 3), dtype=np.int32)
    for c in range(3):
        bank_counts[:, c] = (bank_arr == c).sum(axis=1)
    bank_maj_train = bank_counts.argmax(axis=1)
    bank_max_train = bank_counts.max(axis=1)

    bank_agree_train = bank_maj_train == bagged_am

    flip_mask_train = diff_mask_train & rt_unan_train & bank_agree_train
    n_flips_train = int(flip_mask_train.sum())
    print(f"\n4b-analog filter fires on TRAIN OOF: {n_flips_train} rows")

    # Direction breakdown
    train_dirs = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            n = int(((b_oof_am == fr) & (bagged_am == to) & flip_mask_train).sum())
            if n > 0:
                train_dirs[f"{LMH[fr]}->{LMH[to]}"] = n
    print(f"  directions: {train_dirs}")

    # Cross-seed stability on TRAIN OOF flips
    seed_train = []
    for s in ["fs7", "fs42", "fs123"]:
        suffix = f"_n1000_{s}" if s == "fs42" else f"_n500_{s}"
        arr = np.load(ART / f"oof_rf_natural_v1{suffix}.npy").astype(np.float32)
        arr = arr / np.clip(arr.sum(1, keepdims=True), 1e-9, None)
        seed_train.append(arr.argmax(1).astype(np.int8))
    seed_train_arr = np.stack(seed_train, axis=1)

    seeds_agree_train = (seed_train_arr == bagged_am[:, None]).sum(axis=1)

    print(f"\n=== Per-stratum precision on TRAIN OOF ===")
    print(f"{'stratum':<25} {'n':>6} {'P(flip-correct)':>18} {'P(anchor-correct)':>20} "
          f"{'overall (M->H)':>15}")
    print("-" * 100)

    BREAK_EVEN = {
        "H->M": 100367 / (100367 + 10174),    # 0.908
        "M->L": 159459 / (159459 + 100367),   # 0.614
        "L->M": 100367 / (100367 + 159459),   # 0.386
        "M->H": 10174 / (10174 + 100367),     # 0.092
    }

    strata = [
        ("all flips", flip_mask_train),
        ("3/3 seeds agree", flip_mask_train & (seeds_agree_train == 3)),
        ("2/3 seeds agree", flip_mask_train & (seeds_agree_train == 2)),
        ("<=1/3 seeds agree", flip_mask_train & (seeds_agree_train <= 1)),
    ]

    for label, mask in strata:
        n = int(mask.sum())
        if n == 0:
            print(f"{label:<25} {n:>6}  (empty)")
            continue
        # Per direction
        bcorr = int((y[mask] == bagged_am[mask]).sum())
        ancorr = int((y[mask] == b_oof_am[mask]).sum())
        per_dir = {}
        for fr in range(3):
            for to in range(3):
                if fr == to: continue
                d_mask = mask & (b_oof_am == fr) & (bagged_am == to)
                d_n = int(d_mask.sum())
                if d_n == 0: continue
                p = (y[d_mask] == to).sum() / d_n
                d_label = f"{LMH[fr]}->{LMH[to]}"
                be = BREAK_EVEN.get(d_label, 0.5)
                per_dir[d_label] = (d_n, p, be)
        s = ", ".join([f"{k}:n={v[0]},p={v[1]:.3f}{'PASS' if v[1]>=v[2] else 'FAIL'}"
                       for k, v in per_dir.items()])
        print(f"{label:<25} {n:>6}  flip-correct={bcorr/n:.3f}  anchor-correct={ancorr/n:.3f}")
        for dl, (dn, dp, dbe) in per_dir.items():
            verdict = "PASS" if dp >= dbe else "FAIL"
            print(f"    {dl:<8} n={dn:>4} P={dp:.3f}  break-even={dbe:.3f}  {verdict}")
        print()

    # Project test-side LB if we drop the lowest-precision-stratum
    print("\n=== Test-side projection: drop seed-fragile (1/3 or 0/3) flips ===")
    # On test, n_drop = flips with 0-1 of 3 seeds agreeing
    n_drop = int((flip_1_of_3 | flip_0_of_3).sum())
    print(f"  Would drop {n_drop} test flips (seed-fragile)")
    n_keep = n_flips - n_drop
    print(f"  Would keep {n_keep} test flips (seed-stable + seed-loose)")

    # Direction breakdown of dropped
    drop_dirs = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            n = int(((b_am == fr) & (fb_am == to) & (flip_1_of_3 | flip_0_of_3)).sum())
            if n > 0:
                drop_dirs[f"{LMH[fr]}->{LMH[to]}"] = n
    print(f"  Dropped directions: {drop_dirs}")


if __name__ == "__main__":
    main()
