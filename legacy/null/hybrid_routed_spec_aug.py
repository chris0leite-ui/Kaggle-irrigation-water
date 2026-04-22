"""Hybrid with specialist augmented by original-{6,7,8} rows.

Compares 4 variants at once:
  baseline : routed-{0,1,2} main XGB alone (tuned)
  hybrid   : baseline + spec-678 (baseline spec) overrides on {6,7,8}
  hybrid_A : baseline + spec-678-aug-w1.0 overrides on {6,7,8}
  hybrid_B : baseline + spec-678-aug-w0.3 overrides on {6,7,8}   (optional)

Writes submission for the best variant only. Uses the rule to
route rows with dgp_score in {0, 1, 2} (matches xgb_dist_routed_v3).

Reads:
  oof_xgb_dist_routed_v3.npy, test_xgb_dist_routed_v3.npy        (main)
  oof_xgb_spec_678.npy, test_xgb_spec_678.npy                     (spec base)
  oof_xgb_spec_678_aug_w10.npy, test_xgb_spec_678_aug_w10.npy     (spec aug full)
  oof_xgb_spec_678_aug_w03.npy, test_xgb_spec_678_aug_w03.npy     (spec aug down)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")
SPEC_SCORES = (6, 7, 8)
ART = Path("scripts/artifacts")
SUB = Path("submissions")


def compute_dgp_score(df):
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(ACTIVE_STAGES).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).values


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(
                    balanced_accuracy_score(y, (log_oof + base).argmax(axis=1))
                )
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def overlay_spec(oof_main, oof_spec, mask):
    out = oof_main.copy()
    out[mask] = oof_spec[mask]
    return out


def main():
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    tr_scores = compute_dgp_score(tr)
    te_scores = compute_dgp_score(te)
    tr_mask = np.isin(tr_scores, SPEC_SCORES)
    te_mask = np.isin(te_scores, SPEC_SCORES)

    oof_main = np.load(ART / "oof_xgb_dist_routed_v3.npy")
    test_main = np.load(ART / "test_xgb_dist_routed_v3.npy")
    oof_spec_base = np.load(ART / "oof_xgb_spec_678.npy")
    test_spec_base = np.load(ART / "test_xgb_spec_678.npy")

    _, main_bal = tune_log_bias(oof_main, y, prior)

    variants = {
        "baseline_main_only": (oof_main, test_main),
        "hybrid_spec_base": (
            overlay_spec(oof_main, oof_spec_base, tr_mask),
            overlay_spec(test_main, test_spec_base, te_mask),
        ),
    }

    for suffix, label in [("aug_w10", "hybrid_spec_aug_w1.0"),
                          ("aug_w03", "hybrid_spec_aug_w0.3")]:
        oof_p = ART / f"oof_xgb_spec_678_{suffix}.npy"
        test_p = ART / f"test_xgb_spec_678_{suffix}.npy"
        if oof_p.exists() and test_p.exists():
            oof_s = np.load(oof_p)
            test_s = np.load(test_p)
            variants[label] = (
                overlay_spec(oof_main, oof_s, tr_mask),
                overlay_spec(test_main, test_s, te_mask),
            )

    results = {}
    best_name = None
    best_bal = -1
    best_pair = None

    print(f"main routed-{{0,1,2}} tuned OOF: {main_bal:.5f}")
    print()
    for name, (oof_v, test_v) in variants.items():
        bias, bal = tune_log_bias(oof_v, y, prior)
        cm = confusion_matrix(
            y, (np.log(np.clip(oof_v, 1e-9, 1.0)) + bias).argmax(axis=1))
        results[name] = {
            "tuned_bal_acc": float(bal),
            "log_bias": bias.tolist(),
            "delta_vs_baseline": float(bal - main_bal),
        }
        print(f"{name:30s}  tuned={bal:.5f}  Δvs_main={bal-main_bal:+.5f}  "
              f"bias={bias.round(3).tolist()}")
        if bal > best_bal:
            best_bal = bal
            best_name = name
            best_pair = (oof_v, test_v, bias)

    with open(ART / "hybrid_aug_results.json", "w") as f:
        json.dump({
            "main_routed_tuned": float(main_bal),
            "variants": results,
            "best_variant": best_name,
            "best_tuned_bal_acc": float(best_bal),
        }, f, indent=2)

    print(f"\nbest variant: {best_name} @ {best_bal:.5f}")

    # only emit submission if best is a hybrid variant (not pure main)
    if best_name != "baseline_main_only":
        oof_b, test_b, bias_b = best_pair
        tuned_idx = (np.log(np.clip(test_b, 1e-9, 1.0)) + bias_b).argmax(axis=1)
        out = SUB / f"submission_{best_name}.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
            out, index=False)
        print(f"submission -> {out}")


if __name__ == "__main__":
    main()
