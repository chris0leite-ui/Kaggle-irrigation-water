"""Hybrid: routed-{1,2} main XGB + specialist on {6,7,8}.

Pipeline:
  - Route dgp_score in {1, 2}  -> rule (Low)  [from xgb_dist_routed.py]
  - Route dgp_score in {6, 7, 8} -> specialist XGB [from xgb_specialist_678.py]
  - All other scores {0, 3, 4, 5, 9} -> main XGB (trained on non-routed)

Reads:
  oof_xgb_dist_routed.npy, test_xgb_dist_routed.npy    (main)
  oof_xgb_spec_678.npy, test_xgb_spec_678.npy          (spec)

Writes:
  oof_xgb_hybrid_routed_spec.npy
  test_xgb_hybrid_routed_spec.npy
  hybrid_routed_spec_results.json
  submissions/submission_xgb_hybrid_routed_spec.csv
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


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(ACTIVE_STAGES).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).values


def main() -> None:
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    tr_scores = compute_dgp_score(tr)
    te_scores = compute_dgp_score(te)

    oof_main = np.load(ART / "oof_xgb_dist_routed.npy")
    test_main = np.load(ART / "test_xgb_dist_routed.npy")
    oof_spec = np.load(ART / "oof_xgb_spec_678.npy")
    test_spec = np.load(ART / "test_xgb_spec_678.npy")

    tr_spec_mask = np.isin(tr_scores, SPEC_SCORES)
    te_spec_mask = np.isin(te_scores, SPEC_SCORES)
    print(f"rows where specialist overrides main:  "
          f"train {tr_spec_mask.sum()} / test {te_spec_mask.sum()}")

    oof_hybrid = oof_main.copy()
    oof_hybrid[tr_spec_mask] = oof_spec[tr_spec_mask]
    test_hybrid = test_main.copy()
    test_hybrid[te_spec_mask] = test_spec[te_spec_mask]

    _, main_bal = tune_log_bias(oof_main, y, prior)
    bias, hyb_bal = tune_log_bias(oof_hybrid, y, prior)

    print(f"main routed-{{1,2}} tuned OOF:       {main_bal:.5f}")
    print(f"hybrid (+ spec on {{6,7,8}}) tuned:   {hyb_bal:.5f}")
    print(f"Δ hybrid - main:                      {hyb_bal - main_bal:+.5f}")
    print(f"bias = {dict(zip(CLASSES, bias.round(4)))}")

    cm = confusion_matrix(
        y, (np.log(np.clip(oof_hybrid, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    print(f"hybrid OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_xgb_hybrid_routed_spec.npy", oof_hybrid)
    np.save(ART / "test_xgb_hybrid_routed_spec.npy", test_hybrid)
    with open(ART / "hybrid_routed_spec_results.json", "w") as f:
        json.dump({
            "main_routed_tuned": float(main_bal),
            "hybrid_tuned": float(hyb_bal),
            "delta": float(hyb_bal - main_bal),
            "log_bias": bias.tolist(),
            "spec_scores": list(SPEC_SCORES),
            "n_train_overridden": int(tr_spec_mask.sum()),
            "n_test_overridden": int(te_spec_mask.sum()),
        }, f, indent=2)

    tuned_idx = (np.log(np.clip(test_hybrid, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        SUB / "submission_xgb_hybrid_routed_spec.csv", index=False
    )
    print(f"submission -> {SUB/'submission_xgb_hybrid_routed_spec.csv'}")


if __name__ == "__main__":
    main()
