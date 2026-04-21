"""Reconstruct hybrid_v3 (routed-{0,1,2} + specialist-{6,7,8}).

Same glue logic as hybrid_routed_spec.py but loads the v3 routing
variant (routed_v3 = drops training rows where dgp_score in {0,1,2}
from XGB's fit; at predict time, routes those to the rule).

Inputs (must exist):
  oof_xgb_dist_routed_v3.npy, test_xgb_dist_routed_v3.npy
  oof_xgb_spec_678.npy, test_xgb_spec_678.npy

Outputs:
  oof_xgb_hybrid_v3.npy, test_xgb_hybrid_v3.npy
  hybrid_v3_reconstruct_results.json
  submissions/submission_xgb_hybrid_v3_reconstructed.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
SPEC_SCORES = {6, 7, 8}


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).str.strip() == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).values


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid_def = np.linspace(-3.0, 3.0, 61)
    grid_hi = np.linspace(-3.0, 6.0, 91)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            grid = grid_hi if k == 2 else grid_def
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


def main():
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    tr_scores = compute_dgp_score(tr)
    te_scores = compute_dgp_score(te)

    oof_main = np.load(ART / "oof_xgb_dist_routed_v3.npy")
    test_main = np.load(ART / "test_xgb_dist_routed_v3.npy")
    oof_spec = np.load(ART / "oof_xgb_spec_678.npy")
    test_spec = np.load(ART / "test_xgb_spec_678.npy")

    tr_mask = np.isin(tr_scores, list(SPEC_SCORES))
    te_mask = np.isin(te_scores, list(SPEC_SCORES))
    print(f"spec override rows: train {tr_mask.sum()} test {te_mask.sum()}")

    oof = oof_main.copy(); oof[tr_mask] = oof_spec[tr_mask]
    test = test_main.copy(); test[te_mask] = test_spec[te_mask]

    _, main_bal = tune_log_bias(oof_main, y, prior)
    bias, hyb_bal = tune_log_bias(oof, y, prior)
    print(f"main routed-v3 tuned:  {main_bal:.5f}")
    print(f"hybrid v3 tuned:        {hyb_bal:.5f}  (Δ={hyb_bal - main_bal:+.5f})")
    print(f"bias = {dict(zip(CLASSES, bias.round(3)))}")

    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    print(f"OOF confusion:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_xgb_hybrid_v3.npy", oof)
    np.save(ART / "test_xgb_hybrid_v3.npy", test)
    with open(ART / "hybrid_v3_reconstruct_results.json", "w") as f:
        json.dump({
            "main_routed_v3_tuned": float(main_bal),
            "hybrid_v3_tuned": float(hyb_bal),
            "delta": float(hyb_bal - main_bal),
            "log_bias": bias.tolist(),
            "spec_scores": list(SPEC_SCORES),
            "n_train_overridden": int(tr_mask.sum()),
            "n_test_overridden": int(te_mask.sum()),
        }, f, indent=2)

    tuned_idx = (np.log(np.clip(test, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        SUB / "submission_xgb_hybrid_v3_reconstructed.csv", index=False
    )
    print(f"submission -> {SUB/'submission_xgb_hybrid_v3_reconstructed.csv'}")


if __name__ == "__main__":
    main()
