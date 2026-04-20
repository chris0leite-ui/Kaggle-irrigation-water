"""Closed-form DGP for the original Irrigation Prediction dataset.

Reverse-engineered from data/irrigation_prediction.csv: the label is a
deterministic function of 6 features (Soil_Moisture, Rainfall_mm,
Temperature_C, Wind_Speed_kmh, Mulching_Used, Crop_Growth_Stage).
Reaches 100% accuracy on all 10,000 rows.

Rule:
  dry     = Soil_Moisture < 25
  norain  = Rainfall_mm   < 300
  hot     = Temperature_C > 30
  windy   = Wind_Speed_kmh > 10
  nomulch = Mulching_Used == "No"
  Kc      = 2 if Crop_Growth_Stage in {Flowering, Vegetative} else 0

  score = 2*(dry + norain) + (hot + windy + nomulch) + Kc

  Low     if score <= 3
  Medium  if 4 <= score <= 6
  High    if score >= 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVE_STAGES = ("Flowering", "Vegetative")
LABELS = ("Low", "Medium", "High")


def dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"].astype(float).values < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float).values < 300).astype(int)
    hot = (df["Temperature_C"].astype(float).values > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float).values > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(int)
    kc = np.where(np.isin(df["Crop_Growth_Stage"].astype(str).values, ACTIVE_STAGES), 2, 0)
    return 2 * (dry + norain) + (hot + windy + nomulch) + kc


def dgp_predict(df: pd.DataFrame) -> np.ndarray:
    s = dgp_score(df)
    return np.where(s <= 3, "Low", np.where(s <= 6, "Medium", "High"))


def evaluate(path: Path) -> None:
    df = pd.read_csv(path, dtype_backend="numpy_nullable")
    pred = dgp_predict(df)
    if "Irrigation_Need" in df.columns:
        y = df["Irrigation_Need"].astype(str).values
        acc = (pred == y).mean()
        print(f"{path.name}: n={len(df)}  accuracy={acc:.6f}")
        if acc < 1.0:
            mism = pd.DataFrame({"pred": pred, "actual": y, "score": dgp_score(df)})
            bad = mism[mism["pred"] != mism["actual"]]
            print(f"  {len(bad)} mismatches; label dist of mismatches:")
            print(bad["actual"].value_counts().to_string())
    else:
        print(f"{path.name}: n={len(df)}  (no label col; first 5 preds: {pred[:5].tolist()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        default=["data/irrigation_prediction.csv", "data/train.csv"],
        help="CSV files to score with the closed-form DGP",
    )
    args = ap.parse_args()
    for p in args.paths:
        path = Path(p)
        if path.exists():
            evaluate(path)
        else:
            print(f"{p}: missing (skipped)")
