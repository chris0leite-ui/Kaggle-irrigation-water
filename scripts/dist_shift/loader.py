"""Shared loader for distribution-shift research.

Returns three frames with the SAME column ordering and a `source`
column added (so we can `pd.concat` for adversarial validation).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARTI = ROOT / "scripts" / "artifacts" / "dist_shift"
ARTI.mkdir(parents=True, exist_ok=True)

NUMS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
TARGET = "Irrigation_Need"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv").drop(columns=["id"])
    test = pd.read_csv(DATA / "test.csv").drop(columns=["id"])
    with zipfile.ZipFile(DATA / "archive.zip") as z:
        with z.open(z.namelist()[0]) as f:
            orig = pd.read_csv(f)
    return train, test, orig


def with_source() -> pd.DataFrame:
    """Stack orig + train + test with a `source` label."""
    train, test, orig = load()
    orig["source"] = "orig"
    train["source"] = "train"
    test["source"] = "test"
    cols = ["source"] + NUMS + CATS + [TARGET]
    # test has no target; pad with NaN so we can concat uniformly
    test[TARGET] = pd.NA
    return pd.concat([orig[cols], train[cols], test[cols]], ignore_index=True)
