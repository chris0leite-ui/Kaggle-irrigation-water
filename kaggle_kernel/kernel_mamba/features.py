"""Feature engineering for the Mamba kernel.

Same 19 raw features (8 cats + 11 nums) as RealMLP / Trompt kernels —
keeps Jaccard apples-to-apples vs the existing NN-family blend leg
(realmlp.npy on disk). Mamba treats the row as a sequence; cat
embeddings + numerical features are tokenised inside mambular.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42

CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon",
        "Electrical_Conductivity", "Temperature_C", "Humidity",
        "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
        "Field_Area_hectare", "Previous_Irrigation_mm"]


def find_one(root: Path, pattern_lc: str) -> Path:
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == pattern_lc:
            return p
    raise FileNotFoundError(f"no match for {pattern_lc} under {root}")


def load_data(kaggle_input: Path, smoke: bool):
    print("[data] loading train / test / orig", flush=True)
    train = pd.read_csv(find_one(kaggle_input, "train.csv"))
    test = pd.read_csv(find_one(kaggle_input, "test.csv"))
    orig = pd.read_csv(find_one(kaggle_input, "irrigation_prediction.csv"))
    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    for df in (train, test):
        df.drop(columns=["id"], inplace=True, errors="ignore")
    if smoke:
        print("[data] SMOKE=1 - subsampling", flush=True)
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test_sub = test.sample(10_000, random_state=SEED)
        test_ids = test_ids[test_sub.index.to_numpy()]
        test = test_sub.reset_index(drop=True)
    print(f"[data] train={len(train):,} test={len(test):,} "
          f"orig={len(orig):,}", flush=True)
    return train, test, orig, test_ids


def build_frame(train: pd.DataFrame, test: pd.DataFrame,
                orig: pd.DataFrame):
    """cats as strings (mambular handles dtype detection),
    nums as float32. Drop unused columns. Returns (train, test, orig)
    with only CATS + NUMS + TARGET remaining.
    """
    for c in CATS:
        for df in (train, test, orig):
            df[c] = df[c].astype(str)
    for c in NUMS:
        for df in (train, test, orig):
            df[c] = df[c].astype(np.float32)
    keep_tr = CATS + NUMS + [TARGET]
    keep_te = CATS + NUMS
    return train[keep_tr].copy(), test[keep_te].copy(), orig[keep_tr].copy()
