"""Feature engineering for the ExcelFormer kernel.

ExcelFormer ONLY accepts numerical features (its semi-permutation-invariant
attention is defined per-numeric-column). The 8 categoricals are
label-encoded (factorized) to integer codes BEFORE building the
DataFrame, so all 19 features become numerical.

Encoding strategy: pd.factorize on the train+test+orig union for each
cat ensures consistent codes across all 3 data sources. Same approach
used in `nonrule_features_only.py` and `seed_bag_dist.py`.
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
RAW_NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon",
            "Electrical_Conductivity", "Temperature_C", "Humidity",
            "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
            "Field_Area_hectare", "Previous_Irrigation_mm"]
# After factorize, all features become NUMS:
NUMS = [f"{c}_code" for c in CATS] + RAW_NUMS


def find_one(root: Path, pattern: str) -> Path:
    for p in root.rglob(pattern):
        return p
    raise FileNotFoundError(f"no match for {pattern} under {root}")


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
    """Factorize cats → integer codes, treat all as numerics.

    pd.factorize on the union of train+test+orig ensures consistent codes
    (same value gets same code in all 3 sources). Cats become
    `<colname>_code` integer columns; original cat columns dropped.
    """
    print("[data] factorizing cats to numerics for ExcelFormer", flush=True)
    for c in CATS:
        train_str = train[c].astype(str)
        test_str = test[c].astype(str)
        orig_str = orig[c].astype(str)
        union = pd.concat([train_str, test_str, orig_str], ignore_index=True)
        codes_union, _ = pd.factorize(union)
        n_tr, n_te = len(train_str), len(test_str)
        train[f"{c}_code"] = codes_union[:n_tr].astype(np.float32)
        test[f"{c}_code"] = codes_union[n_tr:n_tr + n_te].astype(np.float32)
        orig[f"{c}_code"] = codes_union[n_tr + n_te:].astype(np.float32)
    for c in RAW_NUMS:
        for df in (train, test, orig):
            df[c] = df[c].astype(np.float32)
    keep_tr = NUMS + [TARGET]
    keep_te = NUMS
    return train[keep_tr].copy(), test[keep_te].copy(), orig[keep_tr].copy()
