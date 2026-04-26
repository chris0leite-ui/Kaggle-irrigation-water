"""Feature engineering for the KAN kernel.

Same 19 raw features (8 cats + 11 nums) as RealMLP / Trompt / Mamba —
keeps Jaccard apples-to-apples vs the existing NN-family blend legs.

For KAN: cats one-hot encoded, nums standardised (zero-mean, unit-var
on train+orig combined). KAN spline activations expect bounded inputs;
we use grid_range=[-1,1] in config and rely on spline regularisation
to handle outliers — no clip.
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


def build_arrays(train: pd.DataFrame, test: pd.DataFrame,
                 orig: pd.DataFrame):
    """One-hot cats + standardise nums on (train ∪ orig).

    Returns (X_train, X_test, X_orig, y_train, y_orig, feat_dim).
    All arrays are float32 numpy.
    """
    # Cats: build categorical maps from union of train/test/orig values.
    cat_arrays = {"train": [], "test": [], "orig": []}
    for c in CATS:
        vals = sorted(set(train[c].astype(str)) |
                      set(test[c].astype(str)) |
                      set(orig[c].astype(str)))
        idx = {v: i for i, v in enumerate(vals)}
        K = len(vals)
        for tag, df in (("train", train), ("test", test), ("orig", orig)):
            col = df[c].astype(str).map(idx).to_numpy()
            oh = np.zeros((len(df), K), dtype=np.float32)
            oh[np.arange(len(df)), col] = 1.0
            cat_arrays[tag].append(oh)
        print(f"[fe] cat {c}: card={K}", flush=True)

    # Nums: standardise on train ∪ orig (test stats unseen for fit).
    num_arrays = {"train": [], "test": [], "orig": []}
    fit_data = pd.concat([train[NUMS], orig[NUMS]], axis=0,
                         ignore_index=True)
    means = fit_data.mean().to_numpy(dtype=np.float32)
    stds = fit_data.std().to_numpy(dtype=np.float32)
    stds = np.where(stds < 1e-8, 1.0, stds)
    for tag, df in (("train", train), ("test", test), ("orig", orig)):
        x = df[NUMS].to_numpy(dtype=np.float32)
        num_arrays[tag] = (x - means) / stds

    # Stack: cats one-hot ++ nums.
    def stack(tag):
        return np.concatenate(cat_arrays[tag] + [num_arrays[tag]], axis=1)

    X_train = stack("train").astype(np.float32)
    X_test = stack("test").astype(np.float32)
    X_orig = stack("orig").astype(np.float32)
    y_train = train[TARGET].to_numpy(dtype=np.int64)
    y_orig = orig[TARGET].to_numpy(dtype=np.int64)
    feat_dim = X_train.shape[1]
    print(f"[fe] feat_dim={feat_dim}  X_train={X_train.shape} "
          f"X_test={X_test.shape} X_orig={X_orig.shape}", flush=True)
    return X_train, X_test, X_orig, y_train, y_orig, feat_dim
