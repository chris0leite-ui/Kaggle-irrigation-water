"""Feature engineering blocks for the public-notebook recipe (XGB + OTE + FE).

Adapted from:
- cdeotte/original-data-exact-formula  (LR-formula coefs on 10k original)
- aliafzal9323/s6e4-0-978-xgb-cat-pairwise-te-magic  (cat-pair combos + TE_ORIG)
- yunsuxiaozi/pss6e4-xgb-cv-0-979805  (digit FE + OrderedTE)
- include4eto/ps6e4-xgb-cudf-pseudo-labels  (composed pipeline)

Public `build_feature_sets(train, test, orig, target='Irrigation_Need')` returns
  (train_fe, test_fe, orig_fe, feat_info) where feat_info carries the column
lists the downstream pipeline needs (NUMS, CATS, NEW_NUMS, NEW_CATS,
NUM_AS_CAT, TE_COLUMNS, TO_REMOVE). All transforms are deterministic and
leak-free (external aggregations are computed on orig only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations


# Chris Deotte's LR coefficients on the 10k original (logits per class), used
# verbatim as 3 numeric features so the tree can combine them with TE/digits.
# Source: include4eto/ps6e4-xgb-cudf-pseudo-labels cell-17.
_LOGIT_COEFS = {
    "Low":    dict(bias=16.3173,
                   soil_lt_25=-11.0237, temp_gt_30=-5.8559,
                   rain_lt_300=-10.8500, wind_gt_10=-5.8284,
                   stage=dict(Flowering=-5.4155, Harvest=5.5073,
                              Sowing=5.2299, Vegetative=-5.4617),
                   mulch=dict(No=-3.0014, Yes=2.8613)),
    "Medium": dict(bias=4.6524,
                   soil_lt_25=0.3290, temp_gt_30=-0.0204,
                   rain_lt_300=0.1542, wind_gt_10=0.0841,
                   stage=dict(Flowering=0.3586, Harvest=-0.1348,
                              Sowing=-0.3547, Vegetative=0.3334),
                   mulch=dict(No=0.1883, Yes=0.0142)),
    "High":   dict(bias=-20.9697,
                   soil_lt_25=10.6947, temp_gt_30=5.8763,
                   rain_lt_300=10.6958, wind_gt_10=5.7444,
                   stage=dict(Flowering=5.0569, Harvest=-5.3725,
                              Sowing=-4.8752, Vegetative=5.1283),
                   mulch=dict(No=2.8131, Yes=-2.8755)),
}


def add_threshold_flags(df: pd.DataFrame) -> list[str]:
    """Add the 4 DGP threshold boolean flags; mutates df in-place."""
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(np.int8)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(np.int8)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(np.int8)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(np.int8)
    return ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]


def add_lr_formula_logits(df: pd.DataFrame) -> list[str]:
    """Compute the 3 LR-formula logit features (Chris Deotte coefficients)."""
    stage = df["Crop_Growth_Stage"].astype(str).values
    mulch = df["Mulching_Used"].astype(str).values
    soil = df["soil_lt_25"].values
    temp = df["temp_gt_30"].values
    rain = df["rain_lt_300"].values
    wind = df["wind_gt_10"].values

    cols: list[str] = []
    for cls, coefs in _LOGIT_COEFS.items():
        logit = (coefs["bias"]
                 + coefs["soil_lt_25"] * soil
                 + coefs["temp_gt_30"] * temp
                 + coefs["rain_lt_300"] * rain
                 + coefs["wind_gt_10"] * wind)
        stage_vals = np.array([coefs["stage"].get(s, 0.0) for s in stage])
        mulch_vals = np.array([coefs["mulch"].get(m, 0.0) for m in mulch])
        name = f"logit_P_{cls}"
        df[name] = (logit + stage_vals + mulch_vals).astype(np.float32)
        cols.append(name)
    return cols


def add_cat_pair_combos(train: pd.DataFrame, test: pd.DataFrame,
                        orig: pd.DataFrame, cats: list[str]) -> list[str]:
    """Concat every (c1, c2) cat pair into a single string-encoded combo col.

    Assigns the same integer code across train+test+orig (factorized on the
    concatenation). Returns list of new combo column names.
    """
    new_cols: list[str] = []
    for c1, c2 in combinations(cats, 2):
        col = f"COMBO_{c1}_{c2}"
        for df in (train, test, orig):
            df[col] = df[c1].astype(str) + "_" + df[c2].astype(str)
        combined = pd.concat([train[col], test[col], orig[col]])
        codes, _ = pd.factorize(combined)
        split_tr = len(train)
        split_te = split_tr + len(test)
        train[col] = codes[:split_tr]
        test[col] = codes[split_tr:split_te]
        orig[col] = codes[split_te:]
        new_cols.append(col)
    return new_cols


def add_digit_features(train: pd.DataFrame, test: pd.DataFrame,
                       orig: pd.DataFrame, nums: list[str],
                       digit_range=range(-4, 4)) -> list[str]:
    """Extract digit-position features for every numeric column.

    `floor(v * 10^(-k)) mod 10` for k in `digit_range` (default -4..+3).
    Drops positions that are constant across the test set.
    """
    cols: list[str] = []
    for c in nums:
        for k in digit_range:
            name = f"{c}_digit{k}"
            for df in (train, test, orig):
                df[name] = (df[c] // (10.0 ** k) % 10).astype("int8")
            cols.append(name)
    # Drop positions that are constant on test (zero variance — no signal).
    drop = [c for c in cols if test[c].nunique() == 1]
    for c in drop:
        for df in (train, test, orig):
            df.drop(columns=[c], inplace=True)
    return [c for c in cols if c not in drop]


def add_freq_features(train: pd.DataFrame, test: pd.DataFrame,
                      orig: pd.DataFrame, cats: list[str]) -> list[str]:
    """Per-cat relative-frequency computed on the combined (train+test+orig).

    Uses float32 for memory. Tolerant of unseen categories (fill 0).
    """
    new_cols: list[str] = []
    for c in cats:
        freq = pd.concat([train[c], test[c], orig[c]]).value_counts(normalize=True)
        name = f"FREQ_{c}"
        for df in (train, test, orig):
            df[name] = df[c].map(freq).fillna(0).astype(np.float32)
        new_cols.append(name)
    return new_cols


def add_orig_mean_std(train: pd.DataFrame, test: pd.DataFrame,
                      orig: pd.DataFrame, cols_to_aggregate: list[str],
                      target: str) -> list[str]:
    """For each col, groupby its value on ORIG and compute mean+std of target.

    Join those aggregated values onto train/test. Fill unseen with (0.5, 0).
    This is leak-free for the synthetic train because orig is an external
    dataset. Returns the names of the 2*len(cols) new numeric features.
    """
    new_cols: list[str] = []
    for c in cols_to_aggregate:
        stats = orig.groupby(c)[target].agg(["mean", "std"]).reset_index()
        stats.columns = [c,
                         f"ORIG_{c}_mean",
                         f"ORIG_{c}_std"]
        for df_name in ("train", "test"):
            df = {"train": train, "test": test}[df_name]
            merged = df.merge(stats, on=c, how="left")
            df[f"ORIG_{c}_mean"] = merged[f"ORIG_{c}_mean"].fillna(0.5).astype(np.float32).values
            df[f"ORIG_{c}_std"]  = merged[f"ORIG_{c}_std"].fillna(0).astype(np.float32).values
        new_cols += [f"ORIG_{c}_mean", f"ORIG_{c}_std"]
    return new_cols


def add_num_as_cat(train: pd.DataFrame, test: pd.DataFrame,
                   orig: pd.DataFrame, nums: list[str]) -> list[str]:
    """Duplicate each numeric as a string-cast categorical column.

    Factorized across train+test+orig so codes are consistent. These columns
    are intended for target-encoding, not raw XGB input.
    """
    new_cols: list[str] = []
    for c in nums:
        name = f"CAT_{c}"
        for df in (train, test, orig):
            df[name] = df[c].astype(str)
        combined = pd.concat([train[name], test[name], orig[name]])
        codes, _ = pd.factorize(combined)
        s = len(train)
        t = s + len(test)
        train[name] = codes[:s]
        test[name] = codes[s:t]
        orig[name] = codes[t:]
        new_cols.append(name)
    return new_cols
