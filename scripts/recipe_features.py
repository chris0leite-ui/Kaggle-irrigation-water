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


def add_domain_interactions(df: pd.DataFrame) -> list[str]:
    """11 ratio/product features from utaazu kernel (0.979 CV).

    Mutates df in-place. All safe additions (+1/+0.1 denominators prevent
    div-by-zero). No groupby or cross-split stats — purely row-wise.
    """
    d = df
    d["moist_rain"]    = d["Soil_Moisture"]  / (d["Rainfall_mm"]      + 1)
    d["moist_temp"]    = d["Soil_Moisture"]  / (d["Temperature_C"]    + 1)
    d["moist_wind"]    = d["Soil_Moisture"]  / (d["Wind_Speed_kmh"]   + 1)
    d["ET_proxy"]      = (d["Temperature_C"] * d["Wind_Speed_kmh"] *
                          d["Sunlight_Hours"]) / (d["Humidity"]       + 1)
    d["heat_stress"]   = d["Temperature_C"]  * d["Sunlight_Hours"]
    d["drying_force"]  = (d["Wind_Speed_kmh"] * d["Temperature_C"]) / (
                          d["Humidity"] + 1)
    d["water_supply"]  = d["Rainfall_mm"]    + d["Previous_Irrigation_mm"]
    d["water_deficit"] = d["Soil_Moisture"]  - d["water_supply"] * 0.1
    d["soil_quality"]  = d["Organic_Carbon"] / (d["Electrical_Conductivity"]
                                                + 0.1)
    d["moist_x_temp"]  = d["Soil_Moisture"]  * d["Temperature_C"]
    d["wind_x_temp"]   = d["Wind_Speed_kmh"] * d["Temperature_C"]
    return ["moist_rain", "moist_temp", "moist_wind", "ET_proxy",
            "heat_stress", "drying_force", "water_supply", "water_deficit",
            "soil_quality", "moist_x_temp", "wind_x_temp"]


def add_decimal_fractions(df: pd.DataFrame,
                          nums: list[str] | None = None) -> list[str]:
    """Extract 2-dp decimal fraction `(col % 1).round(2)` per numeric.

    Structurally distinct from digit extraction: digit FE captures INTEGER
    digit positions, this captures the FRACTIONAL portion. Float rounding
    noise below the 2nd decimal is suppressed via .round(2).
    """
    if nums is None:
        nums = ["Temperature_C", "Organic_Carbon", "Soil_Moisture",
                "Soil_pH", "Sunlight_Hours"]
    cols: list[str] = []
    for c in nums:
        if c not in df.columns:
            continue
        name = f"{c}_dec"
        df[name] = (df[c] % 1).round(2).astype(np.float32)
        cols.append(name)
    return cols


def add_groupby_cat_num_stats(train: pd.DataFrame, test: pd.DataFrame,
                               cats: list[str], nums: list[str],
                               stats=("mean", "std")) -> list[str]:
    """Per-cat-group mean/std of each numeric, computed on SYNTHETIC TRAIN only.

    Port of rohit8527kmr7518/ps-s6e4-lgbm-with-target-encoding-group-stats:
    for each (cat, num) pair, group train by cat → aggregate num via mean/std,
    merge back onto train and test. Leak-free for the synthetic train fold
    structure (computed on FULL train once, not per-fold; but downstream
    target encoding handles the leak path via OrderedTE).

    Distinct from `add_orig_mean_std` (which aggregates TARGET on 10k
    original) — this aggregates NUMERIC distributions on the full 630k
    synthetic pool. Mutates train, test in-place.

    Returns list of new numeric column names (2 × |cats| × |nums| = 176
    for 8 cats + 11 nums + mean/std).
    """
    new_cols: list[str] = []
    for c in cats:
        for n in nums:
            stats_df = train.groupby(c, observed=False)[n].agg(list(stats)).reset_index()
            stats_df.columns = [c] + [f"GBY_{c}_{n}_{s}" for s in stats]
            for df_name in ("train", "test"):
                df = {"train": train, "test": test}[df_name]
                merged = df.merge(stats_df, on=c, how="left")
                for s in stats:
                    col = f"GBY_{c}_{n}_{s}"
                    df[col] = merged[col].fillna(0).astype(np.float32).values
            new_cols += [f"GBY_{c}_{n}_{s}" for s in stats]
    return new_cols


def add_w8_block(df: pd.DataFrame, y: np.ndarray | None = None) -> list[str]:
    """W8 — novel-on-recipe FE block from 2026-04-25 LLM-FE smoke survivors.

    14 cols spanning ideas not already in `add_domain_interactions` /
    `add_decimal_fractions`:
      - I6 humidity_water (4 NEW): hum_x_dry, hum_minus_50, pi_x_norain, pi_x_dry
      - I9 soil_chemistry (3 NEW): ph_dev_neutral, oc_x_ph, ec_x_sm
        (ph_dev_optimal omitted — too similar to dev_neutral)
      - I16 kc_x_water (3): kc_x_dry, kc_x_norain, kc_x_sm_dist
      - I18 three_axis_interaction (3): dry_nor_hot, dry_nor_kc, all_critical
      - I20 + I21 per-score-band z-scores (2): humidity_z_in_score, prev_irrig_z_in_score

    Mutates df in-place. y argument unused (no target encoding here).
    """
    sm = df["Soil_Moisture"].astype(float).values
    rf = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    h  = df["Humidity"].astype(float).values
    pi = df["Previous_Irrigation_mm"].astype(float).values
    ph = df["Soil_pH"].astype(float).values
    oc = df["Organic_Carbon"].astype(float).values
    ec = df["Electrical_Conductivity"].astype(float).values
    mu = (df["Mulching_Used"].astype(str).values == "No").astype(np.float32)
    cgs = df["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(cgs, ("Flowering", "Vegetative")), 2.0, 0.0).astype(np.float32)
    dry = (sm < 25).astype(np.float32)
    nor = (rf < 300).astype(np.float32)
    hot = (tc > 30).astype(np.float32)
    win = (ws > 10).astype(np.float32)
    score = (2 * (dry + nor) + (hot + win + mu) + kc).astype(np.float32)

    # I6 (4 NEW — water_supply/deficit already in add_domain_interactions)
    df["W8_hum_x_dry"]    = (h * dry).astype(np.float32)
    df["W8_hum_minus_50"] = (h - 50).astype(np.float32)
    df["W8_pi_x_norain"]  = (pi * nor).astype(np.float32)
    df["W8_pi_x_dry"]     = (pi * dry).astype(np.float32)
    # I9 (3 NEW — soil_quality already in add_domain_interactions)
    df["W8_ph_dev_neutral"] = np.abs(ph - 7.0).astype(np.float32)
    df["W8_oc_x_ph"]        = (oc * ph).astype(np.float32)
    df["W8_ec_x_sm"]        = (ec * sm).astype(np.float32)
    # I16 (3)
    df["W8_kc_x_dry"]       = (kc * dry).astype(np.float32)
    df["W8_kc_x_norain"]    = (kc * nor).astype(np.float32)
    df["W8_kc_x_sm_dist"]   = (kc * (sm - 25)).astype(np.float32)
    # I18 (3)
    df["W8_dry_nor_hot"]    = (dry * nor * hot).astype(np.float32)
    df["W8_dry_nor_kc"]     = (dry * nor * (kc / 2)).astype(np.float32)
    df["W8_all_critical"]   = (dry * nor * hot * win * mu * (kc / 2)).astype(np.float32)
    # I20, I21 — per-score-band z-scores
    z_h = np.zeros(len(df), dtype=np.float32)
    z_pi = np.zeros(len(df), dtype=np.float32)
    s_int = score.astype(int)
    for sk in range(10):
        m = s_int == sk
        if m.sum() > 1:
            mu_h = h[m].mean(); sd_h = h[m].std() + 1e-6
            mu_p = pi[m].mean(); sd_p = pi[m].std() + 1e-6
            z_h[m] = (h[m] - mu_h) / sd_h
            z_pi[m] = (pi[m] - mu_p) / sd_p
    df["W8_humidity_z_in_score"] = z_h
    df["W8_prev_irrig_z_in_score"] = z_pi

    return [
        "W8_hum_x_dry", "W8_hum_minus_50", "W8_pi_x_norain", "W8_pi_x_dry",
        "W8_ph_dev_neutral", "W8_oc_x_ph", "W8_ec_x_sm",
        "W8_kc_x_dry", "W8_kc_x_norain", "W8_kc_x_sm_dist",
        "W8_dry_nor_hot", "W8_dry_nor_kc", "W8_all_critical",
        "W8_humidity_z_in_score", "W8_prev_irrig_z_in_score",
    ]


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
