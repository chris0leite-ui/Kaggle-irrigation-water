"""Feature builders for the NN pipeline.

Two passes over the raw competition frame:
  - `add_distance_features`: mirrors scripts/benchmark_dist.py exactly.
  - `add_digit_features_inline`: mirrors scripts/digit_features.py exactly.

Kept self-contained so the Kaggle kernel has no repo-relative imports.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ACTIVE_STAGES = ("Flowering", "Vegetative")
RAW_NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
DIGIT_POS = (-3, -2, -1, 0, 1, 2, 3)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    kc = np.where(np.isin(out["Crop_Growth_Stage"].astype(str).values,
                          ACTIVE_STAGES), 2, 0).astype(np.int8)
    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    for name, v in (("sm_abs", out["sm_dist"].values),
                    ("rf_abs", out["rf_dist"].values),
                    ("tc_abs", out["tc_dist"].values),
                    ("ws_abs", out["ws_dist"].values)):
        out[name] = np.abs(v).astype(np.float32)
    out["dry"] = dry; out["norain"] = norain; out["hot"] = hot
    out["windy"] = windy; out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0,
                                np.where(score <= 6, 1, 2)).astype(np.int8)
    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)
    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)
    return out


def _digit_at(values: np.ndarray, pos: int) -> np.ndarray:
    scale = 10.0 ** (-pos)
    return (np.floor(values.astype(np.float64) * scale + 1e-9).astype(np.int64)
            % 10).astype(np.int8)


def add_digit_features_inline(df: pd.DataFrame) -> list[str]:
    """Add digit cols in place; return the list of digit col names."""
    cols = []
    for c in RAW_NUMERIC:
        v = df[c].astype(float).values
        for d in DIGIT_POS:
            name = f"dig_{c}_{d}" if d >= 0 else f"dig_{c}_n{-d}"
            df[name] = _digit_at(v, d)
            cols.append(name)
    return cols


def drop_const_digit_cols(tr: pd.DataFrame, te: pd.DataFrame,
                          digit_cols: list[str]) -> list[str]:
    keep, drop = [], []
    for c in digit_cols:
        if tr[c].nunique(dropna=False) > 1:
            keep.append(c)
        else:
            drop.append(c)
    if drop:
        tr.drop(columns=drop, inplace=True)
        te.drop(columns=drop, inplace=True)
    return keep
