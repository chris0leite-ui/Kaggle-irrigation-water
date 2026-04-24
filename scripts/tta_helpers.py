"""TTA helpers for threshold-axis perturbation.

Perturb the 4 rule-threshold numerics (Soil_Moisture, Rainfall_mm,
Temperature_C, Wind_Speed_kmh) with Gaussian noise sigma * feature_IQR,
then recompute ONLY the features directly derived from those numerics:
  - threshold flags (soil_lt_25, rain_lt_300, temp_gt_30, wind_gt_10)
  - LR-formula logits (logit_P_Low/Medium/High)
  - digit-position cols for those 4 numerics

Factorization-dependent features (OTE / FREQ / num_as_cat / combos) are
left at their unperturbed values. They carry row-identity signal; a
perturbed numeric would become an unknown key and degenerate to prior,
which adds noise rather than smoothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from recipe_features import _LOGIT_COEFS

THRESHOLD_NUMS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C",
                  "Wind_Speed_kmh"]


def compute_iqr(train_df: pd.DataFrame) -> dict[str, float]:
    """IQR per threshold numeric, measured on training data only."""
    out = {}
    for c in THRESHOLD_NUMS:
        q25, q75 = np.quantile(train_df[c].to_numpy(), [0.25, 0.75])
        out[c] = float(q75 - q25)
    return out


def perturb(raw_nums: dict[str, np.ndarray], sigma_iqr: float,
            iqr: dict[str, float], rng: np.random.Generator
            ) -> dict[str, np.ndarray]:
    """Return perturbed copies of the 4 threshold numerics.

    Noise scale per-feature: sigma_iqr * IQR[feature]. Raw arrays are
    not mutated.
    """
    n = len(next(iter(raw_nums.values())))
    out = {}
    for c in THRESHOLD_NUMS:
        noise = rng.normal(0.0, sigma_iqr * iqr[c], size=n).astype(np.float32)
        out[c] = (raw_nums[c] + noise).astype(np.float32)
    return out


def recompute_threshold_derived(
    perturbed: dict[str, np.ndarray],
    stage: np.ndarray, mulch: np.ndarray,
    digit_range=range(-4, 4),
) -> pd.DataFrame:
    """Rebuild the threshold-derived feature matrix from perturbed nums.

    Returns a DataFrame with columns:
      soil_lt_25 / rain_lt_300 / temp_gt_30 / wind_gt_10
      logit_P_Low / logit_P_Medium / logit_P_High
      {num}_digit{k} for num in THRESHOLD_NUMS, k in digit_range
    Caller is responsible for dropping digit cols that were filtered out
    during the original FE pass (by matching column names against the
    pre-computed "surviving digit cols" list from the original FE).
    """
    soil = (perturbed["Soil_Moisture"] < 25).astype(np.int8)
    rain = (perturbed["Rainfall_mm"] < 300).astype(np.int8)
    temp = (perturbed["Temperature_C"] > 30).astype(np.int8)
    wind = (perturbed["Wind_Speed_kmh"] > 10).astype(np.int8)

    out = {
        # Raw numerics — included so XGB's splits on the raw cols are also
        # smoothed. Otherwise the tree can bypass TTA via the un-perturbed
        # Soil_Moisture / Rainfall_mm / Temperature_C / Wind_Speed_kmh.
        "Soil_Moisture": perturbed["Soil_Moisture"],
        "Rainfall_mm": perturbed["Rainfall_mm"],
        "Temperature_C": perturbed["Temperature_C"],
        "Wind_Speed_kmh": perturbed["Wind_Speed_kmh"],
        "soil_lt_25": soil,
        "rain_lt_300": rain,
        "temp_gt_30": temp,
        "wind_gt_10": wind,
    }

    # LR logits — same formula as add_lr_formula_logits, rewritten vector-wise
    # with the perturbed flags.
    for cls, coefs in _LOGIT_COEFS.items():
        stage_vals = np.array(
            [coefs["stage"].get(s, 0.0) for s in stage], dtype=np.float32)
        mulch_vals = np.array(
            [coefs["mulch"].get(m, 0.0) for m in mulch], dtype=np.float32)
        logit = (coefs["bias"]
                 + coefs["soil_lt_25"] * soil
                 + coefs["rain_lt_300"] * rain
                 + coefs["temp_gt_30"] * temp
                 + coefs["wind_gt_10"] * wind
                 + stage_vals + mulch_vals)
        out[f"logit_P_{cls}"] = logit.astype(np.float32)

    # Digit features for the 4 perturbed numerics only.
    for c in THRESHOLD_NUMS:
        v = perturbed[c]
        for k in digit_range:
            out[f"{c}_digit{k}"] = (v // (10.0 ** k) % 10).astype(np.int8)

    return pd.DataFrame(out)


def apply_tta_override(X: pd.DataFrame, tta_df: pd.DataFrame,
                       surviving_digit_cols: set[str]) -> pd.DataFrame:
    """Replace threshold-derived columns in X with the TTA-perturbed versions.

    Mutates X in place for speed (call on a copy at the caller level).
    Only replaces columns that exist in X — handles the case where
    add_digit_features dropped some positions as test-constant.
    """
    for col in tta_df.columns:
        if col in X.columns:
            # digit cols may have been dropped during original FE; skip those.
            if col.endswith(tuple(f"_digit{k}" for k in range(-4, 4))):
                if col not in surviving_digit_cols:
                    continue
            X[col] = tta_df[col].values
    return X
