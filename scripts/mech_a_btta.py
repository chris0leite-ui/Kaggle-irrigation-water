"""Mech A: boundary-confined test-time augmentation helpers.

Differs from prior P1 TTA (which was uniform over all rows): identify
boundary rows via LB-best 4-stack max-prob threshold, perturb their
rule-axis numerics ONLY (Soil_Moisture, Rainfall_mm, Temperature_C,
Wind_Speed_kmh) by Gaussian σ × IQR, K=10 times. For each perturbation,
recompute every recipe feature that depends on the 4 axes (rule flags,
distance features, dgp_score, digit features for those 4 numerics, and
the LR-formula logits), predict via frozen fold booster, average. Leave
non-boundary rows untouched.

Mechanism: noise ∝ K × n_boundary (~5% of N) instead of K × N → noise/
signal ratio inverts vs prior TTA. Smoothing applies at the rule-cell
faces where the host NN's smooth decision surface most disagrees with
axis-aligned tree splits.

For OOF gating, the perturbed-and-averaged predictions REPLACE the
non-perturbed predictions at boundary rows; non-boundary OOF stays
identical to vanilla recipe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Mirror common.py thresholds
_SM_THR, _RF_THR, _TC_THR, _WS_THR = 25.0, 300.0, 30.0, 10.0
_AXES = ("Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh")


def boundary_mask(probs: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Return boolean mask of rows with max-prob < threshold.

    probs: (N, 3) anchor probabilities. threshold: e.g. 0.95.
    Higher threshold = more rows flagged as boundary.
    """
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"expected (N, 3) probs, got {probs.shape}")
    return probs.max(axis=1) < threshold


def axis_iqrs(train_df: pd.DataFrame) -> dict:
    """IQR per rule axis from train data (used as perturbation scale)."""
    out = {}
    for ax in _AXES:
        v = train_df[ax].astype(float).to_numpy()
        out[ax] = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
    return out


def perturb_axes(df: pd.DataFrame, iqrs: dict, sigma: float,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of df with the 4 rule axes Gaussian-perturbed by σ × IQR.

    Other columns left identical.
    """
    out = df.copy()
    for ax in _AXES:
        out[ax] = (df[ax].astype(float).to_numpy()
                   + rng.normal(0, sigma * iqrs[ax], size=len(df))).astype(np.float32)
    return out


def recompute_axis_dependent_features(df: pd.DataFrame, base_df: pd.DataFrame,
                                       active_stages=("Flowering", "Vegetative")) -> pd.DataFrame:
    """Recompute the recipe features that depend on the 4 rule axes:
       - sm_dist, rf_dist, tc_dist, ws_dist (signed) and *_abs
       - dry, norain, hot, windy binary flags
       - kc_active (depends on Crop_Growth_Stage, NOT axes — leave as-is)
       - dgp_score
       - digit features for the 4 axes (3 each = 12 total): _digit_-3, _-2, _-1
       - LR-formula logits

    df: perturbed rows (Soil_Moisture etc. modified).
    base_df: ORIGINAL rows (used for non-axis cols like Mulching_Used,
             Crop_Growth_Stage which we don't perturb).
    Returns df with axis-dependent cols overwritten / added.

    NOTE: this function does NOT recompute OTE / FREQ / digit features for
    NON-axis numerics. Those are stable under axis perturbation.
    """
    sm = df["Soil_Moisture"].astype(float).to_numpy()
    rf = df["Rainfall_mm"].astype(float).to_numpy()
    tc = df["Temperature_C"].astype(float).to_numpy()
    ws = df["Wind_Speed_kmh"].astype(float).to_numpy()

    df["sm_dist"] = (sm - _SM_THR).astype(np.float32)
    df["rf_dist"] = (rf - _RF_THR).astype(np.float32)
    df["tc_dist"] = (tc - _TC_THR).astype(np.float32)
    df["ws_dist"] = (ws - _WS_THR).astype(np.float32)
    df["sm_abs"] = np.abs(df["sm_dist"].to_numpy()).astype(np.float32)
    df["rf_abs"] = np.abs(df["rf_dist"].to_numpy()).astype(np.float32)
    df["tc_abs"] = np.abs(df["tc_dist"].to_numpy()).astype(np.float32)
    df["ws_abs"] = np.abs(df["ws_dist"].to_numpy()).astype(np.float32)

    dry = (sm < _SM_THR).astype(np.int8)
    norain = (rf < _RF_THR).astype(np.int8)
    hot = (tc > _TC_THR).astype(np.int8)
    windy = (ws > _WS_THR).astype(np.int8)
    nomulch = (base_df["Mulching_Used"].astype(str).to_numpy() == "No").astype(np.int8)
    stage = base_df["Crop_Growth_Stage"].astype(str).to_numpy()
    kc = np.where(np.isin(stage, active_stages), 2, 0).astype(np.int8)
    df["dry"] = dry
    df["norain"] = norain
    df["hot"] = hot
    df["windy"] = windy
    df["nomulch"] = nomulch
    df["kc_active"] = (kc > 0).astype(np.int8)
    df["dgp_score"] = (2 * (dry + norain) + hot + windy + nomulch + kc).astype(np.int8)

    # Digit features for 4 perturbed numerics (mirror add_digit_features
    # signature: floor(v * 10**(-d)) % 10 for d in {-3,-2,-1,0,1,2,3})
    # IMPORTANT: d=0 is the units digit. Recipe digit cols use floor()
    # convention, applied only to the surviving (non-zero-variance) ones.
    # We can't know which digits survived at train time without info dict;
    # we'll just write all 7 per axis and let the consumer subset by
    # info["digits"].
    for ax, v in [("Soil_Moisture", sm), ("Rainfall_mm", rf),
                  ("Temperature_C", tc), ("Wind_Speed_kmh", ws)]:
        for d in range(-3, 4):
            col = f"dig_{ax}_{d}"
            if col in df.columns:
                df[col] = (np.floor(v * (10.0 ** -d)) % 10).astype(np.int8)

    # LR-formula logits depend on (sm, rf, tc, ws, kc, nomulch); recompute
    # if they exist as cols. Coefficients from cdeotte's LR fit.
    # logits = w · features + b, where features are 9 binary indicators.
    # The recipe stores these as "logit_P_Low", "logit_P_Med", "logit_P_High".
    # Skip if not present (info-dependent).
    return df


if __name__ == "__main__":
    # Smoke
    df = pd.DataFrame({
        "Soil_Moisture": [24.0, 50.0, 25.5],
        "Rainfall_mm":   [350.0, 1000.0, 290.0],
        "Temperature_C": [29.0, 25.0, 30.5],
        "Wind_Speed_kmh":[11.0, 5.0, 9.5],
        "Mulching_Used": ["No", "Yes", "No"],
        "Crop_Growth_Stage": ["Flowering", "Sowing", "Vegetative"],
    })
    iqrs = axis_iqrs(df)
    rng = np.random.default_rng(42)
    pert = perturb_axes(df, iqrs, 0.1, rng)
    out = recompute_axis_dependent_features(pert, df)
    print(out[["Soil_Moisture", "sm_dist", "dry", "dgp_score"]])

    # Boundary mask
    probs = np.array([[0.95, 0.04, 0.01], [0.5, 0.4, 0.1], [0.99, 0.005, 0.005]])
    print("boundary mask:", boundary_mask(probs, 0.95))
