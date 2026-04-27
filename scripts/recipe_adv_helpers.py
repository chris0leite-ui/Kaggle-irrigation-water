"""Adversarial-robustness recipe XGB: helpers.

Mechanism (different from prior TTA / mixup / SMOTE / cleanlab):
  Perturb the 11 RAW numeric columns (Soil_Moisture, Rainfall_mm, ...) by
  σ × IQR Gaussian noise on training rows ONLY, AFTER FE has been computed
  from clean values. Derived features (rule flags, OTE, digits, dist) stay
  clean; XGB sees noisy raw numerics alongside clean rule-derived columns.
  Forces splits on raw numeric axes to be robust to small perturbations
  (different from tree-axis-aligned-step lever that closed inference TTA).

Single-pass noise (K=1). No row duplication. Simply replaces tr's raw
numerics with perturbed copies right before model.fit. Test inference
uses CLEAN raw numerics — robustness is purely a training-time property.

Distinct from:
  - Inference TTA (Mech A in CLAUDE.md): inference-side perturbation, NULL
    via boundary-confined family. This is training-side.
  - Mixup (Angle C2/C3): pair-based interpolation, magnitude trap or
    redundancy. This is single-row perturbation.
  - SMOTE-NC: minority oversampling via k-NN; this is symmetric noise.
  - Cleanlab downweight: row-importance reweighting. This perturbs values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 11 raw numerics in the competition feature set.
RAW_NUMS = [
    "Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm",
    "Wind_Speed_kmh", "Soil_pH", "Organic_Carbon", "Electrical_Conductivity",
    "Sunlight_Hours", "Field_Area_hectare", "Previous_Irrigation_mm",
]


def compute_iqrs(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Per-column IQR from a clean reference frame."""
    out = {}
    for c in cols:
        v = df[c].astype(float).to_numpy()
        q1, q3 = np.quantile(v, 0.25), np.quantile(v, 0.75)
        out[c] = float(q3 - q1)
    return out


def perturb_train_inplace(X_tr: pd.DataFrame, iqrs: dict[str, float],
                           sigma: float, rng: np.random.Generator) -> None:
    """Add σ × IQR Gaussian noise to raw numeric columns in-place.

    Only modifies the 11 raw numeric columns. Derived features
    (digits / OTE / dist / rule flags / freq / orig_stats / etc.) remain
    untouched because they were computed pre-fold from clean values.
    """
    for c in RAW_NUMS:
        if c not in X_tr.columns:
            continue
        scale = sigma * iqrs.get(c, 0.0)
        if scale <= 0:
            continue
        clean = X_tr[c].astype(np.float32).to_numpy()
        noise = rng.normal(0, scale, size=len(clean)).astype(np.float32)
        X_tr[c] = clean + noise
