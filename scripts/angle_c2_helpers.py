"""Pair generation + mixup synthesis for Angle C2.

Three improvements over Angle C:
  1. Drop Medium↔High pairs entirely (Medium-protection: M↔H mixup
     dragged Medium recall −0.046 in v1).
  2. Confidence-gate pairs: only generate when primary's max_prob is
     < 0.95 on at least one donor (skip clean-confident rows).
  3. K=1 + β(0.2, 0.2) for sharper labels closer to one parent.

The recipe FE pipeline runs on (train + mixup_rows, test, orig)
transparently — combos / digits / num_as_cat / freq / orig_stats are
all per-row deterministic on raw cats + nums, which mixup preserves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUM_COLS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
            "Soil_pH", "Organic_Carbon", "Electrical_Conductivity",
            "Humidity", "Sunlight_Hours", "Field_Area_hectare",
            "Previous_Irrigation_mm"]
CAT_COLS = ["Region", "Crop_Type", "Soil_Type", "Crop_Growth_Stage",
            "Mulching_Used", "Season", "Irrigation_Type", "Water_Source"]


def cell_id(df: pd.DataFrame) -> np.ndarray:
    """6-bit rule-cell packing: dry|nor|hot|win|nom|kc."""
    dry = (df["Soil_Moisture"].values < 25).astype(np.int8)
    nor = (df["Rainfall_mm"].values < 300).astype(np.int8)
    hot = (df["Temperature_C"].values > 30).astype(np.int8)
    win = (df["Wind_Speed_kmh"].values > 10).astype(np.int8)
    nom = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stg = df["Crop_Growth_Stage"].astype(str).values
    kc = np.isin(stg, ("Flowering", "Vegetative")).astype(np.int8)
    return (dry | (nor << 1) | (hot << 2) | (win << 3)
            | (nom << 4) | (kc << 5)).astype(np.int8)


def build_pairs_v2(train: pd.DataFrame, y: np.ndarray, primary_max: np.ndarray,
                   rng: np.random.Generator, conf_thresh: float = 0.95,
                   cap_per_cell: int = 4000, drop_mh: bool = True
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Filtered cross-class within-cell pairs.

    - Drops M↔H if drop_mh (y∈{1,2} both, but different).
    - Keeps only pairs where min(primary_max[i], primary_max[j]) < conf_thresh.
    """
    cell = cell_id(train)
    pi_l, pj_l = [], []
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        if len(idx) < 2:
            continue
        if len(idx) > cap_per_cell:
            idx = rng.choice(idx, size=cap_per_cell, replace=False)
        ys = y[idx]
        i2 = idx[rng.permutation(len(idx))]
        cross = ys != y[i2]
        if drop_mh:
            mh = ((ys == 1) & (y[i2] == 2)) | ((ys == 2) & (y[i2] == 1))
            cross = cross & ~mh
        # confidence gate: at least one donor uncertain
        conf_keep = np.minimum(primary_max[idx], primary_max[i2]) < conf_thresh
        cross = cross & conf_keep
        if cross.sum() == 0:
            continue
        pi_l.append(idx[cross]); pj_l.append(i2[cross])
    if not pi_l:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(pi_l), np.concatenate(pj_l)


def synthesize_mixup(train: pd.DataFrame, y: np.ndarray, pi: np.ndarray,
                     pj: np.ndarray, rng: np.random.Generator,
                     k: int = 1, beta_a: float = 0.2):
    """Build K mixup rows per pair on raw frame (numerics + cats).

    Returns:
      mixed (DataFrame): same columns as train (numerics + cats, no target)
      hard_y (int array): argmax of soft label
      conf_w (float array): max(soft_label) — confidence weight
      pi_r, pj_r (arrays): donor indices repeated K times
    """
    if len(pi) == 0:
        empty = pd.DataFrame(columns=NUM_COLS + CAT_COLS)
        return empty, np.array([], dtype=np.int64), np.array([], dtype=np.float32), \
               np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    n = len(pi) * k
    alpha = rng.beta(beta_a, beta_a, size=n).astype(np.float32)
    pi_r = np.repeat(pi, k); pj_r = np.repeat(pj, k)
    mixed = pd.DataFrame()
    for c in NUM_COLS:
        a = train[c].values[pi_r].astype(np.float32)
        b = train[c].values[pj_r].astype(np.float32)
        mixed[c] = (1 - alpha) * a + alpha * b
    for c in CAT_COLS:
        sel = rng.random(n) < alpha
        mixed[c] = np.where(sel, train[c].values[pj_r], train[c].values[pi_r])
    onehot = np.eye(3, dtype=np.float32)
    soft = (1 - alpha)[:, None] * onehot[y[pi_r]] + alpha[:, None] * onehot[y[pj_r]]
    hard_y = soft.argmax(1).astype(np.int64)
    conf_w = soft.max(1).astype(np.float32)
    return mixed, hard_y, conf_w, pi_r, pj_r
