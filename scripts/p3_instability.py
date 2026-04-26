"""P3 counterfactual rule-instability features.

For each row, perturb each rule axis (Soil_Moisture, Rainfall_mm,
Temperature_C, Wind_Speed_kmh) by relative deltas {±2%, ±5%, ±10%, ±20%}.
For each perturbation, recompute the dgp_score and check whether it
differs from the base score.

Mechanism (P3 from 2026-04-26 EDA proposals):
  Existing per-axis distance features (sm_dist, rf_dist, ...) capture
  closeness-to-threshold on ONE axis at a time. They do NOT capture
  multi-axis simultaneous closeness — when a row sits near multiple
  rule-cell boundaries, the host NN's smooth decision surface is most
  likely to disagree with the discrete rule. Instability quantifies
  this joint position directly via cell-flip count under perturbation.

Output features (5 total, all int32 / float32):
  rule_instability       — total flip count over 32 perturbations (0..32)
  rule_inst_sm           — flips when perturbing Soil_Moisture (0..8)
  rule_inst_rf           — flips when perturbing Rainfall_mm  (0..8)
  rule_inst_tc           — flips when perturbing Temperature_C (0..8)
  rule_inst_ws           — flips when perturbing Wind_Speed_kmh (0..8)

Smoke: instability(row at SM=24.0, RF=350, TC=29, WS=11) — sm_lt_25
crosses if sm × (1+δ) ≥ 25 for δ ≥ +0.042; flips at +5%, +10%, +20%
positive = 3 flips. tc_gt_30 crosses if tc × (1+δ) > 30 for δ ≥ +0.035;
flips at +5%, +10%, +20% = 3 flips. Similar logic for rf/ws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Mirror common.py constants
_SM_THR, _RF_THR, _TC_THR, _WS_THR = 25.0, 300.0, 30.0, 10.0
_DELTAS = (0.02, 0.05, 0.10, 0.20)
_ACTIVE_STAGES = ("Flowering", "Vegetative")


def _score_from_axes(sm, rf, tc, ws, nomulch, kc):
    """Vectorized DGP score: 2*(dry+norain) + (hot+windy+nomulch) + Kc."""
    dry = (sm < _SM_THR).astype(np.int8)
    norain = (rf < _RF_THR).astype(np.int8)
    hot = (tc > _TC_THR).astype(np.int8)
    windy = (ws > _WS_THR).astype(np.int8)
    return (2 * (dry + norain) + hot + windy + nomulch + kc).astype(np.int8)


def add_instability(df: pd.DataFrame) -> pd.DataFrame:
    """Append 5 instability columns to df. Returns new DataFrame."""
    out = df.copy()
    sm = df["Soil_Moisture"].astype(float).to_numpy()
    rf = df["Rainfall_mm"].astype(float).to_numpy()
    tc = df["Temperature_C"].astype(float).to_numpy()
    ws = df["Wind_Speed_kmh"].astype(float).to_numpy()
    nomulch = (df["Mulching_Used"].astype(str).to_numpy() == "No").astype(np.int8)
    stage = df["Crop_Growth_Stage"].astype(str).to_numpy()
    kc = np.where(np.isin(stage, _ACTIVE_STAGES), 2, 0).astype(np.int8)

    base = _score_from_axes(sm, rf, tc, ws, nomulch, kc)
    flips_sm = np.zeros(len(df), dtype=np.int16)
    flips_rf = np.zeros_like(flips_sm)
    flips_tc = np.zeros_like(flips_sm)
    flips_ws = np.zeros_like(flips_sm)

    for d in _DELTAS:
        for sign in (-1.0, +1.0):
            # Perturb Soil_Moisture only
            sm_p = sm * (1 + sign * d)
            flips_sm += (_score_from_axes(sm_p, rf, tc, ws, nomulch, kc) != base).astype(np.int16)
            # Perturb Rainfall only
            rf_p = rf * (1 + sign * d)
            flips_rf += (_score_from_axes(sm, rf_p, tc, ws, nomulch, kc) != base).astype(np.int16)
            # Perturb Temperature only
            tc_p = tc * (1 + sign * d)
            flips_tc += (_score_from_axes(sm, rf, tc_p, ws, nomulch, kc) != base).astype(np.int16)
            # Perturb Wind_Speed only
            ws_p = ws * (1 + sign * d)
            flips_ws += (_score_from_axes(sm, rf, tc, ws_p, nomulch, kc) != base).astype(np.int16)

    out["rule_inst_sm"] = flips_sm.astype(np.int16)
    out["rule_inst_rf"] = flips_rf.astype(np.int16)
    out["rule_inst_tc"] = flips_tc.astype(np.int16)
    out["rule_inst_ws"] = flips_ws.astype(np.int16)
    out["rule_instability"] = (flips_sm + flips_rf + flips_tc + flips_ws).astype(np.int16)
    return out


if __name__ == "__main__":
    # Smoke: tiny synthetic frame
    df = pd.DataFrame({
        "Soil_Moisture":   [24.0, 50.0, 25.5],
        "Rainfall_mm":     [350.0, 1000.0, 290.0],
        "Temperature_C":   [29.0, 25.0, 30.5],
        "Wind_Speed_kmh":  [11.0, 5.0, 9.5],
        "Mulching_Used":   ["No", "Yes", "No"],
        "Crop_Growth_Stage": ["Flowering", "Sowing", "Vegetative"],
    })
    out = add_instability(df)
    print(out[["rule_inst_sm","rule_inst_rf","rule_inst_tc","rule_inst_ws","rule_instability"]])
