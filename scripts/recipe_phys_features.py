"""Physical / agronomic FE block — Option A from 2026-04-28 brainstorm.

Three blocks, all untested in the recipe pipeline (saturation evidence
lives downstream of recipe FE, not at the FE layer itself):

1. FAO-56 Penman-Monteith reference evapotranspiration ETo, plus
   ETc = ETo × Kc and a soil-water-balance proxy.
2. Crop-coefficient (Kc) values that respect both stage AND crop
   type (existing recipe collapses Kc to {0, 2}).
3. A 3-way `Region × Soil_Type × Crop_Type` categorical (cardinality
   ≤ 5×5×6 = 150) for OrderedTE.

Why this is novel on the V10 recipe:
- The DGP is a deterministic NN trained on the 10k original. The
  10k labels are agronomic (irrigation_need); the host's NN almost
  certainly saw physically-meaningful combinations during training.
- Existing recipe FE includes `ET_proxy = T·W·S / (H+1)` (utaazu) but
  NOT the calibrated FAO-56 equation that climate scientists actually
  use. Calibrated ETo aligns with the real-world ET signal much
  more cleanly than the heuristic proxy.
- Recipe has C(8,2)=28 cat-PAIR combos but no cat-TRIPLE combos
  with all-categorical inputs. `three_way_keys.py` covers
  cat×cat×digit triples; this is cat×cat×cat.

All transforms are deterministic and row-wise (no groupby / no
cross-split aggregation). Mutates df in-place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FAO-56 Penman-Monteith
#
# Reference: Allen, Pereira, Raes, Smith (1998), FAO Irrigation and Drainage
# Paper 56. Reference equation for daily ETo (mm/day):
#
#   ETo = [ 0.408·Δ·(Rn − G) + γ · (900 / (T + 273)) · u₂ · (es − ea) ]
#         / [ Δ + γ · (1 + 0.34·u₂) ]
#
# with
#   es     = 0.6108 · exp(17.27·T / (T + 237.3))    saturated vapor pressure
#   ea     = es · RH / 100                          actual vapor pressure
#   Δ      = 4098·es / (T + 237.3)²                 slope of es-T curve
#   γ      ≈ 0.067 kPa/°C                           psychrometric constant
#   u₂     = u_kmh / 3.6                            wind speed at 2 m (m/s)
#   Rn     ≈ 0.77 · Rs                              net radiation (no Rs0 term)
#   Rs     ≈ 0.5 · sunlight_hours                   crude radiation proxy
#   G      = 0                                      soil heat flux (daily ≈ 0)
#
# We don't have lat/lon/altitude → we approximate Rs from sunlight hours
# directly. Calibration is approximate but the SHAPE of ETo across the
# (T, RH, wind, sun) manifold matches what the host NN's labels would
# correlate with much better than `T·W·S/(H+1)`.
# ---------------------------------------------------------------------------
GAMMA = 0.067  # kPa/°C, psychrometric constant near sea level


def _fao56_eto(T: np.ndarray, RH: np.ndarray, u_kmh: np.ndarray,
               sun_hours: np.ndarray) -> np.ndarray:
    """Vectorized FAO-56 ETo (mm/day). Inputs are 1-D float arrays."""
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    ea = es * (RH / 100.0)
    delta = 4098.0 * es / np.square(T + 237.3)
    u2 = u_kmh / 3.6
    Rs = 0.5 * sun_hours              # crude solar-radiation proxy (MJ/m²/day)
    Rn = 0.77 * Rs                    # net radiation
    numerator = (0.408 * delta * Rn
                 + GAMMA * (900.0 / (T + 273.0)) * u2 * (es - ea))
    denominator = delta + GAMMA * (1.0 + 0.34 * u2)
    eto = numerator / np.clip(denominator, 1e-6, None)
    return eto.astype(np.float32)


# Crop-coefficient table — FAO-56 Annex 7 ranges, conservative midpoints.
# Falls back to 0.85 (mid-development default) when stage / crop unknown.
_KC_TABLE = {
    # crop_type            → (initial, vegetative, flowering, harvest)
    "Wheat":     (0.30, 0.85, 1.15, 0.40),
    "Maize":     (0.30, 0.80, 1.20, 0.50),
    "Rice":      (1.05, 1.10, 1.20, 0.90),
    "Cotton":    (0.35, 0.70, 1.20, 0.65),
    "Sugarcane": (0.40, 0.85, 1.25, 0.75),
    "Soybean":   (0.40, 0.80, 1.15, 0.50),
}
_STAGE_IDX = {"Sowing": 0, "Vegetative": 1, "Flowering": 2, "Harvest": 3}
_KC_DEFAULT = 0.85


def _kc_lookup(crop: np.ndarray, stage: np.ndarray) -> np.ndarray:
    """Per-row crop coefficient from FAO-56 lookup. Float32, length len(crop)."""
    out = np.full(len(crop), _KC_DEFAULT, dtype=np.float32)
    for i, (c, s) in enumerate(zip(crop, stage)):
        if c in _KC_TABLE and s in _STAGE_IDX:
            out[i] = _KC_TABLE[c][_STAGE_IDX[s]]
    return out


def add_phys_block(df: pd.DataFrame) -> list[str]:
    """8 physically-grounded numeric features. Mutates df in-place."""
    T  = df["Temperature_C"].astype(float).values
    RH = df["Humidity"].astype(float).values
    u  = df["Wind_Speed_kmh"].astype(float).values
    sun = df["Sunlight_Hours"].astype(float).values
    sm  = df["Soil_Moisture"].astype(float).values
    rf  = df["Rainfall_mm"].astype(float).values
    pi  = df["Previous_Irrigation_mm"].astype(float).values
    fa  = df["Field_Area_hectare"].astype(float).values
    crop = df["Crop_Type"].astype(str).values
    stage = df["Crop_Growth_Stage"].astype(str).values
    mulch = (df["Mulching_Used"].astype(str).values == "Yes").astype(np.float32)

    eto = _fao56_eto(T, RH, u, sun)
    kc  = _kc_lookup(crop, stage)
    etc = (eto * kc).astype(np.float32)

    # Effective rainfall after mulching: mulched fields conserve more water,
    # so reduce ET demand (heuristic 25% reduction).
    eff_rain = (rf * (1.0 - 0.25 * mulch)).astype(np.float32)

    # Daily soil-water balance proxy:
    #   change ≈ effective_rain + previous_irrigation − ETc
    #   stock  = current Soil_Moisture (capacity proxy)
    swb_change = (eff_rain + pi - etc * 30.0).astype(np.float32)  # 30-day window
    swb_stock  = (sm + swb_change * 0.001).astype(np.float32)     # rescaled

    # Demand vs stock — the irrigation-need signal in physical terms.
    demand = (etc * 30.0 - eff_rain - pi).astype(np.float32)
    demand_ratio = (demand / np.clip(sm + 1e-3, 1e-3, None)).astype(np.float32)

    # Per-area demand normalisation (Field_Area_hectare).
    demand_per_ha = (demand / np.clip(fa + 1e-3, 1e-3, None)).astype(np.float32)

    df["PHYS_ETO"]          = eto
    df["PHYS_KC"]           = kc
    df["PHYS_ETC"]          = etc
    df["PHYS_EFF_RAIN"]     = eff_rain
    df["PHYS_SWB_CHANGE"]   = swb_change
    df["PHYS_SWB_STOCK"]    = swb_stock
    df["PHYS_DEMAND"]       = demand
    df["PHYS_DEMAND_RATIO"] = demand_ratio
    # demand_per_ha is exposed but not in the standard 8-list (untested)
    df["PHYS_DEMAND_PER_HA"] = demand_per_ha
    return [
        "PHYS_ETO", "PHYS_KC", "PHYS_ETC", "PHYS_EFF_RAIN",
        "PHYS_SWB_CHANGE", "PHYS_SWB_STOCK", "PHYS_DEMAND",
        "PHYS_DEMAND_RATIO", "PHYS_DEMAND_PER_HA",
    ]


def add_three_way_cat_combo(train: pd.DataFrame, test: pd.DataFrame,
                            orig: pd.DataFrame) -> list[str]:
    """Build 1 cat-triple combo: Region × Soil_Type × Crop_Type.

    Cardinality ≤ 5×5×6 = 150 unique values, well within OTE's healthy
    range. Returns ["COMBO3_Region_Soil_Crop"]; factorized across
    train+test+orig.
    """
    col = "COMBO3_Region_Soil_Crop"
    for df in (train, test, orig):
        df[col] = (df["Region"].astype(str) + "_"
                   + df["Soil_Type"].astype(str) + "_"
                   + df["Crop_Type"].astype(str))
    combined = pd.concat([train[col], test[col], orig[col]])
    codes, _ = pd.factorize(combined)
    s = len(train); t = s + len(test)
    train[col] = codes[:s]
    test[col]  = codes[s:t]
    orig[col]  = codes[t:]
    return [col]
