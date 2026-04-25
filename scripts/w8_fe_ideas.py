"""W8 — 50+ feature-engineering ideas as add-on columns to recipe XGB.

Each idea returns a single (or small bundle of) numeric/binary columns.
The grader (`scripts/w8_grade.py`) tests each on a 1-fold smoke vs
recipe baseline; survivors → 5-fold full + meta-stacker bank addition.

Idea categories:
  - Pairwise threshold-distance products
  - Triple-product interactions (rule × non-rule × non-rule)
  - Domain ratios / agronomy proxies
  - Boundary-band binary indicators
  - Cell × non-rule cross stats
  - Raw-feature transforms (log/sqrt/log1p)
  - Per-cat rank / per-row z-score
  - Rule-cell distance metrics (L1, L2 to cell centroid)
  - Test-row-similar-to-original-10k flags
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# Helper: rule features
# ============================================================
def rule_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    sm  = df["Soil_Moisture"].astype(float).values
    rf  = df["Rainfall_mm"].astype(float).values
    tc  = df["Temperature_C"].astype(float).values
    ws  = df["Wind_Speed_kmh"].astype(float).values
    mu  = (df["Mulching_Used"].astype(str).values == "No").astype(np.float32)
    cgs = df["Crop_Growth_Stage"].astype(str).values
    kc  = np.where(np.isin(cgs, ("Flowering", "Vegetative")), 2.0, 0.0).astype(np.float32)
    dry = (sm < 25).astype(np.float32)
    nor = (rf < 300).astype(np.float32)
    hot = (tc > 30).astype(np.float32)
    win = (ws > 10).astype(np.float32)
    score = (2 * (dry + nor) + (hot + win + mu) + kc).astype(np.float32)
    return {"sm": sm, "rf": rf, "tc": tc, "ws": ws, "mu": mu, "kc": kc,
            "dry": dry, "nor": nor, "hot": hot, "win": win, "score": score}


# ============================================================
# Idea bank — each function returns dict[name -> np.ndarray]
# ============================================================
def idea_threshold_dists(df):
    r = rule_features(df)
    return {
        "I1_sm_dist": r["sm"] - 25,
        "I1_rf_dist": r["rf"] - 300,
        "I1_tc_dist": r["tc"] - 30,
        "I1_ws_dist": r["ws"] - 10,
    }

def idea_pairwise_dist_products(df):
    r = rule_features(df)
    sm_d = r["sm"] - 25; rf_d = r["rf"] - 300
    tc_d = r["tc"] - 30; ws_d = r["ws"] - 10
    return {
        "I2_sm_x_rf": sm_d * rf_d,
        "I2_sm_x_tc": sm_d * tc_d,
        "I2_sm_x_ws": sm_d * ws_d,
        "I2_rf_x_tc": rf_d * tc_d,
        "I2_rf_x_ws": rf_d * ws_d,
        "I2_tc_x_ws": tc_d * ws_d,
    }

def idea_min_dist_to_any_threshold(df):
    r = rule_features(df)
    s = np.abs(r["sm"] - 25)
    return {
        "I3_min_axis_dist": np.minimum.reduce([s, np.abs(r["rf"]-300), np.abs(r["tc"]-30), np.abs(r["ws"]-10)]),
        "I3_max_axis_dist": np.maximum.reduce([s, np.abs(r["rf"]-300), np.abs(r["tc"]-30), np.abs(r["ws"]-10)]),
        "I3_l2_dist": np.sqrt(s**2 + (r["rf"]-300)**2/100 + (r["tc"]-30)**2 + (r["ws"]-10)**2),
    }

def idea_score_band_indicators(df):
    r = rule_features(df)
    s = r["score"].astype(int)
    return {f"I4_score_eq_{k}": (s == k).astype(np.float32) for k in [0,1,2,3,4,5,6,7,8,9]}

def idea_in_boundary_band(df):
    r = rule_features(df)
    s = r["score"].astype(int)
    return {
        "I5_boundary_lm": ((s == 3) | (s == 4)).astype(np.float32),
        "I5_boundary_mh": ((s == 6) | (s == 7) | (s == 8)).astype(np.float32),
        "I5_boundary_any": ((s >= 3) & (s <= 8)).astype(np.float32),
        "I5_far_extreme": ((s <= 1) | (s == 9)).astype(np.float32),
    }

def idea_humidity_water_interactions(df):
    r = rule_features(df)
    h = df["Humidity"].astype(float).values
    pi = df["Previous_Irrigation_mm"].astype(float).values
    return {
        "I6_hum_x_dry": h * r["dry"],
        "I6_hum_minus_50": h - 50,
        "I6_pi_x_norain": pi * r["nor"],
        "I6_pi_x_dry": pi * r["dry"],
        "I6_water_supply": pi + r["rf"],
        "I6_water_deficit": (100 - h) + (1 - r["dry"]) * (r["sm"] - 25),
    }

def idea_vpd_proxies(df):
    tc = df["Temperature_C"].astype(float).values
    h = df["Humidity"].astype(float).values
    return {
        "I7_vpd_proxy": tc * (100 - h) / 100,
        "I7_vpd_x_dry": tc * (100 - h) / 100 * (df["Soil_Moisture"].astype(float).values < 25),
        "I7_temp_humidity_ratio": tc / (h + 1),
    }

def idea_field_area_interactions(df):
    r = rule_features(df)
    fa = df["Field_Area_hectare"].astype(float).values
    return {
        "I8_fa_x_score": fa * r["score"],
        "I8_log_fa": np.log1p(fa),
        "I8_fa_x_dry": fa * r["dry"],
    }

def idea_soil_chemistry_interactions(df):
    ph = df["Soil_pH"].astype(float).values
    oc = df["Organic_Carbon"].astype(float).values
    ec = df["Electrical_Conductivity"].astype(float).values
    return {
        "I9_ph_dev_from_neutral": np.abs(ph - 7.0),
        "I9_ph_dev_from_optimal": np.abs(ph - 6.5),
        "I9_oc_x_ph": oc * ph,
        "I9_ec_x_sm": ec * df["Soil_Moisture"].astype(float).values,
    }

def idea_sunlight_x_dryness(df):
    r = rule_features(df)
    sh = df["Sunlight_Hours"].astype(float).values
    return {
        "I10_sun_x_hot": sh * r["hot"],
        "I10_sun_x_dry": sh * r["dry"],
        "I10_sun_per_water": sh / (r["sm"] + r["rf"] + 1),
    }

def idea_decimal_fractions(df):
    out = {}
    for c in ["Temperature_C", "Soil_Moisture", "Soil_pH", "Organic_Carbon", "Sunlight_Hours"]:
        v = df[c].astype(float).values
        out[f"I11_dec_{c}"] = (v % 1).round(2)
    return out

def idea_floor_round(df):
    """Quantize raw nums to 5/10/15 bin granularities."""
    out = {}
    for c, mod in [("Soil_Moisture", 5), ("Rainfall_mm", 50), ("Temperature_C", 5),
                   ("Humidity", 5), ("Previous_Irrigation_mm", 10)]:
        v = df[c].astype(float).values
        out[f"I12_q{mod}_{c}"] = (v // mod) * mod
    return out

def idea_per_cat_rank(df):
    """Rank of soil_moisture within each (Crop_Type, Soil_Type) cell."""
    out = {}
    for cat in ["Crop_Type", "Soil_Type", "Region", "Season"]:
        for num in ["Soil_Moisture", "Humidity", "Previous_Irrigation_mm"]:
            key = df[cat].astype(str).values
            v = df[num].astype(float).values
            df_tmp = pd.DataFrame({"k": key, "v": v})
            ranks = df_tmp.groupby("k")["v"].rank(pct=True).values
            out[f"I13_rank_{num}_in_{cat}"] = ranks.astype(np.float32)
    return out

def idea_log_transforms(df):
    out = {}
    for c in ["Field_Area_hectare", "Previous_Irrigation_mm", "Electrical_Conductivity"]:
        v = df[c].astype(float).values
        out[f"I14_log1p_{c}"] = np.log1p(np.clip(v, 0, None))
    return out

def idea_score_residuals(df):
    """How much each row exceeds the next-band score threshold."""
    r = rule_features(df)
    return {
        "I15_score_above_lm_thresh": np.maximum(0, r["score"] - 3),
        "I15_score_above_mh_thresh": np.maximum(0, r["score"] - 6),
        "I15_score_below_lm_thresh": np.maximum(0, 4 - r["score"]),
        "I15_score_below_mh_thresh": np.maximum(0, 7 - r["score"]),
    }

def idea_kc_x_water(df):
    """Active transpiration × water status."""
    r = rule_features(df)
    return {
        "I16_kc_x_dry": r["kc"] * r["dry"],
        "I16_kc_x_norain": r["kc"] * r["nor"],
        "I16_kc_x_sm_dist": r["kc"] * (r["sm"] - 25),
    }

def idea_polynomial_score(df):
    r = rule_features(df)
    return {
        "I17_score_sq": r["score"] ** 2,
        "I17_score_cube": r["score"] ** 3,
        "I17_score_sqrt": np.sqrt(r["score"]),
        "I17_score_x_kc": r["score"] * r["kc"],
    }

def idea_three_axis_interaction(df):
    r = rule_features(df)
    return {
        "I18_dry_nor_hot": r["dry"] * r["nor"] * r["hot"],
        "I18_dry_nor_kc": r["dry"] * r["nor"] * (r["kc"] / 2),
        "I18_dry_hot_win": r["dry"] * r["hot"] * r["win"],
        "I18_all_critical": r["dry"] * r["nor"] * r["hot"] * r["win"] * r["mu"] * (r["kc"] / 2),
    }

def idea_climatic_index(df):
    """Composite stress index a domain expert might write."""
    tc = df["Temperature_C"].astype(float).values
    h = df["Humidity"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    sh = df["Sunlight_Hours"].astype(float).values
    return {
        "I19_pet_proxy": (tc * sh) / (h + 1),
        "I19_evap_index": ws * (100 - h) / 10,
        "I19_drought_index": (tc + sh - h/2) * (1 + ws/10),
    }

def idea_per_score_humidity_z(df):
    """Z-score of Humidity within score band."""
    r = rule_features(df)
    h = df["Humidity"].astype(float).values
    out_z = np.zeros(len(df), dtype=np.float32)
    for s in range(10):
        m = r["score"].astype(int) == s
        if m.sum() > 1:
            mu = h[m].mean(); sd = h[m].std() + 1e-6
            out_z[m] = (h[m] - mu) / sd
    return {"I20_humidity_z_in_score": out_z}

def idea_per_score_prev_irrig_z(df):
    r = rule_features(df)
    pi = df["Previous_Irrigation_mm"].astype(float).values
    out_z = np.zeros(len(df), dtype=np.float32)
    for s in range(10):
        m = r["score"].astype(int) == s
        if m.sum() > 1:
            mu = pi[m].mean(); sd = pi[m].std() + 1e-6
            out_z[m] = (pi[m] - mu) / sd
    return {"I21_prev_irrig_z_in_score": out_z}


# Master list — each entry is (id, name, function returning dict)
ALL_IDEAS = [
    ("I1",  "threshold_dists",          idea_threshold_dists),
    ("I2",  "pairwise_dist_products",   idea_pairwise_dist_products),
    ("I3",  "min_dist_to_threshold",    idea_min_dist_to_any_threshold),
    ("I4",  "score_band_indicators",    idea_score_band_indicators),
    ("I5",  "in_boundary_band",         idea_in_boundary_band),
    ("I6",  "humidity_water",           idea_humidity_water_interactions),
    ("I7",  "vpd_proxies",              idea_vpd_proxies),
    ("I8",  "field_area",               idea_field_area_interactions),
    ("I9",  "soil_chemistry",           idea_soil_chemistry_interactions),
    ("I10", "sunlight_x_dryness",       idea_sunlight_x_dryness),
    ("I11", "decimal_fractions",        idea_decimal_fractions),
    ("I12", "floor_round_quantize",     idea_floor_round),
    ("I13", "per_cat_rank",             idea_per_cat_rank),
    ("I14", "log_transforms",           idea_log_transforms),
    ("I15", "score_residuals",          idea_score_residuals),
    ("I16", "kc_x_water",               idea_kc_x_water),
    ("I17", "polynomial_score",         idea_polynomial_score),
    ("I18", "three_axis_interaction",   idea_three_axis_interaction),
    ("I19", "climatic_index",           idea_climatic_index),
    ("I20", "humidity_z_in_score",      idea_per_score_humidity_z),
    ("I21", "prev_irrig_z_in_score",    idea_per_score_prev_irrig_z),
]


if __name__ == "__main__":
    # Sanity check: run all ideas on a tiny sample
    df = pd.read_csv("data/train.csv", nrows=1000, dtype_backend="numpy_nullable")
    print(f"sanity-checking {len(ALL_IDEAS)} idea functions on 1000 rows")
    total_cols = 0
    for iid, name, fn in ALL_IDEAS:
        cols = fn(df)
        n = len(cols); total_cols += n
        sample = next(iter(cols.values()))
        assert len(sample) == len(df), f"{iid} size mismatch"
        print(f"  {iid} {name:<28} → {n:3d} cols (sample mean={float(np.nanmean(sample)):.3f})")
    print(f"TOTAL cols across all ideas: {total_cols}")
