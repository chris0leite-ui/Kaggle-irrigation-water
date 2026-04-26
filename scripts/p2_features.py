"""P2 bucket-aware FE specialists — feature builders.

Two binary heads, each on its own dgp_score bucket:

  score=3 head (n~102k, 95% Low / 5% Medium):
    target = (y == Medium)
    targets the 4303 M→L errors that are 45.7% of LB-best 4-stack's mass.
    FE engineered for the empirical Cohen's d patterns from 2026-04-26 EDA:
      Soil_Moisture (d=+0.27), Rainfall_mm (d=+0.09), Humidity (d=-0.13),
      Previous_Irrigation (d=-0.10).

  score=6 head (n~38k, 96% Medium / 4% High):
    target = (y == High)
    targets the 1773 M→H + 324 missed-H errors at the M↔H boundary.
    FE engineered for: Rainfall_mm (d=+0.43 missed-H), Soil_Moisture
    (d=-0.29), Temperature_C (d=-0.24), Soil_pH (d=+0.19), Wind_Speed
    (d=-0.15).

Both heads consume:
  - 11 raw numerics (recipe baseline)
  - 8 raw cats factorized
  - 4 signed dist + 4 abs dist (per-axis distance to threshold)
  - dgp_score
  - Bucket-specific engineered FE (5-7 cols)

Output: pd.DataFrame ready for XGBoost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]


def _add_dist(df: pd.DataFrame) -> dict:
    """Per-axis signed/abs distances to rule thresholds."""
    sm = df["Soil_Moisture"].astype(float).to_numpy()
    rf = df["Rainfall_mm"].astype(float).to_numpy()
    tc = df["Temperature_C"].astype(float).to_numpy()
    ws = df["Wind_Speed_kmh"].astype(float).to_numpy()
    out = dict(
        sm_dist=(sm - 25.0).astype(np.float32),
        rf_dist=(rf - 300.0).astype(np.float32),
        tc_dist=(tc - 30.0).astype(np.float32),
        ws_dist=(ws - 10.0).astype(np.float32),
    )
    for k, v in list(out.items()):
        out[k.replace("_dist", "_abs")] = np.abs(v)
    return out


def _factorize_cats_combined(train, test, cats):
    """Factorize cat columns over combined train+test for stable codes."""
    out_tr = {}
    out_te = {}
    for c in cats:
        combined = pd.concat([train[c], test[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        out_tr[c] = codes[:len(train)].astype(np.int32)
        out_te[c] = codes[len(train):].astype(np.int32)
    return out_tr, out_te


def build_features(train: pd.DataFrame, test: pd.DataFrame, dgp_score_train,
                   dgp_score_test, bucket: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build feature matrices for the bucket's binary head.

    bucket ∈ {"score3", "score6"}: drives engineered FE selection.
    Returns (X_train, X_test, feature_names).
    """
    if bucket not in ("score3", "score6"):
        raise ValueError(f"bucket must be 'score3' or 'score6', got {bucket!r}")

    cat_tr, cat_te = _factorize_cats_combined(train, test, CATS)
    dist_tr = _add_dist(train)
    dist_te = _add_dist(test)

    Xtr = pd.DataFrame()
    Xte = pd.DataFrame()
    for c in NUMS:
        Xtr[c] = train[c].astype(np.float32).to_numpy()
        Xte[c] = test[c].astype(np.float32).to_numpy()
    for c, v in cat_tr.items():
        Xtr[c] = v
        Xte[c] = cat_te[c]
    for c, v in dist_tr.items():
        Xtr[c] = v
        Xte[c] = dist_te[c]
    Xtr["dgp_score"] = np.asarray(dgp_score_train, dtype=np.int8)
    Xte["dgp_score"] = np.asarray(dgp_score_test, dtype=np.int8)

    # Bucket-engineered FE
    sm_tr = train["Soil_Moisture"].to_numpy(); sm_te = test["Soil_Moisture"].to_numpy()
    rf_tr = train["Rainfall_mm"].to_numpy(); rf_te = test["Rainfall_mm"].to_numpy()
    h_tr = train["Humidity"].to_numpy(); h_te = test["Humidity"].to_numpy()
    pi_tr = train["Previous_Irrigation_mm"].to_numpy(); pi_te = test["Previous_Irrigation_mm"].to_numpy()
    pH_tr = train["Soil_pH"].to_numpy(); pH_te = test["Soil_pH"].to_numpy()
    ws_tr = train["Wind_Speed_kmh"].to_numpy(); ws_te = test["Wind_Speed_kmh"].to_numpy()
    tc_tr = train["Temperature_C"].to_numpy(); tc_te = test["Temperature_C"].to_numpy()

    if bucket == "score3":
        # score=3 = NOT dry, NOT no-rain → SM ≥ 25, RF ≥ 300; "excess over thr"
        Xtr["log_sm_excess"] = np.log1p(np.maximum(sm_tr - 25.0, 0)).astype(np.float32)
        Xte["log_sm_excess"] = np.log1p(np.maximum(sm_te - 25.0, 0)).astype(np.float32)
        Xtr["log_rf_excess"] = np.log1p(np.maximum(rf_tr - 300.0, 0)).astype(np.float32)
        Xte["log_rf_excess"] = np.log1p(np.maximum(rf_te - 300.0, 0)).astype(np.float32)
        Xtr["humidity_x_sm"] = (h_tr * sm_tr).astype(np.float32)
        Xte["humidity_x_sm"] = (h_te * sm_te).astype(np.float32)
        Xtr["humidity_x_prevIrrig"] = (h_tr * pi_tr).astype(np.float32)
        Xte["humidity_x_prevIrrig"] = (h_te * pi_te).astype(np.float32)
        Xtr["sm_x_prevIrrig"] = (sm_tr * pi_tr).astype(np.float32)
        Xte["sm_x_prevIrrig"] = (sm_te * pi_te).astype(np.float32)
    else:  # score6
        # score=6 = dry+no-rain (sm<25, rf<300) + 0 secondary
        Xtr["log_sm_deficit"] = np.log1p(np.maximum(25.0 - sm_tr, 0)).astype(np.float32)
        Xte["log_sm_deficit"] = np.log1p(np.maximum(25.0 - sm_te, 0)).astype(np.float32)
        Xtr["log_rf_deficit"] = np.log1p(np.maximum(300.0 - rf_tr, 0)).astype(np.float32)
        Xte["log_rf_deficit"] = np.log1p(np.maximum(300.0 - rf_te, 0)).astype(np.float32)
        Xtr["soil_pH_dev"] = (pH_tr - 6.5).astype(np.float32)
        Xte["soil_pH_dev"] = (pH_te - 6.5).astype(np.float32)
        Xtr["wind_excess"] = (ws_tr - 10.0).astype(np.float32)
        Xte["wind_excess"] = (ws_te - 10.0).astype(np.float32)
        Xtr["rainfall_x_temp"] = (rf_tr * tc_tr).astype(np.float32)
        Xte["rainfall_x_temp"] = (rf_te * tc_te).astype(np.float32)

    return Xtr, Xte, list(Xtr.columns)


if __name__ == "__main__":
    import pandas as pd
    tr = pd.read_csv("data/train.csv").head(100)
    te = pd.read_csv("data/test.csv").head(50)
    score_tr = np.zeros(len(tr), dtype=np.int8)
    score_te = np.zeros(len(te), dtype=np.int8)
    for bucket in ("score3", "score6"):
        Xtr, Xte, feats = build_features(tr, te, score_tr, score_te, bucket)
        print(f"{bucket}: train {Xtr.shape}, test {Xte.shape}, feats {len(feats)}")
        print(f"  cols: {feats[-7:]}")
