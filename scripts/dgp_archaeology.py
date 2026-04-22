"""Noise-seed archaeology for the Playground S6E4 irrigation DGP.

Context: the reverse-engineered rule
    score = 2*(dry + norain) + (hot + windy + nomulch) + Kc
with thresholds 25 / 300 / 30 / 10 already maps 619,696 / 630,000
rows correctly (raw acc 0.98364). The remaining 10,304 "flipped"
rows sit strictly in the score-boundary bands.

The flip-detector at AUC 0.899 using DGP features shows the flip is
mostly-but-not-entirely a learnable function of the features — 10 %
residual that distance-geometry FE cannot crack. If that residual is
a deterministic function of row-level inputs (id, categorical
combination, hash, ...), we can find it.

This diagnostic sweeps the obvious deterministic patterns and
produces a compact report. Each section is cheap — we either find a
pattern worth promoting to a feature, or rule it out.

Sections:
  1. id-based patterns: monotonicity, modulo-K, parity, id ranges
  2. Float-quantization artefacts: do flipped rows have different
     mantissa / rounding patterns than clean rows?
  3. Single-feature flip rates for non-rule numeric features
     (Previous_Irrigation_mm, Humidity, Soil_pH, Sunlight_Hours,
     Electrical_Conductivity, Organic_Carbon, Field_Area_hectare)
  4. Single-feature flip rates for non-rule categoricals
     (Soil_Type, Crop_Type, Season, Irrigation_Type, Water_Source,
     Region)
  5. Hash(non-rule categoricals) mod K as a flip predictor
  6. `id` mod K as a flip predictor, K ∈ {2..20, 50, 100, 128, 256, 997}
  7. Score-conditional flip pattern per boundary band
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART_DIR = Path("scripts/artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dgp_score_and_pred(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sm = df["Soil_Moisture"].astype(float).values
    rm = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    um = df["Mulching_Used"].astype(str).values
    stg = df["Crop_Growth_Stage"].astype(str).values
    dry = (sm < 25).astype(int)
    norain = (rm < 300).astype(int)
    hot = (tc > 30).astype(int)
    windy = (ws > 10).astype(int)
    nomulch = (um == "No").astype(int)
    kc = np.where(np.isin(stg, ["Flowering", "Vegetative"]), 2, 0)
    s = 2 * (dry + norain) + (hot + windy + nomulch) + kc
    pred = np.where(s <= 3, "Low", np.where(s <= 6, "Medium", "High"))
    return s, pred


def single_feature_auc(feat: np.ndarray, target: np.ndarray) -> float:
    try:
        return max(roc_auc_score(target, feat), 1 - roc_auc_score(target, feat))
    except Exception:
        return float("nan")


log("loading data")
tr = pd.read_csv("data/train.csv")

score, rule_pred = dgp_score_and_pred(tr)
y_true = tr["Irrigation_Need"].astype(str).values
is_flipped = (rule_pred != y_true).astype(np.int32)
log(f"flip rate = {is_flipped.mean():.5f}  ({is_flipped.sum()}/{len(is_flipped)})")

report: dict = {
    "n_rows": int(len(tr)),
    "flip_rate": float(is_flipped.mean()),
    "n_flipped": int(is_flipped.sum()),
}


# ---------- 1. id-based patterns ----------
log("=== section 1: id-based patterns ===")
ids = tr["id"].values
log(f"  id range: [{ids.min()}, {ids.max()}]  monotonic={np.all(np.diff(ids) == 1)}")
log(f"  id parity AUC for flip: {single_feature_auc((ids % 2).astype(float), is_flipped):.5f}")

# Flip rate for each id-decile
deciles = pd.qcut(ids, 10, labels=False, duplicates="drop")
flip_by_decile = pd.Series(is_flipped).groupby(deciles).mean().round(5).tolist()
log(f"  flip rate per id-decile: {flip_by_decile}")

# id mod K
id_mod_scan = {}
for K in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 32, 50, 64, 100, 128, 256, 997, 1024]:
    # chi-square-like: is flip rate uniform across residue classes?
    res = ids % K
    rates = pd.Series(is_flipped).groupby(res).mean()
    spread = float(rates.max() - rates.min())
    id_mod_scan[K] = {"min": float(rates.min()), "max": float(rates.max()), "spread": spread}
log("  id mod K spread (max-min flip rate across residue classes):")
for K, stats in sorted(id_mod_scan.items(), key=lambda kv: -kv[1]["spread"])[:6]:
    log(f"    K={K:5d}  spread={stats['spread']:.6f}  min={stats['min']:.5f}  max={stats['max']:.5f}")
report["id_mod_scan"] = id_mod_scan

# autocorrelation of flip sequence at small lags (after sort by id)
flip_sorted = is_flipped[np.argsort(ids)]
autocorr = {}
for lag in [1, 2, 3, 5, 7, 10, 100]:
    x = flip_sorted[:-lag].astype(np.float64)
    y = flip_sorted[lag:].astype(np.float64)
    m = x.mean(); v = x.std() * y.std()
    rho = float(((x - m) * (y - y.mean())).mean() / (v if v > 0 else 1.0))
    autocorr[lag] = rho
log(f"  flip autocorr by lag: {autocorr}")
report["flip_autocorr"] = autocorr


# ---------- 2. float-quantization artefacts ----------
log("=== section 2: float-quantization signatures ===")
num_cols = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
            "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
            "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
quant_info = {}
for c in num_cols:
    v = tr[c].values
    # parity of cents: last digit when scaled
    scaled = np.round(v * 100).astype(np.int64)
    last_digit = scaled % 10
    rates = pd.Series(is_flipped).groupby(last_digit).mean()
    spread = float(rates.max() - rates.min())
    quant_info[c] = {
        "n_unique": int(len(np.unique(v))),
        "last_digit_flip_rate_spread": spread,
    }
log("  last-decimal-digit flip-rate spread (higher = more suspicious):")
for c, info in sorted(quant_info.items(), key=lambda kv: -kv[1]["last_digit_flip_rate_spread"])[:6]:
    log(f"    {c:<28s}  spread={info['last_digit_flip_rate_spread']:.6f}  n_unique={info['n_unique']}")
report["quantization_signatures"] = quant_info


# ---------- 3. single-feature flip AUC for non-rule numeric ----------
log("=== section 3: single-feature flip AUC (non-rule numeric) ===")
non_rule_num = ["Soil_pH", "Organic_Carbon", "Electrical_Conductivity",
                "Humidity", "Sunlight_Hours", "Field_Area_hectare",
                "Previous_Irrigation_mm"]
auc_num = {}
for c in non_rule_num:
    v = tr[c].values.astype(float)
    auc_num[c] = single_feature_auc(v, is_flipped)
for c, a in sorted(auc_num.items(), key=lambda kv: -kv[1]):
    log(f"    {c:<28s}  flip-AUC={a:.5f}")
report["single_feature_auc_numeric"] = auc_num


# ---------- 4. single-feature flip AUC for non-rule categorical ----------
log("=== section 4: flip rate by non-rule categorical ===")
cat_cols = ["Soil_Type", "Crop_Type", "Season", "Irrigation_Type", "Water_Source", "Region"]
cat_info = {}
for c in cat_cols:
    ser = tr[c].astype(str)
    rates = pd.Series(is_flipped).groupby(ser).agg(["mean", "size"])
    spread = float(rates["mean"].max() - rates["mean"].min())
    cat_info[c] = {
        "n_levels": int(ser.nunique()),
        "flip_rate_spread": spread,
        "rates": rates["mean"].round(5).to_dict(),
    }
for c, info in sorted(cat_info.items(), key=lambda kv: -kv[1]["flip_rate_spread"]):
    log(f"    {c:<28s}  levels={info['n_levels']}  spread={info['flip_rate_spread']:.6f}")
    log(f"        rates={info['rates']}")
report["categorical_flip_rates"] = cat_info


# ---------- 5. hash(non-rule categoricals) mod K ----------
log("=== section 5: hash(non-rule categoricals) mod K as flip predictor ===")
combo = (
    tr["Soil_Type"].astype(str) + "|"
    + tr["Crop_Type"].astype(str) + "|"
    + tr["Season"].astype(str) + "|"
    + tr["Irrigation_Type"].astype(str) + "|"
    + tr["Water_Source"].astype(str) + "|"
    + tr["Region"].astype(str)
).values
combo_hash = np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in combo])
hash_mod_scan = {}
for K in [2, 3, 4, 5, 7, 8, 10, 11, 13, 16, 32, 64, 100, 128]:
    res = combo_hash % K
    rates = pd.Series(is_flipped).groupby(res).mean()
    hash_mod_scan[K] = {"min": float(rates.min()), "max": float(rates.max()),
                       "spread": float(rates.max() - rates.min())}
log("  hash mod K spread:")
for K, stats in sorted(hash_mod_scan.items(), key=lambda kv: -kv[1]["spread"])[:6]:
    log(f"    K={K:3d}  spread={stats['spread']:.6f}")
report["hash_mod_scan"] = hash_mod_scan


# ---------- 6. score-conditional flip rate ----------
log("=== section 6: flip rate per score band ===")
score_flip = pd.DataFrame({"score": score, "flip": is_flipped}).groupby("score").agg(
    n=("flip", "size"), flip_rate=("flip", "mean")
)
log(f"\n{score_flip.round(5).to_string()}")
report["flip_rate_by_score"] = score_flip.reset_index().to_dict(orient="records")


# ---------- 7. per-boundary-score: flip direction ----------
log("=== section 7: score x true-label contingency (diagnostic of noise shape) ===")
ct = pd.crosstab(score, y_true, margins=False)
log(f"\n{ct.to_string()}")
report["score_true_label_crosstab"] = ct.to_dict()


with open(ART_DIR / "dgp_archaeology_results.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
log(f"report saved to {ART_DIR}/dgp_archaeology_results.json")
