"""Targeted follow-up: is `id mod K` a real flip predictor or noise?

dgp_archaeology.py flagged `id mod 1024` with flip-rate spread
0.0341 (max 0.03415, min 0.00000) across residue classes, roughly
7x the expected spread under H0 of ~0.005. This script:

  1. Bootstraps the null: shuffle `is_flipped` 500x and compute
     spread for id mod {2,3,5,7,8,16,64,128,256,997,1024}. Report
     observed p-value.
  2. Runs a flip-detector LGBM augmented with `id` + `id mod K`
     features (K ∈ {2,3,5,7,8,16,32,64,128,256,997,1024}), 5-fold
     OOF AUC. Compares to baseline 0.899.
  3. Also tries `id // 615` (bucket into 1024 contiguous groups)
     and raw `id` — any of these might encode the noise seed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5

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


log("loading data")
tr = pd.read_csv("data/train.csv")
ids = tr["id"].values
y_true = tr["Irrigation_Need"].astype(str).values
_, rule_pred = dgp_score_and_pred(tr)
is_flipped = (rule_pred != y_true).astype(np.int32)
n_flip = int(is_flipped.sum())
n = len(is_flipped)
log(f"flip rate = {is_flipped.mean():.5f}  ({n_flip}/{n})")


# -------- 1. bootstrap null ------------
log("=== step 1: bootstrap null for id-mod-K spread ===")
Ks = [2, 3, 5, 7, 8, 16, 32, 64, 128, 256, 512, 997, 1024, 2048]
B = 500
rng = np.random.default_rng(SEED)

observed = {}
for K in Ks:
    res = ids % K
    rates = pd.Series(is_flipped).groupby(res).mean()
    observed[K] = float(rates.max() - rates.min())

# pre-compute residues
residues = {K: (ids % K) for K in Ks}

null_spreads = {K: [] for K in Ks}
for b in range(B):
    perm = rng.permutation(n)
    sf = is_flipped[perm]
    # For each K compute spread fast: group sums via np.bincount
    for K in Ks:
        r = residues[K]
        n_in_bin = np.bincount(r, minlength=K)
        sum_in_bin = np.bincount(r, weights=sf, minlength=K)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(n_in_bin > 0, sum_in_bin / n_in_bin, 0.0)
        null_spreads[K].append(float(rate.max() - rate.min()))
    if (b + 1) % 100 == 0:
        log(f"  bootstrap progress: {b+1}/{B}")

log("K: observed_spread / null_median / null_p95 / p-value (frac null>=observed)")
pvals = {}
for K in Ks:
    arr = np.array(null_spreads[K])
    med = float(np.median(arr))
    p95 = float(np.quantile(arr, 0.95))
    pval = float((arr >= observed[K]).mean())
    pvals[K] = pval
    log(
        f"  K={K:5d}  observed={observed[K]:.5f}  "
        f"null_median={med:.5f}  null_p95={p95:.5f}  p-value={pval:.4f}"
    )


# -------- 2. flip-detector with id features ----------
log("=== step 2: LGBM flip-detector with id + id-mod-K features ===")
from sklearn.model_selection import StratifiedKFold as SKF

num_cols = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
            "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
            "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
cat_cols_raw = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
                "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

# DGP features (same as flip_detector.py)
df = tr.copy()
sm = df["Soil_Moisture"].astype(float)
rm = df["Rainfall_mm"].astype(float)
tc = df["Temperature_C"].astype(float)
ws = df["Wind_Speed_kmh"].astype(float)
df["dgp_dry"] = (sm < 25).astype(np.int8)
df["dgp_norain"] = (rm < 300).astype(np.int8)
df["dgp_hot"] = (tc > 30).astype(np.int8)
df["dgp_windy"] = (ws > 10).astype(np.int8)
df["dgp_nomulch"] = (df["Mulching_Used"].astype(str) == "No").astype(np.int8)
df["dgp_kc"] = np.where(
    df["Crop_Growth_Stage"].astype(str).isin(["Flowering", "Vegetative"]), 2, 0
).astype(np.int8)
df["dgp_score"] = (
    2 * (df["dgp_dry"] + df["dgp_norain"])
    + (df["dgp_hot"] + df["dgp_windy"] + df["dgp_nomulch"])
    + df["dgp_kc"]
).astype(np.int8)
df["dgp_dist_moist"] = sm - 25.0
df["dgp_dist_rain"] = rm - 300.0
df["dgp_dist_temp"] = tc - 30.0
df["dgp_dist_wind"] = ws - 10.0
df["dgp_abs_moist"] = df["dgp_dist_moist"].abs()
df["dgp_abs_rain"] = df["dgp_dist_rain"].abs()
df["dgp_abs_temp"] = df["dgp_dist_temp"].abs()
df["dgp_abs_wind"] = df["dgp_dist_wind"].abs()

# id features
df["id_raw"] = ids
for K in [2, 3, 5, 7, 8, 16, 32, 64, 128, 256, 512, 997, 1024, 2048]:
    df[f"id_mod_{K}"] = (ids % K).astype(np.int32)
df["id_div_1024"] = (ids // 1024).astype(np.int32)  # contiguous blocks of 1024

for c in cat_cols_raw:
    mapping = {v: i for i, v in enumerate(sorted(df[c].unique()))}
    df[c] = df[c].map(mapping).astype("int32")

feat_base = (
    num_cols
    + ["dgp_dry", "dgp_norain", "dgp_hot", "dgp_windy", "dgp_nomulch", "dgp_kc",
       "dgp_score", "dgp_dist_moist", "dgp_dist_rain", "dgp_dist_temp", "dgp_dist_wind",
       "dgp_abs_moist", "dgp_abs_rain", "dgp_abs_temp", "dgp_abs_wind"]
    + cat_cols_raw
)
feat_with_id = feat_base + [
    "id_raw",
    "id_mod_2", "id_mod_3", "id_mod_5", "id_mod_7", "id_mod_8", "id_mod_16",
    "id_mod_32", "id_mod_64", "id_mod_128", "id_mod_256", "id_mod_512",
    "id_mod_997", "id_mod_1024", "id_mod_2048", "id_div_1024",
]

X_base = df[feat_base].copy()
X_full = df[feat_with_id].copy()

params_bin = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=127,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=200,
    verbose=-1,
    seed=SEED,
    is_unbalance=True,
)

results = {}
importances_full = None
for label, Xf in [("baseline_dgp_only", X_base), ("plus_id_mods", X_full)]:
    log(f"--- running flip-detector: {label} ({len(Xf.columns)} features) ---")
    skf = SKF(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(n, dtype=np.float64)
    gain_accum = np.zeros(len(Xf.columns))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xf, is_flipped)):
        t0 = time.time()
        dtr = lgb.Dataset(Xf.iloc[tr_idx], label=is_flipped[tr_idx], categorical_feature=cat_cols_raw)
        dva = lgb.Dataset(
            Xf.iloc[va_idx], label=is_flipped[va_idx],
            categorical_feature=cat_cols_raw, reference=dtr,
        )
        model = lgb.train(
            params_bin, dtr, num_boost_round=2000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(Xf.iloc[va_idx], num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(is_flipped[va_idx], oof[va_idx])
        gain_accum += model.feature_importance(importance_type="gain")
        log(
            f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
            f"AUC={fold_auc:.5f}  ({time.time()-t0:.1f}s)"
        )
    overall = roc_auc_score(is_flipped, oof)
    log(f"  OVERALL OOF AUC = {overall:.5f}")
    results[label] = {"auc": float(overall)}
    if label == "plus_id_mods":
        imp_df = pd.DataFrame({"feature": Xf.columns, "gain": gain_accum}).sort_values(
            "gain", ascending=False
        )
        importances_full = imp_df
        log("Top 20 features (plus_id_mods) by gain:")
        print(imp_df.head(20).to_string(index=False))

report = {
    "n": n,
    "n_flip": n_flip,
    "observed_spread": observed,
    "bootstrap_pvals": pvals,
    "null_medians": {K: float(np.median(null_spreads[K])) for K in Ks},
    "null_p95": {K: float(np.quantile(null_spreads[K], 0.95)) for K in Ks},
    "flip_detector_auc": results,
    "top_importance_plus_id_mods":
        importances_full.head(30).to_dict(orient="records") if importances_full is not None else None,
}
with open(ART_DIR / "archaeology_id_mod_results.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
log(f"results -> {ART_DIR}/archaeology_id_mod_results.json")
