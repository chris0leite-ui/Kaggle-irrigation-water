"""Diagnostic: are rule-vs-label flips feature-predictable?

Train a binary LGBM on `is_flipped = (rule_pred != true_label)` using
all available features (raw + DGP-derived). 5-fold stratified CV, OOF
AUC reported. If AUC > 0.7 the noise is learnable and there is more
lift to extract past the DGP rule; if AUC ~ 0.5 the flips are IID and
we have hit the irreducible ceiling on this problem.

Also trains a 3-class classifier restricted to the 10,304 flipped rows
to see if flip *direction* is predictable among rows we already know
are flipped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

ART_DIR = Path("scripts/artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dgp_predict(df: pd.DataFrame) -> np.ndarray:
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
    return np.where(s <= 3, "Low", np.where(s <= 6, "Medium", "High"))


def add_dgp_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float)
    rm = out["Rainfall_mm"].astype(float)
    tc = out["Temperature_C"].astype(float)
    ws = out["Wind_Speed_kmh"].astype(float)
    out["dgp_dry"] = (sm < 25).astype(np.int8)
    out["dgp_norain"] = (rm < 300).astype(np.int8)
    out["dgp_hot"] = (tc > 30).astype(np.int8)
    out["dgp_windy"] = (ws > 10).astype(np.int8)
    out["dgp_nomulch"] = (out["Mulching_Used"].astype(str) == "No").astype(np.int8)
    out["dgp_kc"] = np.where(
        out["Crop_Growth_Stage"].astype(str).isin(["Flowering", "Vegetative"]), 2, 0
    ).astype(np.int8)
    out["dgp_score"] = (
        2 * (out["dgp_dry"] + out["dgp_norain"])
        + (out["dgp_hot"] + out["dgp_windy"] + out["dgp_nomulch"])
        + out["dgp_kc"]
    ).astype(np.int8)
    out["dgp_dist_moist"] = sm - 25.0
    out["dgp_dist_rain"] = rm - 300.0
    out["dgp_dist_temp"] = tc - 30.0
    out["dgp_dist_wind"] = ws - 10.0
    out["dgp_abs_moist"] = out["dgp_dist_moist"].abs()
    out["dgp_abs_rain"] = out["dgp_dist_rain"].abs()
    out["dgp_abs_temp"] = out["dgp_dist_temp"].abs()
    out["dgp_abs_wind"] = out["dgp_dist_wind"].abs()
    return out


log("loading data")
tr = pd.read_csv("data/train.csv")
tr = add_dgp_features(tr)

rule_pred = dgp_predict(tr)
y_true = tr[TARGET].astype(str).values
is_flipped = (rule_pred != y_true).astype(np.int32)
log(f"flip rate: {is_flipped.mean():.5f}  ({is_flipped.sum()}/{len(is_flipped)})")

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
for c in cat_cols:
    mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
    tr[c] = tr[c].map(mapping).astype("int32")

feature_cols = num_cols + cat_cols
X = tr[feature_cols].copy()

log(f"features ({len(feature_cols)})")


# ----- experiment A: binary flip detector ------------------------------------
log("flip detector: 5-fold stratified CV")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_flip = np.zeros(len(tr), dtype=np.float64)

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

importances = np.zeros(len(feature_cols))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, is_flipped)):
    t0 = time.time()
    dtr = lgb.Dataset(X.iloc[tr_idx], label=is_flipped[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(
        X.iloc[va_idx], label=is_flipped[va_idx], categorical_feature=cat_cols, reference=dtr
    )
    model = lgb.train(
        params_bin,
        dtr,
        num_boost_round=2000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof_flip[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
    fold_auc = roc_auc_score(is_flipped[va_idx], oof_flip[va_idx])
    log(
        f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
        f"AUC={fold_auc:.5f}  ({time.time()-t0:.1f}s)"
    )
    importances += model.feature_importance(importance_type="gain")

overall_auc = roc_auc_score(is_flipped, oof_flip)
log(f"FLIP DETECTOR OOF AUC = {overall_auc:.5f}")
print()

imp_df = (
    pd.DataFrame({"feature": feature_cols, "gain": importances})
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)
print("Top 15 features by gain (binary flip detector):")
print(imp_df.head(15).to_string(index=False))


# ----- experiment B: flip-direction classifier (only flipped rows) ----------
log("flip-direction 3-class classifier (restricted to flipped rows)")
flip_mask = is_flipped == 1
Xf = X[flip_mask].reset_index(drop=True)
yf = pd.Series(y_true[flip_mask]).map(CLS2IDX).values.astype(np.int32)
log(f"  flipped subset: n={len(Xf)}  label dist={np.bincount(yf).tolist()}")

skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_dir = np.zeros((len(Xf), len(CLASSES)), dtype=np.float64)
params_multi = dict(
    objective="multiclass",
    num_class=len(CLASSES),
    metric="multi_logloss",
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=50,
    verbose=-1,
    seed=SEED,
)
for fold, (tr_idx, va_idx) in enumerate(skf2.split(Xf, yf)):
    dtr = lgb.Dataset(Xf.iloc[tr_idx], label=yf[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(
        Xf.iloc[va_idx], label=yf[va_idx], categorical_feature=cat_cols, reference=dtr
    )
    model = lgb.train(
        params_multi,
        dtr,
        num_boost_round=1500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof_dir[va_idx] = model.predict(Xf.iloc[va_idx], num_iteration=model.best_iteration)

dir_argmax = oof_dir.argmax(axis=1)
dir_bal = balanced_accuracy_score(yf, dir_argmax)
dir_acc = (dir_argmax == yf).mean()
log(f"FLIP-DIRECTION classifier OOF: raw_acc={dir_acc:.5f}  bal_acc={dir_bal:.5f}")
print("confusion (rows=actual, cols=pred) on flipped rows:")
print(pd.DataFrame(
    confusion_matrix(yf, dir_argmax, labels=[0, 1, 2]),
    index=CLASSES, columns=CLASSES,
))


# ----- save artefacts --------------------------------------------------------
np.save(ART_DIR / "oof_flip_prob.npy", oof_flip)
np.save(ART_DIR / "flipped_rows_mask.npy", flip_mask)
np.save(ART_DIR / "oof_flip_direction.npy", oof_dir)
with open(ART_DIR / "flip_detector_results.json", "w") as f:
    json.dump(
        {
            "flip_rate": float(is_flipped.mean()),
            "flip_detector_auc": float(overall_auc),
            "flip_direction_bal_acc": float(dir_bal),
            "flip_direction_raw_acc": float(dir_acc),
            "top_importance": imp_df.head(20).to_dict(orient="records"),
        },
        f,
        indent=2,
    )
log(f"results saved to {ART_DIR}/flip_detector_results.json")
