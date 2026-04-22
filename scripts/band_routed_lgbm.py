"""Band-routed LGBM — one model per rule band.

Partition every row by the rule's prediction:
  band 0: score <= 3       (rule says Low)      369,917 train rows
  band 1: 4 <= score <= 6  (rule says Medium)   235,390 train rows
  band 2: score >= 7       (rule says High)      21,009 train rows

Train three independent LGBM classifiers, one per band, each with the
full feature set. Within a band, flips are higher-frequency (up to 4%
vs 1.6% globally) and always one-step, so the conditional task is
simpler: "given the rule said X, which class is the row actually?"

Everything else (5-fold stratified CV on the full-train labels,
early-stopping, tuned log-bias on the combined OOF) mirrors the
baseline LGBM+DGP pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
BAND_NAMES = ["rule_Low", "rule_Medium", "rule_High"]

OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
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
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)


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
    out["dgp_score"] = compute_dgp_score(df)
    out["dgp_dist_moist"] = sm - 25.0
    out["dgp_dist_rain"] = rm - 300.0
    out["dgp_dist_temp"] = tc - 30.0
    out["dgp_dist_wind"] = ws - 10.0
    out["dgp_abs_moist"] = out["dgp_dist_moist"].abs()
    out["dgp_abs_rain"] = out["dgp_dist_rain"].abs()
    out["dgp_abs_temp"] = out["dgp_dist_temp"].abs()
    out["dgp_abs_wind"] = out["dgp_dist_wind"].abs()
    return out


def band_of(score: np.ndarray) -> np.ndarray:
    return np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)


log("loading data")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
tr = add_dgp_features(tr)
te = add_dgp_features(te)

y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
tr_band = band_of(tr["dgp_score"].values)
te_band = band_of(te["dgp_score"].values)

log("train band distribution:")
for b, name in enumerate(BAND_NAMES):
    mask = tr_band == b
    log(f"  {name}: n={mask.sum():>6d}  label_dist={np.bincount(y[mask], minlength=3).tolist()}")

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
for c in cat_cols:
    mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
    tr[c] = tr[c].map(mapping).astype("int32")
    te[c] = te[c].map(mapping).astype("int32")
feature_cols = num_cols + cat_cols
X = tr[feature_cols].copy()
X_test = te[feature_cols].copy()
prior = np.bincount(y) / len(y)

params = dict(
    objective="multiclass", num_class=len(CLASSES), metric="multi_logloss",
    learning_rate=0.05, num_leaves=127,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    min_data_in_leaf=50, verbose=-1, seed=SEED,
)

oof_probs = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
test_probs = np.zeros((len(te), len(CLASSES)), dtype=np.float64)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t0 = time.time()
    for b in range(3):
        band_tr = tr_idx[tr_band[tr_idx] == b]
        band_va = va_idx[tr_band[va_idx] == b]
        if len(band_tr) == 0 or len(band_va) == 0:
            continue
        Xb = X.iloc[band_tr]
        yb = y[band_tr]
        Xbv = X.iloc[band_va]
        ybv = y[band_va]
        dtr = lgb.Dataset(Xb, label=yb, categorical_feature=cat_cols)
        dva = lgb.Dataset(Xbv, label=ybv, categorical_feature=cat_cols, reference=dtr)
        model = lgb.train(
            params, dtr, num_boost_round=4000,
            valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof_probs[band_va] = model.predict(Xbv, num_iteration=model.best_iteration)
        test_sel = te_band == b
        if test_sel.any():
            test_probs[test_sel] += (
                model.predict(X_test[test_sel], num_iteration=model.best_iteration)
                / N_FOLDS
            )
    fold_bal = balanced_accuracy_score(y[va_idx], oof_probs[va_idx].argmax(axis=1))
    log(f"  fold {fold+1}/{N_FOLDS}  combined_argmax_bal={fold_bal:.5f}  ({time.time()-t0:.1f}s)")


# ---- diagnostics ------------------------------------------------------------
print()
argmax_pred = oof_probs.argmax(axis=1)
argmax_bal = balanced_accuracy_score(y, argmax_pred)
log(f"band-routed LGBM argmax bal_acc = {argmax_bal:.5f}")

for b, name in enumerate(BAND_NAMES):
    mask = tr_band == b
    bal = balanced_accuracy_score(y[mask], argmax_pred[mask])
    raw = (argmax_pred[mask] == y[mask]).mean()
    print(f"  band {name}: raw={raw:.5f}  bal={bal:.5f}  support={mask.sum()}")


def tune_bias(probs: np.ndarray) -> tuple[float, np.ndarray]:
    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_p + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 2.5, 51)
    for _ in range(20):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_p + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return best, bias


tuned_bal, bias = tune_bias(oof_probs)
log(f"band-routed LGBM tuned log-bias: {bias.round(4).tolist()}  bal_acc={tuned_bal:.5f}")

print("\n=== summary (OOF balanced accuracy) ===")
for name, val in [
    ("rule-only argmax",                   0.96097),
    ("LGBM+DGP tuned log-bias",            0.97271),
    ("band-routed argmax",                 argmax_bal),
    ("band-routed tuned log-bias",         tuned_bal),
]:
    print(f"  {name:<38s} {val:.5f}")

print("\nconfusion matrix (OOF, tuned):")
tuned_pred = (np.log(np.clip(oof_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
print(pd.DataFrame(confusion_matrix(y, tuned_pred), index=CLASSES, columns=CLASSES))

# ---- test submission --------------------------------------------------------
tuned_test_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "submission_band_routed_tuned.csv", index=False
)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in test_probs.argmax(axis=1)]}).to_csv(
    OUT_DIR / "submission_band_routed_argmax.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")

np.save(ART_DIR / "oof_band_routed.npy", oof_probs)
np.save(ART_DIR / "test_band_routed.npy", test_probs)
with open(ART_DIR / "band_routed_results.json", "w") as f:
    json.dump(
        {"argmax_bal": float(argmax_bal), "tuned_bal": float(tuned_bal),
         "tuned_bias": bias.tolist()},
        f, indent=2,
    )
log(f"artefacts saved to {ART_DIR}/band_routed_results.json")
