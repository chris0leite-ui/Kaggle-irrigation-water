"""Boundary-band LGBM — v3 of the DGP-exploit pipeline.

Insight: train flips happen only for rows with dgp_score in {1..9}.
Score 0 rows (no stress, no active stage) are 100% Low in the data
and 100% correct by the rule -- forcing Low avoids any model noise
there. All other rows are candidates for a one-step flip driven by
the "unused" features (Humidity, Soil_pH, Previous_Irrigation_mm,
Sunlight_Hours, EC, Field_Area, Organic_Carbon, etc.).

Pipeline:
  1. Compute dgp_score for train and test.
  2. For score == 0 rows: predict Low directly.
  3. For score in {1..9} rows: train a 3-class LGBM on all 19 raw
     features + DGP indicators + distance-to-threshold features,
     restricted to the boundary subset (~237k rows in train).
  4. Stratified 5-fold CV aligned with the full-train label
     distribution. Within each fold, train only on boundary rows of
     the training indices; predict on the full val indices (rule for
     score=0, model for score>0).
  5. Tune log-bias on OOF. Produce submission.
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


log("loading data")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
tr = add_dgp_features(tr)
te = add_dgp_features(te)

y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
score_tr = tr["dgp_score"].values
score_te = te["dgp_score"].values

log(f"train score distribution:")
print(pd.Series(score_tr).value_counts().sort_index().to_string())

boundary_tr = (score_tr > 0) & (score_tr < 10)  # rule-uncertain
log(f"boundary train rows: {boundary_tr.sum()} / {len(tr)}  ({100*boundary_tr.mean():.2f}%)")
log(f"label dist on boundary: {np.bincount(y[boundary_tr]).tolist()}")
log(f"label dist on interior (score=0): {np.bincount(y[~boundary_tr]).tolist()}")

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


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
# OOF probs for ALL train rows. Interior rows (score==0) get a one-hot on Low;
# boundary rows get the model's 3-class probs.
oof_probs = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
test_probs = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

# seed the one-hot for score==0 rows
score0_mask_tr = score_tr == 0
score0_mask_te = score_te == 0
oof_probs[score0_mask_tr, 0] = 1.0
test_probs[score0_mask_te, 0] = 1.0

params = dict(
    objective="multiclass", num_class=len(CLASSES), metric="multi_logloss",
    learning_rate=0.05, num_leaves=127,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    min_data_in_leaf=200, verbose=-1, seed=SEED,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t0 = time.time()
    bnd_tr = tr_idx[boundary_tr[tr_idx]]
    Xf = X.iloc[bnd_tr]
    yf = y[bnd_tr]
    dtr = lgb.Dataset(Xf, label=yf, categorical_feature=cat_cols)
    dva_mask = va_idx[boundary_tr[va_idx]]
    dva = lgb.Dataset(
        X.iloc[dva_mask], label=y[dva_mask], categorical_feature=cat_cols, reference=dtr
    )
    model = lgb.train(
        params, dtr, num_boost_round=4000,
        valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    # write model probs on boundary rows; interior rows already have one-hot Low
    oof_probs[dva_mask] = model.predict(X.iloc[dva_mask], num_iteration=model.best_iteration)

    # test: only boundary rows need the model; score==0 rows are already Low
    bnd_te = ~score0_mask_te
    test_probs[bnd_te] += model.predict(X_test[bnd_te], num_iteration=model.best_iteration) / N_FOLDS

    fold_bal = balanced_accuracy_score(y[va_idx], oof_probs[va_idx].argmax(axis=1))
    log(
        f"  fold {fold+1}/{N_FOLDS}  iter={model.best_iteration}  "
        f"n_tr={len(bnd_tr)}  n_va={len(dva_mask)}  overall_bal={fold_bal:.5f}  "
        f"({time.time()-t0:.1f}s)"
    )

# ---- diagnostics ------------------------------------------------------------
argmax_pred = oof_probs.argmax(axis=1)
argmax_bal = balanced_accuracy_score(y, argmax_pred)
print()
log(f"boundary-LGBM argmax (interior=rule):         bal_acc = {argmax_bal:.5f}")

# accuracy on boundary rows only
bnd_bal = balanced_accuracy_score(y[boundary_tr], argmax_pred[boundary_tr])
bnd_raw = (argmax_pred[boundary_tr] == y[boundary_tr]).mean()
log(f"  on boundary rows: raw={bnd_raw:.5f}  bal={bnd_bal:.5f}")
log(f"  on interior rows: raw={(argmax_pred[~boundary_tr] == y[~boundary_tr]).mean():.5f}")


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
log(f"boundary-LGBM tuned log-bias: {bias.round(4).tolist()}  bal_acc={tuned_bal:.5f}")

print("\n=== summary v3 (OOF balanced accuracy) ===")
for name, val in [
    ("rule-only argmax",                  0.96097),
    ("LGBM+DGP argmax (v0)",              0.96349),
    ("LGBM+DGP tuned log-bias (v0)",      0.97271),
    ("boundary-LGBM argmax (v3)",         argmax_bal),
    ("boundary-LGBM tuned log-bias (v3)", tuned_bal),
]:
    print(f"  {name:<38s} {val:.5f}")

print("\nconfusion (OOF, best rule = tuned):")
tuned_pred = (np.log(np.clip(oof_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
print(pd.DataFrame(confusion_matrix(y, tuned_pred), index=CLASSES, columns=CLASSES))

# ---- test submission --------------------------------------------------------
tuned_test_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "submission_boundary_lgbm_tuned.csv", index=False
)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in test_probs.argmax(axis=1)]}).to_csv(
    OUT_DIR / "submission_boundary_lgbm_argmax.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")

np.save(ART_DIR / "oof_boundary_lgbm.npy", oof_probs)
np.save(ART_DIR / "test_boundary_lgbm.npy", test_probs)
with open(ART_DIR / "boundary_lgbm_results.json", "w") as f:
    json.dump(
        {
            "argmax_bal": float(argmax_bal),
            "tuned_bal": float(tuned_bal),
            "tuned_bias": bias.tolist(),
            "boundary_bal_on_boundary_rows": float(bnd_bal),
            "boundary_raw_on_boundary_rows": float(bnd_raw),
        },
        f, indent=2,
    )
log(f"artefacts saved to {ART_DIR}/boundary_lgbm_results.json")
