"""Weighted LGBM+DGP — v4 of the DGP-exploit pipeline.

Diagnostic showed LGBM+DGP achieves only 0.12 bal_acc on the 10,304
flipped rows (vs the specialist's 0.99). Root cause: flipped rows are
1.6% of the training data and contribute only 1.6% of the loss, so the
tree optimises the majority at their expense.

Fix: sample_weight = w on flipped rows, 1 on clean rows. Swept across
w in {10, 30, 60, 100, 200}. w=62 roughly equalises total-loss
contribution between the two populations.

Pipeline mirrors scripts/benchmark_dgp.py (same features, 5-fold
stratified CV, early-stopping, tuned log-bias). Best weight by OOF
bal_acc is written as submission_weighted_lgbm_dgp_tuned.csv.
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
WEIGHTS = [10, 30, 60, 100, 200]

OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dgp_rule_int(df: pd.DataFrame) -> np.ndarray:
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
    return np.where(s <= 3, 0, np.where(s <= 6, 1, 2)).astype(np.int32)


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
te = pd.read_csv("data/test.csv")
tr = add_dgp_features(tr)
te = add_dgp_features(te)

rule_tr = dgp_rule_int(tr)
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
is_flipped = (rule_tr != y).astype(np.int32)
log(f"flip_rate={float(is_flipped.mean()):.5f}  n_flipped={int(is_flipped.sum())}")

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
    min_data_in_leaf=200, verbose=-1, seed=SEED,
)


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


all_results: dict[int, dict] = {}
best_weight = None
best_tuned = -1.0
best_oof = None
best_test = None
best_bias = None

for w in WEIGHTS:
    log(f"=== sample-weight w={w} on flipped rows ===")
    sample_w = np.where(is_flipped == 1, float(w), 1.0).astype(np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = lgb.Dataset(
            X.iloc[tr_idx], label=y[tr_idx], weight=sample_w[tr_idx],
            categorical_feature=cat_cols,
        )
        dva = lgb.Dataset(
            X.iloc[va_idx], label=y[va_idx],  # unweighted val loss
            categorical_feature=cat_cols, reference=dtr,
        )
        model = lgb.train(
            params, dtr, num_boost_round=4000,
            valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(
            f"  w={w} fold {fold+1}/{N_FOLDS}  iter={model.best_iteration}  "
            f"argmax_bal={fold_bal:.5f}  ({time.time()-t0:.1f}s)"
        )

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    flip_mask = is_flipped == 1
    spec_bal = balanced_accuracy_score(y[flip_mask], oof[flip_mask].argmax(axis=1))
    spec_raw = (oof[flip_mask].argmax(axis=1) == y[flip_mask]).mean()
    clean_raw = (oof[~flip_mask].argmax(axis=1) == y[~flip_mask]).mean()
    tuned_bal, bias = tune_bias(oof)
    log(
        f"  w={w}  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}  "
        f"on_flipped={spec_bal:.5f} (raw {spec_raw:.5f})  on_clean_raw={clean_raw:.5f}"
    )
    all_results[w] = {
        "argmax_bal": float(argmax_bal),
        "tuned_bal": float(tuned_bal),
        "tuned_bias": bias.tolist(),
        "on_flipped_bal": float(spec_bal),
        "on_flipped_raw": float(spec_raw),
        "on_clean_raw": float(clean_raw),
    }
    if tuned_bal > best_tuned:
        best_tuned = tuned_bal
        best_weight = w
        best_oof = oof.copy()
        best_test = test_pred.copy()
        best_bias = bias.copy()


print("\n=== sweep summary (OOF bal_acc) ===")
print(f"  baseline LGBM+DGP tuned = 0.97271 (no weighting)")
for w, r in all_results.items():
    print(
        f"  w={w:>3d}  argmax={r['argmax_bal']:.5f}  tuned={r['tuned_bal']:.5f}  "
        f"on_flipped={r['on_flipped_bal']:.5f}  on_clean_raw={r['on_clean_raw']:.5f}"
    )
print(f"\n  best w={best_weight}  tuned bal_acc={best_tuned:.5f}")

# Save best-weight OOF + test + submission
log(f"writing artefacts for w={best_weight}")
np.save(ART_DIR / f"oof_weighted_w{best_weight}.npy", best_oof)
np.save(ART_DIR / f"test_weighted_w{best_weight}.npy", best_test)

tuned_test_idx = (np.log(np.clip(best_test, 1e-9, 1.0)) + best_bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / f"submission_weighted_lgbm_dgp_w{best_weight}_tuned.csv", index=False
)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in best_test.argmax(axis=1)]}).to_csv(
    OUT_DIR / f"submission_weighted_lgbm_dgp_w{best_weight}_argmax.csv", index=False
)

with open(ART_DIR / "weighted_lgbm_dgp_results.json", "w") as f:
    json.dump({"best_weight": best_weight, "sweep": all_results}, f, indent=2)
log(f"artefacts saved to {ART_DIR}/weighted_lgbm_dgp_results.json")
