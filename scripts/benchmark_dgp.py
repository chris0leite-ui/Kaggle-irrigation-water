"""LGBM with DGP-derived features + distance-to-threshold features.

Baseline recap:
  - Pure DGP rule on train (630k): bal_acc 0.96097, raw acc 0.98364.
    10,304 mismatches, all in score-boundary bands.
  - Vanilla LGBM argmax: 0.96135. Tuned log-bias: 0.97097.

Hypothesis: the synthetic DGP is `rule + label-flip noise`, where the
flip probability depends on the row's distance-to-threshold. If so,
exposing those distances (plus the rule itself) to LGBM should let it
learn the noise model and recover the flipped rows.

Features added:
  - score, dry, norain, hot, windy, nomulch, Kc (the rule itself)
  - Soil_Moisture - 25, Rainfall_mm - 300, Temperature_C - 30,
    Wind_Speed_kmh - 10 (signed distance to each threshold)
  - |distance| versions of the same four (absolute proximity to boundary)
  - score * nomulch, score * Kc (interaction with weight-carrying flags)

Keeps the same 5-fold stratified pipeline as scripts/benchmark.py so
the OOF number is directly comparable.
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
        out["Crop_Growth_Stage"].astype(str).isin(["Flowering", "Vegetative"]),
        2, 0,
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

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

for c in cat_cols:
    mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
    tr[c] = tr[c].map(mapping).astype("int32")
    te[c] = te[c].map(mapping).astype("int32")

feature_cols = num_cols + cat_cols
log(f"features ({len(feature_cols)}): {feature_cols}")

X = tr[feature_cols].copy()
X_test = te[feature_cols].copy()
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)
log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")


log("running 5-fold stratified LGBM on DGP-enriched features")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

params = dict(
    objective="multiclass",
    num_class=len(CLASSES),
    metric="multi_logloss",
    learning_rate=0.05,
    num_leaves=127,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=200,
    verbose=-1,
    seed=SEED,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t0 = time.time()
    dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(
        X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_cols, reference=dtr,
    )
    model = lgb.train(
        params,
        dtr,
        num_boost_round=4000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
    fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
    log(
        f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
        f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)"
    )


def bench(name: str, pred_idx: np.ndarray) -> dict:
    return {
        "name": name,
        "bal_acc": balanced_accuracy_score(y, pred_idx),
        "cm": confusion_matrix(y, pred_idx).tolist(),
    }


results = [
    bench("LGBM+DGP argmax", oof.argmax(axis=1)),
    bench("LGBM+DGP prior-reweight argmax", (oof / prior).argmax(axis=1)),
]

log("coord-ascent over per-class log-bias")
log_oof = np.log(np.clip(oof, 1e-9, 1.0))


def score_bias(bias: np.ndarray) -> float:
    return balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))


bias = -np.log(prior)
best = score_bias(bias)
grid = np.linspace(-2.5, 2.5, 51)
for _ in range(20):
    improved = False
    for k in range(len(CLASSES)):
        base = bias.copy()
        scores = []
        for g in grid:
            base[k] = bias[k] + g
            scores.append(score_bias(base))
        j = int(np.argmax(scores))
        if scores[j] > best + 1e-6:
            bias[k] = bias[k] + grid[j]
            best = scores[j]
            improved = True
    if not improved:
        break
log(f"  best bias = {dict(zip(CLASSES, bias.round(4)))}  oof_bal_acc={best:.5f}")
results.append(bench("LGBM+DGP tuned log-bias", (log_oof + bias).argmax(axis=1)))

print("\n=== LGBM+DGP summary (OOF balanced accuracy) ===")
w = max(len(r["name"]) for r in results)
for r in results:
    print(f"  {r['name']:<{w}}  {r['bal_acc']:.5f}")

print("\nconfusion matrix (rows=true, cols=pred) for best rule:")
best_rule = max(results, key=lambda r: r["bal_acc"])
print(f"best: {best_rule['name']}")
print(pd.DataFrame(best_rule["cm"], index=CLASSES, columns=CLASSES))

np.save(ART_DIR / "oof_lgbm_dgp.npy", oof)
np.save(ART_DIR / "test_lgbm_dgp.npy", test_pred)
with open(ART_DIR / "bench_dgp_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "class_priors": prior.tolist(),
            "log_bias": bias.tolist(),
            "feature_cols": feature_cols,
            "results": [{"name": r["name"], "bal_acc": r["bal_acc"]} for r in results],
            "best_rule": best_rule["name"],
        },
        f,
        indent=2,
    )
log(f"OOF + test probs saved to {ART_DIR}/")

argmax_test_idx = test_pred.argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in argmax_test_idx]}).to_csv(
    OUT_DIR / "submission_lgbm_dgp_argmax.csv", index=False
)
tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "submission_lgbm_dgp_tuned.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")
