"""
Benchmark LGBM with engineered domain features.

Extends scripts/benchmark.py by injecting the engineered columns already
validated by MNLogit-F2 and heuristic-H3 (DOMAIN.md, NEXT_STEPS.md §3):

  - ET0_proxy = Temperature_C * (1 - Humidity/100) * Wind_Speed_kmh
  - Kc_stage  = FAO-56 lookup by Crop_Growth_Stage
  - ETc_proxy = ET0_proxy * Kc_stage * (1 - 0.30 * Is_Mulched)
  - Soil_deficit    = max(0, capacity[Soil_Type] - Soil_Moisture)
  - Is_Rainfed      = (Irrigation_Type == "Rainfed")
  - Eff_Rainfall_active = 0.80 * Rainfall_mm * (1 - Is_Rainfed)
  - Crop_x_Stage    = Crop_Type ⊕ Crop_Growth_Stage   (new categorical)
  - Season_x_Region = Season ⊕ Region                 (new categorical)

Compares 5-fold stratified OOF balanced accuracy against the baseline
(0.97097 tuned log-bias) and writes the submission CSVs.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

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

KC_STAGE = {"Sowing": 0.35, "Vegetative": 0.85, "Flowering": 1.15, "Harvest": 0.55}
SOIL_CAP = {"Sandy": 18.0, "Loamy": 33.0, "Silt": 40.0, "Clay": 45.0}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns in-place-style; returns the augmented frame.

    Inputs must still hold the original string categoricals (run BEFORE
    integer-encoding).
    """
    out = df.copy()
    out["ET0_proxy"] = (
        out["Temperature_C"] * (1 - out["Humidity"] / 100) * out["Wind_Speed_kmh"]
    )
    kc = out["Crop_Growth_Stage"].map(KC_STAGE)
    is_mulched = (out["Mulching_Used"] == "Yes").astype(np.float32)
    is_rainfed = (out["Irrigation_Type"] == "Rainfed").astype(np.float32)
    cap = out["Soil_Type"].map(SOIL_CAP)

    out["Kc_stage"] = kc.astype(np.float32)
    out["ETc_proxy"] = (out["ET0_proxy"] * kc * (1 - 0.30 * is_mulched)).astype(np.float32)
    out["Soil_deficit"] = (cap - out["Soil_Moisture"]).clip(lower=0).astype(np.float32)
    out["Is_Rainfed"] = is_rainfed.astype(np.int8)
    out["Eff_Rainfall_active"] = (0.80 * out["Rainfall_mm"] * (1 - is_rainfed)).astype(np.float32)

    out["Crop_x_Stage"] = (
        out["Crop_Type"].astype(str) + "_" + out["Crop_Growth_Stage"].astype(str)
    )
    out["Season_x_Region"] = (
        out["Season"].astype(str) + "_" + out["Region"].astype(str)
    )
    return out


# ------------------------------------------------------------------ data ----
log("loading data")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")

log("building engineered features")
tr = add_engineered(tr)
te = add_engineered(te)

# Column grouping AFTER engineering. Target and id stay out.
num_cols = [
    c for c in tr.select_dtypes(include=[np.number]).columns
    if c not in (TARGET, ID)
]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

# integer-encode categoricals (LGBM handles via categorical_feature). Fit
# mapping on the union of train+test values so no unseen categories leak
# into test prediction.
for c in cat_cols:
    vocab = sorted(set(tr[c].unique()) | set(te[c].unique()))
    mapping = {v: i for i, v in enumerate(vocab)}
    tr[c] = tr[c].map(mapping).astype("int32")
    te[c] = te[c].map(mapping).astype("int32")

feat_cols = num_cols + cat_cols
X = tr[feat_cols].copy()
X_test = te[feat_cols].copy()
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)

log(f"n_features={len(feat_cols)}  (numeric={len(num_cols)}, categorical={len(cat_cols)})")
log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")


# --------------------------------------------------------------------- CV ---
log("running 5-fold stratified LGBM (engineered features)")
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

fold_bal = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t0 = time.time()
    dtr = lgb.Dataset(
        X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols
    )
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
    fb = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
    fold_bal.append(fb)
    log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
        f"bal_acc(argmax)={fb:.5f}  ({time.time()-t0:.1f}s)")


# ------------------------------------------------- decision rules on OOF ----
def bench(name: str, pred_idx: np.ndarray) -> dict:
    return {
        "name": name,
        "bal_acc": balanced_accuracy_score(y, pred_idx),
        "cm": confusion_matrix(y, pred_idx).tolist(),
    }


results = []
results.append(bench("LGBM+FE argmax", oof.argmax(axis=1)))
results.append(bench("LGBM+FE prior-reweight argmax", (oof / prior).argmax(axis=1)))

log("coord-ascent over per-class log-bias")
log_oof = np.log(np.clip(oof, 1e-9, 1.0))


def score_bias(b: np.ndarray) -> float:
    return balanced_accuracy_score(y, (log_oof + b).argmax(axis=1))


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
results.append(bench("LGBM+FE tuned log-bias", (log_oof + bias).argmax(axis=1)))


# ------------------------------------------------------------------- report ----
print("\n=== LGBM+FE benchmark (OOF balanced accuracy, 5-fold, seed=42) ===")
w = max(len(r["name"]) for r in results)
for r in results:
    print(f"  {r['name']:<{w}}  {r['bal_acc']:.5f}")
print(f"  fold std (argmax)     {np.std(fold_bal):.5f}")

best_rule = max(results, key=lambda r: r["bal_acc"])
print(f"\nbest rule: {best_rule['name']}  bal_acc={best_rule['bal_acc']:.5f}")
print("confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(best_rule["cm"], index=CLASSES, columns=CLASSES))

print("\n=== delta vs baseline (LGBM tuned log-bias = 0.97097) ===")
delta = best_rule["bal_acc"] - 0.97097
print(f"  {delta:+.5f}")


# ------------------------------------------------------------- artifacts -----
np.save(ART_DIR / "oof_lgbm_fe.npy", oof)
np.save(ART_DIR / "test_lgbm_fe.npy", test_pred)
with open(ART_DIR / "bench_fe_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "n_features": len(feat_cols),
            "feat_cols": feat_cols,
            "class_priors": prior.tolist(),
            "log_bias": bias.tolist(),
            "fold_bal_argmax": fold_bal,
            "results": [{"name": r["name"], "bal_acc": r["bal_acc"]} for r in results],
            "best_rule": best_rule["name"],
            "delta_vs_baseline_tuned": delta,
        },
        f,
        indent=2,
    )
log(f"OOF + test probs saved to {ART_DIR}/")


# ------------------------------------------------------------- submissions ---
argmax_test_idx = test_pred.argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in argmax_test_idx]}).to_csv(
    OUT_DIR / "submission_lgbm_fe_argmax.csv", index=False
)
tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "submission_lgbm_fe_tuned.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")
