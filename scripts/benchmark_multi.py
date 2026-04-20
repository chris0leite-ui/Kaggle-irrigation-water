"""
Trains XGBoost + CatBoost with the same 5-fold stratified protocol used
by scripts/benchmark.py, then blends with the existing LGBM OOF.

For every model:
  1. 5-fold stratified CV, early stopping on log-loss.
  2. Save OOF probs + averaged test probs to scripts/artifacts/.
  3. Score three decision rules on OOF: argmax, prior-reweight, tuned
     log-bias (coord-ascent on balanced accuracy).

Blend: geometric mean of the three models' OOF probs, then apply a
fresh log-bias search on the blend.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from catboost import CatBoostClassifier, Pool

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


# ----------------------------------------------------------------- load -----
log("loading data")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

# integer-encode categoricals for xgb (cb can eat strings directly)
tr_enc = tr.copy()
te_enc = te.copy()
for c in cat_cols:
    mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
    tr_enc[c] = tr[c].map(mapping).astype("int32")
    te_enc[c] = te[c].map(mapping).astype("int32")

X_enc = tr_enc[num_cols + cat_cols]
X_test_enc = te_enc[num_cols + cat_cols]
X_str = tr[num_cols + cat_cols]            # cb will use strings for cat_cols
X_test_str = te[num_cols + cat_cols]
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)


# ------------------------------------------------------------ tuning util ---
def tune_bias(oof_probs: np.ndarray, y: np.ndarray,
              grid_span: float = 2.5, grid_n: int = 51,
              passes: int = 20) -> tuple[np.ndarray, float]:
    """Coord-ascent over an additive log-bias, max balanced accuracy."""
    log_p = np.log(np.clip(oof_probs, 1e-9, 1.0))

    def score(bias: np.ndarray) -> float:
        return balanced_accuracy_score(y, (log_p + bias).argmax(axis=1))

    bias = -np.log(prior)
    best = score(bias)
    grid = np.linspace(-grid_span, grid_span, grid_n)
    for _ in range(passes):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(score(base))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def report(name: str, oof: np.ndarray) -> dict:
    argmax = oof.argmax(axis=1)
    reweight = (oof / prior).argmax(axis=1)
    bias, tuned_bal = tune_bias(oof, y)
    tuned = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    row = {
        "model": name,
        "argmax": balanced_accuracy_score(y, argmax),
        "prior_reweight": balanced_accuracy_score(y, reweight),
        "tuned_log_bias": tuned_bal,
        "bias": bias.tolist(),
    }
    log(f"  {name:10s}  argmax={row['argmax']:.5f}  "
        f"reweight={row['prior_reweight']:.5f}  "
        f"tuned={row['tuned_log_bias']:.5f}  "
        f"bias={[round(b,3) for b in bias]}")
    return row


# -------------------------------------------------------------- XGBoost -----
log("training XGBoost")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_xgb = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
test_xgb = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

xgb_params = dict(
    objective="multi:softprob",
    num_class=len(CLASSES),
    eval_metric="mlogloss",
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    verbosity=0,
    seed=SEED,
)

# tell xgb which cols are categorical
X_enc_cat = X_enc.copy()
X_test_enc_cat = X_test_enc.copy()
for c in cat_cols:
    X_enc_cat[c] = X_enc_cat[c].astype("category")
    X_test_enc_cat[c] = X_test_enc_cat[c].astype("category")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_enc_cat, y)):
    t0 = time.time()
    dtr = xgb.DMatrix(X_enc_cat.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
    dva = xgb.DMatrix(X_enc_cat.iloc[va_idx], label=y[va_idx], enable_categorical=True)
    booster = xgb.train(
        xgb_params, dtr, num_boost_round=4000,
        evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=0,
    )
    oof_xgb[va_idx] = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    dte = xgb.DMatrix(X_test_enc_cat, enable_categorical=True)
    test_xgb += booster.predict(dte, iteration_range=(0, booster.best_iteration + 1)) / N_FOLDS
    fold_bal = balanced_accuracy_score(y[va_idx], oof_xgb[va_idx].argmax(axis=1))
    log(f"  xgb fold {fold+1}/{N_FOLDS}  best_iter={booster.best_iteration}  "
        f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

np.save(ART_DIR / "oof_xgb_baseline.npy", oof_xgb)
np.save(ART_DIR / "test_xgb_baseline.npy", test_xgb)


# -------------------------------------------------------------- CatBoost ----
log("training CatBoost")
oof_cb = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
test_cb = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

cb_params = dict(
    loss_function="MultiClass",
    iterations=4000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=SEED,
    early_stopping_rounds=100,
    verbose=0,
    task_type="CPU",
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_str, y)):
    t0 = time.time()
    model = CatBoostClassifier(**cb_params)
    tr_pool = Pool(X_str.iloc[tr_idx], label=y[tr_idx], cat_features=cat_cols)
    va_pool = Pool(X_str.iloc[va_idx], label=y[va_idx], cat_features=cat_cols)
    te_pool = Pool(X_test_str, cat_features=cat_cols)
    model.fit(tr_pool, eval_set=va_pool, verbose=0)
    oof_cb[va_idx] = model.predict_proba(va_pool)
    test_cb += model.predict_proba(te_pool) / N_FOLDS
    fold_bal = balanced_accuracy_score(y[va_idx], oof_cb[va_idx].argmax(axis=1))
    log(f"  cb  fold {fold+1}/{N_FOLDS}  best_iter={model.tree_count_}  "
        f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

np.save(ART_DIR / "oof_cb_baseline.npy", oof_cb)
np.save(ART_DIR / "test_cb_baseline.npy", test_cb)


# -------------------------------------------------- evaluate each + blend ---
oof_lgb = np.load(ART_DIR / "oof_lgbm_baseline.npy")
test_lgb = np.load(ART_DIR / "test_lgbm_baseline.npy")

log("single-model decision-rule report")
rows = [
    report("LGBM", oof_lgb),
    report("XGB", oof_xgb),
    report("CatBoost", oof_cb),
]

# geometric-mean blend (equivalent to mean of log-probs)
log_blend = (
    np.log(np.clip(oof_lgb, 1e-9, 1.0))
    + np.log(np.clip(oof_xgb, 1e-9, 1.0))
    + np.log(np.clip(oof_cb,  1e-9, 1.0))
) / 3
oof_blend = np.exp(log_blend - log_blend.max(axis=1, keepdims=True))
oof_blend = oof_blend / oof_blend.sum(axis=1, keepdims=True)

log_blend_te = (
    np.log(np.clip(test_lgb, 1e-9, 1.0))
    + np.log(np.clip(test_xgb, 1e-9, 1.0))
    + np.log(np.clip(test_cb,  1e-9, 1.0))
) / 3
test_blend = np.exp(log_blend_te - log_blend_te.max(axis=1, keepdims=True))
test_blend = test_blend / test_blend.sum(axis=1, keepdims=True)

rows.append(report("blend3", oof_blend))

# persist summary
with open(ART_DIR / "bench_multi_results.json", "w") as f:
    json.dump({"seed": SEED, "n_folds": N_FOLDS, "results": rows}, f, indent=2)

# pick overall best rule → write submission
best = max(rows, key=lambda r: r["tuned_log_bias"])
log(f"best model: {best['model']}  tuned_bal_acc={best['tuned_log_bias']:.5f}")

name_to_test = {"LGBM": test_lgb, "XGB": test_xgb, "CatBoost": test_cb, "blend3": test_blend}
best_test = name_to_test[best["model"]]
bias = np.array(best["bias"])
pred = (np.log(np.clip(best_test, 1e-9, 1.0)) + bias).argmax(axis=1)
sub_path = OUT_DIR / f"baseline_{best['model'].lower()}_tuned.csv"
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in pred]}).to_csv(sub_path, index=False)
log(f"submission written: {sub_path}")

# confusion matrix for best model
print(f"\nconfusion matrix (rows=true, cols=pred) for {best['model']} + tuned log-bias:")
best_oof = {"LGBM": oof_lgb, "XGB": oof_xgb, "CatBoost": oof_cb, "blend3": oof_blend}[best["model"]]
best_pred = (np.log(np.clip(best_oof, 1e-9, 1.0)) + bias).argmax(axis=1)
cm = confusion_matrix(y, best_pred)
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

print("\n=== summary (OOF balanced accuracy, tuned log-bias) ===")
for r in rows:
    print(f"  {r['model']:10s}  {r['tuned_log_bias']:.5f}")
