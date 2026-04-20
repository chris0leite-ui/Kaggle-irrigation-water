"""
Benchmark suite for playground-series-s6e4 (Irrigation_Need).

Pipeline:
  1. Dummy baselines (majority class, stratified random) for a floor.
  2. LGBM multiclass with 5-fold stratified CV; save OOF probs & test probs.
  3. Decision rules evaluated on OOF:
       - argmax
       - class-weighted argmax (reweight probs by 1/prior to hit balanced accuracy)
       - learned per-class additive log-bias (coordinate ascent)
  4. Print a comparison table, save the chosen rule's submission to
     submissions/baseline_lgbm.csv.

Balanced-accuracy metric: macro-recall (sklearn.metrics.balanced_accuracy_score).
"""
from __future__ import annotations
import json
import os
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
CLASSES = ["Low", "Medium", "High"]  # class_to_idx ordering used end-to-end
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

OUT_DIR = Path("submissions")
OOF_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
OOF_DIR.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data ----
log("loading data")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

# integer-encode categoricals (LGBM handles these natively via categorical_feature)
for c in cat_cols:
    mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
    tr[c] = tr[c].map(mapping).astype("int32")
    te[c] = te[c].map(mapping).astype("int32")

X = tr[num_cols + cat_cols].copy()
X_test = te[num_cols + cat_cols].copy()
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)
log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")


# -------------------------------------------------------------- benchmarks ---
def bench(name: str, pred_idx: np.ndarray) -> dict:
    bal = balanced_accuracy_score(y, pred_idx)
    cm = confusion_matrix(y, pred_idx)
    return {"name": name, "bal_acc": bal, "cm": cm.tolist()}


results: list[dict] = []

# (a) majority-class
results.append(bench("majority-class (all Low)", np.zeros_like(y)))

# (b) stratified random draws
rng = np.random.default_rng(SEED)
rand_pred = rng.choice(len(CLASSES), size=len(y), p=prior)
results.append(bench("stratified-random", rand_pred))


# --------------------------------------------------------------------- CV ---
log("running 5-fold stratified LGBM")
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
    dtr = lgb.Dataset(
        X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols
    )
    dva = lgb.Dataset(
        X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_cols,
        reference=dtr,
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
    fold_bal = balanced_accuracy_score(
        y[va_idx], oof[va_idx].argmax(axis=1)
    )
    log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
        f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")


# ------------------------------------------------- decision rules on OOF ----
# rule 1: argmax
argmax_pred = oof.argmax(axis=1)
results.append(bench("LGBM + argmax", argmax_pred))

# rule 2: prior-reweighted argmax  —  divides prob by class prior so every
# class has an equal effective prior at decision time.
reweight_pred = (oof / prior).argmax(axis=1)
results.append(bench("LGBM + prior-reweight argmax", reweight_pred))

# rule 3: learn additive log-bias per class that maximizes balanced accuracy
# via coordinate ascent on OOF.
log("coord-ascent over per-class log-bias")
log_oof = np.log(np.clip(oof, 1e-9, 1.0))


def score_bias(bias: np.ndarray) -> float:
    return balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))


bias = -np.log(prior)  # start at prior-reweight solution
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
tuned_pred = (log_oof + bias).argmax(axis=1)
results.append(bench("LGBM + tuned log-bias", tuned_pred))


# ----------------------------------------------------------------- report ----
print("\n=== benchmark summary (OOF balanced accuracy) ===")
w = max(len(r["name"]) for r in results)
for r in results:
    print(f"  {r['name']:<{w}}  {r['bal_acc']:.5f}")

print("\nconfusion matrix (rows=true, cols=pred) for best rule:")
best_rule = max(results, key=lambda r: r["bal_acc"])
print(f"best: {best_rule['name']}")
print(pd.DataFrame(best_rule["cm"], index=CLASSES, columns=CLASSES))


# ---------------------------------------------------------- persist artifacts
np.save(OOF_DIR / "oof_lgbm_baseline.npy", oof)
np.save(OOF_DIR / "test_lgbm_baseline.npy", test_pred)
with open(OOF_DIR / "bench_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "class_priors": prior.tolist(),
            "log_bias": bias.tolist(),
            "results": [{"name": r["name"], "bal_acc": r["bal_acc"]} for r in results],
            "best_rule": best_rule["name"],
        },
        f,
        indent=2,
    )
log(f"OOF + test probs saved to {OOF_DIR}/")


# --------------------------------------------------------- submission -------
# argmax submission (pure-model sanity check)
argmax_test_idx = test_pred.argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in argmax_test_idx]}).to_csv(
    OUT_DIR / "baseline_lgbm_argmax.csv", index=False
)
# tuned-bias submission (applies the same bias learned on OOF)
tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "baseline_lgbm_tuned.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")
