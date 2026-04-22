"""
Ablation: does the original Irrigation Prediction dataset (data/archive.zip,
10k rows) improve LGBM OOF balanced accuracy when concatenated with the
synthetic training set?

Design (apples-to-apples with baseline 0.97097):
  - 5-fold stratified CV on synthetic train (same seed as benchmark.py).
  - Each fold: train on (synthetic_train_fold ∪ all_original); validate on
    synthetic_val_fold only.
  - OOF predictions computed only on synthetic rows — directly comparable
    with the synthetic-only baseline.
  - Test predictions (synthetic test set) averaged across the 5 folds.
  - Apply the same decision-rule suite (argmax / prior-reweight / tuned
    log-bias via coordinate ascent).

Schema check (already done): categorical vocabularies match exactly.
Numeric distributions align within ~1%, except Rainfall_mm is ~15 % lower
in the original dataset. Prior distributions match to 3 decimals.
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data ----
log("loading synthetic train/test + original dataset")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
orig = pd.read_csv("data/archive.zip")

log(f"synthetic train={len(tr)}  test={len(te)}  original={len(orig)}")

# Column grouping (original has no `id`). Integer-encode across the union
# of all three sources so every label stays consistent and no unseen value
# appears at predict time.
num_cols = [
    c for c in tr.select_dtypes(include=[np.number]).columns
    if c not in (TARGET, ID)
]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

for c in cat_cols:
    vocab = sorted(
        set(tr[c].unique()) | set(te[c].unique()) | set(orig[c].unique())
    )
    mapping = {v: i for i, v in enumerate(vocab)}
    tr[c] = tr[c].map(mapping).astype("int32")
    te[c] = te[c].map(mapping).astype("int32")
    orig[c] = orig[c].map(mapping).astype("int32")

feat_cols = num_cols + cat_cols

X_syn = tr[feat_cols].copy()
y_syn = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
X_test = te[feat_cols].copy()

X_orig = orig[feat_cols].copy()
y_orig = orig[TARGET].map(CLS2IDX).values.astype(np.int32)

prior = np.bincount(y_syn) / len(y_syn)
log(f"n_features={len(feat_cols)}  class priors (synthetic): "
    f"{dict(zip(CLASSES, prior.round(4)))}")


# --------------------------------------------------------------------- CV ---
log("running 5-fold stratified LGBM (synthetic folds + full original in train)")
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

fold_bal_argmax = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_syn, y_syn)):
    t0 = time.time()
    # concat synthetic fold-train with ALL original rows
    X_fit = pd.concat([X_syn.iloc[tr_idx], X_orig], axis=0, ignore_index=True)
    y_fit = np.concatenate([y_syn[tr_idx], y_orig])

    dtr = lgb.Dataset(X_fit, label=y_fit, categorical_feature=cat_cols)
    dva = lgb.Dataset(
        X_syn.iloc[va_idx], label=y_syn[va_idx],
        categorical_feature=cat_cols, reference=dtr,
    )
    model = lgb.train(
        params,
        dtr,
        num_boost_round=4000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof[va_idx] = model.predict(X_syn.iloc[va_idx], num_iteration=model.best_iteration)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
    fb = balanced_accuracy_score(y_syn[va_idx], oof[va_idx].argmax(axis=1))
    fold_bal_argmax.append(fb)
    log(f"  fold {fold+1}/{N_FOLDS}  n_fit={len(y_fit)}  "
        f"best_iter={model.best_iteration}  "
        f"bal_acc(argmax)={fb:.5f}  ({time.time()-t0:.1f}s)")


# ------------------------------------------------- decision rules on OOF ----
def bench(name: str, pred_idx: np.ndarray) -> dict:
    return {
        "name": name,
        "bal_acc": balanced_accuracy_score(y_syn, pred_idx),
        "cm": confusion_matrix(y_syn, pred_idx).tolist(),
    }


results = []
results.append(bench("LGBM+EXT argmax", oof.argmax(axis=1)))
results.append(bench("LGBM+EXT prior-reweight argmax", (oof / prior).argmax(axis=1)))

log("coord-ascent over per-class log-bias")
log_oof = np.log(np.clip(oof, 1e-9, 1.0))


def score_bias(b: np.ndarray) -> float:
    return balanced_accuracy_score(y_syn, (log_oof + b).argmax(axis=1))


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
results.append(bench("LGBM+EXT tuned log-bias", (log_oof + bias).argmax(axis=1)))


# ----------------------------------------------------------------- report ----
print("\n=== LGBM+EXT benchmark (OOF on synthetic folds, 5-fold, seed=42) ===")
w = max(len(r["name"]) for r in results)
for r in results:
    print(f"  {r['name']:<{w}}  {r['bal_acc']:.5f}")
print(f"  fold std (argmax)     {np.std(fold_bal_argmax):.5f}")

best_rule = max(results, key=lambda r: r["bal_acc"])
print(f"\nbest rule: {best_rule['name']}  bal_acc={best_rule['bal_acc']:.5f}")
print("confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(best_rule["cm"], index=CLASSES, columns=CLASSES))

print("\n=== delta vs baseline (LGBM tuned log-bias = 0.97097) ===")
delta = best_rule["bal_acc"] - 0.97097
print(f"  {delta:+.5f}")


# ------------------------------------------------------------- artifacts -----
np.save(ART_DIR / "oof_lgbm_ext.npy", oof)
np.save(ART_DIR / "test_lgbm_ext.npy", test_pred)
with open(ART_DIR / "bench_ext_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "n_features": len(feat_cols),
            "n_synthetic": int(len(tr)),
            "n_original": int(len(orig)),
            "class_priors_synthetic": prior.tolist(),
            "log_bias": bias.tolist(),
            "fold_bal_argmax": fold_bal_argmax,
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
    OUT_DIR / "submission_lgbm_ext_argmax.csv", index=False
)
tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "submission_lgbm_ext_tuned.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")
