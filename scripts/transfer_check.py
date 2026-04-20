"""
Transfer check: train LGBM on the 10k-row original Irrigation Prediction
dataset (data/archive.zip) and predict on the full 630k synthetic train
set. Reports balanced accuracy under each decision rule.

Diagnostic for the concat ablation in benchmark_external.py:
  - high bal_acc (≈ 0.97) → DGPs overlap; concatenation should help.
  - substantially lower  → DGPs diverge; concatenation likely hurts or is flat.

Cheap (~20 s). No test predictions or submission — this is pure
generalization measurement, not a model we would submit.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import lightgbm as lgb

SEED = 42
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

ART_DIR = Path("scripts/artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data ----
log("loading synthetic train + original dataset")
tr = pd.read_csv("data/train.csv")
orig = pd.read_csv("data/archive.zip")
log(f"synthetic train={len(tr)}  original={len(orig)}")

num_cols = [
    c for c in tr.select_dtypes(include=[np.number]).columns
    if c not in (TARGET, ID)
]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

for c in cat_cols:
    vocab = sorted(set(tr[c].unique()) | set(orig[c].unique()))
    mapping = {v: i for i, v in enumerate(vocab)}
    tr[c] = tr[c].map(mapping).astype("int32")
    orig[c] = orig[c].map(mapping).astype("int32")

feat_cols = num_cols + cat_cols
X_orig = orig[feat_cols]
y_orig = orig[TARGET].map(CLS2IDX).values.astype(np.int32)
X_syn = tr[feat_cols]
y_syn = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

prior_syn = np.bincount(y_syn) / len(y_syn)


# ---------------------------------------------------------------- train ----
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

# No eval set on synthetic — that would leak information. Train to a
# fixed number of rounds matched to the synthetic benchmark's best iter
# range (~260). Keep a chunk of original for internal early stopping.
log("training LGBM on original only (train/val split inside original)")
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(orig))
va_n = int(0.2 * len(orig))
va_idx = perm[:va_n]
tr_idx = perm[va_n:]

dtr = lgb.Dataset(X_orig.iloc[tr_idx], label=y_orig[tr_idx], categorical_feature=cat_cols)
dva = lgb.Dataset(
    X_orig.iloc[va_idx], label=y_orig[va_idx],
    categorical_feature=cat_cols, reference=dtr,
)
t0 = time.time()
model = lgb.train(
    params,
    dtr,
    num_boost_round=4000,
    valid_sets=[dva],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
log(f"trained on {len(tr_idx)} original rows in {time.time()-t0:.1f}s  "
    f"best_iter={model.best_iteration}")


# -------------------------------------------------------------- predict ----
log("predicting on the full synthetic train set (630k rows)")
t0 = time.time()
probs_syn = model.predict(X_syn, num_iteration=model.best_iteration)
log(f"inference in {time.time()-t0:.1f}s")


# ------------------------------------------------------- decision rules ----
def bench(name: str, pred_idx: np.ndarray) -> dict:
    return {
        "name": name,
        "bal_acc": balanced_accuracy_score(y_syn, pred_idx),
        "cm": confusion_matrix(y_syn, pred_idx).tolist(),
    }


results = []
results.append(bench("orig→syn argmax", probs_syn.argmax(axis=1)))
results.append(bench("orig→syn prior-reweight argmax", (probs_syn / prior_syn).argmax(axis=1)))

# Coord-ascent log-bias, same harness as benchmark.py. Tuned on the
# synthetic labels directly — this is a ceiling on what the
# trained-on-original model could achieve with the best decision rule.
log("coord-ascent over per-class log-bias")
log_probs = np.log(np.clip(probs_syn, 1e-9, 1.0))


def score_bias(b: np.ndarray) -> float:
    return balanced_accuracy_score(y_syn, (log_probs + b).argmax(axis=1))


bias = -np.log(prior_syn)
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
log(f"  best bias = {dict(zip(CLASSES, bias.round(4)))}  "
    f"synthetic_bal_acc={best:.5f}")
results.append(bench("orig→syn tuned log-bias", (log_probs + bias).argmax(axis=1)))


# ---------------------------------------------------------------- report ----
print("\n=== transfer check (train on original 10k → predict 630k synthetic) ===")
w = max(len(r["name"]) for r in results)
for r in results:
    print(f"  {r['name']:<{w}}  {r['bal_acc']:.5f}")

best_rule = max(results, key=lambda r: r["bal_acc"])
print(f"\nbest rule: {best_rule['name']}  bal_acc={best_rule['bal_acc']:.5f}")
print("confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(best_rule["cm"], index=CLASSES, columns=CLASSES))

print("\n=== interpretation ===")
print(f"  synthetic-only baseline (5-fold OOF) = 0.97097")
print(f"  transfer score                       = {best_rule['bal_acc']:.5f}")
gap = 0.97097 - best_rule["bal_acc"]
print(f"  gap (baseline − transfer)            = {gap:+.5f}")
if gap < 0.02:
    verdict = "DGPs overlap; concatenation should help."
elif gap < 0.10:
    verdict = "moderate divergence; concat could go either way."
else:
    verdict = "large divergence; concatenation likely flat or harmful."
print(f"  verdict: {verdict}")


# --------------------------------------------------------------- artifacts ---
with open(ART_DIR / "transfer_check_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_original_train": int(len(tr_idx)),
            "n_original_val": int(len(va_idx)),
            "n_synthetic": int(len(tr)),
            "best_iter": int(model.best_iteration),
            "log_bias": bias.tolist(),
            "results": [{"name": r["name"], "bal_acc": r["bal_acc"]} for r in results],
            "best_rule": best_rule["name"],
            "gap_vs_synthetic_baseline": gap,
        },
        f,
        indent=2,
    )
log(f"results saved to {ART_DIR}/transfer_check_results.json")
