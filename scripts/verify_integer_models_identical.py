"""Sanity checks on the 743-integer-models result.

Claim: all 743 separating models produce identical predictions on
630k synthetic. The earlier script only saved top-50 predictions;
this re-scores every model from scratch and tests for row-level
disagreement directly.

Checks:
  1. Generate predictions for ALL 743 models. Pairwise-verify that
     every pair gives identical predictions row-by-row, not just
     identical aggregate bal_acc.
  2. For a few hand-picked "very different looking" models (lowest
     hinge vs highest hinge; different sign patterns on Mulching),
     print predictions on a random sample of synthetic rows so a
     human can eyeball them.
  3. Confirm every synthetic row maps to one of the 128 cells from
     the 10k original (the mechanism underlying the claim).
  4. Confirm CP enumeration actually enforces the cell-label
     constraints by independently checking a random model against
     the 128-cell lookup.
  5. Permute a model's weights into something that should NOT
     separate (introduce a deliberate misclassification) and
     confirm its synthetic bal_acc drops -- to prove the scoring
     pipeline isn't stuck at 0.96097 independent of the model.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ART_DIR = Path("scripts/artifacts")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_features(df: pd.DataFrame) -> np.ndarray:
    f = np.zeros((len(df), 9), dtype=np.int8)
    f[:, 0] = (df["Soil_Moisture"].values < 25).astype(np.int8)
    f[:, 1] = (df["Temperature_C"].values > 30).astype(np.int8)
    f[:, 2] = (df["Rainfall_mm"].values < 300).astype(np.int8)
    f[:, 3] = (df["Wind_Speed_kmh"].values > 10).astype(np.int8)
    f[:, 4] = (df["Mulching_Used"].values == "Yes").astype(np.int8)
    stages = df["Crop_Growth_Stage"].values
    f[:, 5] = (stages == "Flowering").astype(np.int8)
    f[:, 6] = (stages == "Harvest").astype(np.int8)
    f[:, 7] = (stages == "Sowing").astype(np.int8)
    f[:, 8] = (stages == "Vegetative").astype(np.int8)
    return f


def predict_class(w: np.ndarray, theta: int, X: np.ndarray) -> np.ndarray:
    eta = X @ w.astype(np.int64)
    return np.where(eta <= 0, 0, np.where(eta <= theta, 1, 2)).astype(np.int8)


log("loading data + saved enumeration output")
orig = pd.read_csv("data/archive.zip")
tr = pd.read_csv("data/train.csv")
X_orig = make_features(orig)
X_syn = make_features(tr)
y_orig = orig[TARGET].map(CLS2IDX).values.astype(np.int32)
y_syn = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

models_all = np.load(ART_DIR / "integer_models_raw.npy")   # (743, 10): 9 w + theta
assert models_all.shape == (743, 10)
log(f"  n_models = {models_all.shape[0]}")


# ---------------- check 1: every pair produces identical predictions ---------
log("check 1: generate predictions for all 743 models and compare")
t0 = time.time()
all_preds = np.stack([
    predict_class(m[:9], int(m[9]), X_syn)
    for m in models_all
], axis=1).astype(np.int8)  # (n_syn, 743)
log(f"  prediction matrix shape: {all_preds.shape}  ({time.time() - t0:.1f}s)")

# Column 0 as reference, compare every column to it.
ref = all_preds[:, 0:1]
row_agreements = (all_preds == ref).all(axis=0)  # (743,)
print(f"  models producing identical predictions to model 0: "
      f"{row_agreements.sum()} / {len(row_agreements)}")
if row_agreements.sum() < len(row_agreements):
    # Find a disagreeing model.
    disagree_idx = np.where(~row_agreements)[0]
    for j in disagree_idx[:3]:
        diff_rows = np.where(all_preds[:, j] != ref[:, 0])[0]
        print(f"    model {j}: differs on {len(diff_rows)} rows, "
              f"first few row-indices: {diff_rows[:5].tolist()}")
else:
    print("  -> ALL pairs identical.")

# Stronger check: number of unique prediction columns.
# Use tobytes on each column for O(n) dedup.
unique_cols = len({all_preds[:, j].tobytes() for j in range(all_preds.shape[1])})
print(f"  unique prediction vectors across 743 models: {unique_cols}")


# ---------------- check 2: eyeball two extremal models on random rows -----
log("check 2: hand-pick lowest-hinge vs highest-hinge and inspect")
# Re-compute hinge quickly using the same multiclass_hinge as the main script.
from sklearn.metrics import hinge_loss


def multiclass_hinge(w: np.ndarray, theta: int,
                     X: np.ndarray, y: np.ndarray) -> float:
    coef = np.array([
        -w, np.zeros_like(w), w - np.array([0] * 5 + [theta] * 4),
    ], dtype=np.float64)
    return float(hinge_loss(y, X.astype(np.float64) @ coef.T, labels=[0, 1, 2]))


hinges = np.array([
    multiclass_hinge(m[:9], int(m[9]), X_orig, y_orig) for m in models_all
])
lo_idx = int(np.argmin(hinges))
hi_idx = int(np.argmax(hinges))
print(f"  lowest  hinge = {hinges[lo_idx]:.4f}  model {lo_idx}: "
      f"w={models_all[lo_idx, :9].tolist()} theta={int(models_all[lo_idx, 9])}")
print(f"  highest hinge = {hinges[hi_idx]:.4f}  model {hi_idx}: "
      f"w={models_all[hi_idx, :9].tolist()} theta={int(models_all[hi_idx, 9])}")

rng = np.random.default_rng(123)
sample = rng.choice(len(tr), 10, replace=False)
pred_lo = predict_class(models_all[lo_idx, :9], int(models_all[lo_idx, 9]), X_syn[sample])
pred_hi = predict_class(models_all[hi_idx, :9], int(models_all[hi_idx, 9]), X_syn[sample])

# For each of those 10 rows, print (feature vector, lo pred, hi pred, y).
print("\n  row | features (9 bits) | eta_lo | eta_hi | pred_lo | pred_hi | y_true")
for idx in sample:
    x = X_syn[idx]
    eta_lo = int(x @ models_all[lo_idx, :9])
    eta_hi = int(x @ models_all[hi_idx, :9])
    plo = predict_class(models_all[lo_idx, :9], int(models_all[lo_idx, 9]), x[None, :])[0]
    phi = predict_class(models_all[hi_idx, :9], int(models_all[hi_idx, 9]), x[None, :])[0]
    print(f"    {idx:>6}  {x.tolist()}  eta_lo={eta_lo:>3}  eta_hi={eta_hi:>3}  "
          f"pred_lo={CLASSES[plo]}  pred_hi={CLASSES[phi]}  y={CLASSES[y_syn[idx]]}")

# Aggregate bal_acc.
print(f"\n  bal_acc(lo) = {balanced_accuracy_score(y_syn, predict_class(models_all[lo_idx, :9], int(models_all[lo_idx, 9]), X_syn)):.5f}")
print(f"  bal_acc(hi) = {balanced_accuracy_score(y_syn, predict_class(models_all[hi_idx, :9], int(models_all[hi_idx, 9]), X_syn)):.5f}")


# ---------------- check 3: every synthetic row maps to a 10k cell ----------
log("check 3: do all synthetic rows map to one of the 128 training cells?")
cells_10k = {tuple(x) for x in X_orig}
missing = 0
for x in X_syn[:50000]:     # sample; the feature-space is discrete finite
    if tuple(x) not in cells_10k:
        missing += 1
print(f"  of 50k sampled synthetic rows, cells not in 10k: {missing}")
# Full check on deduplicated synthetic.
uniq_syn = np.unique(X_syn, axis=0)
missing_all = sum(1 for x in uniq_syn if tuple(x) not in cells_10k)
print(f"  unique synthetic cells: {len(uniq_syn)}  (expected <= 128)")
print(f"  synthetic cells NOT in 10k: {missing_all}")


# ---------------- check 4: independent verification against cell lookup ----
log("check 4: independent 128-cell lookup vs a random model's predictions")
# Build cell -> majority label from 10k (since 10k is separable, majority
# per cell is unambiguous).
from collections import Counter
cell_label: dict[tuple, int] = {}
for x, yi in zip(X_orig, y_orig):
    cell_label.setdefault(tuple(x), yi)
cell_pred = np.array([cell_label[tuple(x)] for x in X_syn], dtype=np.int8)
# Compare to a random model.
rand_idx = int(rng.integers(0, 743))
rand_pred = predict_class(models_all[rand_idx, :9], int(models_all[rand_idx, 9]), X_syn)
agree = (cell_pred == rand_pred).mean()
print(f"  random model idx {rand_idx} agreement with cell-lookup on 630k: "
      f"{agree:.6f}")
print(f"  bal_acc(cell-lookup) = {balanced_accuracy_score(y_syn, cell_pred):.5f}")
print(f"  bal_acc(rand model)  = {balanced_accuracy_score(y_syn, rand_pred):.5f}")


# ---------------- check 5: break a model deliberately and confirm drop ------
log("check 5: corrupt a valid model's weights -> bal_acc should drop")
broken = models_all[rand_idx].copy()
broken[0] = -broken[0] if broken[0] != 0 else 5   # flip Soil weight sign
bro_w = broken[:9]
bro_t = int(broken[9])
bro_pred = predict_class(bro_w, bro_t, X_syn)
bro_bal = balanced_accuracy_score(y_syn, bro_pred)
print(f"  corrupted-Soil-weight model: w={bro_w.tolist()} theta={bro_t}")
print(f"  bal_acc(broken) = {bro_bal:.5f}   (should be far below 0.96097)")
