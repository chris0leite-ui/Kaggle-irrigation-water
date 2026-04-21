"""Enumerate all integer linear models that perfectly separate the original
10k dataset, then score each one's extrapolation to the 630k synthetic.

Reproduces @broccoli-beef's OR-Tools CP search from the competition
discussion (post 692754). Our existing integer rule is ONE of these
solutions; there are ~743 of them under |w| <= 10, theta <= 10, each
with 100.0000 % training accuracy on the 10k original.

They differ in **hinge loss**. Max-margin argument (VC / margin
bounds): the lowest-hinge-loss solution is the best single candidate
for extrapolation to unseen data. The question this script answers:

  Does lower hinge loss on the 10k correlate with higher balanced
  accuracy on the 630k synthetic? Which specific (w, theta) wins?

If the winner has meaningfully different weights from cdeotte's rule
(which our DGP features are built on), we have a drop-in replacement
at zero additional inference cost.

Feature space (9-dim, exactly as in the discussion -- the discussion's
column-name "Soil<26" is a display label; the actual separating
inequality is `Soil_Moisture < 25`, confirmed by sweeping +/- 0.5
around each threshold):
  x1 = Soil_Moisture < 25
  x2 = Temperature_C > 30
  x3 = Rainfall_mm   < 300
  x4 = Wind_Speed_kmh > 10
  x5 = Mulching_Used == "Yes"
  x6 = Crop_Growth_Stage == "Flowering"
  x7 = Crop_Growth_Stage == "Harvest"
  x8 = Crop_Growth_Stage == "Sowing"
  x9 = Crop_Growth_Stage == "Vegetative"

Model
  lambda  = sum_i w_i * x_i
  class   = Low    if lambda <= 0
            Medium if 0 < lambda <= theta
            High   if lambda > theta
(theta_0 = 0 without loss of generality -- any constant shift can be
 absorbed because x6+x7+x8+x9 = 1.)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, hinge_loss

SEED = 42
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART_DIR = Path("scripts/artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

FEAT_COLS = [
    "Soil<25", "Temp>30", "Rain<300", "Wind>10", "Mulching=Yes",
    "Crop=Flowering", "Crop=Harvest", "Crop=Sowing", "Crop=Vegetative",
]


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


# --------------------------------------------------------------------- data ---
log("loading data")
orig = pd.read_csv("data/archive.zip")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")

X_orig = make_features(orig)
y_orig = orig[TARGET].map(CLS2IDX).values
X_syn = make_features(tr)
y_syn = tr[TARGET].map(CLS2IDX).values
X_test = make_features(te)

prior_syn = np.bincount(y_syn) / len(y_syn)

# De-duplicate original feature vectors for CP (labels must agree per cell).
uniq_rows, first_idx = np.unique(X_orig, axis=0, return_index=True)
uniq_labels = y_orig[first_idx]
# Check: within each unique row vector, do all 10k original rows share a label?
from collections import defaultdict
row_to_labels: dict[tuple, set] = defaultdict(set)
for r, lab in zip(X_orig, y_orig):
    row_to_labels[tuple(r)].add(int(lab))
conflicts = [k for k, v in row_to_labels.items() if len(v) > 1]
log(f"  original unique feature-rows: {len(uniq_rows)} / {len(X_orig)}  label-conflicts: {len(conflicts)}")
assert len(conflicts) == 0, "original is not exactly separable in this feature space"


# ------------------------------------------------------------ CP enumeration ---
log("enumerating all integer models with |w| <= 10, theta in [1, 10]")
cp = cp_model.CpModel()
w = [cp.NewIntVar(-10, 10, f"w_{j}") for j in range(9)]
theta = cp.NewIntVar(1, 10, "theta")

for i, (x_row, y_lab) in enumerate(zip(uniq_rows, uniq_labels)):
    eta = sum(int(x_row[j]) * w[j] for j in range(9))
    if y_lab == 0:      # Low
        cp.Add(eta <= 0)
    elif y_lab == 1:    # Medium
        cp.Add(eta > 0)
        cp.Add(eta <= theta)
    else:               # High
        cp.Add(eta > theta)

solutions: list[np.ndarray] = []


class _Collect(cp_model.CpSolverSolutionCallback):
    def on_solution_callback(self):
        wv = np.array([self.Value(v) for v in w], dtype=np.int64)
        tv = int(self.Value(theta))
        # Canonicalise: drop solutions whose (w, theta) share a common
        # factor -- those are scale copies of a smaller solution.
        if math.gcd(*wv.tolist(), tv) == 1:
            solutions.append(np.concatenate([wv, [tv]]))


solver = cp_model.CpSolver()
solver.parameters.enumerate_all_solutions = True
solver.parameters.num_search_workers = 1
t0 = time.time()
status = solver.Solve(cp, _Collect())
log(f"  CP finished in {time.time() - t0:.1f}s  status={solver.StatusName(status)}  models={len(solutions)}")


# --------------------------------------------------- score every solution ---
def predict_synth(w_vec: np.ndarray, theta_val: int, X: np.ndarray) -> np.ndarray:
    eta = X @ w_vec.astype(np.int64)
    pred = np.where(eta <= 0, 0, np.where(eta <= theta_val, 1, 2))
    return pred.astype(np.int32)


def multiclass_hinge(w_vec: np.ndarray, theta_val: int,
                     X: np.ndarray, y: np.ndarray) -> float:
    """Construct the 3-class linear decision matrix Lambda per the
    discussion:
        row0 = -w  (class Low score)
        row1 =  0  (class Medium score; arbitrary zero-point)
        row2 =  w - theta*one_hot_stage_offset (class High score)

    Then use sklearn.metrics.hinge_loss's multiclass formula on the
    row-scored matrix X @ Lambda.T.  Replicates the hinge_loss used
    in the discussion post.
    """
    coef = np.array([
        -w_vec,
        np.zeros_like(w_vec),
        w_vec - np.array([0] * 5 + [theta_val] * 4),
    ], dtype=np.float64)
    return float(hinge_loss(y, X.astype(np.float64) @ coef.T, labels=[0, 1, 2]))


log("scoring every integer model on synthetic 630k")
records = []
t0 = time.time()
for idx, sol in enumerate(solutions):
    wv = sol[:9]
    tv = int(sol[9])
    pred_syn = predict_synth(wv, tv, X_syn)
    pred_orig = predict_synth(wv, tv, X_orig)
    records.append({
        "id": idx,
        "w": wv.tolist(),
        "theta": tv,
        "hinge": multiclass_hinge(wv, tv, X_orig, y_orig),
        "train_acc_orig": float((pred_orig == y_orig).mean()),
        "bal_acc_syn": balanced_accuracy_score(y_syn, pred_syn),
        "raw_acc_syn": float((pred_syn == y_syn).mean()),
    })
log(f"  scored {len(records)} models in {time.time() - t0:.1f}s")

df = pd.DataFrame([
    {
        "id": r["id"],
        "hinge": r["hinge"],
        "train_acc_orig": r["train_acc_orig"],
        "bal_acc_syn": r["bal_acc_syn"],
        "raw_acc_syn": r["raw_acc_syn"],
        **{fc: r["w"][i] for i, fc in enumerate(FEAT_COLS)},
        "theta": r["theta"],
    }
    for r in records
])
df = df.sort_values(["hinge", "theta"]).reset_index(drop=True)

assert (df["train_acc_orig"] == 1.0).all(), "non-separating solution leaked in"


# --------------------------------------------------------------- reporting ---
print("\n=== integer separating models: top-10 by LOW hinge loss ===")
cols = ["hinge", "theta", "bal_acc_syn", "raw_acc_syn"] + FEAT_COLS
print(df[cols].head(10).to_string(index=False))

print("\n=== integer separating models: top-10 by HIGH synthetic bal_acc ===")
best = df.sort_values("bal_acc_syn", ascending=False)[cols].head(10)
print(best.to_string(index=False))

print("\n=== integer separating models: LARGEST hinge loss (for contrast) ===")
print(df[cols].tail(5).to_string(index=False))

# Correlation between hinge loss on 10k and synthetic bal_acc.
rho = df[["hinge", "bal_acc_syn"]].corr(method="spearman").iloc[0, 1]
print(f"\nSpearman rank correlation  hinge <-> bal_acc_syn  = {rho:+.4f}")
print("  (expected NEGATIVE under the max-margin story: lower hinge => better transfer.)")

# Rule used by our DGP features (legacy):
#   Soil<25 +2, Temp>30 +1, Rain<300 +2, Wind>10 +1,
#   Mulching=Yes -1, Crop in {Flowering, Vegetative} +2
legacy_ours = np.array([2, 1, 2, 1, -1, 2, 0, 0, 2])  # theta=3 (Low if score<=0)
# cdeotte's exactly as posted:
cdeotte = np.array([2, 1, 2, 1, -1, 0, -2, -2, 0])
svm     = np.array([4, 2, 4, 2, -2, -1, -5, -5, -1])

for name, wv, tv in [("legacy_ours (Soil<25)", legacy_ours, 3),
                     ("cdeotte", cdeotte, 3),
                     ("SVM (discussion)", svm, 6)]:
    # legacy_ours uses Soil<25, others use Soil<26 -- rescore on same
    # X_orig for apples-to-apples.  (Boundary rows may be mislabeled
    # under the Soil<25 parameterisation but same applies everywhere.)
    pred_orig = predict_synth(wv, tv, X_orig)
    pred_syn = predict_synth(wv, tv, X_syn)
    h = multiclass_hinge(wv, tv, X_orig, y_orig)
    print(
        f"\nreference model: {name:<24}  "
        f"w={wv.tolist()} theta={tv}\n"
        f"  train_acc_orig={(pred_orig==y_orig).mean():.5f}  "
        f"hinge={h:.4f}  "
        f"bal_acc_syn={balanced_accuracy_score(y_syn, pred_syn):.5f}"
    )


# ----------------------------------------------------------- artifacts -------
df.to_csv(ART_DIR / "integer_separating_models.csv", index=False)
np.save(ART_DIR / "integer_models_raw.npy", np.array([np.concatenate([r["w"], [r["theta"]]]) for r in records]))

# Save per-row predictions (synthetic + test) for the top-K by bal_acc so
# we can stack or blend them later without re-enumerating.
TOP_K = 50
top_ids = df.sort_values("bal_acc_syn", ascending=False).head(TOP_K)["id"].tolist()
pred_syn_topk = np.stack([
    predict_synth(np.array(records[i]["w"]), records[i]["theta"], X_syn)
    for i in top_ids
], axis=1).astype(np.int8)  # (n_syn, TOP_K)
pred_test_topk = np.stack([
    predict_synth(np.array(records[i]["w"]), records[i]["theta"], X_test)
    for i in top_ids
], axis=1).astype(np.int8)
np.save(ART_DIR / "integer_models_topk_pred_syn.npy", pred_syn_topk)
np.save(ART_DIR / "integer_models_topk_pred_test.npy", pred_test_topk)
np.save(ART_DIR / "integer_models_topk_ids.npy", np.array(top_ids, dtype=np.int64))

with open(ART_DIR / "integer_models_summary.json", "w") as f:
    json.dump({
        "n_models": len(records),
        "n_unique_feature_rows_orig": int(len(uniq_rows)),
        "spearman_hinge_vs_bal_acc": rho,
        "top1_by_bal_acc": df.sort_values("bal_acc_syn", ascending=False).head(1).to_dict("records")[0],
        "top1_by_low_hinge": df.head(1).to_dict("records")[0],
        "top_k_saved": TOP_K,
    }, f, indent=2, default=float)

log(f"artifacts written to {ART_DIR}/")
