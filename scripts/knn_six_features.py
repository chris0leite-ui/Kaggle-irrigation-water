"""kNN sanity check on the 6 DGP features.

Encodes the 6 rule features into a common scale:
  Soil_Moisture, Rainfall_mm, Temperature_C, Wind_Speed_kmh  -> z-scored
  Mulching_Used                                              -> 0/1
  Crop_Growth_Stage                                          -> Kc value
                                                                (0, 0, 2, 2)
Reports:
  - 5-fold OOF bal_acc (argmax)
  - tuned log-bias bal_acc
  - comparison to rule (0.96097) and LGBM+DGP (0.97271)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
K = 50  # neighbourhood size

ART_DIR = Path("scripts/artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode6(df: pd.DataFrame) -> np.ndarray:
    cont = np.column_stack([
        df["Soil_Moisture"].astype(float).values,
        df["Rainfall_mm"].astype(float).values,
        df["Temperature_C"].astype(float).values,
        df["Wind_Speed_kmh"].astype(float).values,
    ])
    mulch = (df["Mulching_Used"].astype(str) == "No").astype(int).values
    kc = np.where(df["Crop_Growth_Stage"].astype(str).isin(["Flowering", "Vegetative"]), 2, 0)
    return np.column_stack([cont, mulch, kc])


log("loading data")
tr = pd.read_csv("data/train.csv")
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)

X6 = encode6(tr)
log(f"shape: {X6.shape}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X6, y)):
    t0 = time.time()
    scaler = StandardScaler().fit(X6[tr_idx])
    Xs_tr = scaler.transform(X6[tr_idx])
    Xs_va = scaler.transform(X6[va_idx])
    knn = KNeighborsClassifier(n_neighbors=K, n_jobs=-1, algorithm="ball_tree")
    knn.fit(Xs_tr, y[tr_idx])
    oof_probs[va_idx] = knn.predict_proba(Xs_va)
    fold_bal = balanced_accuracy_score(y[va_idx], oof_probs[va_idx].argmax(axis=1))
    log(f"  fold {fold+1}/{N_FOLDS}  k={K}  bal_acc={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

argmax_bal = balanced_accuracy_score(y, oof_probs.argmax(axis=1))
log(f"kNN argmax bal_acc = {argmax_bal:.5f}")


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
log(f"kNN tuned log-bias = {bias.round(4).tolist()}  bal_acc={tuned_bal:.5f}")

print("\n=== summary kNN vs prior results (OOF balanced accuracy) ===")
for name, val in [
    ("rule-only argmax",                   0.96097),
    ("kNN 6-feat (k=50) argmax",           argmax_bal),
    ("kNN 6-feat (k=50) tuned log-bias",   tuned_bal),
    ("LGBM+DGP tuned log-bias",            0.97271),
    ("boundary-LGBM tuned log-bias",       0.97284),
]:
    print(f"  {name:<38s} {val:.5f}")

print("\nconfusion (OOF argmax):")
print(pd.DataFrame(
    confusion_matrix(y, oof_probs.argmax(axis=1)),
    index=CLASSES, columns=CLASSES,
))

np.save(ART_DIR / "oof_knn6.npy", oof_probs)
with open(ART_DIR / "knn6_results.json", "w") as f:
    json.dump(
        {"k": K, "argmax_bal": float(argmax_bal), "tuned_bal": float(tuned_bal),
         "tuned_bias": bias.tolist()},
        f, indent=2,
    )
log(f"artefacts saved to {ART_DIR}/knn6_results.json")
