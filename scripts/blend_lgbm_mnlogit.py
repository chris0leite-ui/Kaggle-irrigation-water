"""
Blend the LGBM OOF with each MNLogit formula on OOF balanced accuracy.

Blend type: geometric mean in log-prob space with a single mixing weight
w ∈ [0, 0.5] on the MNLogit side:

    log_blend = (1 − w) · log(p_lgbm) + w · log(p_mnlogit)

For each formula, we sweep w, tune the log-bias at the best w, and
report whether any MNLogit contribution improves the tuned balanced
accuracy over LGBM-alone (0.97097 baseline).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ART = Path("scripts/artifacts")
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

tr = pd.read_csv("data/train.csv")
y = tr["Irrigation_Need"].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)


def tune_bias(oof: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    log_p = np.log(np.clip(oof, 1e-9, 1.0))

    def score(b: np.ndarray) -> float:
        return balanced_accuracy_score(y, (log_p + b).argmax(axis=1))

    b = -np.log(prior)
    best = score(b)
    grid = np.linspace(-2.5, 2.5, 51)
    for _ in range(20):
        improved = False
        for k in range(len(CLASSES)):
            base = b.copy()
            scores = []
            for g in grid:
                base[k] = b[k] + g
                scores.append(score(base))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                b[k] = b[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return b, best


def log_blend(p1: np.ndarray, p2: np.ndarray, w: float) -> np.ndarray:
    lb = (1 - w) * np.log(np.clip(p1, 1e-9, 1.0)) + w * np.log(np.clip(p2, 1e-9, 1.0))
    z = np.exp(lb - lb.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


lgb = np.load(ART / "oof_lgbm_baseline.npy")
_, baseline = tune_bias(lgb, y)
print(f"LGBM alone (tuned)                 bal_acc={baseline:.5f}")

rows = [{"blend": "LGBM-only", "w": 0.0, "tuned": baseline}]

for name in ["F1_minimal_balance", "F2_balance_plus_management", "F3_full_structural"]:
    mn = np.load(ART / f"oof_mnlogit_{name}.npy")
    best_w, best_s = 0.0, baseline
    for w in np.linspace(0.0, 0.5, 26):
        blend = log_blend(lgb, mn, w)
        _, s = tune_bias(blend, y)
        if s > best_s + 1e-6:
            best_s, best_w = s, w
    print(f"LGBM + {name:30s} best_w={best_w:.2f}  tuned={best_s:.5f}  "
          f"Δ={best_s - baseline:+.5f}")
    rows.append({"blend": f"LGBM + {name}", "w": float(best_w), "tuned": float(best_s),
                 "delta_vs_lgbm": float(best_s - baseline)})

with open(ART / "blend_lgbm_mnlogit_results.json", "w") as f:
    json.dump(rows, f, indent=2)
