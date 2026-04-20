"""Blend LGBM-dist and empirical-Bayes-cell OOF probs.

Tests whether cell-level Bayesian probs bring orthogonal signal to the
distance-feature LGBM. Sweeps mixing weight on LGBM (α ∈ [0,1]), then
tunes log-bias on the blended probs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def main() -> None:
    tr = pd.read_csv("data/train.csv", usecols=[ID, TARGET])
    te_ids = pd.read_csv("data/test.csv", usecols=[ID])[ID].values
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    oof_dist = np.load(ART / "oof_lgbm_dist.npy")
    oof_eb = np.load(ART / "oof_eb_cell.npy")
    test_dist = np.load(ART / "test_lgbm_dist.npy")
    test_eb = np.load(ART / "test_eb_cell.npy")

    print(f"oof_dist shape {oof_dist.shape}, oof_eb shape {oof_eb.shape}")

    # sweep alpha in [0,1]: blend = alpha*LGBM + (1-alpha)*EB, in prob space
    alphas = np.linspace(0.0, 1.0, 21)
    rows = []
    for a in alphas:
        blend = a * oof_dist + (1 - a) * oof_eb
        bias, best = tune_log_bias(blend, y, prior)
        argmax_bal = balanced_accuracy_score(y, blend.argmax(axis=1))
        rows.append({"alpha": float(a), "argmax": float(argmax_bal),
                     "tuned": float(best), "bias": bias.tolist()})
        print(f"  alpha={a:0.2f}  argmax={argmax_bal:.5f}  tuned={best:.5f}")

    best_row = max(rows, key=lambda r: r["tuned"])
    print(f"\nbest blend: alpha={best_row['alpha']:.2f}  tuned={best_row['tuned']:.5f}")

    # also try log-space blend
    eps = 1e-9
    log_dist = np.log(np.clip(oof_dist, eps, 1))
    log_eb = np.log(np.clip(oof_eb, eps, 1))
    log_rows = []
    for a in alphas:
        logits = a * log_dist + (1 - a) * log_eb
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        bias, best = tune_log_bias(p, y, prior)
        log_rows.append({"alpha": float(a), "tuned": float(best),
                         "bias": bias.tolist()})
    best_log = max(log_rows, key=lambda r: r["tuned"])
    print(f"best log-blend: alpha={best_log['alpha']:.2f}  tuned={best_log['tuned']:.5f}")

    # build best submission (from the better of the two blends)
    if best_log["tuned"] > best_row["tuned"]:
        a = best_log["alpha"]
        test_blend_log = a * np.log(np.clip(test_dist, eps, 1)) + (1 - a) * np.log(np.clip(test_eb, eps, 1))
        test_p = np.exp(test_blend_log - test_blend_log.max(axis=1, keepdims=True))
        test_p /= test_p.sum(axis=1, keepdims=True)
        bias = np.array(best_log["bias"])
        tag = "log"
        best_tuned = best_log["tuned"]
    else:
        a = best_row["alpha"]
        test_p = a * test_dist + (1 - a) * test_eb
        bias = np.array(best_row["bias"])
        tag = "prob"
        best_tuned = best_row["tuned"]

    test_pred_idx = (np.log(np.clip(test_p, eps, 1)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te_ids, TARGET: [IDX2CLS[i] for i in test_pred_idx]}).to_csv(
        SUB / "submission_blend_eb_dist.csv", index=False
    )

    with open(ART / "blend_eb_dist_results.json", "w") as f:
        json.dump({
            "rows_prob_blend": rows,
            "rows_log_blend": log_rows,
            "best_prob": best_row,
            "best_log": best_log,
            "chosen_tag": tag,
            "chosen_alpha": float(a),
            "chosen_tuned": float(best_tuned),
        }, f, indent=2)
    print(f"submission written to {SUB/'submission_blend_eb_dist.csv'} (tag={tag}, alpha={a:.2f})")


if __name__ == "__main__":
    main()
