"""Blend FE-enriched LGBM-dist 5-seed bag + XGBoost-dist OOF probs.

Mirrors scripts/blend_lgbm_xgb_dist.py but reads the FE-enriched
artefacts produced by scripts/seed_bag_dist_fe.py and
scripts/benchmark_xgb_dist_fe.py.

Inputs:
    scripts/artifacts/oof_lgbm_dist_fe_bag.npy
    scripts/artifacts/test_lgbm_dist_fe_bag.npy
    scripts/artifacts/oof_xgb_dist_fe.npy
    scripts/artifacts/test_xgb_dist_fe.npy

Outputs:
    scripts/artifacts/blend_lgbm_xgb_dist_fe_results.json
    submissions/submission_blend_lgbm_xgb_dist_fe.csv (best α)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

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

    oof_lgbm = np.load(ART / "oof_lgbm_dist_fe_bag.npy")
    oof_xgb = np.load(ART / "oof_xgb_dist_fe.npy")
    test_lgbm = np.load(ART / "test_lgbm_dist_fe_bag.npy")
    test_xgb = np.load(ART / "test_xgb_dist_fe.npy")
    print(f"oof_lgbm {oof_lgbm.shape}  oof_xgb {oof_xgb.shape}")

    _, lgbm_bal = tune_log_bias(oof_lgbm, y, prior)
    _, xgb_bal = tune_log_bias(oof_xgb, y, prior)
    print(f"standalone tuned  lgbm_fe_bag={lgbm_bal:.5f}  xgb_fe={xgb_bal:.5f}")

    alphas = np.linspace(0.0, 1.0, 21)
    rows_prob = []
    for a in alphas:
        blend = a * oof_lgbm + (1 - a) * oof_xgb
        bias, tuned = tune_log_bias(blend, y, prior)
        rows_prob.append({"alpha": float(a), "tuned": float(tuned),
                          "bias": bias.tolist()})
        print(f"  prob-blend  alpha={a:0.2f}  tuned={tuned:.5f}")
    best_prob = max(rows_prob, key=lambda r: r["tuned"])

    eps = 1e-9
    log_lgbm = np.log(np.clip(oof_lgbm, eps, 1.0))
    log_xgb = np.log(np.clip(oof_xgb, eps, 1.0))
    rows_log = []
    for a in alphas:
        logits = a * log_lgbm + (1 - a) * log_xgb
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        bias, tuned = tune_log_bias(p, y, prior)
        rows_log.append({"alpha": float(a), "tuned": float(tuned),
                         "bias": bias.tolist()})
    best_log = max(rows_log, key=lambda r: r["tuned"])
    print(f"\nbest prob-blend: alpha={best_prob['alpha']:.2f}  tuned={best_prob['tuned']:.5f}")
    print(f"best log-blend : alpha={best_log['alpha']:.2f}  tuned={best_log['tuned']:.5f}")

    if best_log["tuned"] > best_prob["tuned"]:
        a = best_log["alpha"]
        tag = "log"
        bias = np.array(best_log["bias"])
        tlog = a * np.log(np.clip(test_lgbm, eps, 1.0)) + (1 - a) * np.log(np.clip(test_xgb, eps, 1.0))
        test_p = np.exp(tlog - tlog.max(axis=1, keepdims=True))
        test_p /= test_p.sum(axis=1, keepdims=True)
        winner_tuned = best_log["tuned"]
    else:
        a = best_prob["alpha"]
        tag = "prob"
        bias = np.array(best_prob["bias"])
        test_p = a * test_lgbm + (1 - a) * test_xgb
        winner_tuned = best_prob["tuned"]

    cm = confusion_matrix(
        y, (np.log(np.clip(
            (a * oof_lgbm + (1 - a) * oof_xgb) if tag == "prob" else
            np.exp(a * log_lgbm + (1 - a) * log_xgb - (a * log_lgbm + (1 - a) * log_xgb).max(axis=1, keepdims=True)),
            eps, 1.0)) + bias).argmax(axis=1)
    )
    print(f"\nwinner: {tag}-blend alpha={a:.2f} tuned={winner_tuned:.5f}")
    print(f"OOF confusion matrix (winner):\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    test_pred_idx = (np.log(np.clip(test_p, eps, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te_ids, TARGET: [IDX2CLS[i] for i in test_pred_idx]}).to_csv(
        SUB / "submission_blend_lgbm_xgb_dist_fe.csv", index=False
    )

    with open(ART / "blend_lgbm_xgb_dist_fe_results.json", "w") as f:
        json.dump({
            "standalone_lgbm_fe_bag_tuned": float(lgbm_bal),
            "standalone_xgb_fe_tuned": float(xgb_bal),
            "prob_blend_rows": rows_prob,
            "log_blend_rows": rows_log,
            "best_prob": best_prob,
            "best_log": best_log,
            "winner_tag": tag,
            "winner_alpha": float(a),
            "winner_tuned": float(winner_tuned),
        }, f, indent=2)
    print(f"submission written to {SUB/'submission_blend_lgbm_xgb_dist_fe.csv'}")


if __name__ == "__main__":
    main()
