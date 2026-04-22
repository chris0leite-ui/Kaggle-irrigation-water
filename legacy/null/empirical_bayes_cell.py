"""128-cell empirical Bayes classifier.

Implements idea #1 from the balanced-accuracy brainstorm:

    The reverse-engineered DGP partitions every row into one of
    2^5 * 4 = 128 cells defined by the (dry, norain, hot, windy,
    nomulch) boolean tuple times the 4-valued Crop_Growth_Stage.
    On the 10k original dataset each cell has a single label. On
    630k synthetic, each cell has a *distribution* over labels
    induced by the boundary-band noise process. For each row we
    output P(y | cell) measured out-of-fold on the synthetic train
    set. That is the Bayes-optimal predictor given only the 6 rule
    features, and with log-bias tuning on top it sets the ceiling
    reachable without using any of the remaining 13 features.

Pipeline:
  - Build cell_id from the 6 rule features.
  - 5-fold stratified CV. In each fold, estimate P(y | cell) from
    the training portion using Laplace smoothing (alpha = 1/|Y|).
  - Write OOF probs to scripts/artifacts/oof_eb_cell.npy and test
    probs (trained on all synthetic train) to test_eb_cell.npy.
  - Tune per-class additive log-bias on OOF via coord-ascent.
  - Save summary to scripts/artifacts/eb_cell_results.json.

Balanced-accuracy metric is sklearn.metrics.balanced_accuracy_score.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")
OUT_DIR = Path("scripts/artifacts")
SUB_DIR = Path("submissions")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cell_id(df: pd.DataFrame) -> np.ndarray:
    """Return an integer in [0, 128) identifying the DGP cell of each row."""
    dry = (df["Soil_Moisture"].astype(float).values < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float).values < 300).astype(int)
    hot = (df["Temperature_C"].astype(float).values > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float).values > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(int)
    # stage categories observed on the synthetic competition data.
    # (Note: the 10k original uses "Fruiting" instead of "Sowing"; active
    #  set { Flowering, Vegetative } is unchanged between the two.)
    stage_cats = ["Flowering", "Harvest", "Sowing", "Vegetative"]
    stage_vals = pd.Categorical(
        df["Crop_Growth_Stage"].astype(str).values,
        categories=stage_cats,
        ordered=False,
    ).codes.astype(int)
    if (stage_vals < 0).any():
        raise ValueError(
            f"unseen Crop_Growth_Stage values: "
            f"{set(df['Crop_Growth_Stage'].astype(str)) - set(stage_cats)}"
        )
    # pack into a single int: stage * 32 + dry*16 + norain*8 + hot*4 + windy*2 + nomulch
    return (
        stage_vals * 32
        + dry * 16
        + norain * 8
        + hot * 4
        + windy * 2
        + nomulch
    ).astype(np.int32)


def cell_probs(cell_train: np.ndarray, y_train: np.ndarray, n_cells: int = 128) -> np.ndarray:
    """P(y | cell) with Laplace smoothing."""
    alpha = 1.0 / len(CLASSES)
    counts = np.full((n_cells, len(CLASSES)), alpha, dtype=np.float64)
    for c, yy in zip(cell_train, y_train):
        counts[c, yy] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, float]:
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
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    cell_tr = cell_id(tr)
    cell_te = cell_id(te)
    n_cells = 128
    log(f"cells used in train: {len(np.unique(cell_tr))}/{n_cells}")

    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(cell_tr, y)):
        probs = cell_probs(cell_tr[tr_idx], y[tr_idx], n_cells=n_cells)
        oof[va_idx] = probs[cell_tr[va_idx]]
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  argmax_bal_acc={fold_bal:.5f}")

    # full-train probs for the test set
    full_probs = cell_probs(cell_tr, y, n_cells=n_cells)
    test_pred = full_probs[cell_te]

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    log("tuning log-bias on OOF")
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"  bias = {dict(zip(CLASSES, bias.round(4)))}")

    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log(f"OOF confusion matrix (rows=true, cols=pred):\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== empirical-Bayes 128-cell summary (OOF bal_acc) ===")
    print(f"  argmax               : {argmax_bal:.5f}")
    print(f"  prior-reweight       : {reweight_bal:.5f}")
    print(f"  tuned log-bias       : {tuned_bal:.5f}")

    np.save(OUT_DIR / "oof_eb_cell.npy", oof)
    np.save(OUT_DIR / "test_eb_cell.npy", test_pred)
    with open(OUT_DIR / "eb_cell_results.json", "w") as f:
        json.dump(
            {
                "seed": SEED,
                "n_folds": N_FOLDS,
                "class_priors": prior.tolist(),
                "log_bias": bias.tolist(),
                "argmax_bal_acc": float(argmax_bal),
                "reweight_bal_acc": float(reweight_bal),
                "tuned_bal_acc": float(tuned_bal),
                "cells_used_in_train": int(len(np.unique(cell_tr))),
            },
            f,
            indent=2,
        )

    # submission using tuned bias
    log_test = np.log(np.clip(test_pred, 1e-9, 1.0))
    tuned_test_idx = (log_test + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        SUB_DIR / "submission_eb_cell_tuned.csv", index=False
    )
    log(f"artifacts written to {OUT_DIR}/; submission to {SUB_DIR}/")


if __name__ == "__main__":
    main()
