"""Shared utilities for the irrigation pipeline.

Pinned conventions:
    5-fold StratifiedKFold, shuffle=True, random_state=42
    classes: Low=0, Medium=1, High=2
    OOF shape: (n_train, 3)  test shape: (n_test, 3)  rows sum to 1
"""
from __future__ import annotations

import numpy as np

SEED = 42
N_FOLDS = 5
CLASSES = ("Low", "Medium", "High")
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def fast_bal_acc(y: np.ndarray, pred: np.ndarray, n_class: int = 3,
                 class_counts: np.ndarray | None = None) -> float:
    """Vectorised macro-recall. ~30x faster than sklearn on 630k rows."""
    if class_counts is None:
        class_counts = np.bincount(y, minlength=n_class)
    matches = (pred == y)
    hit = np.array([matches[y == k].sum() for k in range(n_class)], dtype=np.int64)
    return float((hit / np.maximum(class_counts, 1)).mean())


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray,
                  eps: float = 1e-9):
    """Coord-ascent per-class log-bias on full OOF.

    Grid `-3..+6` for High — empirical optimum is ~+3.4 under severe class
    imbalance. Returns (bias, best_tuned_bal_acc).
    """
    log_oof = np.log(np.clip(oof, eps, 1.0))
    bias = -np.log(prior)
    cc = np.bincount(y, minlength=3)
    best = fast_bal_acc(y, (log_oof + bias).argmax(1), class_counts=cc)
    grid_default = np.linspace(-3.0, 3.0, 61)
    grid_high = np.linspace(-3.0, 6.0, 91)
    for _ in range(25):
        improved = False
        for k in range(3):
            grid = grid_high if k == 2 else grid_default
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(fast_bal_acc(y, (log_oof + base).argmax(1), class_counts=cc))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def log_blend(oofs: list[np.ndarray], weights: np.ndarray,
              eps: float = 1e-9) -> np.ndarray:
    """Weighted geometric mean in probability space."""
    w = weights / weights.sum()
    logits = np.zeros_like(oofs[0])
    for wi, o in zip(w, oofs):
        logits += wi * np.log(np.clip(o, eps, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)
