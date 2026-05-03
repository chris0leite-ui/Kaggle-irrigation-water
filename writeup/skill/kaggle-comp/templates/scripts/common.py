"""Shared CV / OOF / metric utilities for a Kaggle tabular comp.

Generic across competitions. The kickoff agent edits CLASSES and
TASK_TYPE in comp-context.md and (if needed) updates this file's
defaults to match. New scripts should import from here rather than
copy-pasting fold/seed conventions.

Pinned conventions (matches every committed OOF / test artifact):
    SEED = 42
    N_FOLDS = 5
    StratifiedKFold(shuffle=True, random_state=SEED) for classification
    KFold(shuffle=True, random_state=SEED) for regression
    OOF shape: (n_train, n_class) for classification (rows sum to 1)
    test shape: (n_test, n_class) (averaged across folds, rows sum to 1)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

SEED = 42
N_FOLDS = 5
ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def folds(y, task: str = "classification"):
    """Yield (fold_idx, train_idx, val_idx) for the pinned 5-fold split."""
    if task == "classification":
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield k, tr, va
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (tr, va) in enumerate(kf.split(np.zeros(len(y)))):
            yield k, tr, va


def fast_bal_acc(y: np.ndarray, pred: np.ndarray, n_class: int) -> float:
    """Vectorised macro-recall. ~30x faster than sklearn on 1M+ rows."""
    class_counts = np.bincount(y, minlength=n_class)
    matches = (pred == y)
    hit = np.array([matches[y == k].sum() for k in range(n_class)],
                   dtype=np.int64)
    return float((hit / np.maximum(class_counts, 1)).mean())


def tune_log_bias(y: np.ndarray, proba: np.ndarray, metric: str = "bal_acc",
                  init: np.ndarray | None = None,
                  n_iter: int = 200, lr: float = 0.05) -> np.ndarray:
    """Coordinate-ascent log-bias tuning for balanced accuracy.

    Adds a per-class constant to log(p) before argmax. Equivalent to a
    class-conditional prior shift; for balanced accuracy this picks
    the macro-recall-optimal operating point.
    """
    n_class = proba.shape[1]
    bias = np.zeros(n_class) if init is None else np.asarray(init).copy()
    log_p = np.log(np.clip(proba, 1e-12, None))

    def score_at(b):
        pred = (log_p + b).argmax(1)
        if metric == "bal_acc":
            return fast_bal_acc(y, pred, n_class)
        elif metric == "acc":
            return float((pred == y).mean())
        else:
            raise ValueError(f"unknown metric {metric}")

    best = score_at(bias)
    for _ in range(n_iter):
        improved = False
        for c in range(n_class):
            for delta in (-lr, lr):
                trial = bias.copy()
                trial[c] += delta
                s = score_at(trial)
                if s > best + 1e-9:
                    bias = trial
                    best = s
                    improved = True
        if not improved:
            lr *= 0.5
            if lr < 1e-4:
                break
    return bias


def score(y, pred, metric: str, n_class: int | None = None) -> float:
    """Generic dispatch for the comp metric."""
    if metric == "bal_acc":
        return float(balanced_accuracy_score(y, pred))
    if metric == "log_loss":
        return float(log_loss(y, pred))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y, pred)))
    if metric == "auc":
        return float(roc_auc_score(y, pred))
    raise ValueError(f"unknown metric {metric}")


def save_oof(name: str, oof: np.ndarray, test: np.ndarray, results: dict):
    """Atomic save: write to tmp, then rename. Never leave half-written."""
    import json
    for arr, kind in [(oof, "oof"), (test, "test")]:
        tmp = ART / f"_tmp_{kind}_{name}.npy"
        final = ART / f"{kind}_{name}.npy"
        np.save(tmp, arr)
        tmp.rename(final)
    (ART / f"{name}_results.json").write_text(json.dumps(results, indent=2))
