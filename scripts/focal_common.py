"""Focal loss helpers for multi-class XGB custom objective.

Focal loss (Lin et al. 2017):
    FL(p_y) = -alpha_y * (1 - p_y)^gamma * log(p_y)

We use the standard "focal-weighted CE" implementation (widely used on XGB):
    per-sample weight  w = alpha_y * (1 - p_y)^gamma
    grad   = w * (softmax(z) - one_hot(y))
    hess   = w * p * (1 - p)  + EPS    (diagonal softmax approx)

This is the "outer-weight" approximation: it treats w as constant in the
gradient step (ignoring dw/dz). Exact focal gradient has extra terms
from chain rule through (1-p_y)^gamma; empirically the approximation is
indistinguishable for gamma <= 3 and keeps the Hessian PSD.

Rare-class focus: alpha = [1.0, 1.0, HIGH_ALPHA] upweights High errors
per-sample. Combined with sample_weight='balanced' this stacks on top
of the prior-based correction; typically we use EITHER balanced
sample_weight OR focal alpha, not both (to avoid double-counting).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def make_focal_obj(y: np.ndarray, gamma: float = 2.0,
                   alpha: tuple[float, float, float] = (1.0, 1.0, 3.0),
                   n_class: int = 3):
    """Multi-class focal-weighted CE objective for xgb.train.

    y: hard labels (N,) int.
    gamma: focal exponent. 0 -> standard weighted CE; 2 is the Lin et al default.
    alpha: per-class weight. (1,1,3) gives High 3x leverage on top of (1-p)^gamma.
    """
    y = y.astype(np.int64)
    N = len(y)
    K = n_class
    alpha_np = np.asarray(alpha, dtype=np.float32)
    assert alpha_np.shape == (K,), alpha_np.shape
    one_hot = np.eye(K, dtype=np.float32)[y]  # (N, K)
    alpha_y = alpha_np[y]  # (N,)

    def obj(preds: np.ndarray, dtrain):
        assert preds.size == N * K, (
            f"unexpected preds size {preds.size}, expected {N*K}"
        )
        logits = preds.reshape(N, K)
        probs = softmax(logits)
        p_y = probs[np.arange(N), y]  # (N,)
        # Focal weight per-sample
        w = alpha_y * np.power(1.0 - p_y, gamma)  # (N,)
        w = w.astype(np.float32)
        # Weighted CE grad / hess
        grad = (probs - one_hot) * w[:, None]  # (N, K)
        hess = probs * (1.0 - probs) * w[:, None] + EPS  # (N, K)
        return grad.astype(np.float32), hess.astype(np.float32)

    return obj


def make_val_bal_acc(y_val: np.ndarray, n_class: int = 3):
    """Hard-label balanced accuracy (maximize). For early stopping on bal_acc
    directly rather than mlogloss, since we want macro-recall optimal."""
    y_val = y_val.astype(np.int64)
    N = len(y_val)
    K = n_class
    cc = np.bincount(y_val, minlength=K)

    def metric(preds: np.ndarray, dtrain):
        logits = preds.reshape(N, K)
        pred = logits.argmax(1)
        matches = (pred == y_val)
        hit = np.array([matches[y_val == k].sum() for k in range(K)],
                       dtype=np.int64)
        bal = float((hit / np.maximum(cc, 1)).mean())
        return "bal_acc", bal

    return metric


def margin_to_prob(raw_margin: np.ndarray, n_class: int = 3) -> np.ndarray:
    if raw_margin.ndim == 1:
        raw_margin = raw_margin.reshape(-1, n_class)
    return softmax(raw_margin)
