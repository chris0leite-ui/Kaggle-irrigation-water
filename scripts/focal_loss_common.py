"""Multi-class focal-loss objective for XGBoost with per-class alpha weights.

Focal loss (Lin et al. 2017) for multi-class:
    L_i = -alpha[y_i] * (1 - p_{i,y_i})^gamma * log(p_{i,y_i})

where p = softmax(logits).

Under balanced accuracy with a 58/38/3 class prior, the rare High class drives
macro-recall. Every prior attempt to help High (log-bias, router, detector
override) is post-hoc on a model trained with class-balanced sample_weight.
Focal loss moves capacity at training time toward low-confidence / rare-class
rows, producing genuinely different error geometry — not a re-mapping of the
existing bank.

Gradient derivation:
    Let q = p_y. dL/dq = alpha_y * [gamma (1-q)^{gamma-1} log(q) - (1-q)^gamma / q]
    Softmax Jacobian: dp_y/dz_j = q * ([j==y] - p_j)
    grad_j = S(q,gamma) * (p_j - [j==y])
    where  S(q,gamma) = alpha_y * [(1-q)^gamma - gamma * q * (1-q)^{gamma-1} * log(q)]

At gamma=0 this reduces to standard CE: grad_j = alpha_y * (p_j - [j==y]).

Hessian: use the standard softmax-CE diagonal approximation scaled by focal
factor: h_ij = alpha_y * (1-q)^gamma * p_ij * (1 - p_ij) + EPS. Empirically
safer than the true 2nd derivative (which can be negative for gamma > 1).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def make_focal_obj(y_true: np.ndarray, alpha: np.ndarray,
                   gamma: float = 2.0, n_class: int = 3):
    """Factory: return an XGB custom-obj closure for multi-class focal.

    Args:
        y_true: (N,) int32 class labels in [0, n_class).
        alpha:  (K,) per-class weight. Typical: alpha = 1 / prior (inverse
                frequency), with optional extra boost on the rare class.
        gamma:  focal modulating exponent. 2.0 is the Lin et al. default.
        n_class: number of classes.
    """
    y_true = y_true.astype(np.int64)
    alpha = alpha.astype(np.float32)
    N = len(y_true)
    K = n_class
    assert alpha.shape == (K,), alpha.shape
    eye = np.eye(K, dtype=np.float32)[y_true]  # (N, K) one-hot
    alpha_per_row = alpha[y_true]              # (N,)

    def obj(preds: np.ndarray, dtrain):
        assert preds.size == N * K, f"preds {preds.size} != N*K ({N*K})"
        logits = preds.reshape(N, K)
        probs = softmax(logits)
        q = probs[np.arange(N), y_true]                # (N,) p_{i, y_i}
        one_minus_q = np.clip(1.0 - q, EPS, 1.0)
        log_q = np.log(np.clip(q, EPS, 1.0))

        # Scalar focal factor per row (broadcast to K later).
        focal_pow_gamma = one_minus_q ** gamma                       # (N,)
        focal_pow_gm1 = one_minus_q ** (gamma - 1.0) if gamma > 0 else np.ones(N, np.float32)
        S = alpha_per_row * (focal_pow_gamma - gamma * q * focal_pow_gm1 * log_q)

        grad = (S[:, None] * (probs - eye)).astype(np.float32)
        # Hessian: scaled softmax-CE diag.
        h_scale = (alpha_per_row * focal_pow_gamma).astype(np.float32)
        hess = (h_scale[:, None] * probs * (1.0 - probs) + EPS).astype(np.float32)
        return grad, hess

    return obj


def make_hard_val_metric(y_val: np.ndarray, n_class: int = 3):
    """Hard-label multi-class log-loss for early stopping. Lower is better."""
    N = len(y_val)
    K = n_class
    eye = np.eye(K, dtype=np.float32)[y_val]

    def metric(preds: np.ndarray, dtrain):
        logits = preds.reshape(N, K)
        probs = softmax(logits)
        ll = -(eye * np.log(np.clip(probs, EPS, 1.0))).sum(1).mean()
        return "hard_logloss", float(ll)

    return metric


def margin_to_prob(raw_margin: np.ndarray, n_class: int = 3) -> np.ndarray:
    if raw_margin.ndim == 1:
        raw_margin = raw_margin.reshape(-1, n_class)
    return softmax(raw_margin)
