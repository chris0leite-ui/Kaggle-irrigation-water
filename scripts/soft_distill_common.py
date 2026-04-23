"""Helpers for soft-target distillation from the LB-best blend teacher.

Teacher = softmax(0.5 * log(recipe_full_te) + 0.5 * log(recipe_pseudolabel))
on OOF and test. This is the RAW posterior of the LB-0.97998 50/50 log-blend
(log-bias is an inference-time decision-rule, not part of the distribution
we distill). The student learns to reproduce this posterior on the same
recipe feature matrix; at inference we tune our own log-bias on the
student's OOF.

Why distillation might break the stacking ceiling:
  Every blend family so far compresses via argmax log-averaging of
  near-equivalent predictors. Distillation trains the student on the
  teacher's FULL per-row posterior — including the boundary uncertainty
  that drives macro-recall. Pseudo-label at tau=0.98 throws away 15 % of
  test rows AND collapses the kept rows to argmax — this keeps 100 % and
  uses full distribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ART = Path("scripts/artifacts")
EPS = 1e-9


def softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def build_teacher_oof(
    oof_recipe_path: Path = ART / "oof_recipe_full_te.npy",
    oof_pseudo_path: Path = ART / "oof_recipe_pseudolabel.npy",
    w_recipe: float = 0.5,
) -> np.ndarray:
    """Recipe × pseudolabel equal-weight log-blend, softmaxed (no bias)."""
    p_r = np.load(oof_recipe_path)
    p_p = np.load(oof_pseudo_path)
    assert p_r.shape == p_p.shape, f"shape mismatch: {p_r.shape} vs {p_p.shape}"
    log_r = np.log(np.clip(p_r, EPS, 1.0))
    log_p = np.log(np.clip(p_p, EPS, 1.0))
    logit_sum = w_recipe * log_r + (1.0 - w_recipe) * log_p
    return softmax(logit_sum)


def build_teacher_test(
    test_recipe_path: Path = ART / "test_recipe_full_te.npy",
    test_pseudo_path: Path = ART / "test_recipe_pseudolabel.npy",
    w_recipe: float = 0.5,
) -> np.ndarray:
    p_r = np.load(test_recipe_path)
    p_p = np.load(test_pseudo_path)
    log_r = np.log(np.clip(p_r, EPS, 1.0))
    log_p = np.log(np.clip(p_p, EPS, 1.0))
    logit_sum = w_recipe * log_r + (1.0 - w_recipe) * log_p
    return softmax(logit_sum)


def make_soft_xent_obj(y_soft: np.ndarray, n_class: int = 3):
    """Factory: returns an XGB custom obj closure that targets y_soft (N, K).

    Gradient: probs - y_soft   (K outputs per row, flat for XGB).
    Hessian:  probs * (1 - probs) + EPS   (diagonal softmax approx).
    """
    y_soft = y_soft.astype(np.float32)
    N, K = y_soft.shape
    assert K == n_class

    def obj(preds: np.ndarray, dtrain):
        # With num_class=K and custom obj, XGB stores preds flat in row-major
        # order over (sample, class). For single-tree-per-class boosters,
        # preds.shape == (N*K,). reshape to (N, K).
        assert preds.size == N * K, (
            f"unexpected preds size {preds.size}, expected {N*K} (N={N}, K={K})"
        )
        logits = preds.reshape(N, K)
        probs = softmax(logits)
        grad = (probs - y_soft).astype(np.float32)
        hess = (probs * (1.0 - probs) + EPS).astype(np.float32)
        # XGB >=2.1 wants (n_samples, n_classes) shape; older accepts flat.
        return grad, hess

    return obj


def make_val_metric(y_val: np.ndarray, n_class: int = 3):
    """Hard-label multi-class log-loss on val (for early stopping).

    Returns a custom_metric(preds, dtrain) -> (name, value). Lower is better.
    """
    N = len(y_val)
    K = n_class
    eye = np.eye(K, dtype=np.float32)[y_val]  # (N, K) one-hot

    def metric(preds: np.ndarray, dtrain):
        logits = preds.reshape(N, K)
        probs = softmax(logits)
        ll = -(eye * np.log(np.clip(probs, EPS, 1.0))).sum(1).mean()
        return "hard_logloss", float(ll)

    return metric


def margin_to_prob(raw_margin: np.ndarray, n_class: int = 3) -> np.ndarray:
    """XGB with custom obj returns raw margins. Softmax to get probs."""
    if raw_margin.ndim == 1:
        raw_margin = raw_margin.reshape(-1, n_class)
    return softmax(raw_margin)
