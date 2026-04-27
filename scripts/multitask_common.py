"""Multi-task XGB helpers: shared-tree joint y + aux objective.

XGB native API with `num_class=K` produces K trees per round. With a custom
objective, we own the gradient/Hessian shape (N, K). Here K=6:
  cols 0-2 → main 3-class softmax over y (Low / Medium / High)
  col  3   → binary sigmoid for `flipped_from_rule`  = (y != rule_pred)
  col  4   → binary sigmoid for `missed_high`        = (y==2) & (rule!=2)
  col  5   → binary sigmoid for `missed_medium`      = (y==1) & (rule!=1)

Auxiliary supervision is INSERTED AT TRAINING TIME — trees split to
minimize the joint loss, so the shared tree structure must serve both
tasks. This is structurally distinct from meta-stacker insertion (where
aux outputs are post-hoc features the meta-XGB overfits to OOF noise).

Why aux helps in principle: the host's NN-noise process flips ~10k rows.
Aux heads provide direct supervision on "is this row likely flipped?",
which is information the main softmax CE only sees indirectly. AUC
estimates from prior runs: flipped 0.90, missed_high 0.98, missed_med 0.95.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def softmax3(z: np.ndarray) -> np.ndarray:
    """Softmax over first 3 columns of a (N, 6) logit matrix."""
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))).astype(np.float32)


def build_aux_targets(y: np.ndarray, rule_pred: np.ndarray) -> np.ndarray:
    """Returns (N, 3) binary matrix of [flipped, missed_high, missed_med]."""
    flipped = (y != rule_pred).astype(np.float32)
    missed_h = ((y == 2) & (rule_pred != 2)).astype(np.float32)
    missed_m = ((y == 1) & (rule_pred != 1)).astype(np.float32)
    return np.stack([flipped, missed_h, missed_m], axis=1)


def make_multitask_obj(
    y_main: np.ndarray,           # (N,) int in {0,1,2}
    aux_targets: np.ndarray,       # (N, 3) float in {0,1}
    main_weight: float = 1.0,
    aux_weights: tuple = (0.3, 0.3, 0.3),
    sample_weight: np.ndarray | None = None,
):
    """Joint multi-task objective. Returns (grad, hess) in shape (N, 6).

    Main loss: weighted cross-entropy on softmax over cols 0-2.
    Aux losses: per-head weighted binary CE on sigmoid of cols 3-5.

    Hessian is the diagonal softmax/sigmoid approximation. XGB's hist
    splitter only uses the diagonal so this is exact for split selection.
    """
    y_main = y_main.astype(np.int64)
    aux_targets = aux_targets.astype(np.float32)
    N = len(y_main)
    one_hot = np.zeros((N, 3), dtype=np.float32)
    one_hot[np.arange(N), y_main] = 1.0

    if sample_weight is None:
        sw = np.ones(N, dtype=np.float32)
    else:
        sw = sample_weight.astype(np.float32)

    a1, a2, a3 = aux_weights

    def obj(preds: np.ndarray, dmat):
        # XGB ≥ 2.1 returns (N, K) for multi-output.  Older returns flat (N*K,).
        if preds.ndim == 1:
            z = preds.reshape(N, 6)
        else:
            z = preds
        # Main 3-class softmax.
        p_main = softmax3(z[:, :3])
        # Aux 3-binary sigmoids.
        p_aux = sigmoid(z[:, 3:])

        grad = np.empty((N, 6), dtype=np.float32)
        hess = np.empty((N, 6), dtype=np.float32)

        # Main task gradients (weighted by main_weight + per-row sw).
        grad[:, :3] = main_weight * sw[:, None] * (p_main - one_hot)
        hess[:, :3] = main_weight * sw[:, None] * (p_main * (1.0 - p_main) + EPS)

        # Aux task gradients (weighted by per-head weight + per-row sw).
        for k, w in enumerate((a1, a2, a3)):
            grad[:, 3 + k] = w * sw * (p_aux[:, k] - aux_targets[:, k])
            hess[:, 3 + k] = w * sw * (p_aux[:, k] * (1.0 - p_aux[:, k]) + EPS)

        return grad, hess

    return obj


def make_multitask_metric(y_val: np.ndarray):
    """Hard-label main-task mlogloss on val for early stopping."""
    N = len(y_val)
    eye = np.eye(3, dtype=np.float32)[y_val.astype(np.int64)]

    def metric(preds: np.ndarray, dmat):
        z = preds.reshape(N, 6) if preds.ndim == 1 else preds
        p_main = softmax3(z[:, :3])
        ll = -(eye * np.log(np.clip(p_main, EPS, 1.0))).sum(1).mean()
        return "main_logloss", float(ll)

    return metric


def margin_to_main_prob(raw_margin: np.ndarray, n_rows: int) -> np.ndarray:
    """Extract main 3-class softmax from the raw margin output."""
    z = raw_margin.reshape(n_rows, 6) if raw_margin.ndim == 1 else raw_margin
    return softmax3(z[:, :3])
