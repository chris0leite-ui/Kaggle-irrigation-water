"""Multiclass focal-loss custom objective for XGBoost.

Focal loss (Lin et al. 2017, generalized to multiclass):
    L = -α_y * (1 - p_y)^γ * log(p_y)     where p = softmax(z)

Gradient (w.r.t. logit z_c):
    ∂L/∂z_c = α_y * A * (p_c - [c==y])
where
    A = (1-p_y)^γ - γ * p_y * (1-p_y)^(γ-1) * log(p_y)
    (A ≥ 0 because log p_y ≤ 0)

Hessian (diagonal approximation, standard in focal-XGB implementations):
    ∂²L/∂z_c² ≈ α_y * A * p_c * (1 - p_c)

Sanity: when γ=0, A=1 and we recover standard softmax CE.

The objective is a CLOSURE over (γ, per-class α, per-row sample_weight).
Sample weights are applied INSIDE the objective (multiplied into
grad/hess); we do NOT pass them to DMatrix, so xgboost's internal
scaling semantics don't interfere with the focal modulation.

Also exposes a `feval_bal_acc` that returns MACRO-RECALL at a temporary
log-bias that is re-solved each call (expensive on full data; we restrict
to a subsample for speed).
"""
from __future__ import annotations

import numpy as np


def make_focal_multi_obj(
    gamma: float = 2.0,
    alpha: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    K: int = 3,
    eps: float = 1e-7,
):
    """Build an xgb obj closure for multi:softprob with per-class α and γ.

    sample_weight: optional (N,) array; applied as a multiplier on top of α_y.
    """
    if alpha is None:
        alpha = np.ones(K, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    def obj(preds, dtrain):
        y = dtrain.get_label().astype(np.int64)
        N = y.shape[0]
        # preds for multi:softprob is shape (N, K) already in xgboost ≥1.x
        if preds.ndim == 1:
            z = preds.reshape(N, K)
        else:
            z = preds
        # Stable softmax
        zmax = z.max(axis=1, keepdims=True)
        ez = np.exp(z - zmax)
        p = ez / ez.sum(axis=1, keepdims=True)
        p = np.clip(p, eps, 1.0 - eps)

        rows = np.arange(N)
        p_y = p[rows, y]
        one_minus = 1.0 - p_y

        if gamma == 0.0:
            A = np.ones(N, dtype=np.float64)
        else:
            mod = one_minus ** gamma
            dmod = gamma * (one_minus ** (gamma - 1.0)) * p_y * np.log(p_y)
            A = mod - dmod  # positive

        w = alpha[y] * A
        if sample_weight is not None:
            w = w * sample_weight

        grad = p.copy()
        grad[rows, y] -= 1.0
        grad = grad * w[:, None]

        hess = (w[:, None] * p * (1.0 - p))
        # Safety floor — xgboost needs strictly positive hessian for
        # split finding to behave; clip at a small value.
        hess = np.maximum(hess, 1e-12)

        grad = grad.astype(np.float32, copy=False)
        hess = hess.astype(np.float32, copy=False)
        if preds.ndim == 1:
            return grad.reshape(-1), hess.reshape(-1)
        return grad, hess

    return obj


# ------------------------------- unit test --------------------------------

def _check_gradient_recovers_ce():
    """γ=0, uniform α, no sample_weight → focal obj ≡ standard softmax CE.

    Standard CE grad: p_c - [c==y]. Hessian diag: p_c * (1-p_c).
    """
    rng = np.random.default_rng(0)
    N, K = 8, 3
    z = rng.standard_normal((N, K))
    y = rng.integers(0, K, size=N)

    class FakeDMat:
        def get_label(self):
            return y.astype(np.float32)

    obj = make_focal_multi_obj(gamma=0.0, K=K)
    grad, hess = obj(z, FakeDMat())

    # Reference
    zmax = z.max(axis=1, keepdims=True)
    ez = np.exp(z - zmax); p = ez / ez.sum(axis=1, keepdims=True)
    ref_grad = p.copy()
    ref_grad[np.arange(N), y] -= 1.0
    ref_hess = p * (1.0 - p)

    assert np.allclose(grad, ref_grad, atol=1e-5), \
        f"focal γ=0 gradient does not match CE: max diff {np.abs(grad - ref_grad).max()}"
    assert np.allclose(hess, ref_hess, atol=1e-5), \
        f"focal γ=0 hessian does not match CE: max diff {np.abs(hess - ref_hess).max()}"
    print("[focal] γ=0 grad/hess match softmax CE ✓")


def _check_gradient_numeric(gamma=2.0):
    """Numerical gradient check for γ>0 via finite differences."""
    rng = np.random.default_rng(1)
    N, K = 5, 3
    z = rng.standard_normal((N, K)).astype(np.float64)
    y = rng.integers(0, K, size=N)

    class FakeDMat:
        def get_label(self):
            return y.astype(np.float32)

    obj = make_focal_multi_obj(gamma=gamma, K=K)
    grad_ana, _ = obj(z, FakeDMat())
    grad_ana = np.asarray(grad_ana, dtype=np.float64)

    def loss_total(zz):
        zmax = zz.max(axis=1, keepdims=True)
        ez = np.exp(zz - zmax); p = ez / ez.sum(axis=1, keepdims=True)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        p_y = p[np.arange(N), y]
        return float(np.sum(-((1 - p_y) ** gamma) * np.log(p_y)))

    # Finite-difference per element
    h = 1e-4
    grad_num = np.zeros_like(z)
    for i in range(N):
        for c in range(K):
            zp = z.copy(); zp[i, c] += h
            zm = z.copy(); zm[i, c] -= h
            grad_num[i, c] = (loss_total(zp) - loss_total(zm)) / (2 * h)

    err = np.abs(grad_ana - grad_num).max()
    rel = err / (np.abs(grad_num).max() + 1e-12)
    print(f"[focal] γ={gamma} max abs grad err = {err:.2e}  rel = {rel:.2e}")
    assert rel < 1e-3, "focal gradient doesn't match numerical FD"


if __name__ == "__main__":
    _check_gradient_recovers_ce()
    _check_gradient_numeric(gamma=1.0)
    _check_gradient_numeric(gamma=2.0)
    _check_gradient_numeric(gamma=3.0)
    print("[focal] all unit tests passed")
