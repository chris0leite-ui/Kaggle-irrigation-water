"""Shared helpers for the gated mixture-of-experts ensemble (#1).

Linear gate over low-dim features. Forward + backward in pure numpy;
fit per fold via L-BFGS on cross-entropy of the blended posterior.

Genuinely new on this competition: every prior blend used CONSTANT weights.
Here w_k(x) varies per row and is learned end-to-end against the metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import add_distance_features

EPS = 1e-9


def gate_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  expert_oof: list[np.ndarray],
                  expert_test: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build low-dim per-row gating features.

    The gate sees ~ (4 dist + 4 abs + dgp_score + rule_pred onehot
    + per-expert max_prob + per-expert argmax onehot) — small enough
    that a linear gate generalises.
    """
    train_eng = add_distance_features(train_df)
    test_eng = add_distance_features(test_df)
    base_cols = [
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "min_axis_abs", "min_boundary_dist",
        "dgp_score", "rule_pred",
    ]

    def block(df, ps):
        feats = [df[c].astype(np.float32).to_numpy() for c in base_cols]
        # Per-expert max_prob + argmax (one-hot 3 cls)
        for p in ps:
            feats.append(p.max(1).astype(np.float32))
            am = p.argmax(1)
            feats.append((am == 0).astype(np.float32))
            feats.append((am == 1).astype(np.float32))
            feats.append((am == 2).astype(np.float32))
        return np.stack(feats, axis=1)

    Xtr = block(train_eng, expert_oof)
    Xte = block(test_eng, expert_test)

    # Standardise per column on train.
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True).clip(EPS, None)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    # Bias column.
    Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1), dtype=np.float32)])
    Xte = np.hstack([Xte, np.ones((Xte.shape[0], 1), dtype=np.float32)])
    return Xtr.astype(np.float32), Xte.astype(np.float32)


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def forward_blend(W: np.ndarray, X: np.ndarray,
                  experts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """W: (K, d). X: (N, d). experts: (K, N, 3). Returns (gate (N, K), P (N, 3))."""
    g = X @ W.T                # (N, K)
    w = softmax_rows(g)        # (N, K)
    P = np.einsum("nk,knc->nc", w, experts)
    return w, np.clip(P, EPS, 1.0)


def loss_and_grad(W_flat: np.ndarray, X: np.ndarray, experts: np.ndarray,
                  y: np.ndarray, K: int, d: int,
                  l2: float = 1e-3) -> tuple[float, np.ndarray]:
    """Mean cross-entropy + L2 on W.

    P[i,c]  = sum_k w_ik * E_kic
    L       = -mean_i log P[i, y_i]
    dL/dP_yi = -1 / (N * P[i, y_i])
    dP[i,c] / dw_ik = E_kic
    dw_ik / dg_ij  = w_ik * (1[k==j] - w_ij)   (softmax jacobian)
    dg / dW_kj    = X[i, j]
    """
    W = W_flat.reshape(K, d)
    N = X.shape[0]
    g = X @ W.T               # (N, K)
    w = softmax_rows(g)       # (N, K)
    # P[i,c] = sum_k w[i,k] * experts[k,i,c]
    P = np.einsum("nk,knc->nc", w, experts)
    P = np.clip(P, EPS, 1.0)
    py = P[np.arange(N), y]
    loss = -np.log(py).mean() + 0.5 * l2 * (W * W).sum()

    # dL/dP[i, y_i] = -1/(N * py)
    # dL/dw[i, k] = sum_c dL/dP[i,c] * dP[i,c]/dw[i,k] = -1/(N*py) * experts[k,i,y_i]
    inv_py = 1.0 / (N * py)
    # gradient w.r.t. w: shape (N, K)
    dL_dw = -inv_py[:, None] * np.array(
        [experts[k, np.arange(N), y] for k in range(K)]
    ).T  # transpose to (N, K)
    # gradient through softmax: dL/dg[i,k] = w[i,k] * (dL/dw[i,k] - sum_j w[i,j]*dL/dw[i,j])
    s = (w * dL_dw).sum(1, keepdims=True)
    dL_dg = w * (dL_dw - s)   # (N, K)
    # dL/dW[k,j] = sum_i dL/dg[i,k] * X[i,j]
    dL_dW = dL_dg.T @ X        # (K, d)
    dL_dW += l2 * W
    return float(loss), dL_dW.ravel()


def fit_gate(X: np.ndarray, experts: np.ndarray, y: np.ndarray,
             K: int, l2: float = 1e-3, maxiter: int = 200,
             seed: int = 42) -> np.ndarray:
    """Fit linear gate via L-BFGS-B. Returns W (K, d)."""
    from scipy.optimize import minimize
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    W0 = (rng.standard_normal(K * d) * 0.01).astype(np.float64)
    res = minimize(
        loss_and_grad, W0,
        args=(X.astype(np.float32), experts.astype(np.float32), y.astype(np.int32), K, d, l2),
        jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "disp": False, "ftol": 1e-7, "gtol": 1e-6},
    )
    return res.x.reshape(K, d)
