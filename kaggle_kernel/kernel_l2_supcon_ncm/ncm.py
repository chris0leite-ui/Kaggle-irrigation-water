"""L2 — Mahalanobis NCM with macro-recall-Bayes-optimal decision rule.

Per-class LedoitWolf-shrunk covariance + uniform-prior posterior.
Mathematically: argmax_k log p(z | y=k) - log(1/3) ≡ argmax_k log p(z | y=k).
ZERO post-hoc bias retune (eliminates the leak channel that bounds every
prior +OOF/-LB carryover).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class MahalanobisNCM:
    def __init__(self, n_classes: int = 3, eps: float = 1e-6):
        self.n_classes = n_classes
        self.eps = eps
        self.means_: list[np.ndarray] = []
        self.precisions_: list[np.ndarray] = []
        self.logdet_: list[float] = []
        self.dim_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MahalanobisNCM":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        D = X.shape[1]
        self.dim_ = D
        self.means_.clear(); self.precisions_.clear(); self.logdet_.clear()
        for k in range(self.n_classes):
            mask = y == k
            n_k = int(mask.sum())
            if n_k < 2:
                mu = np.zeros(D, dtype=np.float64)
                cov = np.eye(D, dtype=np.float64)
            else:
                mu = X[mask].mean(axis=0)
                lw = LedoitWolf(assume_centered=False).fit(X[mask])
                cov = lw.covariance_ + self.eps * np.eye(D, dtype=np.float64)
            L = np.linalg.cholesky(cov)
            logdet = 2.0 * np.log(np.diag(L)).sum()
            inv_L = np.linalg.solve(L, np.eye(D, dtype=np.float64))
            precision = inv_L.T @ inv_L
            self.means_.append(mu)
            self.precisions_.append(precision)
            self.logdet_.append(float(logdet))
        return self

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        if D != self.dim_:
            raise ValueError(f"dim mismatch {D} vs {self.dim_}")
        ll = np.empty((N, self.n_classes), dtype=np.float64)
        const = -0.5 * D * np.log(2.0 * np.pi)
        for k in range(self.n_classes):
            diff = X - self.means_[k]
            mah = np.einsum("ni,ij,nj->n", diff, self.precisions_[k], diff)
            ll[:, k] = const - 0.5 * self.logdet_[k] - 0.5 * mah
        return ll

    def predict_proba_macro_recall(self, X: np.ndarray) -> np.ndarray:
        ll = self.log_likelihood(X)
        ll = ll - ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        p /= p.sum(axis=1, keepdims=True)
        return p.astype(np.float32)
