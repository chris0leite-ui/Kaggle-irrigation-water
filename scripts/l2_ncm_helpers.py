"""Mahalanobis nearest-class-mean classifier with macro-recall-Bayes-optimal
decision rule.

Mechanism (the load-bearing differentiator vs all 15 prior NN nulls):
  - Per-class Gaussian density in embedding space
  - Posterior under UNIFORM prior 1/3 (not empirical class frequency) is
    Bayes-optimal under macro-recall by construction
  - argmax of softmax(log_likelihood_k) → class prediction
  - ZERO post-hoc log-bias retune (this IS the calibration mechanism)

Why not sklearn QDA? Because sklearn's predict_proba shrinks via reg_param
to (1-λ)Σ + λ·trace(Σ)/D·I, which is fine, but we want explicit per-class
LedoitWolf shrinkage for robustness on small classes (High = 3.3% of train).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class MahalanobisNCM:
    """Per-class Gaussian likelihood + uniform-prior posterior."""

    def __init__(self, n_classes: int = 3, eps: float = 1e-6):
        self.n_classes = n_classes
        self.eps = eps
        self.means_: list[np.ndarray] = []
        self.precisions_: list[np.ndarray] = []  # Σ⁻¹
        self.logdet_: list[float] = []  # log |Σ|
        self.dim_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MahalanobisNCM":
        """Fit per-class LedoitWolf-shrunk covariance.

        Args:
            X: (N, D) float32 embedding matrix.
            y: (N,) int class labels in [0, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        D = X.shape[1]
        self.dim_ = D
        self.means_.clear()
        self.precisions_.clear()
        self.logdet_.clear()

        for k in range(self.n_classes):
            mask = y == k
            n_k = int(mask.sum())
            if n_k < 2:
                # Degenerate class → use identity
                mu = np.zeros(D, dtype=np.float64)
                cov = np.eye(D, dtype=np.float64)
            else:
                mu = X[mask].mean(axis=0)
                lw = LedoitWolf(assume_centered=False).fit(X[mask])
                cov = lw.covariance_
                # Numerical floor: ensure positive definite
                cov = cov + self.eps * np.eye(D, dtype=np.float64)
            # Cholesky-based logdet + precision (more stable than np.linalg.inv).
            L = np.linalg.cholesky(cov)
            logdet = 2.0 * np.log(np.diag(L)).sum()
            # precision = (L L^T)^{-1}; solve via triangular back-substitution
            inv_L = np.linalg.solve(L, np.eye(D, dtype=np.float64))
            precision = inv_L.T @ inv_L
            self.means_.append(mu)
            self.precisions_.append(precision)
            self.logdet_.append(float(logdet))
        return self

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log p(x | y=k) for each row x and each class k.

        Returns:
            (N, n_classes) float64 matrix.
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        if D != self.dim_:
            raise ValueError(f"dim mismatch: X has {D}, fit had {self.dim_}")
        ll = np.empty((N, self.n_classes), dtype=np.float64)
        const = -0.5 * D * np.log(2.0 * np.pi)
        for k in range(self.n_classes):
            diff = X - self.means_[k]
            # Mahalanobis distance: diff @ precision @ diff.T (diagonal)
            mah = np.einsum("ni,ij,nj->n", diff, self.precisions_[k], diff)
            ll[:, k] = const - 0.5 * self.logdet_[k] - 0.5 * mah
        return ll

    def predict_proba_macro_recall(self, X: np.ndarray) -> np.ndarray:
        """Posterior under UNIFORM prior 1/3 (Bayes-optimal under macro-recall).

        posterior_k(x) = exp(log_lik_k(x)) / sum_j exp(log_lik_j(x))

        This is mathematically equivalent to QDA with priors=[1/n_classes]*K
        but avoids sklearn's internal regularization and uses LedoitWolf
        shrinkage instead.
        """
        ll = self.log_likelihood(X)
        # Numerical-safe softmax across classes (per row).
        ll = ll - ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        p /= p.sum(axis=1, keepdims=True)
        return p.astype(np.float32)
