"""SVGP (Sparse Variational Gaussian Process) helpers for meta-stacking.

Architectural choice: GPC is the only meta-stacker family with a
non-parametric, kernel-based, smooth-posterior, uncertainty-quantified
inductive bias. Every prior meta tested (XGB / LR / MLP / RF) is
parametric. Hypothesis: smooth Matern-3/2 surface produces errors
orthogonal to the existing meta family on the saturated bank.

Multi-class via SoftmaxLikelihood + IndependentMultitask of K independent
GPs (one per class). Inducing points selected by stratified k-means on
training rows. ARD kernel: one length-scale per feature.

Theory-only HPs (no grid search on OOF, per CLAUDE.md leakage rule):
  - kernel: Matern(nu=1.5) with ONE shared length-scale (no ARD)
    Rationale: 200+ correlated meta features → ARD per-dim length-scales
    overfit fold-specific structure. Single length-scale = stronger
    Bayesian prior, the LB-honest default.
  - inducing points: M=300 stratified k-means centers
  - mini-batch: 4096
  - epochs: 20
  - lr: 5e-3 (Adam)
  - mll: VariationalELBO with num_data=N_train

ARD can be enabled via ARD_DIMS env var (e.g. ARD_DIMS=50 for 50-d PCA);
default None = no ARD.
"""
from __future__ import annotations

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  VariationalStrategy,
                                  IndependentMultitaskVariationalStrategy)
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO
from sklearn.cluster import MiniBatchKMeans

NUM_CLASSES = 3
DEFAULT_M = 300
DEFAULT_BATCH = 4096
DEFAULT_EPOCHS = 20
DEFAULT_LR = 5e-3


class SVGPModel(ApproximateGP):
    """K-task independent SVGP. Each task is one of {Low, Medium, High}.

    Independent variational strategies (not multitask correlated) — keeps
    per-class GPs decoupled, mirrors the multi:softprob assumption used
    everywhere else in this pipeline. SoftmaxLikelihood ties them through
    the cross-class prediction.
    """
    def __init__(self, inducing_points: torch.Tensor, num_classes: int = NUM_CLASSES):
        # inducing_points: (num_classes, M, D)
        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
            batch_shape=torch.Size([num_classes]),
        )
        var_strat = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(self, inducing_points, var_dist,
                                learn_inducing_locations=True),
            num_tasks=num_classes,
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_classes]))
        # ARD off by default: single shared length-scale per class. Cheaper
        # AND a stronger Bayesian prior on a 200+ dim correlated feature space.
        import os
        ard = os.environ.get("ARD_DIMS", "")
        ard_dim = int(ard) if ard.isdigit() else None
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5,
                         ard_num_dims=ard_dim,
                         batch_shape=torch.Size([num_classes])),
            batch_shape=torch.Size([num_classes]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def select_inducing(X: np.ndarray, y: np.ndarray, M: int, seed: int = 42
                    ) -> np.ndarray:
    """Stratified k-means: M/3 centers per class on the (whitened) features.

    Same M=300 budget split equally across classes ensures rare-class
    (High) is well-represented in the inducing set despite class imbalance.
    """
    M_per = M // NUM_CLASSES
    centers = []
    for c in range(NUM_CLASSES):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        # Subsample for k-means speed: use up to 50k rows per class.
        if len(idx) > 50_000:
            rng = np.random.default_rng(seed + c)
            idx = rng.choice(idx, 50_000, replace=False)
        km = MiniBatchKMeans(n_clusters=M_per, random_state=seed + c,
                             batch_size=4096, n_init=3, max_iter=50)
        km.fit(X[idx])
        centers.append(km.cluster_centers_)
    return np.vstack(centers).astype(np.float32)


def fit_svgp(X_tr: np.ndarray, y_tr: np.ndarray, *,
             M: int = DEFAULT_M, epochs: int = DEFAULT_EPOCHS,
             batch_size: int = DEFAULT_BATCH, lr: float = DEFAULT_LR,
             seed: int = 42, log=print
             ) -> tuple[SVGPModel, SoftmaxLikelihood]:
    """Train an SVGP on standardized features. Returns trained (model, likelihood).

    Caller is responsible for standardization (fit on tr only).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, D = X_tr.shape
    log(f"  SVGP: N={N} D={D} M={M} epochs={epochs} batch={batch_size}")

    Z = select_inducing(X_tr, y_tr, M=M, seed=seed)  # (M_eff, D)
    Z_t = torch.from_numpy(Z).float().unsqueeze(0).expand(NUM_CLASSES, -1, -1)
    model = SVGPModel(Z_t.contiguous())
    likelihood = SoftmaxLikelihood(num_classes=NUM_CLASSES, mixing_weights=False)
    model.train()
    likelihood.train()
    optim = torch.optim.Adam([{"params": model.parameters()},
                              {"params": likelihood.parameters()}], lr=lr)
    mll = VariationalELBO(likelihood, model, num_data=N)

    X_t = torch.from_numpy(X_tr).float()
    y_t = torch.from_numpy(y_tr).long()
    n_batches = (N + batch_size - 1) // batch_size
    for ep in range(epochs):
        perm = torch.randperm(N)
        loss_sum = 0.0
        for b in range(n_batches):
            sel = perm[b * batch_size:(b + 1) * batch_size]
            optim.zero_grad()
            out = model(X_t[sel])
            loss = -mll(out, y_t[sel])
            loss.backward()
            optim.step()
            loss_sum += float(loss.detach()) * sel.size(0)
        log(f"    ep {ep+1}/{epochs} loss={loss_sum/N:.4f}")
    return model, likelihood


@torch.no_grad()
def predict_proba(model: SVGPModel, likelihood: SoftmaxLikelihood,
                  X: np.ndarray, batch_size: int = 8192,
                  n_samples: int = 32) -> np.ndarray:
    """Return (N, NUM_CLASSES) calibrated class probabilities.

    SoftmaxLikelihood requires Monte-Carlo over the predictive latent;
    n_samples=32 is enough at 3 classes for tight per-row probabilities.
    """
    model.eval(); likelihood.eval()
    X_t = torch.from_numpy(X).float()
    out_list = []
    with gpytorch.settings.num_likelihood_samples(n_samples), \
         gpytorch.settings.fast_pred_var():
        for i in range(0, X.shape[0], batch_size):
            xb = X_t[i:i + batch_size]
            f = model(xb)
            samples = likelihood(f).probs.mean(0)  # (B, NUM_CLASSES)
            out_list.append(samples.cpu().numpy())
    return np.vstack(out_list).astype(np.float32)
