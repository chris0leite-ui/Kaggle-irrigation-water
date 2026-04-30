"""T2 helpers — split-conformal nonconformity & coverage tools."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ART = Path("scripts/artifacts")

BANK_NAMES = [
    "sklearn_rf_meta_natural",
    "sklearn_rf_meta_natural_a1lgbm",
    "sklearn_rf_meta_natural_r10_with_tier1b",
    "rawashishsin_2600",
    "tier1b_greedy_meta",
    "recipe_full_te",
    "recipe_pseudolabel",
    "recipe_pseudolabel_seed7labeler",
    "realmlp",
    "xgb_nonrule",
    "xgb_metastack",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "lgbm_meta_natural",
]


def _norm(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    s = np.clip(p.sum(axis=1, keepdims=True), eps, None)
    return p / s


def load_bank(side: str) -> np.ndarray:
    """Stack of (B, n_rows, 3) probs for the 14 bank components.

    side: 'oof' or 'test'.
    """
    arrs = []
    for n in BANK_NAMES:
        a = np.load(ART / f"{side}_{n}.npy").astype(np.float32)
        arrs.append(_norm(a))
    return np.stack(arrs, axis=0)


def bank_mean_probs(bank: np.ndarray) -> np.ndarray:
    """Average per-class probability across the 14 bank models."""
    return _norm(bank.mean(axis=0))


def nonconformity(probs: np.ndarray, target_class: np.ndarray) -> np.ndarray:
    """Split-conformal score = 1 - P(target_class | x). Smaller = more conformal."""
    n = probs.shape[0]
    return 1.0 - probs[np.arange(n), target_class]


def conformal_threshold(
    cal_scores: np.ndarray,
    alpha: float,
) -> float:
    """Quantile threshold q_hat such that Pr(score <= q_hat) >= 1 - alpha."""
    n = len(cal_scores)
    # Inflate by (n+1)/n for finite-sample correction (Vovk 2005).
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    k = min(k, n) - 1
    return float(np.sort(cal_scores)[k])


def in_prediction_set(
    probs: np.ndarray, q_hat: float
) -> np.ndarray:
    """For each row, return boolean mask (n, 3) of classes in the prediction set."""
    return (1.0 - probs) <= q_hat
