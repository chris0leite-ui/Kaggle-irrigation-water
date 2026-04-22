"""Ordered Target Encoding (OTE) — per-row cumulative target stats with K-shuffle.

Mechanism (CatBoost / public-notebook recipe):
  For each shuffle s in 1..K:
    Permute train row order with seed s.
    For each train row i (in shuffle order), encode key K(x_i) as the
    cumulative class-count of rows j < i with K(x_j) == K(x_i),
    smoothed toward the global class prior with Laplace alpha.
  Average the K per-row encodings to reduce variance.

Critical contrast with fold-level OOF target encoding:
  fold-level: every row in fold f gets the SAME TE value per category
              (computed from folds != f). Model sees ~5 distinct values
              per (category, fold) pair.
  ordered:    every row gets a DIFFERENT TE value (cumulative noise).
              Model sees a per-row noisy signal that converges to the
              true category mean as rows pile up.

Test rows get the full-train per-category mean (no shuffle, no leak).

Usage inside 5-fold CV:
  for tr_idx, va_idx in skf.split(X, y):
      otes = []
      for spec in key_specs:
          ote = OTE(spec, n_shuffles=8, alpha=10.0, seed=42)
          ote.fit_transform_train(df.iloc[tr_idx], y[tr_idx])
          otes.append(ote)
      X_tr_ote = np.hstack([o.train_block() for o in otes])
      X_va_ote = np.hstack([o.transform(df.iloc[va_idx]) for o in otes])
      # ... train XGB on X_tr + X_tr_ote, predict on X_va + X_va_ote
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _key_strings(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.Series:
    if len(key_cols) == 1:
        return df[key_cols[0]].astype(str)
    return df[list(key_cols)].astype(str).agg("\x1f".join, axis=1)


@dataclass
class OTE:
    """Per-row OTE with K-shuffle averaging over a single key tuple.

    One encoder per key (for multi-key OTE, instantiate multiple).
    Output: 3 columns per encoder (per-class probability over [Low, Medium, High]).
    """
    key_cols: Sequence[str]
    n_shuffles: int = 8
    alpha: float = 10.0
    seed: int = 42
    n_classes: int = 3

    def __post_init__(self) -> None:
        self._train_oof: Optional[np.ndarray] = None  # (n_train, n_classes) K-averaged
        self._train_uniques: Optional[np.ndarray] = None  # sorted unique train key strings
        self._full_probs: Optional[np.ndarray] = None  # (n_keys, n_classes) full-train lookup
        self._prior: Optional[np.ndarray] = None  # (n_classes,) global prior

    @property
    def name(self) -> str:
        return "_x_".join(self.key_cols)

    def feature_names(self) -> List[str]:
        return [f"ote_{self.name}_p{c}" for c in range(self.n_classes)]

    def fit_transform_train(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Fit OTE on (df, y) and return per-row K-averaged encodings.

        df: training rows for this fold (or full train for test-OTE).
        y: int labels in [0, n_classes).
        """
        n = len(df)
        key_strs = _key_strings(df, self.key_cols)
        codes, uniques = pd.factorize(key_strs, sort=True)
        codes = codes.astype(np.int64)
        self._train_uniques = np.array(uniques)
        n_keys = len(uniques)
        K = self.n_classes

        prior = np.bincount(y, minlength=K).astype(np.float64)
        prior /= prior.sum()
        self._prior = prior.astype(np.float32)
        smooth = self.alpha * prior  # (K,)

        accum = np.zeros((n, K), dtype=np.float64)
        rng = np.random.default_rng(self.seed)

        # Pre-build a (n, K) one-hot for y (used per shuffle).
        y_onehot = np.zeros((n, K), dtype=np.float64)
        y_onehot[np.arange(n), y] = 1.0

        for _ in range(self.n_shuffles):
            order = rng.permutation(n)
            shuffled_codes = codes[order]
            shuffled_onehot = y_onehot[order]

            # We want: for each shuffled position p, encoding[p] = (excl_cum_counts[p] + smooth)
            #                                                       / (excl_tot[p] + alpha)
            # where excl_cum_counts[p] = count of positions q<p with same key as p, per class.
            # Compute per-key by sorting once (stable). To keep order, we do it via
            # group-by-key with boolean masks (O(n) per key); n_keys is small for our cards.
            probs_shuf = np.empty((n, K), dtype=np.float64)
            for k in range(n_keys):
                mask = shuffled_codes == k
                if not mask.any():
                    continue
                grp_onehot = shuffled_onehot[mask]  # (n_k, K) in shuffle order
                n_k = grp_onehot.shape[0]
                excl_cum = np.cumsum(grp_onehot, axis=0) - grp_onehot
                excl_tot = np.arange(n_k, dtype=np.float64)
                enc = (excl_cum + smooth) / (excl_tot[:, None] + self.alpha)
                probs_shuf[mask] = enc
            # Scatter back to original row order.
            probs_orig = np.empty_like(probs_shuf)
            probs_orig[order] = probs_shuf
            accum += probs_orig

        self._train_oof = (accum / self.n_shuffles).astype(np.float32)

        # Full-train lookup (no shuffle — pure category mean for test rows).
        full_counts = np.zeros((n_keys, K), dtype=np.float64)
        full_tot = np.zeros(n_keys, dtype=np.float64)
        np.add.at(full_counts, codes, y_onehot)
        np.add.at(full_tot, codes, 1.0)
        self._full_probs = ((full_counts + smooth) / (full_tot[:, None] + self.alpha)).astype(
            np.float32
        )
        return self._train_oof

    def train_block(self) -> np.ndarray:
        if self._train_oof is None:
            raise RuntimeError("fit_transform_train must be called first")
        return self._train_oof

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply full-train per-key lookup. Unseen keys → global prior."""
        if self._full_probs is None or self._prior is None:
            raise RuntimeError("fit_transform_train must be called first")
        key_strs = _key_strings(df, self.key_cols).values
        train_to_code = {v: i for i, v in enumerate(self._train_uniques)}
        out = np.empty((len(df), self.n_classes), dtype=np.float32)
        prior = self._prior
        for i, s in enumerate(key_strs):
            code = train_to_code.get(s)
            if code is None:
                out[i] = prior
            else:
                out[i] = self._full_probs[code]
        return out


def build_ote_block(
    df_fit: pd.DataFrame,
    y_fit: np.ndarray,
    df_apply: pd.DataFrame,
    key_specs: Sequence[Sequence[str]],
    n_shuffles: int = 8,
    alpha: float = 10.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build OTE for `df_fit` (train) and apply to `df_apply` (val or test).

    Returns:
      fit_block:   (len(df_fit), 3 * len(key_specs)) per-row K-averaged OTE
      apply_block: (len(df_apply), 3 * len(key_specs)) full-train lookup OTE
      col_names:   list of column names in order
    """
    fit_blocks: List[np.ndarray] = []
    apply_blocks: List[np.ndarray] = []
    col_names: List[str] = []
    for spec in key_specs:
        ote = OTE(
            key_cols=list(spec), n_shuffles=n_shuffles, alpha=alpha, seed=seed
        )
        fb = ote.fit_transform_train(df_fit, y_fit)
        ab = ote.transform(df_apply)
        fit_blocks.append(fb)
        apply_blocks.append(ab)
        col_names.extend(ote.feature_names())
    return np.hstack(fit_blocks), np.hstack(apply_blocks), col_names
