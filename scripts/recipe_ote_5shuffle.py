"""5-shuffle OTE concat — yunsuxiaozi's training-augmentation variant.

Mechanism: per fold, fit OrderedTE with K different shuffle seeds, concat
the K resulting DataFrames as augmented training rows (K x train pool).
The model sees the same (row, key) categorical assignment with K
DIFFERENT TE values — turns OTE estimation noise into a data-augmentation
signal, which is structurally distinct from our prior K-shuffle averaging
approach (which smooths the noise away per row).

Reference: yunsuxiaozi/pss6e4-lgb-advanced-cv-0-97997 (kernel audit
round 4). Their CV 0.97997 standalone → suggests this lever materially
moves a single-model OOF.

Transform side: identical to OrderedTE.transform — uses full-train per-key
stats, which are deterministic in the train set (don't depend on shuffle).
So val/test get the standard transform; only train gets augmented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from recipe_ote import OrderedTE  # noqa: E402


def fit_concat_5shuffle(
    df: pd.DataFrame,
    cat_cols: list[str],
    target: str,
    a: float = 1.0,
    n_shuffle: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, OrderedTE]:
    """Returns (5x-augmented train df with TE columns, fitted OrderedTE).

    The returned OTE object's per-key stats are deterministic across
    shuffles — safe to use for transform on val/test.

    Implementation note: each .fit() call is independent (constructs its
    own per-key cumcum / cumsum). We do K independent fits on permuted
    copies of df, each .fit() returns a permuted dataframe with TE
    columns; we concat all K (using their original row order, NOT
    permuted, so positional indices are repeatable) and return.
    """
    rng = np.random.default_rng(seed)
    pieces = []
    last_te: OrderedTE | None = None
    for k in range(n_shuffle):
        perm = rng.permutation(len(df))
        df_shuf = df.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=a)
        df_shuf_with_te = te.fit(df_shuf, cat_cols=cat_cols, target=target)
        # Unshuffle so all K augmented copies share the same row order
        # before concat. This matters because we want each augmented row
        # to be identifiable as "row r, augmentation k" for downstream
        # validation / sample_weight repetition.
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        df_unshuf = df_shuf_with_te.iloc[inv].reset_index(drop=True)
        pieces.append(df_unshuf)
        last_te = te  # for transform on val/test
    augmented = pd.concat(pieces, axis=0, ignore_index=True)
    return augmented, last_te
