"""Quantile binning of numeric features for use as keys in 171-pair OTE.

Companion to `recipe_features.add_cat_pair_combos`: bin each numeric to a
small-cardinality categorical so the existing pair-combo + OrderedTE machinery
extends from 28 cat x cat pairs to all C(19, 2) = 171 pairs (Ali Afzal's
"pairwise-TE magic" lever from public kernel s6e4-0-978-xgb-cat-pairwise-te).

After binning, downstream callers just pass `cats + bin_cols` to the existing
`add_cat_pair_combos(...)` to get all 171 combos string-concat factorized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_quantile_bins(train: pd.DataFrame, test: pd.DataFrame,
                      orig: pd.DataFrame, nums: list[str],
                      n_bins: int = 16) -> list[str]:
    """Add per-numeric quantile-bin categorical columns (BIN_<col>).

    Bin edges fit on combined train+test+orig so codes are consistent across
    the three frames. Uses pd.qcut with duplicates='drop' to handle ties;
    columns with fewer effective bins than n_bins still get sensible bins.
    Mutates the dataframes in-place. Returns list of new column names.
    """
    new_cols: list[str] = []
    for c in nums:
        combined = pd.concat([train[c], test[c], orig[c]], ignore_index=True)
        binned = pd.qcut(combined, q=n_bins, duplicates="drop", labels=False)
        binned = binned.fillna(0).astype(np.int16).to_numpy()
        s = len(train); t = s + len(test)
        name = f"BIN_{c}"
        train[name] = binned[:s]
        test[name] = binned[s:t]
        orig[name] = binned[t:]
        new_cols.append(name)
    return new_cols
