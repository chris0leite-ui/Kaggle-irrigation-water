"""Digit extraction helper for numeric features.

Inspired by public-notebook pipelines (emanuellcs / Mahogany) that extract
per-digit columns from numeric features at positions -4..+3. Trees pick
up quantisation artefacts and non-uniform digit distributions that raw
float splits cannot express.

Digit position convention:
  +3  -> thousands  : floor(v /    1000) % 10
  +2  -> hundreds   : floor(v /     100) % 10
  +1  -> tens       : floor(v /      10) % 10
   0  -> ones       : floor(v)           % 10
  -1  -> tenths     : floor(v *      10) % 10
  -2  -> hundredths : floor(v *     100) % 10
  -3  -> thousandths: floor(v *    1000) % 10

All output columns are int8 in [0, 9]; negative values carry sign via
floor so values like -0.15 -> tenths digit = 9 (floor(-0.15*10)=-2; -2 % 10 = 8
in Python). Features are always non-negative in this dataset so sign
handling is not a concern, but guarded just in case.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def digit_at(values: np.ndarray, pos: int) -> np.ndarray:
    """Return the base-10 digit at position `pos` (see header comment)."""
    v = values.astype(np.float64)
    # Use 10**(-pos) as the scale so: pos=+3 -> 0.001, pos=-3 -> 1000.
    scale = 10.0 ** (-pos)
    # Small epsilon to defend against float rounding like 0.3*10=2.999999
    return (np.floor(v * scale + 1e-9).astype(np.int64) % 10).astype(np.int8)


def add_digit_features(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    digits: Iterable[int] = (-3, -2, -1, 0, 1, 2, 3),
    prefix: str = "dig",
) -> Tuple[pd.DataFrame, List[str]]:
    """Append digit columns for each (col, digit) pair.

    Returns the modified dataframe and the list of new column names.
    """
    out = df.copy()
    new_cols: List[str] = []
    for c in numeric_cols:
        v = out[c].astype(float).values
        for d in digits:
            name = f"{prefix}_{c}_{d}" if d >= 0 else f"{prefix}_{c}_n{-d}"
            out[name] = digit_at(v, d)
            new_cols.append(name)
    return out, new_cols


def drop_zero_variance(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: Iterable[str],
) -> List[str]:
    """Return the subset of `cols` that have >1 unique value in train.

    Also drops from df_train/df_test in place. Keeps the rest of `cols` as-is.
    """
    keep: List[str] = []
    drop: List[str] = []
    for c in cols:
        if df_train[c].nunique(dropna=False) > 1:
            keep.append(c)
        else:
            drop.append(c)
    if drop:
        df_train.drop(columns=drop, inplace=True)
        df_test.drop(columns=drop, inplace=True)
    return keep
