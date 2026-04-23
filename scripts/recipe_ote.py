"""OrderedTE — per-class cumulative shuffled target encoder.

Adapted from include4eto/ps6e4-xgb-cudf-pseudo-labels (public notebook).
For each categorical column and each class c, produces a per-row
target-smoothed probability P(y=c | key) using cumulative statistics
that exclude the current row (no leakage within the training set):

    TE[row, key, c] = (cum_sum_excl + a * prior[c]) / (cum_cnt_excl + a)

Transform for val/test uses the full-train per-key count+sum.
Shrinkage `a` (default 1) biases low-count keys toward the global prior.
Shuffle the dataframe BEFORE calling `.fit(...)` for randomised orders.
"""
from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd


class OrderedTE:
    def __init__(self, a: float = 1.0) -> None:
        self.a = float(a)
        self.classes_: np.ndarray | None = None
        self.prior_: np.ndarray | None = None
        self.stats_: dict[str, pd.DataFrame] = {}
        self.cols_: list[str] = []

    def fit(self, df: pd.DataFrame, cat_cols: list[str],
            target: str) -> pd.DataFrame:
        y = df[target].to_numpy()
        self.classes_ = np.array(sorted(pd.unique(y)))
        counts = np.array([(y == k).sum() for k in self.classes_],
                          dtype=np.float64)
        self.prior_ = counts / counts.sum()
        self.cols_ = list(cat_cols)

        df = df.copy()
        for c in self.cols_:
            stats_list = []
            for k, cls in enumerate(self.classes_):
                y_bin = (df[target] == cls).astype(np.int32).to_numpy()
                grp = pd.DataFrame({c: df[c].to_numpy(), "y": y_bin})
                cum_cnt = grp.groupby(c, observed=True, sort=False)["y"].cumcount().to_numpy()
                cum_sum_incl = grp.groupby(c, observed=True, sort=False)["y"].cumsum().to_numpy()
                cum_sum_excl = cum_sum_incl - y_bin
                prior = self.prior_[k]
                te = (cum_sum_excl + self.a * prior) / (cum_cnt + self.a)
                df[f"{c}_TE_cls{cls}"] = te.astype(np.float32)

                agg = grp.groupby(c, observed=True, sort=False)["y"].agg(
                    ["count", "sum"]).reset_index()
                agg.columns = [c, f"{c}_n_{cls}", f"{c}_s_{cls}"]
                stats_list.append(agg)

            self.stats_[c] = reduce(
                lambda a_df, b_df: a_df.merge(b_df, on=c, how="outer"),
                stats_list,
            )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.cols_:
            stats = self.stats_[c]
            merged = df[[c]].merge(stats, on=c, how="left")
            for k, cls in enumerate(self.classes_):
                n_col = f"{c}_n_{cls}"
                s_col = f"{c}_s_{cls}"
                prior = self.prior_[k]
                n = merged[n_col].fillna(0).to_numpy()
                s = merged[s_col].fillna(0).to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    te = np.where(
                        n > 0,
                        (s + self.a * prior) / (n + self.a),
                        prior,
                    )
                df[f"{c}_TE_cls{cls}"] = te.astype(np.float32)
        return df

    def te_col_names(self) -> list[str]:
        return [f"{c}_TE_cls{cls}" for c in self.cols_ for cls in self.classes_]
