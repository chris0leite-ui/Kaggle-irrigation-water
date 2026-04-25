"""Map-capturing FE blocks for SMOTE per-fold redrive.

Wrappers around recipe_features.add_* that also return vocabulary maps
(str→int factorize codes, freq dicts, orig_stats lookup tables) so the
SMOTE-augmented rows can be FE-encoded without re-factorizing or
re-counting on the val fold.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def combos_with_map(train, test, orig, cats):
    """Returns (new_cols, combo_pairs[name→(c1,c2)], combo_maps[name→{str→int}])."""
    new_cols, combo_pairs, combo_maps = [], {}, {}
    for c1, c2 in combinations(cats, 2):
        col = f"COMBO_{c1}_{c2}"
        for df in (train, test, orig):
            df[col] = df[c1].astype(str) + "_" + df[c2].astype(str)
        combined = pd.concat([train[col], test[col], orig[col]])
        codes, uniques = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[col] = codes[:s]
        test[col] = codes[s:t]
        orig[col] = codes[t:]
        combo_pairs[col] = (c1, c2)
        combo_maps[col] = {v: i for i, v in enumerate(uniques)}
        new_cols.append(col)
    return new_cols, combo_pairs, combo_maps


def num_as_cat_with_map(train, test, orig, nums):
    new_cols, nac_maps = [], {}
    for c in nums:
        name = f"CAT_{c}"
        for df in (train, test, orig):
            df[name] = df[c].astype(str)
        combined = pd.concat([train[name], test[name], orig[name]])
        codes, uniques = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[name] = codes[:s]
        test[name] = codes[s:t]
        orig[name] = codes[t:]
        nac_maps[name] = {v: i for i, v in enumerate(uniques)}
        new_cols.append(name)
    return new_cols, nac_maps


def freq_with_map(train, test, orig, cats):
    new_cols, freq_maps = [], {}
    for c in cats:
        freq = pd.concat([train[c], test[c], orig[c]]).value_counts(normalize=True)
        name = f"FREQ_{c}"
        for df in (train, test, orig):
            df[name] = df[c].map(freq).fillna(0).astype(np.float32)
        new_cols.append(name)
        freq_maps[c] = freq.to_dict()
    return new_cols, freq_maps


def orig_mean_std_with_map(train, test, orig, cols_to_aggregate, target):
    new_cols, orig_stat_maps = [], {}
    for c in cols_to_aggregate:
        stats = orig.groupby(c)[target].agg(["mean", "std"]).reset_index()
        stats.columns = [c, f"ORIG_{c}_mean", f"ORIG_{c}_std"]
        for df_name in ("train", "test"):
            df = {"train": train, "test": test}[df_name]
            merged = df.merge(stats, on=c, how="left")
            df[f"ORIG_{c}_mean"] = merged[f"ORIG_{c}_mean"].fillna(0.5).astype(np.float32).values
            df[f"ORIG_{c}_std"]  = merged[f"ORIG_{c}_std"].fillna(0).astype(np.float32).values
        new_cols += [f"ORIG_{c}_mean", f"ORIG_{c}_std"]
        orig_stat_maps[c] = dict(
            mean=dict(zip(stats[c], stats[f"ORIG_{c}_mean"])),
            std=dict(zip(stats[c], stats[f"ORIG_{c}_std"])),
        )
    return new_cols, orig_stat_maps


def factorize_cats_with_map(train, test, orig, cats):
    cat_maps = {}
    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, uniques = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]
        cat_maps[c] = {v: i for i, v in enumerate(uniques)}
    return cat_maps
