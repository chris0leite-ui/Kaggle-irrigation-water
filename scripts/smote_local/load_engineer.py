"""Reuse recipe FE pipeline but capture vocabulary maps for redrive.

Mirrors recipe_full_te.load_and_engineer order so the resulting train_fe
matches the production OOF bank's column structure exactly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from recipe_features import (  # noqa
    add_threshold_flags, add_lr_formula_logits, add_digit_features,
)
from smote_local.fe_with_maps import (  # noqa
    combos_with_map, num_as_cat_with_map, freq_with_map,
    orig_mean_std_with_map, factorize_cats_with_map,
)


TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_and_engineer():
    """Returns (train, test, raw_train, info, test_ids, maps)."""
    log("loading data")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/archive.zip")

    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)} cats={len(cats)} train={len(train):,} "
        f"test={len(test):,} orig={len(orig):,}")

    # Snapshot raw train (str cats + float nums + y) for SMOTE
    raw_train = train[cats + nums + [TARGET]].copy()

    log("threshold flags + LR logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
        logits = add_lr_formula_logits(df)

    log("cat × cat pair combos")
    combos, combo_pairs, combo_maps = combos_with_map(train, test, orig, cats)

    log("digit features")
    digits = add_digit_features(train, test, orig, nums)

    log("num-as-cat")
    num_as_cat, nac_maps = num_as_cat_with_map(train, test, orig, nums)

    log("FREQ")
    freq, freq_maps = freq_with_map(train, test, orig, cats + combos)

    log("ORIG mean/std")
    orig_stats, orig_stat_maps = orig_mean_std_with_map(
        train, test, orig, nums + cats, TARGET)

    log("factorize raw cats")
    cat_maps = factorize_cats_with_map(train, test, orig, cats)

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: cats={len(cats)} combos={len(combos)} "
        f"digits={len(digits)} num_as_cat={len(num_as_cat)} tres={len(tres)} "
        f"logits={len(logits)} freq={len(freq)} orig_stats={len(orig_stats)} "
        f"te_cols={len(info['te_cols'])}")
    maps = dict(
        combo_pairs=combo_pairs, combo_maps=combo_maps,
        nac_maps=nac_maps, freq_maps=freq_maps,
        orig_stat_maps=orig_stat_maps, cat_maps=cat_maps,
        digit_cols=digits,
    )
    return train, test, raw_train, info, test_ids, maps
