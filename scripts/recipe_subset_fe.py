"""Feature engineering for recipe-subset XGB variants.

Applies the full recipe FE pipeline, then filters feature groups
based on the requested variant. Returns (train, test, info, test_ids).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from recipe_features import (  # noqa: E402
    add_cat_pair_combos, add_digit_features, add_freq_features,
    add_lr_formula_logits, add_num_as_cat, add_orig_mean_std,
    add_threshold_flags,
)

TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_and_engineer(variant: str, smoke: bool):
    _log("loading train / test / orig")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/archive.zip")
    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    if smoke:
        _log("SMOKE=1 -- subsampling")
        train = train.sample(20_000, random_state=42).reset_index(drop=True)
        test = test.sample(10_000, random_state=42).reset_index(drop=True)
        test_ids = test_ids[:10_000]

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    _log(f"  nums={len(nums)} cats={len(cats)} "
         f"train={len(train)} test={len(test)} orig={len(orig)}")

    # Always build: threshold flags, LR-formula logits (cheap, 7 cols)
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)

    combos, digits, num_as_cat, freq, orig_stats = [], [], [], [], []

    if variant != "no_combos":
        _log("adding cat x cat pair combos")
        combos = add_cat_pair_combos(train, test, orig, cats)
    else:
        _log("SKIP: cat pair combos (no_combos)")

    if variant != "no_digits":
        _log("adding digit features")
        digits = add_digit_features(train, test, orig, nums)
    else:
        _log("SKIP: digit features (no_digits)")

    _log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)

    _log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)

    if variant != "no_orig":
        _log("adding ORIG mean/std per col")
        orig_stats = add_orig_mean_std(train, test, orig, nums + cats, TARGET)
    else:
        _log("SKIP: ORIG mean/std (no_orig)")

    # Factorize raw cats after all string-value FE is done
    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]

    te_cols = cats + combos + digits + num_as_cat + tres
    info = dict(
        variant=variant,
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats, te_cols=te_cols,
    )
    _log(f"  feature groups: "
         f"cats={len(cats)} combos={len(combos)} digits={len(digits)} "
         f"num_as_cat={len(num_as_cat)} tres={len(tres)} logits={len(logits)} "
         f"freq={len(freq)} orig_stats={len(orig_stats)} "
         f"te_cols={len(te_cols)}")
    return train, test, info, test_ids
