"""Data preparation: CSV -> (num, dig, cat) arrays ready for the model."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from features import (
    add_distance_features, add_digit_features_inline, drop_const_digit_cols,
)

TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def load_prepared(train_csv: Path, test_csv: Path):
    """Return dict of arrays + metadata. Applies dist + digit FE."""
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    tr = add_distance_features(tr)
    te = add_distance_features(te)
    dig_cols = add_digit_features_inline(tr)
    add_digit_features_inline(te)
    dig_cols = drop_const_digit_cols(tr, te, dig_cols)

    cat_cols = [c for c in tr.columns
                if c not in (TARGET, ID) and tr[c].dtype == object]
    num_cols = [c for c in tr.columns
                if c not in (TARGET, ID) and c not in cat_cols
                and c not in dig_cols]

    cat_cards = []
    for c in cat_cols:
        vocab = sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))
        mp = {v: i for i, v in enumerate(vocab)}
        tr[c] = tr[c].astype(str).map(mp).astype("int64")
        te[c] = te[c].astype(str).map(mp).astype("int64")
        cat_cards.append(len(vocab))

    x_num_tr = tr[num_cols].to_numpy(dtype=np.float32)
    x_num_te = te[num_cols].to_numpy(dtype=np.float32)
    mu = x_num_tr.mean(axis=0, keepdims=True)
    sd = x_num_tr.std(axis=0, keepdims=True) + 1e-6
    x_num_tr = (x_num_tr - mu) / sd
    x_num_te = (x_num_te - mu) / sd

    x_dig_tr = tr[dig_cols].to_numpy(dtype=np.int64) if dig_cols else None
    x_dig_te = te[dig_cols].to_numpy(dtype=np.int64) if dig_cols else None
    x_cat_tr = tr[cat_cols].to_numpy(dtype=np.int64) if cat_cols else None
    x_cat_te = te[cat_cols].to_numpy(dtype=np.int64) if cat_cols else None

    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    prior = np.bincount(y, minlength=3) / len(y)

    return {
        "x_num_tr": x_num_tr, "x_num_te": x_num_te,
        "x_dig_tr": x_dig_tr, "x_dig_te": x_dig_te,
        "x_cat_tr": x_cat_tr, "x_cat_te": x_cat_te,
        "y": y, "prior": prior,
        "digit_cards": [10] * len(dig_cols),
        "cat_cards": cat_cards,
        "num_cols": num_cols, "dig_cols": dig_cols, "cat_cols": cat_cols,
        "id_test": te[ID].values,
    }
