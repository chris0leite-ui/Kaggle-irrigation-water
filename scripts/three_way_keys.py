"""3-way OTE key generation.

Picks 15 (cat1 × cat2 × digit_K) triples by combining:
  - top 3 most informative cats (Crop_Type, Soil_Type, Region) —
    based on prior recipe FE importance (these dominate single-cat
    OTE gains and the cat-pair combo bank).
  - 5 high-signal digit positions: Soil_Moisture_digit0,
    Rainfall_mm_digit1, Temperature_C_digit0, Humidity_digit0,
    Previous_Irrigation_mm_digit1. Picked from the 46 surviving digit
    cols that drove the digits-OTE LB win (2026-04-23 entry).

Outputs string-encoded triple combos that get factorized across
train+test+orig in the same pattern as `add_cat_pair_combos`.

15 triples × 3 classes (when fed through OrderedTE) = 45 OTE features
on top of the existing 351-col recipe OTE.

Wall-time on full 504k train: ~30s for triple construction, then
OrderedTE adds ~12s per triple key on a 16-core CPU (so ~3 min to
fit the 15 keys per fold; ~15 min over 5 folds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Top 3 cats by feature importance in the V10 recipe (~12 % of total
# gain combined).
TOP_CATS = ["Crop_Type", "Soil_Type", "Region"]

# 5 digit-position columns that survived the test-constant filter and
# carried the digits-OTE +0.00014 LB lift (CLAUDE.md 2026-04-23).
TOP_DIGITS = [
    "Soil_Moisture_digit0",
    "Rainfall_mm_digit1",
    "Temperature_C_digit0",
    "Humidity_digit0",
    "Previous_Irrigation_mm_digit1",
]


def add_three_way_combos(train: pd.DataFrame, test: pd.DataFrame,
                         orig: pd.DataFrame) -> list[str]:
    """Build 15 (cat × cat × digit) triple-key combos.

    For each (cat1, cat2) pair from TOP_CATS (3 unique pairs) plus all
    5 digit positions = 3 × 5 = 15 triples. Each triple is a
    string-encoded ID factorized across train+test+orig (same protocol
    as `add_cat_pair_combos`).

    All three input frames must already contain the digit cols; they
    are produced upstream by `add_digit_features`.

    Returns the list of new combo column names. Mutates dfs in-place.
    """
    pairs = [("Crop_Type", "Soil_Type"),
             ("Crop_Type", "Region"),
             ("Soil_Type", "Region")]
    new_cols: list[str] = []
    for c1, c2 in pairs:
        for d in TOP_DIGITS:
            for df in (train, test, orig):
                assert d in df.columns, f"missing digit col {d}"
            col = f"COMBO3_{c1}_{c2}_{d}"
            for df in (train, test, orig):
                df[col] = (
                    df[c1].astype(str) + "_"
                    + df[c2].astype(str) + "_"
                    + df[d].astype(str)
                )
            combined = pd.concat([train[col], test[col], orig[col]])
            codes, _ = pd.factorize(combined)
            split_tr = len(train)
            split_te = split_tr + len(test)
            train[col] = codes[:split_tr]
            test[col] = codes[split_tr:split_te]
            orig[col] = codes[split_te:]
            new_cols.append(col)
    return new_cols


def cardinality_report(train: pd.DataFrame, cols: list[str]) -> dict:
    """Diagnostic: report unique-key count + per-key mean train rows."""
    rep = {}
    n = len(train)
    for c in cols:
        nuniq = int(train[c].nunique())
        rep[c] = dict(unique=nuniq, mean_per_key=round(n / max(nuniq, 1), 2))
    return rep


if __name__ == "__main__":
    # Smoke-test triple construction on a tiny synthetic dataset.
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "Crop_Type": rng.choice(list("ABCD"), n),
        "Soil_Type": rng.choice(list("XYZ"), n),
        "Region": rng.choice(list("PQR"), n),
        **{d: rng.integers(0, 10, n) for d in TOP_DIGITS},
    })
    tr = df.iloc[:120].copy()
    te = df.iloc[120:160].copy()
    orig = df.iloc[160:].copy()
    cols = add_three_way_combos(tr, te, orig)
    print(f"smoke: created {len(cols)} triples on {n} rows total")
    print(cardinality_report(tr, cols))
