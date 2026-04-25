"""SMOTE-NC on raw 19 cols + per-fold FE re-derivation.

Memory-safe replacement for `cat_cols_for_smote = info["cats"] + info["combos"] + ...`
which OOMs on the high-card combo features (45 GiB at production scale).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

sys.path.insert(0, str(Path(__file__).parent.parent))
from recipe_features import add_threshold_flags, add_lr_formula_logits  # noqa


def smote_nc_on_raw(raw_train_df, y_tr, target_n_high, k=5, random_state=42):
    """SMOTE-NC over (8 string cats + 11 float nums). Memory ~91 MB."""
    cat_cols = [c for c in raw_train_df.columns if raw_train_df[c].dtype == "object"]
    cat_idx = [raw_train_df.columns.get_loc(c) for c in cat_cols]
    smote = SMOTENC(
        categorical_features=cat_idx,
        sampling_strategy={2: target_n_high},
        k_neighbors=k,
        random_state=random_state,
    )
    return smote.fit_resample(raw_train_df, y_tr)


def redrive_fe(raw_aug, *, cats, nums, combo_pairs, combo_maps,
               nac_maps, freq_maps, orig_stat_maps, cat_maps,
               digit_cols_keep, digit_range=range(-4, 4)):
    """Build the full FE matrix on SMOTE-augmented raw rows.

    Order matches recipe_full_te.load_and_engineer:
      threshold_flags + lr_logits → combos → digits → num_as_cat →
      freq → orig_stats → factorize raw cats.
    """
    df = raw_aug.copy().reset_index(drop=True)

    # 1. row-wise deterministic
    add_threshold_flags(df)
    add_lr_formula_logits(df)

    # 2. combos: str-concat → lookup
    for combo_name, (c1, c2) in combo_pairs.items():
        keys = df[c1].astype(str) + "_" + df[c2].astype(str)
        df[combo_name] = keys.map(combo_maps[combo_name]).fillna(-1).astype(np.int32)

    # 3. digits: row-wise on raw nums
    for c in nums:
        for k in digit_range:
            df[f"{c}_digit{k}"] = (df[c] // (10.0 ** k) % 10).astype("int8")
    all_names = [f"{c}_digit{k}" for c in nums for k in digit_range]
    for n in all_names:
        if n not in digit_cols_keep and n in df.columns:
            df.drop(columns=[n], inplace=True)

    # 4. num-as-cat: str-cast → lookup
    for c in nums:
        name = f"CAT_{c}"
        df[name] = df[c].astype(str).map(nac_maps[name]).fillna(-1).astype(np.int32)

    # 5. FREQ: raw cat (str-key) + combo (int-key after step 2)
    new_cols = {}
    for c in cats:
        new_cols[f"FREQ_{c}"] = df[c].map(freq_maps[c]).fillna(0).astype(np.float32)
    for combo_name in combo_maps:
        new_cols[f"FREQ_{combo_name}"] = df[combo_name].map(
            freq_maps[combo_name]).fillna(0).astype(np.float32)

    # 6. ORIG mean/std: nums (float-key, mostly miss → 0.5) + cats (str-key)
    for c in nums + cats:
        new_cols[f"ORIG_{c}_mean"] = df[c].map(
            orig_stat_maps[c]["mean"]).fillna(0.5).astype(np.float32)
        new_cols[f"ORIG_{c}_std"] = df[c].map(
            orig_stat_maps[c]["std"]).fillna(0).astype(np.float32)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # 7. final raw-cat factorize
    for c in cats:
        df[c] = df[c].astype(str).map(cat_maps[c]).fillna(-1).astype(np.int32)

    return df
