"""Shared helpers for recipe-FE-based base components: per-fold OTE,
common feature-list construction, and CV loop. Used by:
  recipe_lgbm_native.py  — LightGBM with native categorical handling
  recipe_catboost_v2.py  — CatBoost with distinct config (depth=6, ordered)
  recipe_mlp_ote.py      — MLP with OTE features (vs recipe_mlp without)

All three share the SAME per-fold OTE setup matching recipe_full_te so
their OOFs align with v1's StratifiedKFold(seed=42) component bank.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Force defaults — no env-var FE additions, just core V10 recipe
for k in ("EXTRA_FE", "EXTRA_OOD", "EXTRA_KNN10K", "EXTRA_OOD9",
          "DROP_DETERMINISTIC", "DROP_SCORES", "ANCHOR_WEIGHT_ALPHA",
          "TTA_BOUNDARY", "THREE_WAY_OTE", "NN_DIST_PATH",
          "CLEANLAB_TREATMENT", "DAE_EMBED_PATH", "SMOKE",
          "EXTRA_W8", "EXTRA_INSTAB"):
    os.environ.pop(k, None)

sys.path.insert(0, str(Path(__file__).parent))
from recipe_full_te import load_and_engineer  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
TARGET = "Irrigation_Need"
SEED = 42
N_FOLDS = 5
CLS_MAP = {0: "Low", 1: "Medium", 2: "High"}


def build_fe():
    """Returns (train, test, info, te_keys, feat_cols_static).
    te_keys is the list of categorical keys to OTE-encode per fold.
    feat_cols_static is the list of columns ready for direct use (no OTE)."""
    train, test, info, _ = load_and_engineer()
    te_keys = info.get("te_cols", [])
    static_cols = (info.get("nums", []) + info.get("digits", [])
                   + info.get("num_as_cat", []) + info.get("tres", [])
                   + info.get("logits", []) + info.get("freq", [])
                   + info.get("orig_stats", [])
                   + info.get("extra_domain", [])
                   + info.get("extra_decimal", [])
                   + info.get("extra_w8", [])
                   + info.get("gby_cols", []))
    static_cols = [c for c in static_cols if c in train.columns and c != TARGET]
    return train, test, info, te_keys, static_cols


def fit_per_fold_ote(train_fold, te_keys, target, a=1.0, shuffle=True):
    """Returns a fitted OrderedTE on this fold's train rows."""
    if shuffle:
        df = train_fold.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    else:
        df = train_fold.reset_index(drop=True)
    ote = OrderedTE(a=a)
    df_with_te = ote.fit(df, te_keys, target)
    # Map back to original index order
    df_with_te = df_with_te.reset_index(drop=True)
    return ote, df_with_te
