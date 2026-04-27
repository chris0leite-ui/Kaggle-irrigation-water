"""Shared helpers for Phase A residual Target Encoding.

Three binary residual targets (per-row, computed from y + dgp_score + rule_pred):
  r_global  = y != rule_pred                               (general residual)
  r_mh_s6   = y==High AND rule==Med AND dgp_score==6       (M→H boundary flip)
  r_hm_s78  = y==Med AND rule==High AND dgp_score in {7,8} (H→M boundary flip)

Key set (recipe-tier — features already produced by recipe_features):
  - dgp_score itself                            (1 key)
  - 8 digit cols on rule-axis numerics k∈{-1,0}: sm/rf/tc/ws × {-1,0}
  - 4 high-signal cat-pairs (factorized combos already in recipe)
  - 2 stage/mulch single-cats (already in recipe)
Total ~15 keys; 3 binary targets × 15 keys × 1 col (p(class=1)) = ~45 features.

Fold-safe per-row OrderedTE via OTE class (n_shuffles=8, alpha=10). Train rows
get K-shuffle averaged exclusive cumulative stats; val/test get the full-tr
per-key lookup with prior fallback for unseen keys.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ote_features import OTE  # noqa: E402


def build_residual_targets(y: np.ndarray, dgp_score: np.ndarray,
                            rule_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Return three binary residual targets keyed by name. y in {0,1,2}."""
    r_global = (y != rule_pred).astype(np.int64)
    r_mh_s6 = ((y == 2) & (rule_pred == 1) & (dgp_score == 6)).astype(np.int64)
    r_hm_s78 = ((y == 1) & (rule_pred == 2) & np.isin(dgp_score, (7, 8))).astype(np.int64)
    return dict(r_global=r_global, r_mh_s6=r_mh_s6, r_hm_s78=r_hm_s78)


# Default residual TE key list. Each entry is a list of column names that
# concatenate to form a key (single-col keys = list of length 1).
def default_key_specs(combos: list[str], digits: list[str]) -> list[list[str]]:
    """Build the default key list. Tolerant of missing optional combos."""
    keys: list[list[str]] = [["dgp_score"]]

    # Single high-signal cats from recipe.
    for c in ("Crop_Growth_Stage", "Mulching_Used"):
        keys.append([c])

    # Digit cols on rule-axis numerics. Recipe digit naming convention:
    # f"{col}_digit{k}" for k in [-4..3]. We pick k∈{-1,0} (units & tens).
    digit_targets = []
    for axis in ("Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh"):
        for k in (-1, 0):
            cand = f"{axis}_digit{k}"
            if cand in digits:
                digit_targets.append(cand)
    for d in digit_targets:
        keys.append([d])

    # 4 high-signal cat-pair combos from recipe. Recipe combo naming:
    # f"COMBO_{c1}_{c2}". Pick the ones rule-axis-relevant.
    pair_targets = [
        "COMBO_Crop_Growth_Stage_Mulching_Used",
        "COMBO_Crop_Type_Region",
        "COMBO_Soil_Type_Crop_Growth_Stage",
        "COMBO_Mulching_Used_Crop_Type",
    ]
    for p in pair_targets:
        if p in combos:
            keys.append([p])

    return keys


def fit_residual_ote_block(
    df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame,
    targets: dict[str, np.ndarray],
    keys: list[list[str]],
    n_shuffles: int = 8, alpha: float = 10.0, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fit residual OrderedTE per (target × key) on df_tr; apply to va, te.

    Each (target, key) yields ONE column = P(residual=1 | key) per row.
    Returns: (tr_block, va_block, te_block, col_names).
    Wall budget per call ~45-90 sec for n_keys=15, n_shuffles=8.
    """
    tr_blocks, va_blocks, te_blocks = [], [], []
    col_names: list[str] = []
    for tgt_name, y_bin in targets.items():
        assert y_bin.shape[0] == len(df_tr), (tgt_name, y_bin.shape, len(df_tr))
        for key_cols in keys:
            ote = OTE(
                key_cols=list(key_cols),
                n_shuffles=n_shuffles, alpha=alpha,
                seed=seed, n_classes=2,
            )
            tr_oof = ote.fit_transform_train(df_tr, y_bin)
            va_block = ote.transform(df_va)
            te_block = ote.transform(df_te)
            # Take p(class=1) only.
            tr_blocks.append(tr_oof[:, 1:2].astype(np.float32))
            va_blocks.append(va_block[:, 1:2].astype(np.float32))
            te_blocks.append(te_block[:, 1:2].astype(np.float32))
            key_str = "_x_".join(key_cols)
            col_names.append(f"resTE_{tgt_name}_{key_str}")
    return (
        np.hstack(tr_blocks),
        np.hstack(va_blocks),
        np.hstack(te_blocks),
        col_names,
    )


def compute_rule_pred_score(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute (dgp_score, rule_pred) from raw features.

    rule_pred: 0 if score≤3, 1 if 4≤score≤6, 2 if score≥7. Matches
    common.add_distance_features convention.
    """
    sm = df["Soil_Moisture"].astype(float).to_numpy()
    rf = df["Rainfall_mm"].astype(float).to_numpy()
    tc = df["Temperature_C"].astype(float).to_numpy()
    ws = df["Wind_Speed_kmh"].astype(float).to_numpy()
    dry = (sm < 25.0).astype(np.int8)
    norain = (rf < 300.0).astype(np.int8)
    hot = (tc > 30.0).astype(np.int8)
    windy = (ws > 10.0).astype(np.int8)
    nomulch = (df["Mulching_Used"].astype(str).to_numpy() == "No").astype(np.int8)
    stage = df["Crop_Growth_Stage"].astype(str).to_numpy()
    kc = np.where(np.isin(stage, ("Flowering", "Vegetative")), 2, 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    rule_pred = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    return score, rule_pred
