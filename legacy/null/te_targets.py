"""Step 1/3 of TE-continuous-regression experiment.

Builds per-class TE targets from the 10k rule-perfect original
dataset, keyed by (Crop_Type, Soil_Type, Season, Region,
Crop_Growth_Stage, dgp_score), with Bayesian shrinkage toward the
per-(dgp_score) prior. Output is one continuous (n,3) target matrix
per dataset (train, test) ready to be used as the regression target
for the XGB model in step 2.

Why this target shape (per the design discussion in chat):
  - Discrete residuals (y - rule_pred) are 98.4% zero -> regression
    collapses to "predict 0".
  - Per-class TE probs from the 10k rule-perfect original are
    continuous in [0,1], smoothly varying across (cat-tuple x score)
    cells. Never zero-inflated.
  - Conditioning on dgp_score makes the target depend on the
    continuous rule features (since dgp_score is derived from them),
    so XGB on the full dist feature set learns more than a pure
    categorical lookup.

Shrinkage:
  shrunk = (counts + m * per_score_prior) / (counts.sum() + m)
  m = 30 -> a cell with 5 rows is 14% data, 86% prior. Cells with
  >100 rows are ~77% data.

Outputs:
  scripts/artifacts/te_targets_train.npy  (630_000, 3)
  scripts/artifacts/te_targets_test.npy   (270_000, 3)
  scripts/artifacts/te_targets_meta.json  diagnostic info
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ACTIVE_STAGES = ("Flowering", "Vegetative")

KEY_COLS = [
    "Crop_Type", "Soil_Type", "Season", "Region",
    "Crop_Growth_Stage", "dgp_score",
]
SHRINKAGE_M = 30.0

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_dgp_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add dgp_score column (0..9) per the verified DGP rule."""
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    return out


def build_te(orig: pd.DataFrame, m: float) -> tuple[dict, dict, np.ndarray]:
    """Compute per-(key) class probs on original.

    Returns:
      lookup        dict[key_tuple] -> (3,) shrunk per-class probs
      score_prior   dict[score_int] -> (3,) prior used for shrinkage
      global_prior  (3,) global per-class prior on original
    """
    y = orig[TARGET].map(CLS2IDX).values.astype(np.int32)
    global_prior = np.bincount(y, minlength=3) / len(y)

    score_prior: dict[int, np.ndarray] = {}
    for s, sub in orig.groupby("dgp_score"):
        ys = y[sub.index.values]
        if len(ys) == 0:
            score_prior[int(s)] = global_prior.copy()
        else:
            score_prior[int(s)] = (np.bincount(ys, minlength=3) / len(ys))

    lookup: dict[tuple, np.ndarray] = {}
    grouped = orig.groupby(KEY_COLS, observed=True)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        ys = y[sub.index.values]
        counts = np.bincount(ys, minlength=3).astype(np.float64)
        score_int = int(key[KEY_COLS.index("dgp_score")])
        prior = score_prior.get(score_int, global_prior)
        shrunk = (counts + m * prior) / (counts.sum() + m)
        lookup[key] = shrunk
    return lookup, score_prior, global_prior


def apply_te(
    df: pd.DataFrame, lookup: dict, score_prior: dict, global_prior: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Lookup per-row TE; fall back to per-score prior, then global."""
    n = len(df)
    out = np.zeros((n, 3), dtype=np.float32)
    cols = [df[c].values for c in KEY_COLS]
    keys = list(zip(*cols))
    hits = 0
    score_fallbacks = 0
    for i, key in enumerate(keys):
        v = lookup.get(key)
        if v is not None:
            out[i] = v
            hits += 1
            continue
        score_int = int(key[KEY_COLS.index("dgp_score")])
        prior = score_prior.get(score_int)
        if prior is not None:
            out[i] = prior
            score_fallbacks += 1
        else:
            out[i] = global_prior
    return out, hits, score_fallbacks


def main() -> None:
    t0 = time.time()
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/irrigation_prediction.csv")
    log(f"  train={len(tr)}  test={len(te)}  original={len(orig)}")

    log("computing dgp_score on all three sets")
    tr = add_dgp_score(tr)
    te = add_dgp_score(te)
    orig = add_dgp_score(orig)

    log(f"building TE lookup keyed by {KEY_COLS} with m={SHRINKAGE_M}")
    lookup, score_prior, global_prior = build_te(orig, SHRINKAGE_M)
    log(f"  unique key cells in original: {len(lookup)}")
    cell_sizes = []
    for _, sub in orig.groupby(KEY_COLS, observed=True):
        cell_sizes.append(len(sub))
    cell_sizes = np.array(cell_sizes)
    log(f"  cells per row count: median={int(np.median(cell_sizes))}  "
        f"min={int(cell_sizes.min())}  max={int(cell_sizes.max())}  "
        f"mean={cell_sizes.mean():.1f}")
    log(f"  global_prior on original = {global_prior.round(4).tolist()}")

    log("applying TE to train")
    tt_train, hits_tr, sf_tr = apply_te(tr, lookup, score_prior, global_prior)
    log(f"  hits={hits_tr}/{len(tr)}  score_fallbacks={sf_tr}")
    log("applying TE to test")
    tt_test, hits_te, sf_te = apply_te(te, lookup, score_prior, global_prior)
    log(f"  hits={hits_te}/{len(te)}  score_fallbacks={sf_te}")

    # Sanity: rows sum to 1 by construction.
    s_tr = tt_train.sum(axis=1)
    s_te = tt_test.sum(axis=1)
    log(f"  train target row-sum range = [{s_tr.min():.5f}, {s_tr.max():.5f}]")
    log(f"  test  target row-sum range = [{s_te.min():.5f}, {s_te.max():.5f}]")

    np.save(ART / "te_targets_train.npy", tt_train)
    np.save(ART / "te_targets_test.npy", tt_test)
    meta = {
        "key_cols": KEY_COLS,
        "shrinkage_m": SHRINKAGE_M,
        "n_original": int(len(orig)),
        "n_unique_cells": int(len(lookup)),
        "global_prior": global_prior.tolist(),
        "cell_size_median": int(np.median(cell_sizes)),
        "cell_size_min": int(cell_sizes.min()),
        "cell_size_max": int(cell_sizes.max()),
        "train_hits": int(hits_tr),
        "train_score_fallbacks": int(sf_tr),
        "test_hits": int(hits_te),
        "test_score_fallbacks": int(sf_te),
    }
    with open(ART / "te_targets_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote te_targets_train.npy, te_targets_test.npy, "
        f"te_targets_meta.json   ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
