"""B — Per-row k-NN distance features (test→train + train→train LOO).

Differentiator: every feature in the recipe pipeline is computed from
the row's own values OR from train-aggregate stats. NO feature explicitly
encodes "where in the training manifold does this row sit?". This script
builds 6 such features per row via FAISS k-NN.

Feature representation: standardized 11 raw numerics + one-hot 8 cats =
~30-43 dim space (matches W7 NN-to-original analysis distance metric).

For each row, find k=100 nearest TRAIN rows and compute:
  knn_dist_min     - distance to nearest train row
  knn_dist_mean    - mean distance over top-100 neighbors
  knn_class_low    - fraction of top-100 neighbors with y=Low
  knn_class_med    - fraction with y=Medium
  knn_class_high   - fraction with y=High
  knn_margin       - max(class_frac) - 2nd_max(class_frac)

Train-side: leave-one-out (skip self in neighbor list). For per-fold
leak-safety in downstream recipe XGB use, train rows pull neighbors from
the FULL train set EXCLUDING themselves; an extra fold-safe variant
(skip rows in the same fold) is a future refinement if needed.

Outputs:
  scripts/artifacts/oof_knn_train.npy   (n_train, 6) float32
  scripts/artifacts/test_knn_train.npy  (n_test, 6) float32
  scripts/artifacts/B_knn_features_results.json

Wall budget: ~10-15 min (FAISS HNSW on combined 630k+10k = 640k rows
indexing ~30s + 270k × k=100 query ~2 min + 630k × k=100 LOO query ~5 min).

SMOKE=1: 20k train, 10k test, k=20.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"
SUFFIX = "_smoke" if SMOKE else ""
SEED = 42
K = 20 if SMOKE else 100

CAT_COLS = ["Soil_Type", "Crop_Type", "Season", "Irrigation_Type",
            "Water_Source", "Region", "Mulching_Used", "Crop_Growth_Stage"]
NUM_COLS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
            "Humidity", "Soil_pH", "Electrical_Conductivity",
            "Sunlight_Hours", "Organic_Carbon", "Field_Area_hectare",
            "Previous_Irrigation_mm"]
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}


def log(msg: str) -> None:
    print(f"[B-knn {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def featurize(train_df: pd.DataFrame, test_df: pd.DataFrame, fit_only_train=True):
    """Standardize numerics on train, one-hot cats with fit on combined."""
    # One-hot cats (combined to ensure same vocab on test).
    combined = pd.concat([train_df[CAT_COLS], test_df[CAT_COLS]], axis=0, ignore_index=True)
    cat_dummies = pd.get_dummies(combined.astype(str), columns=CAT_COLS, drop_first=False)
    n_train = len(train_df)
    cat_train = cat_dummies.iloc[:n_train].to_numpy(dtype=np.float32)
    cat_test = cat_dummies.iloc[n_train:].to_numpy(dtype=np.float32)

    # Standardize numerics on train.
    num_mean = train_df[NUM_COLS].mean().to_numpy(dtype=np.float64)
    num_std = train_df[NUM_COLS].std().to_numpy(dtype=np.float64).clip(min=1e-6)
    num_train = ((train_df[NUM_COLS].to_numpy(dtype=np.float64) - num_mean) / num_std).astype(np.float32)
    num_test = ((test_df[NUM_COLS].to_numpy(dtype=np.float64) - num_mean) / num_std).astype(np.float32)

    X_train = np.concatenate([num_train, cat_train], axis=1)
    X_test = np.concatenate([num_test, cat_test], axis=1)
    return X_train, X_test


def main() -> None:
    t0 = time.time()
    log("loading train + test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_train = train[TARGET].map(CLS_MAP).to_numpy(dtype=np.int8)
    log(f"train={len(train):,}  test={len(test):,}  k={K}")

    if SMOKE:
        log("SMOKE=1 — subsampling")
        rng_smoke = np.random.default_rng(SEED)
        tr_idx = rng_smoke.choice(len(train), size=20_000, replace=False)
        te_idx = rng_smoke.choice(len(test), size=10_000, replace=False)
        train = train.iloc[tr_idx].reset_index(drop=True)
        test = test.iloc[te_idx].reset_index(drop=True)
        y_train = y_train[tr_idx]

    log("featurizing (standardize nums + one-hot cats)")
    X_train, X_test = featurize(train, test)
    log(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    # FAISS index: HNSW for fast approximate, or FlatL2 for exact.
    # On 630k×43 dim, FlatL2 search 270k×k=100 = ~5 min on CPU. HNSW would
    # be 10× faster but approximate. Going exact for fidelity.
    log("building FAISS FlatL2 index on train")
    index = faiss.IndexFlatL2(X_train.shape[1])
    index.add(X_train)
    log(f"  index size = {index.ntotal:,}")

    # ---- Test → train k-NN (full neighbors, no LOO needed)
    log(f"querying test → train k={K}")
    t1 = time.time()
    test_dists, test_idxs = index.search(X_test, K)
    log(f"  test query done in {time.time() - t1:.1f}s  test_dists={test_dists.shape}")

    # ---- Train → train k-NN with LOO (k+1, skip self)
    log(f"querying train → train k={K + 1} (LOO)")
    t1 = time.time()
    train_dists, train_idxs = index.search(X_train, K + 1)
    log(f"  train query done in {time.time() - t1:.1f}s")
    # Skip self: the first match is self at distance 0.
    train_dists = train_dists[:, 1:]
    train_idxs = train_idxs[:, 1:]

    log("computing 6 derived features per row")
    def derive(dists, idxs):
        # dists: (n, k) squared L2 distances. Take sqrt for actual distance.
        d = np.sqrt(np.clip(dists, 0, None)).astype(np.float32)
        d_min = d[:, 0]
        d_mean = d.mean(axis=1)
        # Class fractions of neighbors.
        nbr_y = y_train[idxs]  # (n, k) class indices
        cls_frac = np.zeros((len(d), 3), dtype=np.float32)
        for c in range(3):
            cls_frac[:, c] = (nbr_y == c).mean(axis=1)
        # Margin: max - 2nd_max
        sorted_frac = np.sort(cls_frac, axis=1)[:, ::-1]
        margin = sorted_frac[:, 0] - sorted_frac[:, 1]
        feats = np.column_stack([
            d_min, d_mean,
            cls_frac[:, 0], cls_frac[:, 1], cls_frac[:, 2],
            margin,
        ]).astype(np.float32)
        return feats

    train_feats = derive(train_dists, train_idxs)
    test_feats = derive(test_dists, test_idxs)
    log(f"  train_feats={train_feats.shape}  test_feats={test_feats.shape}")

    log("saving features")
    np.save(ART / f"oof_knn_train{SUFFIX}.npy", train_feats)
    np.save(ART / f"test_knn_train{SUFFIX}.npy", test_feats)

    # Save metadata.
    out = {
        "smoke": SMOKE,
        "k": K,
        "feature_dim": int(X_train.shape[1]),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_names": ["knn_dist_min", "knn_dist_mean",
                          "knn_class_low", "knn_class_med", "knn_class_high",
                          "knn_margin"],
        "train_feature_summary": {
            n: {"mean": float(train_feats[:, i].mean()),
                "std": float(train_feats[:, i].std()),
                "p1": float(np.percentile(train_feats[:, i], 1)),
                "p99": float(np.percentile(train_feats[:, i], 99))}
            for i, n in enumerate(["knn_dist_min", "knn_dist_mean",
                                    "knn_class_low", "knn_class_med",
                                    "knn_class_high", "knn_margin"])
        },
        "test_feature_summary": {
            n: {"mean": float(test_feats[:, i].mean()),
                "std": float(test_feats[:, i].std())}
            for i, n in enumerate(["knn_dist_min", "knn_dist_mean",
                                    "knn_class_low", "knn_class_med",
                                    "knn_class_high", "knn_margin"])
        },
        "elapsed_seconds": time.time() - t0,
    }
    out_path = ART / f"B_knn_features{SUFFIX}_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"saved {out_path.name}")

    log("=" * 60)
    log(f"FEATURE SUMMARY (train, n={len(train_feats):,})")
    for i, n in enumerate(["knn_dist_min", "knn_dist_mean", "knn_class_low",
                            "knn_class_med", "knn_class_high", "knn_margin"]):
        log(f"  {n:<20} mean={train_feats[:, i].mean():.4f}  std={train_feats[:, i].std():.4f}")
    log(f"total wall = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
