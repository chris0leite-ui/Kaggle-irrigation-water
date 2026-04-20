#!/usr/bin/env python
"""
5-fold stratified CV of an Explainable Boosting Machine on the irrigation
competition, with domain-knowledge-driven feature engineering.

Engineered features (see domain/ notes):
  - VPD_kPa          : vapor pressure deficit (FAO-56 es formula)
  - ET_proxy         : multiplicative ET surrogate (T * sun * wind * (1-RH))
  - Soil_FC / AWC    : lookup per Soil_Type
  - Soil_Depletion   : (FC - Soil_Moisture) / AWC
  - Crop_Thirst      : per-crop seasonal water tier
  - Stage_Kc         : FAO-56 crop coefficient by growth stage
  - Crop_Demand      : Crop_Thirst * Stage_Kc
  - Irr_Eff          : delivery efficiency by Irrigation_Type
  - Water_Balance    : Rainfall + PrevIrrigation - k*ET*Demand
  - Saline_Flag      : EC > 2 dS/m

Usage:
    python scripts/cv_ebm.py                     # full 630k, 5-fold
    python scripts/cv_ebm.py --sample 50000      # quick dry-run
    python scripts/cv_ebm.py --raw-only          # skip FE baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "scripts" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# ---- domain lookups -------------------------------------------------------

# From domain/03_soil.md (volumetric % at field capacity / PWP).
SOIL_FC = {"Sandy": 20, "Loamy": 40, "Clay": 50, "Silty": 42,
           "Peaty": 48, "Saline": 35, "Black": 50, "Red": 35}
SOIL_PWP = {"Sandy": 8, "Loamy": 12, "Clay": 18, "Silty": 13,
            "Peaty": 20, "Saline": 14, "Black": 18, "Red": 12}

# From domain/04_crops.md — rough seasonal thirst tier.
CROP_THIRST = {
    "Rice": 3.0, "Sugarcane": 3.0, "Banana": 2.5, "Maize": 2.0,
    "Cotton": 1.5, "Wheat": 1.5, "Barley": 1.0, "Sorghum": 0.8,
    "Millet": 0.7, "Pulses": 0.7, "Groundnut": 1.5, "Soybean": 1.5,
    "Tomato": 2.0, "Potato": 1.5, "Onion": 1.2, "Vegetables": 2.0,
}

# FAO-56 single-Kc by growth stage (04_crops.md).
STAGE_KC = {
    "Sowing": 0.35, "Germination": 0.40, "Vegetative": 0.75,
    "Flowering": 1.15, "Fruiting": 1.10, "Maturation": 0.75,
    "Ripening": 0.55, "Harvest": 0.40,
}

# Irrigation efficiency (05_irrigation_systems.md).
IRR_EFF = {"Drip": 0.92, "Sprinkler": 0.80, "Furrow": 0.65,
           "Flood": 0.50, "Manual": 0.60}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    t = out["Temperature_C"]
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    out["VPD_kPa"] = (1 - out["Humidity"] / 100).clip(lower=0) * es

    out["ET_proxy"] = (
        t.clip(lower=0)
        * out["Sunlight_Hours"].clip(lower=0)
        * (1 + out["Wind_Speed_kmh"] / 10)
        * (1 - out["Humidity"] / 100).clip(lower=0)
    )

    fc = out["Soil_Type"].map(SOIL_FC).fillna(40).astype(float)
    pwp = out["Soil_Type"].map(SOIL_PWP).fillna(12).astype(float)
    awc = (fc - pwp).clip(lower=1)
    out["Soil_FC"] = fc
    out["Soil_AWC"] = awc
    out["Soil_Depletion"] = ((fc - out["Soil_Moisture"]) / awc).clip(-2, 2)

    out["Crop_Thirst"] = out["Crop_Type"].map(CROP_THIRST).fillna(1.0)
    out["Stage_Kc"] = out["Crop_Growth_Stage"].map(STAGE_KC).fillna(0.8)
    out["Crop_Demand"] = out["Crop_Thirst"] * out["Stage_Kc"]

    out["Irr_Eff"] = out["Irrigation_Type"].map(IRR_EFF).fillna(0.70)

    out["Water_Balance"] = (
        out["Rainfall_mm"].fillna(0)
        + out["Previous_Irrigation_mm"].fillna(0)
        - 0.02 * out["ET_proxy"] * out["Crop_Demand"]
    )

    out["Saline_Flag"] = (out["Electrical_Conductivity"] > 2).astype(int)

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0,
                   help="if > 0, subsample N rows for a quick dry-run")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-only", action="store_true",
                   help="skip domain feature engineering (baseline)")
    p.add_argument("--interactions", type=int, default=10)
    p.add_argument("--outer-bags", type=int, default=4)
    p.add_argument("--tag", type=str, default="ebm",
                   help="suffix for artifact file names")
    args = p.parse_args()

    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    print(f"loaded train.csv: {train.shape}")

    if args.sample:
        train = train.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"subsampled to {train.shape}")

    y = train["Irrigation_Need"]
    X = train.drop(columns=["id", "Irrigation_Need"])

    orig_cols = set(X.columns)
    if not args.raw_only:
        X = engineer(X)
        new_cols = [c for c in X.columns if c not in orig_cols]
        print(f"engineered {len(new_cols)} features: {new_cols}")
    print(f"X shape: {X.shape}")

    feature_types = [
        "continuous" if pd.api.types.is_numeric_dtype(X[c]) else "nominal"
        for c in X.columns
    ]
    cat_cols = [c for c, t in zip(X.columns, feature_types) if t == "nominal"]
    print(f"categorical features ({len(cat_cols)}): {cat_cols}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_scores = []
    class_order = None
    cumulative_cm = None

    for k, (tr, va) in enumerate(skf.split(X, y)):
        ft = time.time()
        ebm = ExplainableBoostingClassifier(
            feature_types=feature_types,
            interactions=args.interactions,
            outer_bags=args.outer_bags,
            max_bins=256,
            learning_rate=0.02,
            random_state=args.seed + k,
            n_jobs=-1,
        )
        ebm.fit(X.iloc[tr], y.iloc[tr])
        pred = ebm.predict(X.iloc[va])
        score = balanced_accuracy_score(y.iloc[va], pred)
        fold_scores.append(score)

        if class_order is None:
            class_order = list(ebm.classes_)
            cumulative_cm = np.zeros((len(class_order), len(class_order)), dtype=int)
        cumulative_cm += confusion_matrix(y.iloc[va], pred, labels=class_order)

        print(f"  fold {k+1}/{args.folds}: balanced_acc={score:.5f}  "
              f"({time.time()-ft:.1f}s)")

    fold_scores = np.array(fold_scores)
    mean, std = fold_scores.mean(), fold_scores.std()
    recall = cumulative_cm.diagonal() / cumulative_cm.sum(axis=1).clip(1)

    print(f"\n=== {args.folds}-fold balanced accuracy ===")
    print(f"  mean ± std: {mean:.5f} ± {std:.5f}")
    print(f"  per fold:   {fold_scores}")
    print(f"\nCumulative confusion matrix (labels={class_order}):")
    print(cumulative_cm)
    for lbl, r in zip(class_order, recall):
        print(f"  recall[{lbl}] = {r:.5f}")
    print(f"\ntotal time: {time.time()-t0:.1f}s")

    result = {
        "tag": args.tag,
        "sample": args.sample,
        "folds": args.folds,
        "seed": args.seed,
        "raw_only": args.raw_only,
        "interactions": args.interactions,
        "outer_bags": args.outer_bags,
        "n_features": X.shape[1],
        "fold_scores": fold_scores.tolist(),
        "mean": float(mean),
        "std": float(std),
        "classes": class_order,
        "confusion_matrix": cumulative_cm.tolist(),
        "recall": dict(zip(class_order, map(float, recall))),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ART / f"cv_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
