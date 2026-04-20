#!/usr/bin/env python
"""
Pure-domain-knowledge heuristic baseline for the irrigation competition,
evaluated in 5-fold stratified CV (OOF).

No ML model. Each row is reduced to a single scalar `need_score` built
from domain priors:

    need_score = +ET_proxy            # atmospheric demand   (↑need)
                 +Crop_Demand         # crop × stage thirst   (↑need)
                 -Rainfall_mm         # free water supply     (↓need)
                 -Previous_Irr        # recent top-up         (↓need)
                 -Soil_Moisture·AWC   # stored water in soil  (↓need)
                 +(1/Irr_Eff - 1)     # delivery losses       (↑need)
                 -Mulching_Yes        # evaporation cut       (↓need)
                 +Saline_Flag·0.3     # leaching overhead     (↑need)

Each term is standardized on the training fold, so the weights are
dimensionless priors (not learned). The only fitted parameters are the
two thresholds (t1 < t2) that split score → {Low, Medium, High}; these
are picked per-fold to maximize balanced accuracy on the training fold,
then applied to the held-out fold.

Reports OOF balanced accuracy — directly comparable to the EBM result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "scripts" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

SOIL_FC = {"Sandy": 20, "Loamy": 40, "Clay": 50, "Silty": 42,
           "Peaty": 48, "Saline": 35, "Black": 50, "Red": 35}
SOIL_PWP = {"Sandy": 8, "Loamy": 12, "Clay": 18, "Silty": 13,
            "Peaty": 20, "Saline": 14, "Black": 18, "Red": 12}
CROP_THIRST = {
    "Rice": 3.0, "Sugarcane": 3.0, "Banana": 2.5, "Maize": 2.0,
    "Cotton": 1.5, "Wheat": 1.5, "Barley": 1.0, "Sorghum": 0.8,
    "Millet": 0.7, "Pulses": 0.7, "Groundnut": 1.5, "Soybean": 1.5,
    "Tomato": 2.0, "Potato": 1.5, "Onion": 1.2, "Vegetables": 2.0,
}
STAGE_KC = {
    "Sowing": 0.35, "Germination": 0.40, "Vegetative": 0.75,
    "Flowering": 1.15, "Fruiting": 1.10, "Maturation": 0.75,
    "Ripening": 0.55, "Harvest": 0.40,
}
IRR_EFF = {"Drip": 0.92, "Sprinkler": 0.80, "Furrow": 0.65,
           "Flood": 0.50, "Manual": 0.60}

# Dimensionless weights — direction from domain priors, magnitudes
# chosen to reflect relative strength (see domain/07_modeling_implications.md).
W = {
    "ET": 1.0, "Crop_Demand": 1.0,
    "Rainfall": 1.2, "Prev_Irr": 0.4,
    "Soil_Buffer": 0.8,
    "Irr_Loss": 0.3, "Mulch": 0.3, "Saline": 0.2,
}


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-row signal columns (pre-standardization)."""
    t = df["Temperature_C"]
    et = (
        t.clip(lower=0)
        * df["Sunlight_Hours"].clip(lower=0)
        * (1 + df["Wind_Speed_kmh"] / 10)
        * (1 - df["Humidity"] / 100).clip(lower=0)
    )
    fc = df["Soil_Type"].map(SOIL_FC).fillna(40).astype(float)
    pwp = df["Soil_Type"].map(SOIL_PWP).fillna(12).astype(float)
    awc = (fc - pwp).clip(lower=1)
    crop_thirst = df["Crop_Type"].map(CROP_THIRST).fillna(1.0)
    kc = df["Crop_Growth_Stage"].map(STAGE_KC).fillna(0.8)
    crop_demand = crop_thirst * kc
    eff = df["Irrigation_Type"].map(IRR_EFF).fillna(0.70)
    mulch = (df["Mulching_Used"].astype(str).str.lower() == "yes").astype(float)
    saline = (df["Electrical_Conductivity"] > 2).astype(float)

    # Soil water buffer (stored water in root zone, crudely).
    soil_buffer = df["Soil_Moisture"].fillna(0) * awc / 100

    return pd.DataFrame({
        "ET": et,
        "Crop_Demand": crop_demand,
        "Rainfall": df["Rainfall_mm"].fillna(0),
        "Prev_Irr": df["Previous_Irrigation_mm"].fillna(0),
        "Soil_Buffer": soil_buffer,
        "Irr_Loss": 1.0 / eff.clip(lower=0.1) - 1.0,
        "Mulch": mulch,
        "Saline": saline,
    })


def standardize_and_score(
    train_sig: pd.DataFrame, val_sig: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score val using train stats, then combine with W."""
    mu = train_sig.mean(axis=0)
    sd = train_sig.std(axis=0).replace(0, 1)

    def score(sig):
        z = (sig - mu) / sd
        return (
            W["ET"] * z["ET"]
            + W["Crop_Demand"] * z["Crop_Demand"]
            - W["Rainfall"] * z["Rainfall"]
            - W["Prev_Irr"] * z["Prev_Irr"]
            - W["Soil_Buffer"] * z["Soil_Buffer"]
            + W["Irr_Loss"] * z["Irr_Loss"]
            - W["Mulch"] * z["Mulch"]
            + W["Saline"] * z["Saline"]
        )

    return score(train_sig).values, score(val_sig).values


def find_thresholds(
    scores: np.ndarray, y: pd.Series, grid: int = 60,
) -> tuple[float, float, str, str, str]:
    """Grid-search two cut points (t1 < t2) to max balanced accuracy.

    Label order is fixed by the domain prior: ascending score ⇒
    ascending need, so low-bucket=Low, mid=Medium, high=High.
    """
    cuts = np.percentile(scores, np.linspace(2, 98, grid))
    y_arr = np.asarray(y)
    best = (-np.inf, (cuts[len(cuts) // 3], cuts[2 * len(cuts) // 3]))
    for i in range(len(cuts)):
        for j in range(i + 1, len(cuts)):
            t1, t2 = cuts[i], cuts[j]
            pred = np.where(
                scores < t1, "Low",
                np.where(scores < t2, "Medium", "High"),
            )
            score = balanced_accuracy_score(y_arr, pred)
            if score > best[0]:
                best = (score, (t1, t2))
    t1, t2 = best[1]
    return t1, t2, "Low", "Medium", "High"


def apply_thresholds(
    scores: np.ndarray, t1: float, t2: float,
    low: str, mid: str, high: str,
) -> np.ndarray:
    return np.where(scores < t1, low, np.where(scores < t2, mid, high))


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    print(f"loaded train.csv: {train.shape}")

    y = train["Irrigation_Need"]
    sig = build_signals(train)
    classes = sorted(y.unique().tolist())  # ['High','Low','Medium']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    oof_pred = np.empty(len(train), dtype=object)
    cum_cm = np.zeros((3, 3), dtype=int)

    for k, (tr, va) in enumerate(skf.split(sig, y)):
        ft = time.time()
        tr_sig, va_sig = sig.iloc[tr], sig.iloc[va]
        tr_y, va_y = y.iloc[tr], y.iloc[va]

        s_tr, s_va = standardize_and_score(tr_sig, va_sig)
        # Coarse grid search on a subsample of the training fold to save time.
        sub = np.random.default_rng(42 + k).choice(
            len(s_tr), size=min(50000, len(s_tr)), replace=False,
        )
        t1, t2, lo, mi, hi = find_thresholds(
            s_tr[sub], tr_y.iloc[sub].reset_index(drop=True), grid=60,
        )
        pred = apply_thresholds(s_va, t1, t2, lo, mi, hi)
        score = balanced_accuracy_score(va_y, pred)
        fold_scores.append(score)
        oof_pred[va] = pred
        cm = confusion_matrix(va_y, pred, labels=classes)
        cum_cm += cm

        print(f"  fold {k+1}/5: bal_acc={score:.5f}  t=({t1:.3f},{t2:.3f})  "
              f"labels=({lo},{mi},{hi})  ({time.time()-ft:.1f}s)")

    fold_scores = np.array(fold_scores)
    oof_score = balanced_accuracy_score(y, oof_pred)
    recall = cum_cm.diagonal() / cum_cm.sum(axis=1).clip(1)

    print(f"\n=== heuristic 5-fold OOF ===")
    print(f"  per-fold:         {fold_scores}")
    print(f"  mean ± std:       {fold_scores.mean():.5f} ± {fold_scores.std():.5f}")
    print(f"  OOF bal_acc:      {oof_score:.5f}")
    print(f"\nOOF confusion matrix (labels={classes}):")
    print(cum_cm)
    for lbl, r in zip(classes, recall):
        print(f"  recall[{lbl}] = {r:.5f}")
    print(f"\ntotal time: {time.time()-t0:.1f}s")

    result = {
        "tag": "heuristic",
        "fold_scores": fold_scores.tolist(),
        "mean": float(fold_scores.mean()),
        "std": float(fold_scores.std()),
        "oof": float(oof_score),
        "classes": classes,
        "confusion_matrix": cum_cm.tolist(),
        "recall": dict(zip(classes, map(float, recall))),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (ART / "cv_heuristic.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
