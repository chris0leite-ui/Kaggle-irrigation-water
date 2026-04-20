"""
Pure-heuristic predictor for Irrigation_Need, evaluated with 5-fold
stratified CV matching scripts/benchmark.py's protocol.

No model is trained. The row-level score is computed from the domain
water-balance equation (DOMAIN.md). The only learned parameters are
the two percentile cut-points (t1 < t2) separating Low / Medium / High,
chosen per fold on the training portion to maximize balanced accuracy.

Three scores are evaluated:
  H1  Soil_Moisture alone, signed for direction (single best feature).
  H2  Z-scored 3-axis water balance: demand, supply, deficit — no
      crop- or mulch-specific adjustments (pure physics).
  H3  H2 with FAO-56 Kc_stage multiplier, −30 % mulching factor on ET,
      and soil-type-specific field-capacity for the deficit term.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)

# FAO-56 typical values
KC_STAGE = {"Sowing": 0.35, "Vegetative": 0.85, "Flowering": 1.15, "Harvest": 0.55}
SOIL_CAP = {"Sandy": 18.0, "Loamy": 33.0, "Silt": 40.0, "Clay": 45.0}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- score definitions -----------------------------------------------------
def et0_proxy(df: pd.DataFrame) -> np.ndarray:
    return (df["Temperature_C"] * (1 - df["Humidity"] / 100)
            * df["Wind_Speed_kmh"]).values


def score_H1(df: pd.DataFrame) -> np.ndarray:
    """Soil moisture alone (inverted so higher → higher need)."""
    return -df["Soil_Moisture"].values.astype(float)


def score_H2(df: pd.DataFrame) -> np.ndarray:
    """Raw 3-axis water balance: demand − supply − reserve (z-normalized)."""
    demand = et0_proxy(df)
    supply = 0.80 * df["Rainfall_mm"].values + df["Previous_Irrigation_mm"].values
    reserve = df["Soil_Moisture"].values.astype(float)
    axes = np.column_stack([demand, -supply, -reserve])
    # Standardize each axis on the whole pool (per-fold normalization is done
    # inside cv_score below to avoid leakage on train/val boundary).
    return axes  # return components; CV will z-score on train and sum


def score_H3(df: pd.DataFrame) -> np.ndarray:
    """H2 + FAO-56 crop coefficient, mulch ET factor, soil-type deficit."""
    kc = df["Crop_Growth_Stage"].map(KC_STAGE).values.astype(float)
    is_mulched = (df["Mulching_Used"] == "Yes").values.astype(float)
    cap = df["Soil_Type"].map(SOIL_CAP).values.astype(float)
    etc = et0_proxy(df) * kc * (1 - 0.30 * is_mulched)
    supply = 0.80 * df["Rainfall_mm"].values + df["Previous_Irrigation_mm"].values
    deficit = np.clip(cap - df["Soil_Moisture"].values, 0, None)
    return np.column_stack([etc, -supply, deficit])


# ---- fast threshold search -------------------------------------------------
def best_thresholds(score: np.ndarray, y: np.ndarray, n_grid: int = 199
                    ) -> tuple[float, float, float]:
    """Grid-search (t1, t2) over percentiles of `score` to maximize bal_acc."""
    grid = np.percentile(score, np.linspace(0.5, 99.5, n_grid))
    # ECDF per class evaluated at each grid point
    F = np.zeros((3, n_grid), dtype=np.float64)
    for c in range(3):
        s_c = np.sort(score[y == c])
        F[c] = np.searchsorted(s_c, grid, side="right") / len(s_c)
    recall_low = F[0][:, None]                 # (G, 1)    → depends on t1
    recall_high = 1.0 - F[2][None, :]          # (1, G)    → depends on t2
    recall_med = F[1][None, :] - F[1][:, None]  # (G, G)   → F_M(t2) − F_M(t1)
    ba = (recall_low + recall_med + recall_high) / 3.0
    mask = np.arange(n_grid)[None, :] > np.arange(n_grid)[:, None]
    ba = np.where(mask, ba, -np.inf)
    i, j = np.unravel_index(np.argmax(ba), ba.shape)
    return float(grid[i]), float(grid[j]), float(ba[i, j])


def classify(score: np.ndarray, t1: float, t2: float) -> np.ndarray:
    return np.where(score < t1, 0, np.where(score < t2, 1, 2))


# ---- CV driver -------------------------------------------------------------
def cv_score(df: pd.DataFrame, y: np.ndarray, score_fn, name: str) -> dict:
    log(f"--- {name} ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    raw = score_fn(df)
    is_matrix = raw.ndim == 2
    oof_pred = np.full(len(df), -1, dtype=np.int8)
    fold_ba = []
    thresholds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y)):
        if is_matrix:
            mu, sd = raw[tr_idx].mean(0), raw[tr_idx].std(0)
            sd = np.where(sd == 0, 1.0, sd)
            s_tr = ((raw[tr_idx] - mu) / sd).sum(1)
            s_va = ((raw[va_idx] - mu) / sd).sum(1)
        else:
            s_tr = raw[tr_idx]
            s_va = raw[va_idx]
        t1, t2, train_ba = best_thresholds(s_tr, y[tr_idx])
        pred = classify(s_va, t1, t2)
        oof_pred[va_idx] = pred
        val_ba = balanced_accuracy_score(y[va_idx], pred)
        fold_ba.append(val_ba)
        thresholds.append((t1, t2))
        log(f"  fold {fold+1}/{N_FOLDS}  t1={t1:+.3f} t2={t2:+.3f}  "
            f"train_ba={train_ba:.5f}  val_ba={val_ba:.5f}")

    oof_ba = balanced_accuracy_score(y, oof_pred)
    cm = confusion_matrix(y, oof_pred)
    log(f"  OOF bal_acc = {oof_ba:.5f}  (fold mean {np.mean(fold_ba):.5f} "
        f"± {np.std(fold_ba):.5f})")
    return {
        "name": name,
        "oof_bal_acc": float(oof_ba),
        "fold_ba": [float(b) for b in fold_ba],
        "thresholds_per_fold": [list(t) for t in thresholds],
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    log("loading")
    df = pd.read_csv("data/train.csv")
    y = df[TARGET].map(CLS2IDX).values.astype(np.int32)

    results = [
        cv_score(df, y, score_H1, "H1_soil_moisture_only"),
        cv_score(df, y, score_H2, "H2_water_balance_raw"),
        cv_score(df, y, score_H3, "H3_water_balance_Kc_mulch_soilcap"),
    ]

    print("\n=== heuristic summary (OOF balanced accuracy) ===")
    for r in results:
        print(f"  {r['name']:36s} {r['oof_bal_acc']:.5f}")

    print("\nconfusion matrix for best heuristic:")
    best = max(results, key=lambda r: r["oof_bal_acc"])
    print(f"best: {best['name']}")
    print(pd.DataFrame(best["confusion_matrix"], index=CLASSES, columns=CLASSES))

    with open(ART / "heuristic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("saved scripts/artifacts/heuristic_results.json")
