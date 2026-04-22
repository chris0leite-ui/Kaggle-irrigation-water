"""
Three hand-crafted multinomial logistic-regression formulas for
Irrigation_Need, motivated by the soil-water balance equation in
DOMAIN.md.

For each formula:
  - Build a design matrix (engineered numerics + one-hot cats).
  - 5-fold stratified CV of sklearn LogisticRegression(multinomial).
  - Report balanced accuracy under argmax / prior-reweight / tuned
    log-bias decision rules.

Run: python scripts/formula_mnlogit.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- domain lookups (FAO-56 typical values) -------------------------------
KC_STAGE = {"Sowing": 0.35, "Vegetative": 0.85, "Flowering": 1.15, "Harvest": 0.55}
SOIL_CAP = {"Sandy": 18.0, "Loamy": 33.0, "Silt": 40.0, "Clay": 45.0}


def et0_proxy(df: pd.DataFrame) -> pd.Series:
    """Crude Penman–Monteith surrogate: hot, dry, windy → high."""
    return df["Temperature_C"] * (1 - df["Humidity"] / 100) * df["Wind_Speed_kmh"]


# ---- formula 1: minimal water balance -------------------------------------
def build_F1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Irrigation_Need ~ ET0_proxy + Rainfall_mm + Previous_Irrigation_mm
                    + Soil_Moisture
    """
    return pd.DataFrame({
        "ET0_proxy": et0_proxy(df),
        "Rainfall_mm": df["Rainfall_mm"],
        "Previous_Irrigation_mm": df["Previous_Irrigation_mm"],
        "Soil_Moisture": df["Soil_Moisture"],
    })


# ---- formula 2: balance + Kc + soil deficit + management ------------------
def build_F2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Irrigation_Need ~ ETc_proxy + Rainfall_mm + Previous_Irrigation_mm
                    + Soil_deficit + C(Soil_Type) + C(Crop_Type)
                    + C(Crop_Growth_Stage) + C(Irrigation_Type)
                    + C(Mulching_Used)
    """
    kc = df["Crop_Growth_Stage"].map(KC_STAGE)
    cap = df["Soil_Type"].map(SOIL_CAP)
    num = pd.DataFrame({
        "ETc_proxy": et0_proxy(df) * kc,
        "Rainfall_mm": df["Rainfall_mm"],
        "Previous_Irrigation_mm": df["Previous_Irrigation_mm"],
        "Soil_deficit": (cap - df["Soil_Moisture"]).clip(lower=0),
    })
    cats = df[["Soil_Type", "Crop_Type", "Crop_Growth_Stage",
               "Irrigation_Type", "Mulching_Used"]]
    cat = pd.get_dummies(cats, drop_first=True).astype(float)
    return pd.concat([num, cat], axis=1)


# ---- formula 3: full structural (interactions, regime splits) -------------
def build_F3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Irrigation_Need ~ ETc_mulched + Eff_Rainfall
                    + Previous_Irrigation_mm * (1 - Is_Rainfed)
                    + Is_Rainfed
                    + Soil_deficit : C(Soil_Type)
                    + C(Crop_Type) : C(Crop_Growth_Stage)
                    + C(Season) : C(Region)
                    + C(Irrigation_Type)
    """
    kc = df["Crop_Growth_Stage"].map(KC_STAGE)
    cap = df["Soil_Type"].map(SOIL_CAP)
    is_rainfed = (df["Irrigation_Type"] == "Rainfed").astype(float)
    is_mulched = (df["Mulching_Used"] == "Yes").astype(float)
    et0 = et0_proxy(df)
    soil_deficit = (cap - df["Soil_Moisture"]).clip(lower=0)

    num = pd.DataFrame({
        "ETc_mulched": et0 * kc * (1 - 0.30 * is_mulched),
        "Eff_Rainfall": 0.80 * df["Rainfall_mm"],
        "Prev_Irr_active": df["Previous_Irrigation_mm"] * (1 - is_rainfed),
        "Is_Rainfed": is_rainfed,
    })

    # soil_deficit × soil_type slope (one-hot of type multiplied by deficit)
    soil_oh = pd.get_dummies(df["Soil_Type"], prefix="SoilSlope").astype(float)
    soil_slope = soil_oh.multiply(soil_deficit, axis=0)

    # C(Crop_Type):C(Crop_Growth_Stage) — full Kc surface
    crop_stage = df["Crop_Type"].astype(str) + "_" + df["Crop_Growth_Stage"].astype(str)
    crop_stage_oh = pd.get_dummies(crop_stage, prefix="CS", drop_first=True).astype(float)

    # C(Season):C(Region) — climatic regime
    season_region = df["Season"].astype(str) + "_" + df["Region"].astype(str)
    sr_oh = pd.get_dummies(season_region, prefix="SR", drop_first=True).astype(float)

    # main effect for Irrigation_Type (rainfed already absorbed but keep the others)
    irr_oh = pd.get_dummies(df["Irrigation_Type"], prefix="IRR", drop_first=True).astype(float)

    return pd.concat([num, soil_slope, crop_stage_oh, sr_oh, irr_oh], axis=1)


# ---- CV harness + bias tuning ---------------------------------------------
def tune_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, float]:
    log_p = np.log(np.clip(oof, 1e-9, 1.0))

    def score(b: np.ndarray) -> float:
        return balanced_accuracy_score(y, (log_p + b).argmax(axis=1))

    bias = -np.log(prior)
    best = score(bias)
    grid = np.linspace(-2.5, 2.5, 51)
    for _ in range(20):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(score(base))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def run_formula(name: str, X: pd.DataFrame, y: np.ndarray, prior: np.ndarray) -> dict:
    log(f"--- {name}  n_features={X.shape[1]} ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X.iloc[tr_idx])
        Xva = scaler.transform(X.iloc[va_idx])
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1,
            random_state=SEED,
        )  # multinomial is default for K>=3 with lbfgs in sklearn >= 1.5
        t0 = time.time()
        clf.fit(Xtr, y[tr_idx])
        oof[va_idx] = clf.predict_proba(Xva)
        log(f"  fold {fold+1}/{N_FOLDS}  bal_acc(argmax)="
            f"{balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1)):.5f}  "
            f"({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_bias(oof, y, prior)
    log(f"  argmax={argmax_bal:.5f}  reweight={reweight_bal:.5f}  "
        f"tuned={tuned_bal:.5f}  bias={[round(b,3) for b in bias]}")
    np.save(ART / f"oof_mnlogit_{name}.npy", oof)
    return {
        "formula": name,
        "n_features": X.shape[1],
        "argmax": argmax_bal,
        "reweight": reweight_bal,
        "tuned": tuned_bal,
        "bias": bias.tolist(),
    }


# ---- main ------------------------------------------------------------------
if __name__ == "__main__":
    log("loading")
    df = pd.read_csv("data/train.csv")
    y = df[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    rows = [
        run_formula("F1_minimal_balance", build_F1(df), y, prior),
        run_formula("F2_balance_plus_management", build_F2(df), y, prior),
        run_formula("F3_full_structural", build_F3(df), y, prior),
    ]

    print("\n=== formula comparison (OOF balanced accuracy) ===")
    print(f"{'formula':32s} {'p':>4s} {'argmax':>8s} {'reweight':>9s} {'tuned':>7s}")
    for r in rows:
        print(f"  {r['formula']:30s} {r['n_features']:4d} {r['argmax']:8.5f} "
              f"{r['reweight']:9.5f} {r['tuned']:7.5f}")

    import json
    with open(ART / "bench_mnlogit_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    log("done")
