"""Grid-search the 4 DGP rule thresholds directly on synthetic train.

Prior belief: thresholds are [Soil<25, Rain<300, Temp>30, Wind>10] from
reverse-engineering on the 10k original. Rule hits bal_acc 0.96097 /
raw 0.98364 on the 630k synthetic. This script checks whether the host
NN generator drifted any threshold by sweeping around each in a narrow
band and measuring rule-argmax bal_acc on synthetic.

Sweeps are done one-at-a-time (greedy coord ascent) starting from the
known-good thresholds, then a full joint fine grid around the winners.
"""
from __future__ import annotations

from pathlib import Path
import itertools
import json
import sys
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_STAGES = ("Flowering", "Vegetative")
LABELS = ("Low", "Medium", "High")  # int-encoded: Low=0, Medium=1, High=2


def precompute_features(df: pd.DataFrame) -> dict:
    """One-time precompute of numpy arrays (float / int) for fast eval."""
    return dict(
        sm=df["Soil_Moisture"].astype(float).values,
        rf=df["Rainfall_mm"].astype(float).values,
        tc=df["Temperature_C"].astype(float).values,
        ws=df["Wind_Speed_kmh"].astype(float).values,
        nomulch=(df["Mulching_Used"].astype(str).values == "No").astype(np.int8),
        kc=np.where(
            np.isin(df["Crop_Growth_Stage"].astype(str).values, ACTIVE_STAGES), 2, 0
        ).astype(np.int8),
    )


def score_fast(F: dict, t_soil: float, t_rain: float,
               t_temp: float, t_wind: float) -> np.ndarray:
    dry = (F["sm"] < t_soil).astype(np.int8)
    norain = (F["rf"] < t_rain).astype(np.int8)
    hot = (F["tc"] > t_temp).astype(np.int8)
    windy = (F["ws"] > t_wind).astype(np.int8)
    return 2 * (dry + norain) + (hot + windy + F["nomulch"]) + F["kc"]


def predict_int_from_score(s: np.ndarray) -> np.ndarray:
    """Int labels: 0=Low, 1=Medium, 2=High."""
    return np.where(s <= 3, 0, np.where(s <= 6, 1, 2)).astype(np.int8)


def bal_acc_int(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Fast balanced accuracy via confusion-matrix on int labels."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2).reshape(
        n_classes, n_classes
    )
    row_sum = cm.sum(axis=1)
    recall = np.where(row_sum > 0, cm.diagonal() / np.maximum(row_sum, 1), 0.0)
    return float(recall.mean())


def eval_thresholds(F: dict, y_true_int: np.ndarray,
                    t_soil: float, t_rain: float,
                    t_temp: float, t_wind: float) -> float:
    s = score_fast(F, t_soil, t_rain, t_temp, t_wind)
    pred = predict_int_from_score(s)
    return bal_acc_int(y_true_int, pred)


def main():
    print("Loading data...", flush=True)
    train = pd.read_csv(ROOT / "data/train.csv")
    y_str = train["Irrigation_Need"].astype(str).values
    y_int = np.where(y_str == "Low", 0, np.where(y_str == "Medium", 1, 2)).astype(np.int8)
    n = len(train)
    print(f"Rows: {n}", flush=True)

    F = precompute_features(train)

    base = dict(t_soil=25.0, t_rain=300.0, t_temp=30.0, t_wind=10.0)
    base_bal = eval_thresholds(F, y_int, **base)
    print(f"\nBaseline thresholds {base} -> bal_acc = {base_bal:.6f}\n", flush=True)

    # 1D marginal sweeps to identify drift direction
    sweeps = {
        "t_soil": np.arange(22.0, 28.01, 0.1),
        "t_rain": np.arange(280.0, 320.01, 1.0),
        "t_temp": np.arange(28.0, 32.01, 0.1),
        "t_wind": np.arange(8.0, 12.01, 0.1),
    }

    results = {"baseline": {"thresholds": base, "bal_acc": base_bal}}
    best_1d = dict(base)
    for name, grid in sweeps.items():
        print(f"Sweeping {name} (other 3 fixed at baseline)...", flush=True)
        rows = []
        for v in grid:
            args = dict(base)
            args[name] = float(v)
            bal = eval_thresholds(F, y_int, **args)
            rows.append((float(v), bal))
        best_v, best_bal = max(rows, key=lambda t: t[1])
        print(f"  best {name}={best_v:.2f}  bal_acc={best_bal:.6f}  (Δ={best_bal-base_bal:+.6f})",
              flush=True)
        results[f"sweep_{name}"] = {"grid": rows, "best_v": best_v, "best_bal": best_bal}
        best_1d[name] = best_v

    best_1d_bal = eval_thresholds(F, y_int, **best_1d)
    print(f"\nAll 1D winners combined {best_1d} -> bal_acc = {best_1d_bal:.6f}", flush=True)
    print(f"  Δ vs baseline: {best_1d_bal - base_bal:+.6f}", flush=True)
    results["best_1d_combined"] = {"thresholds": best_1d, "bal_acc": best_1d_bal}

    # Joint fine grid around best 1D (±0.3 continuous, ±2 for rain).
    # With fast bal_acc each eval is ~5ms -> 1.5k configs in ~8s.
    print(f"\nJoint fine grid around best 1D point...", flush=True)
    soil_grid = np.arange(best_1d["t_soil"]-0.3, best_1d["t_soil"]+0.31, 0.1)
    rain_grid = np.arange(best_1d["t_rain"]-2.0, best_1d["t_rain"]+2.01, 1.0)
    temp_grid = np.arange(best_1d["t_temp"]-0.3, best_1d["t_temp"]+0.31, 0.1)
    wind_grid = np.arange(best_1d["t_wind"]-0.3, best_1d["t_wind"]+0.31, 0.1)
    total = len(soil_grid) * len(rain_grid) * len(temp_grid) * len(wind_grid)
    print(f"  {total} configurations to evaluate", flush=True)

    best_joint = None
    best_joint_bal = -1.0
    n_done = 0
    for ts, tr, tt, tw in itertools.product(soil_grid, rain_grid, temp_grid, wind_grid):
        bal = eval_thresholds(F, y_int, float(ts), float(tr), float(tt), float(tw))
        if bal > best_joint_bal:
            best_joint_bal = bal
            best_joint = dict(t_soil=float(ts), t_rain=float(tr), t_temp=float(tt), t_wind=float(tw))
        n_done += 1
        if n_done % 200 == 0:
            print(f"  progress {n_done}/{total}  best_so_far={best_joint_bal:.6f}",
                  flush=True)

    print(f"  best joint {best_joint} -> bal_acc = {best_joint_bal:.6f}", flush=True)
    print(f"  Δ vs baseline: {best_joint_bal - base_bal:+.6f}", flush=True)
    results["best_joint"] = {"thresholds": best_joint, "bal_acc": best_joint_bal}

    # Raw accuracy at baseline and best
    s_base = score_fast(F, **base)
    s_best = score_fast(F, **best_joint)
    raw_base = (predict_int_from_score(s_base) == y_int).mean()
    raw_best = (predict_int_from_score(s_best) == y_int).mean()
    print(f"\nRaw accuracy baseline: {raw_base:.6f}  best: {raw_best:.6f}  Δ={raw_best-raw_base:+.6f}",
          flush=True)
    results["raw_acc"] = {"baseline": float(raw_base), "best": float(raw_best)}

    out_path = ROOT / "scripts/artifacts/refit_thresholds_synthetic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
