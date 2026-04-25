"""J3: adversarial-validation classifier — per-train-row test-resemblance.

Output: P(is_test=1 | x) per train row, computed OOF over a 5-fold split
of (train ∪ test) on a binary `is_test` label. Importance-weighting form
w = p / (1 - p) is saved separately for downstream consumers.

Features must NOT use target labels (no OTE / FREQ / ORIG_*) since the
binary target is `is_test`, not `Irrigation_Need`. Stick to:
  - 8 cats (factorized)
  - 11 raw nums
  - 4 DGP rule indicators (target-free derivations)
  - 11 decimal-fraction features `(col % 1).round(2)` per num
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "scripts" / "artifacts"
TARGET = "Irrigation_Need"
N_FOLDS = 5
SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"
SUFFIX = os.environ.get("AV_SUFFIX", "")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Concat + factorize cats + threshold flags + decimal fractions. Return
    combined frame with `is_test` label + feature cols list."""
    train = train.drop(columns=[TARGET]).copy()
    train["is_test"] = 0
    test = test.copy()
    test["is_test"] = 1

    df = pd.concat([train, test], ignore_index=True)
    cats = ["Crop_Type", "Soil_Type", "Region", "Crop_Growth_Stage",
            "Mulching_Used", "Irrigation_Type", "Water_Source", "Season"]
    nums = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
            "Humidity", "Sunlight_Hours", "Soil_pH", "Organic_Carbon",
            "Electrical_Conductivity", "Field_Area_hectare",
            "Previous_Irrigation_mm"]

    feats = []
    for c in cats:
        df[c] = pd.Categorical(df[c]).codes.astype(np.int32)
        feats.append(c)
    for c in nums:
        feats.append(c)

    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(np.int8)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(np.int8)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(np.int8)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(np.int8)
    feats += ["soil_lt_25", "rain_lt_300", "temp_gt_30", "wind_gt_10"]

    for c in nums:
        df[f"{c}_frac"] = (df[c] % 1).round(2).astype(np.float32)
        feats.append(f"{c}_frac")

    return df, feats


def main() -> None:
    log(f"loading train + test (SMOKE={SMOKE})")
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    if SMOKE:
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
    n_train = len(train)
    log(f"  train={len(train):,} test={len(test):,}")

    df, feats = build_features(train, test)
    log(f"  built {len(feats)} target-free features (cats + nums + flags + frac)")

    y_av = df["is_test"].to_numpy().astype(np.int32)
    X = df[feats]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_p = np.zeros(len(df), dtype=np.float32)
    fold_aucs = []
    params = dict(
        objective="binary:logistic",
        n_estimators=500 if not SMOKE else 50,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="auc",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )
    for fold, (tr, va) in enumerate(skf.split(X, y_av), 1):
        t0 = time.time()
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[tr], y_av[tr],
                  eval_set=[(X.iloc[va], y_av[va])], verbose=False)
        p = model.predict_proba(X.iloc[va])[:, 1].astype(np.float32)
        oof_p[va] = p
        auc = roc_auc_score(y_av[va], p)
        fold_aucs.append(auc)
        log(f"  fold {fold}/{N_FOLDS} AUC={auc:.5f} best_iter={model.best_iteration} ({time.time()-t0:.0f}s)")

    overall_auc = roc_auc_score(y_av, oof_p)
    log(f"OVERALL AV AUC = {overall_auc:.5f} (mean={np.mean(fold_aucs):.5f}, std={np.std(fold_aucs):.5f})")

    train_p = oof_p[:n_train]
    eps = 1e-6
    weights = train_p / np.clip(1 - train_p, eps, 1.0)
    log(f"train P(is_test) percentiles: p1={np.percentile(train_p,1):.4f} p50={np.percentile(train_p,50):.4f} p99={np.percentile(train_p,99):.4f}")
    log(f"importance-weight percentiles: p1={np.percentile(weights,1):.4f} p50={np.percentile(weights,50):.4f} p99={np.percentile(weights,99):.4f}")

    out_p = ART / f"j3_av_p{SUFFIX}.npy"
    out_w = ART / f"j3_av_w{SUFFIX}.npy"
    np.save(out_p, train_p)
    np.save(out_w, weights.astype(np.float32))
    res = dict(
        overall_auc=float(overall_auc),
        fold_aucs=[float(x) for x in fold_aucs],
        n_train=int(n_train),
        n_test=int(len(test)),
        n_features=len(feats),
        train_p_pct=[float(np.percentile(train_p, q)) for q in (1, 25, 50, 75, 99)],
        weight_pct=[float(np.percentile(weights, q)) for q in (1, 25, 50, 75, 99)],
        smoke=SMOKE,
    )
    with open(ART / f"j3_av_results{SUFFIX}.json", "w") as f:
        json.dump(res, f, indent=2)
    log(f"saved {out_p.name}, {out_w.name}, j3_av_results{SUFFIX}.json")


if __name__ == "__main__":
    main()
