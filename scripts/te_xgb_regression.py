"""Step 2/3: XGB regression onto continuous TE targets.

Variant switch via env var TE_VARIANT in {orig (default), oof}:
  orig -> reads te_targets_{train,test}.npy        writes oof_xgb_te_reg.npy
  oof  -> reads te_targets_{train,test}_oof.npy    writes oof_xgb_te_reg_oof.npy
The oof variant uses TE built leave-one-fold-out from synthetic
train labels (te_targets_oof.py) instead of the rule-perfect 10k.

Trains three independent reg:squarederror XGB boosters (one per
class) on the 43-col dist feature set, with the TE matrix from
te_targets.py as the regression target. 5-fold stratified on y
(seed=42, matches all other on-disk OOFs).

Output OOF/test matrices are clipped to [eps, 1-eps] and
row-normalised to be valid soft probability vectors that can be
log-blended into greedy.

Wall-time target: ~25 min on CPU. Logs every fold + every booster
to keep the stream alive.

Outputs:
  scripts/artifacts/oof_xgb_te_reg.npy        (630_000, 3)
  scripts/artifacts/test_xgb_te_reg.npy       (270_000, 3)
  scripts/artifacts/te_xgb_regression_results.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
VARIANT = os.environ.get("TE_VARIANT", "orig").lower()
if VARIANT not in ("orig", "oof"):
    raise SystemExit(f"TE_VARIANT must be 'orig' or 'oof', got {VARIANT!r}")
SUFFIX = "" if VARIANT == "orig" else "_oof"
TE_TRAIN_PATH = f"te_targets_train{SUFFIX}.npy"
TE_TEST_PATH = f"te_targets_test{SUFFIX}.npy"
OOF_OUT_PATH = f"oof_xgb_te_reg{SUFFIX}.npy"
TEST_OUT_PATH = f"test_xgb_te_reg{SUFFIX}.npy"
RESULTS_OUT_PATH = f"te_xgb_regression_results{SUFFIX}.json"
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART = Path("scripts/artifacts")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    """43-col dist feature set (matches benchmark_xgb_dist.py)."""
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

    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)
    out["dry"] = dry
    out["norain"] = norain
    out["hot"] = hot
    out["windy"] = windy
    out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)
    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)
    return out


def build_xy(tr: pd.DataFrame, te: pd.DataFrame):
    """Return (X_train, X_test, cat_cols, num_cols)."""
    drop = {ID, TARGET}
    num_cols = [c for c in tr.select_dtypes(include=[np.number]).columns if c not in drop]
    cat_cols = [c for c in tr.columns if c not in num_cols and c not in drop]

    X = tr[num_cols + cat_cols].copy()
    X_test = te[num_cols + cat_cols].copy()
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te[c].map(mapping).astype("int32").astype("category")
    return X, X_test, cat_cols, num_cols


def normalise_probs(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return p / p.sum(axis=1, keepdims=True)


def main() -> None:
    t_all = time.time()
    log(f"variant={VARIANT}  reading {TE_TRAIN_PATH} / {TE_TEST_PATH}")
    log("loading data + TE targets")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    tt_train = np.load(ART / TE_TRAIN_PATH)
    tt_test = np.load(ART / TE_TEST_PATH)  # noqa: F841 (used implicitly via XGB on X_test)
    log(f"  train={len(tr)}  test={len(te)}  te_train shape={tt_train.shape}")

    log("building dist features (train, test)")
    tr = add_dist_features(tr)
    te = add_dist_features(te)
    X, X_test, cat_cols, num_cols = build_xy(tr, te)
    log(f"  features: {X.shape[1]} ({len(num_cols)} num + {len(cat_cols)} cat)")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    xgb_params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=SEED,
    )
    log(f"XGB params: max_depth=7 lr=0.05 reg:squarederror, 3 boosters/fold")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_pred = np.zeros((len(te), 3), dtype=np.float64)
    fold_best_iters: list[list[int]] = []

    dte = xgb.DMatrix(X_test, enable_categorical=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t_fold = time.time()
        log(f"--- fold {fold+1}/{N_FOLDS}  train={len(tr_idx)}  val={len(va_idx)}")
        dtr_data = X.iloc[tr_idx]
        dva_data = X.iloc[va_idx]
        per_fold_iters = []
        for k, cls in enumerate(CLASSES):
            t_b = time.time()
            dtr = xgb.DMatrix(dtr_data, label=tt_train[tr_idx, k],
                              enable_categorical=True)
            dva = xgb.DMatrix(dva_data, label=tt_train[va_idx, k],
                              enable_categorical=True)
            booster = xgb.train(
                xgb_params, dtr, num_boost_round=3000,
                evals=[(dva, "val")],
                early_stopping_rounds=80,
                verbose_eval=0,
            )
            bi = booster.best_iteration
            per_fold_iters.append(int(bi))
            oof[va_idx, k] = booster.predict(dva, iteration_range=(0, bi + 1))
            test_pred[:, k] += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
            log(f"  booster[{cls:6s}]  best_iter={bi:4d}  "
                f"({time.time()-t_b:.1f}s)")
        # Report per-fold proxy bal_acc by argmaxing the un-normalised output.
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold done in {time.time()-t_fold:.1f}s   "
            f"argmax bal_acc(raw) = {fold_bal:.5f}")
        fold_best_iters.append(per_fold_iters)

    log("normalising raw regression outputs to soft probs")
    oof_p = normalise_probs(oof)
    test_p = normalise_probs(test_pred)

    np.save(ART / OOF_OUT_PATH, oof_p)
    np.save(ART / TEST_OUT_PATH, test_p)
    log(f"saved {OOF_OUT_PATH} {oof_p.shape}, {TEST_OUT_PATH} {test_p.shape}")

    raw_argmax = balanced_accuracy_score(y, oof_p.argmax(axis=1))
    log(f"standalone argmax bal_acc = {raw_argmax:.5f}")

    with open(ART / RESULTS_OUT_PATH, "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "n_features": int(X.shape[1]),
            "fold_best_iters": fold_best_iters,
            "standalone_argmax_bal": float(raw_argmax),
            "wall_time_s": time.time() - t_all,
        }, f, indent=2)
    log(f"done in {time.time()-t_all:.1f}s total")


if __name__ == "__main__":
    main()
