"""Multi-fold-seed bagging of the full greedy+nonrule pipeline.

Runs the three greedy components (routed_v3, spec_678, nonrule) at a single
fold-seed and saves per-seed OOF / test arrays. Driver script
`fold_seed_bag_blend.py` averages arrays across seeds, rebuilds hybrid_v3 /
greedy / greedy+nonrule, tunes log-bias, and writes a submission.

Motivation: the 2026-04-22 seed-bag experiment varied only XGB training
seeds with StratifiedKFold(seed=42) fixed, and got LB regression because
XGB at our HPs is near-deterministic across training seeds. The
unexercised lever is varying the FOLD-SPLIT seed itself — each seed gives
every row a different fold neighbour set, so the OOF predictions are
sample-diverse rather than model-stochastic.

Usage:
  FOLD_SEED=42  python3 scripts/fold_seed_bag_pipeline.py
  FOLD_SEED=7   python3 scripts/fold_seed_bag_pipeline.py
  ...

Artefacts per run:
  scripts/artifacts/oof_xgb_dist_routed_v3_fs{seed}.npy
  scripts/artifacts/test_xgb_dist_routed_v3_fs{seed}.npy
  scripts/artifacts/oof_xgb_spec_678_fs{seed}.npy
  scripts/artifacts/test_xgb_spec_678_fs{seed}.npy
  scripts/artifacts/oof_xgb_nonrule_fs{seed}.npy
  scripts/artifacts/test_xgb_nonrule_fs{seed}.npy
  scripts/artifacts/fold_seed_bag_pipeline_fs{seed}.json
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


FOLD_SEED = int(os.environ.get("FOLD_SEED", "42"))
XGB_SEED = 42
N_FOLDS = 5
ROUTED_SCORES = (0, 1, 2)
SPEC_SCORES = (6, 7, 8)
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ACTIVE_STAGES = ("Flowering", "Vegetative")
RULE_COLS = {"Soil_Moisture", "Rainfall_mm", "Temperature_C",
             "Wind_Speed_kmh", "Mulching_Used", "Crop_Growth_Stage"}
DROP_COLS = {ID, TARGET}

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [fs={FOLD_SEED}] {msg}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
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


XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=len(CLASSES),
    eval_metric="mlogloss",
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    verbosity=0,
    seed=XGB_SEED,
)


def build_dist_matrices(tr: pd.DataFrame, te: pd.DataFrame):
    """Returns (X, X_test, y, feat_cols, tr_scores, te_scores)."""
    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    tr = tr.copy()
    te = te.copy()
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")
    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    return X, X_test, y, feat_cols, tr["dgp_score"].values, te["dgp_score"].values


def run_routed_v3(X, X_test, y, tr_scores, te_scores):
    tr_routed_mask = np.isin(tr_scores, ROUTED_SCORES)
    te_routed_mask = np.isin(te_scores, ROUTED_SCORES)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
    test_xgb = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64)
    rule_prob_low = np.array([1.0 - 2e-9, 1e-9, 1e-9], dtype=np.float64)

    dte = xgb.DMatrix(X_test, enable_categorical=True)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_filt = tr_idx[~np.isin(tr_scores[tr_idx], ROUTED_SCORES)]
        va_filt = va_idx[~np.isin(tr_scores[va_idx], ROUTED_SCORES)]
        dtr = xgb.DMatrix(X.iloc[tr_filt], label=y[tr_filt], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_filt], label=y[va_filt], enable_categorical=True)
        booster = xgb.train(
            XGB_PARAMS, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        dva_full = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        val_pred = booster.predict(dva_full, iteration_range=(0, bi + 1))
        va_mask = tr_routed_mask[va_idx]
        oof[va_idx[~va_mask]] = val_pred[~va_mask]
        oof[va_idx[va_mask]] = rule_prob_low
        test_xgb += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        log(f"  routed_v3 fold {fold+1}/{N_FOLDS} bi={bi} "
            f"bal={balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1)):.5f} "
            f"({time.time()-t0:.1f}s)")

    test_pred = test_xgb.copy()
    test_pred[te_routed_mask] = rule_prob_low
    return oof, test_pred, best_iters


def run_spec_678(X, X_test, y, tr_scores, te_scores):
    tr_spec_mask = np.isin(tr_scores, SPEC_SCORES)
    te_spec_mask = np.isin(te_scores, SPEC_SCORES)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    oof = np.zeros((len(X), 3), dtype=np.float64)
    test_spec = np.zeros((len(X_test), 3), dtype=np.float64)
    dte_spec = xgb.DMatrix(X_test.iloc[te_spec_mask], enable_categorical=True)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_spec = tr_idx[np.isin(tr_scores[tr_idx], SPEC_SCORES)]
        va_spec = va_idx[np.isin(tr_scores[va_idx], SPEC_SCORES)]
        if len(tr_spec) == 0 or len(va_spec) == 0:
            continue
        dtr = xgb.DMatrix(X.iloc[tr_spec], label=y[tr_spec], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_spec], label=y[va_spec], enable_categorical=True)
        booster = xgb.train(
            XGB_PARAMS, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        val_pred = booster.predict(dva, iteration_range=(0, bi + 1))
        oof[va_spec] = val_pred
        pred_te_spec = booster.predict(dte_spec, iteration_range=(0, bi + 1))
        spec_idx = np.where(te_spec_mask)[0]
        for i, pos in enumerate(spec_idx):
            test_spec[pos] += pred_te_spec[i] / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_spec], val_pred.argmax(1))
        log(f"  spec_678 fold {fold+1}/{N_FOLDS} bi={bi} "
            f"bal(spec)={fold_bal:.5f} ({time.time()-t0:.1f}s)")
    return oof, test_spec, best_iters


def run_nonrule(tr_df: pd.DataFrame, te_df: pd.DataFrame, y):
    all_cols = [c for c in tr_df.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]

    X = tr_df[nonrule_cols].copy()
    X_test = te_df[nonrule_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr_df[c].unique()))}
        X[c] = tr_df[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te_df[c].map(mapping).astype("int32").astype("category")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        booster = xgb.train(
            XGB_PARAMS, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        log(f"  nonrule fold {fold+1}/{N_FOLDS} bi={bi} "
            f"bal(argmax)={fold_bal:.5f} ({time.time()-t0:.1f}s)")
    return oof, test_pred, best_iters


def main() -> None:
    t_start = time.time()
    log(f"fold-seed bag pipeline starting (FOLD_SEED={FOLD_SEED}, XGB_SEED={XGB_SEED})")

    log("loading data")
    tr_raw = pd.read_csv("data/train.csv")
    te_raw = pd.read_csv("data/test.csv")

    log("building distance features")
    tr_dist = add_distance_features(tr_raw)
    te_dist = add_distance_features(te_raw)
    X, X_test, y, feat_cols, tr_scores, te_scores = build_dist_matrices(tr_dist, te_dist)

    log(f"features: dist={len(feat_cols)}")

    log("=== component 1/3: routed_v3 ===")
    oof_r, test_r, bi_r = run_routed_v3(X, X_test, y, tr_scores, te_scores)
    np.save(ART / f"oof_xgb_dist_routed_v3_fs{FOLD_SEED}.npy", oof_r)
    np.save(ART / f"test_xgb_dist_routed_v3_fs{FOLD_SEED}.npy", test_r)
    bal_r = balanced_accuracy_score(y, oof_r.argmax(1))
    log(f"routed_v3 OOF argmax bal_acc = {bal_r:.5f}")

    log("=== component 2/3: spec_678 ===")
    oof_s, test_s, bi_s = run_spec_678(X, X_test, y, tr_scores, te_scores)
    np.save(ART / f"oof_xgb_spec_678_fs{FOLD_SEED}.npy", oof_s)
    np.save(ART / f"test_xgb_spec_678_fs{FOLD_SEED}.npy", test_s)

    log("=== component 3/3: nonrule ===")
    oof_n, test_n, bi_n = run_nonrule(tr_raw, te_raw, y)
    np.save(ART / f"oof_xgb_nonrule_fs{FOLD_SEED}.npy", oof_n)
    np.save(ART / f"test_xgb_nonrule_fs{FOLD_SEED}.npy", test_n)
    bal_n = balanced_accuracy_score(y, oof_n.argmax(1))
    log(f"nonrule OOF argmax bal_acc = {bal_n:.5f}")

    elapsed = time.time() - t_start
    log(f"pipeline done in {elapsed/60:.1f} min")

    with open(ART / f"fold_seed_bag_pipeline_fs{FOLD_SEED}.json", "w") as f:
        json.dump({
            "fold_seed": FOLD_SEED,
            "xgb_seed": XGB_SEED,
            "n_folds": N_FOLDS,
            "routed_best_iters": [int(x) for x in bi_r],
            "spec_best_iters": [int(x) for x in bi_s],
            "nonrule_best_iters": [int(x) for x in bi_n],
            "routed_argmax_bal": float(bal_r),
            "nonrule_argmax_bal": float(bal_n),
            "elapsed_sec": float(elapsed),
        }, f, indent=2)


if __name__ == "__main__":
    main()
