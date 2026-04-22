"""Training-data-quality experiments on the XGB-dist base model.

Tests two orthogonal levers against the xgb_dist baseline (OOF 0.97304):
  A) Heavy-weight original-dataset augmentation.
     Concat the 10k original rows to each fold's training set with
     a per-row sample_weight so their effective contribution is W×
     a synthetic row. W ∈ {5, 20, 50}.
  B) CV stratification by (target × dgp_score_bin) instead of target
     alone. Equalizes the boundary-band-row density per fold and
     stabilizes log-bias tuning.

Each config runs the full 5-fold XGB-dist pipeline (same 43-feature
dist set, same XGB hyperparams) and reports tuned OOF bal_acc +
per-class recall. Saves OOF/test npy pairs for promising configs so
they can be plugged into the greedy blend.

Baseline reference (already on disk): oof_xgb_dist.npy, 0.97304,
rec_L=0.9953 rec_M=0.9607 rec_H=0.9631.

Success criterion: tuned bal_acc >= 0.97325 (i.e. > fold-std noise
above baseline 0.97304). Anything between +0.0005 and +0.002 is
worth adding to the blend pool.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    out["dry"] = dry; out["norain"] = norain; out["hot"] = hot
    out["windy"] = windy; out["nomulch"] = nomulch
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


def fast_bal_acc(y, pred, cc):
    m = pred == y
    hit = np.array([m[y == k].sum() for k in range(3)])
    return float((hit / np.maximum(cc, 1)).mean())


def per_class_recall(y, pred, cc):
    m = pred == y
    return {CLASSES[k]: float(m[y == k].sum() / cc[k]) for k in range(3)}


def tune_log_bias(oof, y, prior, cc):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = fast_bal_acc(y, (log_oof + bias).argmax(axis=1), cc)
    gd = np.linspace(-3.0, 3.0, 61)
    gh = np.linspace(-3.0, 6.0, 91)
    for _ in range(25):
        improved = False
        for k in range(3):
            grid = gh if k == 2 else gd
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(fast_bal_acc(y, (log_oof + base).argmax(axis=1), cc))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def load_synthetic():
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    tr = add_distance_features(tr)
    te = add_distance_features(te)
    return tr, te


def load_original():
    """Load + feature-engineer the 10k original dataset."""
    with zipfile.ZipFile("data/archive.zip") as z:
        with z.open("irrigation_prediction.csv") as f:
            orig = pd.read_csv(f)
    orig = add_distance_features(orig)
    orig[ID] = -1  # sentinel; won't collide with synthetic ids
    return orig


def build_feature_matrices(tr, te, orig):
    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")
        # original may have same string vocab; apply same mapping but
        # fall back to -1 for any unseen values (shouldn't happen).
        orig[c] = orig[c].map(mapping).fillna(-1).astype("int32")
    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    X_orig = orig[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
        X_orig[c] = X_orig[c].astype("category")
    return X, X_test, X_orig, feat_cols


def run_one_config(X, y, X_test, X_orig, y_orig, prior, cc,
                   orig_weight: float, strat_by_score: bool,
                   config_name: str, dgp_score: np.ndarray | None = None):
    """Train XGB-dist under a single (orig_weight, strat_by_score) config."""
    if strat_by_score and dgp_score is None:
        raise ValueError("need dgp_score for score-stratified CV")

    xgb_params = dict(
        objective="multi:softprob", num_class=len(CLASSES),
        eval_metric="mlogloss", learning_rate=0.05, max_depth=7,
        min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", enable_categorical=True, verbosity=0, seed=SEED,
    )

    oof = np.zeros((len(X), 3), dtype=np.float64)
    test_pred = np.zeros((len(X_test), 3), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)

    # score-stratified CV: bin dgp_score into 3 groups (0-3, 4-6, 7-9)
    # and combine with target to get 9-class strat label, else just y.
    if strat_by_score:
        score_bin = np.where(dgp_score <= 3, 0,
                              np.where(dgp_score <= 6, 1, 2)).astype(np.int8)
        strat_label = y * 3 + score_bin
        log(f"  score-stratified CV: {np.bincount(strat_label)} rows per (target,score_bin)")
    else:
        strat_label = y
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_label)):
        t0 = time.time()
        X_fold_tr = X.iloc[tr_idx]
        y_fold_tr = y[tr_idx]
        w_fold_tr = np.ones(len(tr_idx), dtype=np.float32)

        if orig_weight > 0:
            X_fold_tr = pd.concat([X_fold_tr, X_orig], axis=0, ignore_index=True)
            y_fold_tr = np.concatenate([y_fold_tr, y_orig])
            w_fold_tr = np.concatenate([
                w_fold_tr,
                np.full(len(X_orig), orig_weight, dtype=np.float32),
            ])

        dtr = xgb.DMatrix(X_fold_tr, label=y_fold_tr, weight=w_fold_tr,
                          enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx],
                          enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        best_iter = booster.best_iteration
        oof[va_idx] = booster.predict(dva, iteration_range=(0, best_iter + 1))
        test_pred += booster.predict(dte, iteration_range=(0, best_iter + 1)) / N_FOLDS
        fold_bal = fast_bal_acc(y[va_idx], oof[va_idx].argmax(axis=1),
                                np.bincount(y[va_idx], minlength=3))
        log(f"    fold {fold+1}/{N_FOLDS}  best_iter={best_iter}  "
            f"bal(argmax)={fold_bal:.5f}  ({time.time()-t0:.0f}s)")

    bias, tuned = tune_log_bias(oof, y, prior, cc)
    pred = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    pcr = per_class_recall(y, pred, cc)
    log(f"  {config_name}  tuned={tuned:.5f}  "
        f"rec_L={pcr['Low']:.4f} rec_M={pcr['Medium']:.4f} rec_H={pcr['High']:.4f}  "
        f"bias={np.round(bias, 3).tolist()}")
    return oof, test_pred, bias, tuned, pcr


def main():
    log("loading synthetic + original data")
    tr, te = load_synthetic()
    orig = load_original()
    log(f"  synthetic train={len(tr)}  test={len(te)}  original={len(orig)}")
    log(f"  synthetic target dist: {dict(tr[TARGET].value_counts())}")
    log(f"  original  target dist: {dict(orig[TARGET].value_counts())}")

    X, X_test, X_orig, feat_cols = build_feature_matrices(tr, te, orig)
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    y_orig = orig[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    cc = np.bincount(y, minlength=3)
    dgp_score = tr["dgp_score"].values

    configs = [
        # (orig_weight, strat_by_score, name)
        (0,  False, "baseline_no_orig_targetstrat"),
        (20, False, "orig_w20_targetstrat"),
        (0,  True,  "no_orig_scorestrat"),
        (20, True,  "orig_w20_scorestrat"),
    ]

    results = []
    for orig_w, strat_s, name in configs:
        log(f"\n=== config: {name} (orig_w={orig_w}, strat_by_score={strat_s}) ===")
        oof, test_pred, bias, tuned, pcr = run_one_config(
            X, y, X_test, X_orig, y_orig, prior, cc,
            orig_weight=orig_w, strat_by_score=strat_s,
            config_name=name, dgp_score=dgp_score,
        )
        # save artefacts
        np.save(ART_DIR / f"oof_xgb_dist_{name}.npy", oof)
        np.save(ART_DIR / f"test_xgb_dist_{name}.npy", test_pred)
        results.append({
            "config": name, "orig_weight": orig_w,
            "strat_by_score": strat_s, "tuned_bal": tuned,
            "rec_Low": pcr["Low"], "rec_Med": pcr["Medium"],
            "rec_High": pcr["High"], "bias": bias.tolist(),
        })

    # summary
    print("\n=== SUMMARY ===")
    print(f"  baseline (saved earlier)                           tuned=0.97304")
    for r in results:
        delta = r["tuned_bal"] - 0.97304
        print(f"  {r['config']:40s}  tuned={r['tuned_bal']:.5f}  "
              f"Δ={delta:+.5f}  rec_H={r['rec_High']:.4f}")

    with open(ART_DIR / "data_quality_experiments_results.json", "w") as f:
        json.dump({"configs": results}, f, indent=2)


if __name__ == "__main__":
    main()
