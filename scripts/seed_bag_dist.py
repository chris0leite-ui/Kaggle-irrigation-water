"""Seed-bag LGBM-dist (variance reduction on the best on-branch model).

LGBM-dist (single seed=42) hits OOF bal_acc 0.97266. Each fold's
argmax bal_acc has std ≈ 0.00088 across folds — the irreducible
fold noise. An independent source of noise is the stochastic
LGBM training itself (bagging_fraction, feature_fraction, split
randomisation tied to `seed`). A 3–5 seed bag averages those
out without touching CV structure.

Design:
  - Same 5-fold stratified split (seed=42) for CV comparability.
  - Inside each fold, train N_SEEDS independent LGBMs that differ
    only in LGBM's internal seed. Collect their validation probs
    and test probs. Average.
  - Write the averaged OOF probs (n, 3) + averaged test probs to
    scripts/artifacts/. Tune one global log-bias on the bagged
    OOF, same coord-ascent as the single-seed baseline.

Hypothesis: bag variance reduction should move tuned OOF from
0.97266 toward the ensemble ceiling, +0.0005 to +0.001 is a
typical bagging lift at this OOF scale. Anything above +1σ
(0.00088) is a real signal.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


CV_SEED = 42
N_FOLDS = 5
N_SEEDS = 5
BAG_SEEDS = [42, 7, 123, 2024, 9999]
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


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def main() -> None:
    log(f"loading data (bag seeds = {BAG_SEEDS})")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features: {len(feat_cols)} ({len(num_cols)} numeric + {len(cat_cols)} categorical)")

    log(f"running {N_FOLDS}-fold stratified LGBM-dist × {N_SEEDS}-seed bag")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    # per-seed OOF so we can also report single-seed stats
    oof_per_seed = np.zeros((N_SEEDS, len(tr), len(CLASSES)), dtype=np.float64)
    test_per_seed = np.zeros((N_SEEDS, len(te), len(CLASSES)), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        fold_t0 = time.time()
        dtr = lgb.Dataset(
            X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols
        )
        dva = lgb.Dataset(
            X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_cols,
            reference=dtr,
        )
        for s_idx, seed in enumerate(BAG_SEEDS):
            t0 = time.time()
            params = dict(
                objective="multiclass",
                num_class=len(CLASSES),
                metric="multi_logloss",
                learning_rate=0.05,
                num_leaves=127,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                min_data_in_leaf=200,
                verbose=-1,
                seed=seed,
            )
            model = lgb.train(
                params,
                dtr,
                num_boost_round=4000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            oof_per_seed[s_idx, va_idx] = model.predict(
                X.iloc[va_idx], num_iteration=model.best_iteration
            )
            test_per_seed[s_idx] += model.predict(
                X_test, num_iteration=model.best_iteration
            ) / N_FOLDS
            fold_bal = balanced_accuracy_score(
                y[va_idx], oof_per_seed[s_idx, va_idx].argmax(axis=1)
            )
            log(f"  fold {fold+1}/{N_FOLDS}  seed={seed}  "
                f"best_iter={model.best_iteration}  "
                f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")
        log(f"  fold {fold+1}/{N_FOLDS} DONE in {time.time()-fold_t0:.1f}s")

    # single-seed tuned OOFs for reference
    per_seed_tuned = []
    for s_idx, seed in enumerate(BAG_SEEDS):
        _, best_seed = tune_log_bias(oof_per_seed[s_idx], y, prior)
        per_seed_tuned.append((seed, best_seed))
        log(f"  seed={seed}  tuned OOF = {best_seed:.5f}")

    # bag
    oof_bag = oof_per_seed.mean(axis=0)
    test_bag = test_per_seed.mean(axis=0)

    argmax_bal = balanced_accuracy_score(y, oof_bag.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof_bag / prior).argmax(axis=1))
    log("coord-ascent over per-class log-bias on bagged OOF")
    bias, tuned_bal = tune_log_bias(oof_bag, y, prior)
    log(f"  bias = {dict(zip(CLASSES, bias.round(4)))}  oof_bal_acc={tuned_bal:.5f}")

    cm = confusion_matrix(
        y, (np.log(np.clip(oof_bag, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"OOF confusion matrix (bag):\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== LGBM-dist × seed-bag summary (OOF bal_acc) ===")
    print(f"  {'seed':>6}  {'tuned OOF':>10}")
    for seed, t in per_seed_tuned:
        print(f"  {seed:>6}  {t:>10.5f}")
    print(f"  {'bag':>6}  {tuned_bal:>10.5f}")
    best_single = max(t for _, t in per_seed_tuned)
    print(f"  Δ(bag − best_single) = {tuned_bal - best_single:+.5f}")
    print(f"  Δ(bag − seed=42)     = "
          f"{tuned_bal - dict(per_seed_tuned)[42]:+.5f}")

    np.save(ART_DIR / "oof_lgbm_dist_bag.npy", oof_bag)
    np.save(ART_DIR / "test_lgbm_dist_bag.npy", test_bag)
    with open(ART_DIR / "seed_bag_dist_results.json", "w") as f:
        json.dump({
            "cv_seed": CV_SEED,
            "n_folds": N_FOLDS,
            "bag_seeds": BAG_SEEDS,
            "n_features": len(feat_cols),
            "log_bias": bias.tolist(),
            "per_seed_tuned": [{"seed": s, "tuned_bal_acc": t} for s, t in per_seed_tuned],
            "bag_argmax_bal_acc": float(argmax_bal),
            "bag_reweight_bal_acc": float(reweight_bal),
            "bag_tuned_bal_acc": float(tuned_bal),
        }, f, indent=2)

    tuned_test_idx = (
        np.log(np.clip(test_bag, 1e-9, 1.0)) + bias
    ).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT_DIR / "submission_lgbm_dist_bag_tuned.csv", index=False
    )
    argmax_test_idx = test_bag.argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in argmax_test_idx]}).to_csv(
        OUT_DIR / "submission_lgbm_dist_bag_argmax.csv", index=False
    )
    log(f"OOF + test probs saved to {ART_DIR}/; submissions to {OUT_DIR}/")


if __name__ == "__main__":
    main()
