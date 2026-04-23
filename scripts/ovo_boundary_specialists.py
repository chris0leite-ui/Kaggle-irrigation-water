"""Pairwise one-vs-one boundary specialists.

Motivation: ~74% of greedy+nonrule errors land on score=3 (Low->Medium
flips, Rainfall-driven, Cohen's d=+0.557) and score=6 (Medium->High,
Soil_Moisture-driven, d=-0.526). Prior attempts at score-band
specialists either failed on class-imbalance ({3}: 95/5 Low/Med, null)
or succeeded at {6,7,8} because the domain happened to be 69/31.

This script is a structurally different cut: train two binary OvO
classifiers, each on rows where the two conflicting classes are
roughly balanced by construction:

  Low-vs-Medium specialist:
    Training domain: rows with dgp_score in {2, 3, 4} AND y in {Low, Medium}.
    Output: binary P(Low | x, in_domain).
    Purpose: resolve the Low<->Medium ambiguity in the score=3 boundary band.

  Medium-vs-High specialist:
    Training domain: rows with dgp_score in {5, 6, 7} AND y in {Medium, High}.
    Output: binary P(Medium | x, in_domain).
    Purpose: resolve the Medium<->High ambiguity in the score=6 band.

Feature set: 43-feature dist set + 46 digit cols = 89 features
(mirrors `scripts/xgb_dist_digits.py`).

5-fold StratifiedKFold(shuffle=True, random_state=42) on the FULL
synthetic train (so OOF aligns with every other saved OOF). Within
each fold, the training subset is further filtered to the specialist's
domain; predictions are then emitted for EVERY val row (so the
specialist has an opinion for every row, even outside its training
domain -- a row with dgp_score=0 still gets a Low-vs-Med score, which
the blend logic then ignores because that row isn't in the boundary
band).

Saves:
  oof_xgb_ovo_lowmed.npy     (N_train,) -- P(Low | x, Low-or-Med)
  oof_xgb_ovo_medhigh.npy    (N_train,) -- P(Medium | x, Med-or-High)
  test_xgb_ovo_lowmed.npy    (N_test,)
  test_xgb_ovo_medhigh.npy   (N_test,)
  ovo_boundary_specialists_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_dist import add_distance_features
from digit_features import add_digit_features, drop_zero_variance


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

RAW_NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
DIGITS = (-3, -2, -1, 0, 1, 2, 3)

# Score-band domains. Training is filtered to (domain scores) AND (the two
# relevant classes); inference produces a score for every row.
LOWMED_SCORES = {2, 3, 4}    # Low-vs-Medium specialist
MEDHIGH_SCORES = {5, 6, 7}   # Medium-vs-High specialist

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_features(tr: pd.DataFrame, te: pd.DataFrame):
    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    log(f"adding digit features on {len(RAW_NUMERIC)} numerics x {len(DIGITS)} digits")
    tr, new_digit_cols = add_digit_features(tr, RAW_NUMERIC, DIGITS)
    te, _ = add_digit_features(te, RAW_NUMERIC, DIGITS)
    alive = drop_zero_variance(tr, te, new_digit_cols)
    log(f"  digits kept: {len(alive)}/{len(new_digit_cols)}")

    drop_cols = {ID, TARGET}
    all_cols = [c for c in tr.columns if c not in drop_cols]
    num_cols = tr[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in all_cols if c not in num_cols]

    X = tr[all_cols].copy()
    X_test = te[all_cols].copy()
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te[c].map(mapping).astype("int32").astype("category")

    dgp_tr = tr["dgp_score"].astype(int).values
    dgp_te = te["dgp_score"].astype(int).values
    return X, X_test, dgp_tr, dgp_te, all_cols, num_cols, cat_cols


def train_ovo(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y_full: np.ndarray,
    dgp_tr: np.ndarray,
    name: str,
    domain_scores: set,
    pos_class_idx: int,
    neg_class_idx: int,
    skf_splits,
):
    """Train a 5-fold binary OvO XGB specialist.

    pos_class_idx: the class returned as P(pos | x, pos-or-neg). For
    Low-vs-Med: pos=Low(0), neg=Medium(1). For Med-vs-High: pos=Medium(1),
    neg=High(2).
    """
    log(f"\n=== training specialist: {name} (pos={CLASSES[pos_class_idx]}, neg={CLASSES[neg_class_idx]}) ===")
    in_domain_mask = (
        np.isin(dgp_tr, list(domain_scores))
        & ((y_full == pos_class_idx) | (y_full == neg_class_idx))
    )
    log(f"  domain scores={sorted(domain_scores)}  "
        f"domain rows={in_domain_mask.sum()} / {len(y_full)}")
    y_bin_full = (y_full == pos_class_idx).astype(np.int32)  # 1 if pos else 0 (within domain)
    class_counts = np.bincount(y_bin_full[in_domain_mask])
    log(f"  domain class split: {CLASSES[neg_class_idx]}={class_counts[0]}  "
        f"{CLASSES[pos_class_idx]}={class_counts[1]}  "
        f"(pos_rate={class_counts[1] / in_domain_mask.sum():.4f})")

    xgb_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
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

    oof = np.zeros(len(y_full), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)

    fold_stats = []
    for fold, (tr_idx, va_idx) in enumerate(skf_splits):
        t0 = time.time()
        # Filter training to specialist domain
        tr_mask = in_domain_mask[tr_idx]
        tr_sub_idx = tr_idx[tr_mask]
        # Val mask for in-domain rows only -- to compute a meaningful AUC
        va_mask = in_domain_mask[va_idx]
        va_sub_idx = va_idx[va_mask]

        dtr = xgb.DMatrix(
            X.iloc[tr_sub_idx], label=y_bin_full[tr_sub_idx],
            enable_categorical=True,
        )
        # Eval on in-domain val for early-stopping (logloss proxy)
        dva = xgb.DMatrix(
            X.iloc[va_sub_idx], label=y_bin_full[va_sub_idx],
            enable_categorical=True,
        )
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration

        # Predict for ALL val rows and all test rows
        dva_all = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        oof[va_idx] = booster.predict(dva_all, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS

        # In-domain AUC
        try:
            va_auc = roc_auc_score(
                y_bin_full[va_sub_idx],
                booster.predict(dva, iteration_range=(0, bi + 1)),
            )
        except ValueError:
            va_auc = float("nan")
        fold_stats.append({"fold": fold, "best_iter": bi,
                           "in_domain_auc": va_auc,
                           "n_train": int(tr_mask.sum()),
                           "n_val_in_dom": int(va_mask.sum())})
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"n_train={tr_mask.sum()}  val_AUC={va_auc:.5f}  "
            f"({time.time()-t0:.1f}s)")

    # Full-OOF AUC on in-domain rows
    try:
        full_auc = roc_auc_score(y_bin_full[in_domain_mask], oof[in_domain_mask])
    except ValueError:
        full_auc = float("nan")
    log(f"  OOF in-domain AUC = {full_auc:.5f}  "
        f"(mean fold AUC = {np.nanmean([s['in_domain_auc'] for s in fold_stats]):.5f})")

    return oof, test_pred, fold_stats, full_auc


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    y_full = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    X, X_test, dgp_tr, dgp_te, _, _, _ = build_features(tr, te)
    log(f"features shape: train={X.shape}  test={X_test.shape}")

    # dgp_score distribution
    unique, counts = np.unique(dgp_tr, return_counts=True)
    score_dist = dict(zip(unique.tolist(), counts.tolist()))
    log(f"dgp_score distribution (train): {score_dist}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y_full))

    # Low-vs-Medium specialist (pos=Low, neg=Medium)
    oof_lm, test_lm, stats_lm, auc_lm = train_ovo(
        X, X_test, y_full, dgp_tr,
        name="LowMed",
        domain_scores=LOWMED_SCORES,
        pos_class_idx=CLS2IDX["Low"],
        neg_class_idx=CLS2IDX["Medium"],
        skf_splits=splits,
    )
    # Medium-vs-High specialist (pos=Medium, neg=High)
    oof_mh, test_mh, stats_mh, auc_mh = train_ovo(
        X, X_test, y_full, dgp_tr,
        name="MedHigh",
        domain_scores=MEDHIGH_SCORES,
        pos_class_idx=CLS2IDX["Medium"],
        neg_class_idx=CLS2IDX["High"],
        skf_splits=splits,
    )

    np.save(ART / "oof_xgb_ovo_lowmed.npy", oof_lm)
    np.save(ART / "oof_xgb_ovo_medhigh.npy", oof_mh)
    np.save(ART / "test_xgb_ovo_lowmed.npy", test_lm)
    np.save(ART / "test_xgb_ovo_medhigh.npy", test_mh)

    out = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "lowmed": {
            "domain_scores": sorted(LOWMED_SCORES),
            "pos_class": "Low",
            "neg_class": "Medium",
            "in_domain_auc": auc_lm,
            "folds": stats_lm,
        },
        "medhigh": {
            "domain_scores": sorted(MEDHIGH_SCORES),
            "pos_class": "Medium",
            "neg_class": "High",
            "in_domain_auc": auc_mh,
            "folds": stats_mh,
        },
        "dgp_score_distribution": {str(k): int(v) for k, v in score_dist.items()},
    }
    with open(ART / "ovo_boundary_specialists_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nartefacts written to {ART}/")


if __name__ == "__main__":
    main()
