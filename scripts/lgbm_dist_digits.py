"""LightGBM on the 43-feature dist set + per-digit numeric features.

Mirrors `scripts/xgb_dist_digits.py` — same FE, same 5-fold split
(seed=42), same digit-extraction pipeline. This is the LGBM leg of the
digit-family ensemble; diversity comes from leaf-wise LGBM vs
level-wise XGB hist on the same features.

Baseline references on this feature family:
  XGB-dist             OOF tuned  0.97304
  LGBM-dist            OOF tuned  0.97266  (from legacy benchmark_dist)
  XGB-dist + digits    OOF tuned  0.97449  -> LB 0.97468 (current best)

Hypothesis: LGBM with per-digit features gives enough orthogonal errors
to blend with digit-XGB productively. Even a flat LGBM-digits OOF in
the 0.972-0.973 range is useful if its error Jaccard with digit-XGB is
below ~0.7.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
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
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

RAW_NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
DIGITS = (-3, -2, -1, 0, 1, 2, 3)

ART = Path("scripts/artifacts")
OUT = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
                scores.append(
                    balanced_accuracy_score(
                        y, (log_oof + base).argmax(axis=1)
                    )
                )
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    log(f"adding digit features on {len(RAW_NUMERIC)} numerics × {len(DIGITS)} digits")
    tr, new_digit_cols = add_digit_features(tr, RAW_NUMERIC, DIGITS)
    te, _ = add_digit_features(te, RAW_NUMERIC, DIGITS)
    alive = drop_zero_variance(tr, te, new_digit_cols)
    log(f"  {len(new_digit_cols)} digit cols extracted, {len(alive)} kept")

    drop_cols = {ID, TARGET}
    all_cols = [c for c in tr.columns if c not in drop_cols]
    num_cols = tr[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in all_cols if c not in num_cols]

    # LGBM accepts categorical_feature as a list of column names; for other cols,
    # just int-encode like benchmark_dist did.
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features total={len(feat_cols)}  num={len(num_cols)}  cat={len(cat_cols)}")
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

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
        seed=SEED,
    )

    log("training 5-fold LGBM (dist + digits)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = lgb.Dataset(
            X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols
        )
        dva = lgb.Dataset(
            X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_cols,
            reference=dtr,
        )
        model = lgb.train(
            params,
            dtr,
            num_boost_round=4000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        best_iters.append(model.best_iteration)
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
            f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)

    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"OOF confusion matrix:\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== LGBM-dist + digits (OOF bal_acc) ===")
    print(f"  argmax          : {argmax_bal:.5f}")
    print(f"  prior-reweight  : {reweight_bal:.5f}")
    print(f"  tuned log-bias  : {tuned_bal:.5f}")

    np.save(ART / "oof_lgbm_dist_digits.npy", oof)
    np.save(ART / "test_lgbm_dist_digits.npy", test_pred)
    with open(ART / "lgbm_dist_digits_results.json", "w") as f:
        json.dump(
            {
                "seed": SEED,
                "n_folds": N_FOLDS,
                "n_features": len(feat_cols),
                "class_priors": prior.tolist(),
                "log_bias": bias.tolist(),
                "argmax_bal_acc": float(argmax_bal),
                "reweight_bal_acc": float(reweight_bal),
                "tuned_bal_acc": float(tuned_bal),
                "best_iters": [int(b) for b in best_iters],
            },
            f,
            indent=2,
        )

    tuned_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        OUT / "submission_lgbm_dist_digits_tuned.csv", index=False
    )
    log(f"artefacts written to {ART}/ and {OUT}/")


if __name__ == "__main__":
    main()
