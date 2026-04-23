"""LightGBM on the 43-dist + 46-digit + 48-OTE feature set.

LGBM-digits alone was null vs XGB-digits (Jaccard 0.96 - same splits,
no diversity). But LGBM has never been tried on the digit-key OTE-enriched
feature set. Hypothesis: the 48 OTE cols (46 digit keys x 3 classes)
give LGBM's leaf-wise splits structurally new targets that XGB's
level-wise splits don't exploit, potentially unlocking the diversity
we couldn't find with LGBM on plain digits.

Mirrors `xgb_dist_digits_ote.py` VARIANT=digits but with LGBM:
  - 5-fold StratifiedKFold(seed=42) aligned with every other OOF.
  - Per-fold OTE: fit on tr_idx, apply to va_idx (no leak).
  - Test OTE: fit on full train (one-shot).
  - LGBM HPs: mirror benchmark_dist.py (num_leaves=127, min_data_in_leaf=200,
    lr=0.05, feature/bagging_fraction=0.9, bagging_freq=5).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_dist import add_distance_features
from digit_features import add_digit_features, drop_zero_variance
from ote_features import OTE


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
N_SHUFFLES = 8
OTE_ALPHA = 10.0

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
                    balanced_accuracy_score(y, (log_oof + base).argmax(axis=1))
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
    alive_digits = drop_zero_variance(tr, te, new_digit_cols)
    log(f"  {len(alive_digits)} digit cols kept")

    drop_cols = {ID, TARGET}
    base_cols = [c for c in tr.columns if c not in drop_cols]
    num_cols = tr[base_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in base_cols if c not in num_cols]

    # LGBM takes string/category cols natively; ordinal-encode to int codes.
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].astype(str).unique()))}
        tr[c] = tr[c].astype(str).map(mapping).astype("int32")
        te[c] = te[c].astype(str).map(mapping).astype("int32")

    feat_cols_no_ote = num_cols + cat_cols

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features pre-OTE total={len(feat_cols_no_ote)}  num={len(num_cols)}  cat={len(cat_cols)}")
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    # Digit-key OTE specs (matches OTE_VARIANT=digits).
    key_specs = [[c] for c in alive_digits]
    log(f"OTE keys: {len(key_specs)} × 3 classes = {3 * len(key_specs)} cols "
        f"(shuffles={N_SHUFFLES}, alpha={OTE_ALPHA})")

    # Test-OTE on full train (one-shot outside fold loop).
    log("building test-side OTE on full train")
    t0 = time.time()
    test_ote_blocks = []
    test_ote_cols: list[str] = []
    for spec in key_specs:
        ote = OTE(spec, n_shuffles=N_SHUFFLES, alpha=OTE_ALPHA, seed=SEED)
        ote.fit_transform_train(tr, y)
        test_ote_blocks.append(ote.transform(te))
        test_ote_cols.extend(ote.feature_names())
    test_ote_block = np.hstack(test_ote_blocks)
    log(f"  test OTE built in {time.time()-t0:.1f}s, {test_ote_block.shape[1]} cols")

    lgbm_params = dict(
        objective="multiclass",
        num_class=len(CLASSES),
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
        seed=SEED,
    )

    log("training 5-fold LGBM (dist + digits + OTE)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    best_iters: list[int] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        ote_blocks_tr = []
        ote_blocks_va = []
        ote_cols_fold: list[str] = []
        for spec in key_specs:
            ote = OTE(spec, n_shuffles=N_SHUFFLES, alpha=OTE_ALPHA, seed=SEED)
            ote_blocks_tr.append(ote.fit_transform_train(tr.iloc[tr_idx], y[tr_idx]))
            ote_blocks_va.append(ote.transform(tr.iloc[va_idx]))
            ote_cols_fold.extend(ote.feature_names())
        ote_tr = np.hstack(ote_blocks_tr)
        ote_va = np.hstack(ote_blocks_va)

        X_tr = np.hstack([tr.iloc[tr_idx][feat_cols_no_ote].values.astype(np.float32), ote_tr])
        X_va = np.hstack([tr.iloc[va_idx][feat_cols_no_ote].values.astype(np.float32), ote_va])
        X_te = np.hstack([te[feat_cols_no_ote].values.astype(np.float32), test_ote_block])

        # LGBM categorical_feature takes positions; cat cols live in [len(num_cols), len(feat_cols_no_ote))
        cat_positions = list(range(len(num_cols), len(feat_cols_no_ote)))

        dtr = lgb.Dataset(X_tr, label=y[tr_idx], categorical_feature=cat_positions)
        dva = lgb.Dataset(X_va, label=y[va_idx], categorical_feature=cat_positions,
                          reference=dtr)
        booster = lgb.train(
            lgbm_params, dtr, num_boost_round=4000,
            valid_sets=[dva],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        oof[va_idx] = booster.predict(X_va, num_iteration=bi)
        test_pred += booster.predict(X_te, num_iteration=bi) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)

    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"OOF confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== LGBM dist+digits+OTE (OOF bal_acc) ===")
    print(f"  argmax          : {argmax_bal:.5f}")
    print(f"  prior-reweight  : {reweight_bal:.5f}")
    print(f"  tuned log-bias  : {tuned_bal:.5f}")

    np.save(ART / "oof_lgbm_dist_digits_ote.npy", oof)
    np.save(ART / "test_lgbm_dist_digits_ote.npy", test_pred)
    with open(ART / "lgbm_dist_digits_ote_results.json", "w") as f:
        json.dump(
            {
                "seed": SEED,
                "n_folds": N_FOLDS,
                "n_features_total": len(feat_cols_no_ote) + test_ote_block.shape[1],
                "n_ote_keys": len(key_specs),
                "ote_n_shuffles": N_SHUFFLES,
                "ote_alpha": OTE_ALPHA,
                "class_priors": prior.tolist(),
                "log_bias": bias.tolist(),
                "argmax_bal_acc": float(argmax_bal),
                "reweight_bal_acc": float(reweight_bal),
                "tuned_bal_acc": float(tuned_bal),
                "best_iters": best_iters,
            },
            f,
            indent=2,
        )

    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT / "submission_lgbm_dist_digits_ote_tuned.csv", index=False
    )
    log(f"artefacts written")


if __name__ == "__main__":
    main()
