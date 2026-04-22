"""XGBoost on the 43-feature dist set + per-digit numeric features.

Adds the public-notebook digit-extraction idea (digits -3..+3 on all 11
raw numerics) on top of our existing XGB-dist pipeline. Hypothesis: the
host's NN label generator operates on continuous features, but the
synthetic features may carry quantisation / rounding artefacts that
correlate with flipped rows. Per-digit features give the tree direct
splits on those artefacts that axis-aligned splits on the full float
cannot express.

Pipeline mirrors `scripts/benchmark_xgb_dist.py` + the pattern from
`scripts/nonrule_features_only.py`:
  - 5-fold StratifiedKFold(seed=42) — aligned with every other OOF.
  - Same XGB params as xgb_dist / xgb_nonrule so the blend is apples-to-apples.
  - Saves oof_xgb_dist_digits.npy + test_xgb_dist_digits.npy.
  - Tuned log-bias via coord-ascent for standalone reporting only;
    the blend script is the thing that will decide if this is useful.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
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

    log("building distance features (mirror benchmark_dist)")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    log(f"adding digit features on {len(RAW_NUMERIC)} numerics × {len(DIGITS)} digits")
    tr, new_digit_cols = add_digit_features(tr, RAW_NUMERIC, DIGITS)
    te, _ = add_digit_features(te, RAW_NUMERIC, DIGITS)
    alive = drop_zero_variance(tr, te, new_digit_cols)
    log(f"  {len(new_digit_cols)} digit cols extracted, {len(alive)} kept "
        f"(dropped {len(new_digit_cols) - len(alive)} zero-variance)")

    # Figure out feature sets after FE.
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

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features total={X.shape[1]}  num={len(num_cols)}  cat={len(cat_cols)}")
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    xgb_params = dict(
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
        seed=SEED,
    )

    log("training 5-fold XGB (dist + digits)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    dte = xgb.DMatrix(X_test, enable_categorical=True)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)

    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"OOF confusion matrix:\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== XGB-dist + digits (OOF bal_acc) ===")
    print(f"  argmax          : {argmax_bal:.5f}")
    print(f"  prior-reweight  : {reweight_bal:.5f}")
    print(f"  tuned log-bias  : {tuned_bal:.5f}")

    np.save(ART / "oof_xgb_dist_digits.npy", oof)
    np.save(ART / "test_xgb_dist_digits.npy", test_pred)
    with open(ART / "xgb_dist_digits_results.json", "w") as f:
        json.dump(
            {
                "seed": SEED,
                "n_folds": N_FOLDS,
                "n_features": X.shape[1],
                "n_digit_cols_kept": len(alive),
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

    # Standalone tuned submission (diagnostic; blend script decides real submission).
    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT / "submission_xgb_dist_digits_tuned.csv", index=False
    )
    log(f"artefacts written to {ART}/ and {OUT}/")


if __name__ == "__main__":
    main()
