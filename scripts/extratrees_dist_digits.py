"""ExtraTreesClassifier on the 43-dist + 46-digit feature set.

Novel orthogonal-model test on the LB-best feature set (digit-XGB 0.97468).
Extra Trees uses fully random split thresholds, so error footprint differs
from gradient-boosted trees.

Parameterised via env var OTE_VARIANT (yes, reused for consistency):
  EXTRATREES_VARIANT=default   class_weight='balanced' at training time
  EXTRATREES_VARIANT=v2        no class_weight — let log-bias do all the work

v1 (balanced) produced probs that landed at OOF 0.93023 under digit-XGB's
fixed bias (off-calibration, greedy-rejected). v2 leaves probs on the raw
prior-weighted scale so the log-bias tune shifts them consistently with
XGB's family, giving greedy a chance to use them.

Pipeline mirrors `xgb_dist_digits.py`:
  - 5-fold StratifiedKFold(seed=42), aligned with every other OOF.
  - Ordinal encoding for cats (ExtraTrees doesn't take pandas categoricals).
  - Save oof + test + json + diagnostic submission.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_dist import add_distance_features
from digit_features import add_digit_features, drop_zero_variance

VARIANT = os.environ.get("EXTRATREES_VARIANT", "default")  # 'default' | 'v2'
SUFFIX = "" if VARIANT == "default" else f"_{VARIANT}"

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
    alive = drop_zero_variance(tr, te, new_digit_cols)
    log(f"  {len(alive)} digit cols kept")

    drop_cols = {ID, TARGET}
    all_cols = [c for c in tr.columns if c not in drop_cols]
    num_cols = tr[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in all_cols if c not in num_cols]

    # Ordinal-encode cats for ExtraTrees (no native categorical support).
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].astype(str).unique()))}
        tr[c] = tr[c].astype(str).map(mapping).astype(np.int32)
        te[c] = te[c].astype(str).map(mapping).astype(np.int32)

    X = tr[all_cols].values.astype(np.float32)
    X_test = te[all_cols].values.astype(np.float32)

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features total={X.shape[1]}  num={len(num_cols)}  cat={len(cat_cols)}")
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    # ExtraTrees config. v1 used class_weight='balanced' (flattened probs,
    # off-calibration). v2 drops class_weight and lets the post-training
    # log-bias coord-ascent handle the High-class rebalance, matching XGB's
    # probability scale.
    et_params = dict(
        n_estimators=500,
        criterion="gini",
        max_features="sqrt",
        min_samples_leaf=20,
        class_weight=None if VARIANT == "v2" else "balanced",
        bootstrap=False,  # true for RF, false for ExtraTrees
        n_jobs=-1,
        random_state=SEED,
        verbose=0,
    )
    log(f"ExtraTrees variant={VARIANT}  class_weight={et_params['class_weight']}")

    log(f"training 5-fold ExtraTrees  n_estimators=500")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        et = ExtraTreesClassifier(**et_params)
        et.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = et.predict_proba(X[va_idx])
        test_pred += et.predict_proba(X_test) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  bal_acc(argmax)={fold_bal:.5f}  "
            f"({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)

    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"OOF confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== ExtraTrees dist+digits (OOF bal_acc) ===")
    print(f"  argmax          : {argmax_bal:.5f}")
    print(f"  prior-reweight  : {reweight_bal:.5f}")
    print(f"  tuned log-bias  : {tuned_bal:.5f}")

    np.save(ART / f"oof_extratrees_dist_digits{SUFFIX}.npy", oof)
    np.save(ART / f"test_extratrees_dist_digits{SUFFIX}.npy", test_pred)
    with open(ART / f"extratrees_dist_digits{SUFFIX}_results.json", "w") as f:
        json.dump(
            {
                "seed": SEED,
                "n_folds": N_FOLDS,
                "n_features_total": X.shape[1],
                "class_priors": prior.tolist(),
                "log_bias": bias.tolist(),
                "argmax_bal_acc": float(argmax_bal),
                "reweight_bal_acc": float(reweight_bal),
                "tuned_bal_acc": float(tuned_bal),
            },
            f,
            indent=2,
        )

    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT / f"submission_extratrees_dist_digits{SUFFIX}_tuned.csv", index=False
    )
    log(f"artefacts written  (variant={VARIANT})")


if __name__ == "__main__":
    main()
