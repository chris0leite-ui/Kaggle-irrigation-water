"""CatBoost on the 43-feature dist set + per-digit numeric features.

Mirrors `scripts/xgb_dist_digits.py` + `scripts/lgbm_dist_digits.py`.
Same 5-fold split (seed=42), same dist+digit FE.

Historical note: CatBoost on dist-only was null (OOF 0.97128, 2026-04-21)
with ~1h10m wall time. With 46 extra digit cols, expected wall is ~1.5-2h.
If a fold-1 error-Jaccard check vs digit-XGB comes back above 0.90 this
will abort early.

Gate (after fold 1):
  * Jaccard vs digit-XGB >= 0.90  -> abort, CatBoost is mimicking XGB
  * Jaccard < 0.85                -> run all folds
  * 0.85 <= Jaccard < 0.90        -> run all, treat blend lift ceiling as +0.00015
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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


def tune_log_bias(p, y, prior):
    lp = np.log(np.clip(p, 1e-9, 1.0))
    b = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + b).argmax(axis=1))
    grid = np.linspace(-3, 3, 61)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = b.copy()
            sc = []
            for g in grid:
                base[k] = b[k] + g
                sc.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(sc))
            if sc[j] > best + 1e-6:
                b[k] = b[k] + grid[j]
                best = sc[j]
                imp = True
        if not imp:
            break
    return b, float(best)


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
    cat_cols_raw = [c for c in all_cols if c not in num_cols]

    X = tr[all_cols].copy()
    X_test = te[all_cols].copy()
    # CatBoost needs cat cols as strings.
    for c in cat_cols_raw:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features total={X.shape[1]}  num={len(num_cols)}  cat={len(cat_cols_raw)}")
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    cb_params = dict(
        loss_function="MultiClass",
        iterations=2000,
        learning_rate=0.1,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        early_stopping_rounds=100,
        verbose=0,
        task_type="CPU",
        one_hot_max_size=2,
    )

    # Load digit-XGB OOF for fold-1 Jaccard gate.
    oof_xgb_digits = np.load(ART / "oof_xgb_dist_digits.npy")
    e_xgb = oof_xgb_digits.argmax(axis=1) != y

    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_probs = np.zeros((len(te), 3), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_iters = []
    fold_bals = []

    log("training 5-fold CatBoost (dist + digits)")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        model = CatBoostClassifier(**cb_params)
        tr_pool = Pool(X.iloc[tr_idx], label=y[tr_idx], cat_features=cat_cols_raw)
        va_pool = Pool(X.iloc[va_idx], label=y[va_idx], cat_features=cat_cols_raw)
        model.fit(tr_pool, eval_set=va_pool, verbose=0)
        best_iters.append(int(model.tree_count_))
        oof[va_idx] = model.predict_proba(va_pool)
        test_probs += (
            model.predict_proba(Pool(X_test, cat_features=cat_cols_raw)) / N_FOLDS
        )
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        fold_bals.append(bal)
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.tree_count_}  "
            f"argmax_bal={bal:.5f}  ({time.time()-t0:.1f}s)")

        if fold == 0:
            # Gate: check Jaccard on fold-1 val set only.
            e_cat_fold = oof[va_idx].argmax(axis=1) != y[va_idx]
            e_xgb_fold = e_xgb[va_idx]
            inter = (e_cat_fold & e_xgb_fold).sum()
            union = (e_cat_fold | e_xgb_fold).sum()
            jacc = inter / max(1, union)
            log(f"  fold-1 Jaccard (CatBoost vs XGB-digits) = {jacc:.4f}")
            if jacc >= 0.90:
                log(f"  Jaccard >= 0.90 -> aborting (CatBoost mimics XGB-digits); "
                    f"partial artefacts not saved")
                return

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias, tuned = tune_log_bias(oof, y, prior)
    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )

    print(f"\n=== CatBoost-dist + digits (OOF bal_acc) ===")
    print(f"  argmax          : {argmax_bal:.5f}")
    print(f"  tuned log-bias  : {tuned:.5f}")
    print(f"  fold std        : {np.std(fold_bals):.5f}")
    print(f"  bias            : {bias.round(3).tolist()}")
    print(f"  OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_cat_dist_digits.npy", oof)
    np.save(ART / "test_cat_dist_digits.npy", test_probs)
    with open(ART / "cat_dist_digits_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "n_features": X.shape[1],
            "best_iters": best_iters,
            "argmax_bal": float(argmax_bal),
            "tuned_bal": float(tuned),
            "fold_bals": [float(x) for x in fold_bals],
            "log_bias": bias.tolist(),
        }, f, indent=2)

    tuned_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID],
                  TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        OUT / "submission_cat_dist_digits_tuned.csv", index=False)
    log(f"done")


if __name__ == "__main__":
    main()
