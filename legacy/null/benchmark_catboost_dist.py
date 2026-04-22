"""CatBoost on the 43-feature dist set with native cat_features.

Directly tests whether CatBoost's ordered / row-wise target encoding
(which is what native cat_features triggers internally) buys anything
over LGBM / XGB on this feature set.

Baseline references:
  LGBM-dist OOF tuned        0.97266
  XGBoost-dist OOF tuned     0.97304
  LGBM-dist + XGB-dist blend 0.97327

If CatBoost OOF tuned < ~0.972, CatBoost itself isn't the missing
lever — the rival's 0.977 must rely on FE items beyond CatBoost-
native encoding (digit extraction, InnerCV TE stats beyond mean,
pairwise cat interactions at higher cardinality).

If CatBoost OOF tuned > 0.973, it's legitimately a 3rd blend leg and
should be added to the hybrid pipeline.

Compute budget: lr=0.1, depth=7, iterations=2000, early_stopping=100.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from xgb_specialist_678 import add_distance_features


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
SUB.mkdir(exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
    return b, best


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building dist features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    # CatBoost needs cat cols as strings, not category dtype
    X = tr[num_cols + cat_cols].copy()
    X_test = te[num_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    feat_cols = num_cols + cat_cols
    log(f"features: {len(feat_cols)} ({len(num_cols)} num + {len(cat_cols)} cat)")
    log(f"cat cols (native ordered TE): {cat_cols}")

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
        # explicit ordered TE (the CatBoost default for multiclass is "FeatureFreq")
        one_hot_max_size=2,
    )

    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_probs = np.zeros((len(te), 3), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_iters = []
    fold_bals = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        model = CatBoostClassifier(**cb_params)
        tr_pool = Pool(X.iloc[tr_idx], label=y[tr_idx], cat_features=cat_cols)
        va_pool = Pool(X.iloc[va_idx], label=y[va_idx], cat_features=cat_cols)
        model.fit(tr_pool, eval_set=va_pool, verbose=0)
        best_iters.append(model.tree_count_)
        oof[va_idx] = model.predict_proba(va_pool)
        test_probs += model.predict_proba(Pool(X_test, cat_features=cat_cols)) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        fold_bals.append(bal)
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.tree_count_}  "
            f"argmax_bal={bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias, tuned = tune_log_bias(oof, y, prior)
    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))

    print(f"\n=== CatBoost-dist (43 feats, native ordered TE) ===")
    print(f"  argmax               : {argmax_bal:.5f}")
    print(f"  tuned log-bias       : {tuned:.5f}")
    print(f"  LGBM-dist            : 0.97266")
    print(f"  XGB-dist             : 0.97304")
    print(f"  LGBM×XGB blend       : 0.97327")
    print(f"  Δ vs LGBM-dist       : {tuned - 0.97266:+.5f}")
    print(f"  fold std             : {np.std(fold_bals):.5f}")
    print(f"  bias                 : {bias.round(3).tolist()}")
    print(f"  OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_catboost_dist.npy", oof)
    np.save(ART / "test_catboost_dist.npy", test_probs)
    with open(ART / "catboost_dist_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "n_features": len(feat_cols),
            "best_iters": [int(x) for x in best_iters],
            "argmax_bal": float(argmax_bal),
            "tuned_bal": float(tuned),
            "delta_vs_lgbm_dist": float(tuned - 0.97266),
            "delta_vs_xgb_dist": float(tuned - 0.97304),
            "fold_bals": [float(x) for x in fold_bals],
            "log_bias": bias.tolist(),
        }, f, indent=2)

    tuned_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID],
                  TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        SUB / "submission_catboost_dist_tuned.csv", index=False)
    log(f"done")


if __name__ == "__main__":
    main()
