"""XGB specialist on dgp_score == 3.

Score 3 is the second-largest rule-error concentration (~4,899 flips of
102,157 rows = 4.80% error rate, rule predicts Low but 4.8% are actually
Medium or High). Parallel to xgb_specialist_678.py.

Class distribution inside score=3: ~95.2% Low / ~4.8% Medium / ~0% High.
This is at the edge of the 20–80 specialist heuristic — the minority
class is small but meaningful (4.9k rows vs 56k in {6,7,8}).

Hypothesis: a specialist trained only on score=3 rows will learn the
continuous-feature pattern that flags which score-3 rows are Medium
instead of the rule's Low. If per-row override via specialist > rule,
integrate into hybrid.

Pipeline matches xgb_specialist_678.py (same 43-feature dist set, same
XGB config, same 5-fold stratified split, predicts on spec domain only,
hybrid layer handles gating).

Artefacts:
  oof_xgb_spec_3.npy (630k × 3), test_xgb_spec_3.npy (270k × 3),
  xgb_spec_3_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from xgb_specialist_678 import add_distance_features


SEED = 42
N_FOLDS = 5
SPEC_SCORES = (3,)
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART_DIR = Path("scripts/artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values
    tr_spec_mask = np.isin(tr_scores, SPEC_SCORES)
    te_spec_mask = np.isin(te_scores, SPEC_SCORES)
    log(f"train rows in score 3: {tr_spec_mask.sum()} / {len(tr)}")
    log(f"test  rows in score 3: {te_spec_mask.sum()} / {len(te)}")

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
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    spec_prior = np.bincount(y[tr_spec_mask], minlength=3) / tr_spec_mask.sum()
    log(f"spec-domain priors: {dict(zip(CLASSES, spec_prior.round(4)))}")
    log(f"features: {len(feat_cols)}")

    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=7, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        enable_categorical=True, verbosity=0, seed=SEED,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_spec = np.zeros((len(tr), 3), dtype=np.float64)
    test_spec = np.zeros((len(te), 3), dtype=np.float64)
    dte_spec = xgb.DMatrix(X_test.iloc[te_spec_mask], enable_categorical=True)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_spec = tr_idx[np.isin(tr_scores[tr_idx], SPEC_SCORES)]
        va_spec = va_idx[np.isin(tr_scores[va_idx], SPEC_SCORES)]
        if len(tr_spec) == 0 or len(va_spec) == 0:
            continue
        dtr = xgb.DMatrix(X.iloc[tr_spec], label=y[tr_spec],
                          enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_spec], label=y[va_spec],
                          enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100, verbose_eval=0,
        )
        best_iter = booster.best_iteration
        best_iters.append(best_iter)
        val_pred = booster.predict(dva, iteration_range=(0, best_iter + 1))
        oof_spec[va_spec] = val_pred
        test_pred = booster.predict(dte_spec, iteration_range=(0, best_iter + 1))
        spec_idx = np.where(te_spec_mask)[0]
        for i, pos in enumerate(spec_idx):
            test_spec[pos] += test_pred[i] / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_spec], val_pred.argmax(axis=1))
        raw_acc = (val_pred.argmax(axis=1) == y[va_spec]).mean()
        log(f"  fold {fold+1}/{N_FOLDS}  n_tr={len(tr_spec)}  n_va={len(va_spec)}  "
            f"best_iter={best_iter}  bal={fold_bal:.5f}  raw={raw_acc:.5f}  "
            f"({time.time()-t0:.1f}s)")

    spec_y = y[tr_spec_mask]
    spec_oof = oof_spec[tr_spec_mask]
    argmax_bal = balanced_accuracy_score(spec_y, spec_oof.argmax(axis=1))
    raw_acc = (spec_oof.argmax(axis=1) == spec_y).mean()

    # rule on score=3 predicts Low for all rows
    rule_pred_on_spec = np.zeros(len(spec_y), dtype=np.int32)
    rule_bal = balanced_accuracy_score(spec_y, rule_pred_on_spec)
    cm = confusion_matrix(spec_y, spec_oof.argmax(axis=1), labels=[0, 1, 2])

    print(f"\n=== XGB specialist on score=3 (evaluated on spec domain) ===")
    print(f"  n rows in spec domain      : {len(spec_y)}")
    print(f"  class distribution         : "
          f"{dict(zip(CLASSES, np.bincount(spec_y, minlength=3).tolist()))}")
    print(f"  rule bal_acc (all Low)     : {rule_bal:.5f}")
    print(f"  specialist argmax raw_acc  : {raw_acc:.5f}")
    print(f"  specialist argmax bal_acc  : {argmax_bal:.5f}")
    print(f"  Δ spec vs rule             : {argmax_bal - rule_bal:+.5f}")
    print(f"  OOF confusion matrix (spec domain):\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART_DIR / "oof_xgb_spec_3.npy", oof_spec)
    np.save(ART_DIR / "test_xgb_spec_3.npy", test_spec)
    with open(ART_DIR / "xgb_spec_3_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "spec_scores": list(SPEC_SCORES),
            "train_rows_in_spec": int(tr_spec_mask.sum()),
            "test_rows_in_spec": int(te_spec_mask.sum()),
            "spec_prior": spec_prior.tolist(),
            "best_iters_per_fold": [int(x) for x in best_iters],
            "rule_bal_acc_on_spec": float(rule_bal),
            "specialist_argmax_raw_acc": float(raw_acc),
            "specialist_argmax_bal_acc": float(argmax_bal),
        }, f, indent=2)
    log(f"spec-3 OOF saved to {ART_DIR}/")


if __name__ == "__main__":
    main()
