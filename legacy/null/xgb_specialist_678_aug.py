"""XGB specialist on {6,7,8} augmented with original-dataset rows.

Augmentation: the 10k original Irrigation Prediction dataset has 982
rows with dgp_score in {6,7,8} (666 Medium + 316 High). All labels
match the rule (the rule is 100% on original). Concatenated into each
fold's training subset for the specialist:
  - Specialist training pool grows from ~45k synthetic (in-fold) to
    ~46k (+2% augmentation).
  - Validation folds remain synthetic-only so OOF stays apples-to-apples
    with xgb_specialist_678.py.
  - Added rows are rule-correct examples of the specialist's class-
    balanced boundary. Hypothesis: they sharpen the specialist's
    decision function on cleanly-labeled boundary rows without
    polluting validation.

Two variants controlled by --weight:
  1.0  (default): full-weight concatenation
  0.3           : downweight original rows so they regularize feature
                  importance without dominating synthetic flip signal

Output:
  oof_xgb_spec_678_aug_w{W}.npy, test_xgb_spec_678_aug_w{W}.npy,
  xgb_spec_678_aug_w{W}_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# reuse feature builder from the baseline specialist
from xgb_specialist_678 import add_distance_features


SEED = 42
N_FOLDS = 5
SPEC_SCORES = (6, 7, 8)
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART_DIR = Path("scripts/artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weight", type=float, default=1.0,
                   help="sample weight for original rows (1.0 = full)")
    args = p.parse_args()
    w_orig = float(args.weight)
    suffix = f"w{w_orig:.1f}".replace(".", "")

    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/irrigation_prediction.csv")
    # original has no 'id' column; add one to keep concat clean
    orig[ID] = np.arange(-len(orig), 0)  # negative ids so they never clash

    log("building distance features (train, test, original)")
    tr = add_distance_features(tr)
    te = add_distance_features(te)
    orig = add_distance_features(orig)

    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values
    orig_scores = orig["dgp_score"].values
    tr_spec_mask = np.isin(tr_scores, SPEC_SCORES)
    te_spec_mask = np.isin(te_scores, SPEC_SCORES)
    orig_spec_mask = np.isin(orig_scores, SPEC_SCORES)
    log(f"train rows in spec: {tr_spec_mask.sum()} / {len(tr)}")
    log(f"test  rows in spec: {te_spec_mask.sum()} / {len(te)}")
    log(f"orig  rows in spec: {orig_spec_mask.sum()} / {len(orig)} "
        f"(weight = {w_orig})")

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

    # shared categorical mapping across tr / te / orig
    for c in cat_cols:
        vals = sorted(set(tr[c].unique())
                      | set(te[c].unique())
                      | set(orig[c].unique()))
        mapping = {v: i for i, v in enumerate(vals)}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")
        orig[c] = orig[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    X_orig = orig[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
        X_orig[c] = X_orig[c].astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    y_orig = orig[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    spec_prior = np.bincount(y[tr_spec_mask], minlength=3) / tr_spec_mask.sum()
    log(f"spec-domain synthetic priors: {dict(zip(CLASSES, spec_prior.round(4)))}")
    orig_spec_prior = (np.bincount(y_orig[orig_spec_mask], minlength=3)
                       / max(orig_spec_mask.sum(), 1))
    log(f"spec-domain original  priors: {dict(zip(CLASSES, orig_spec_prior.round(4)))}")
    log(f"features: {len(feat_cols)} ({len(num_cols)} num + {len(cat_cols)} cat)")

    # pre-build the original-spec training block (unchanged across folds)
    orig_spec_idx = np.where(orig_spec_mask)[0]
    X_orig_spec = X_orig.iloc[orig_spec_idx]
    y_orig_spec = y_orig[orig_spec_idx]
    w_orig_spec = np.full(len(y_orig_spec), w_orig, dtype=np.float32)

    log(f"running 5-fold stratified XGB spec-aug on scores {SPEC_SCORES}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_spec = np.zeros((len(tr), 3), dtype=np.float64)
    test_spec = np.zeros((len(te), 3), dtype=np.float64)

    xgb_params = dict(
        objective="multi:softprob",
        num_class=3,
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
    dte_spec = xgb.DMatrix(X_test.iloc[te_spec_mask], enable_categorical=True)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_spec = tr_idx[np.isin(tr_scores[tr_idx], SPEC_SCORES)]
        va_spec = va_idx[np.isin(tr_scores[va_idx], SPEC_SCORES)]
        if len(tr_spec) == 0 or len(va_spec) == 0:
            continue

        # concatenate synthetic-spec + original-spec in training
        X_tr_syn = X.iloc[tr_spec]
        y_tr_syn = y[tr_spec]
        w_tr_syn = np.ones(len(y_tr_syn), dtype=np.float32)

        X_tr_all = pd.concat([X_tr_syn, X_orig_spec], axis=0, ignore_index=True)
        y_tr_all = np.concatenate([y_tr_syn, y_orig_spec])
        w_tr_all = np.concatenate([w_tr_syn, w_orig_spec])

        # keep categorical dtype
        for c in cat_cols:
            X_tr_all[c] = X_tr_all[c].astype("category")

        dtr = xgb.DMatrix(X_tr_all, label=y_tr_all, weight=w_tr_all,
                          enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_spec], label=y[va_spec],
                          enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
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
        log(f"  fold {fold+1}/{N_FOLDS}  n_tr_syn={len(tr_spec)}  "
            f"n_tr_orig={len(y_orig_spec)}  n_va={len(va_spec)}  "
            f"best_iter={best_iter}  bal={fold_bal:.5f}  raw={raw_acc:.5f}  "
            f"({time.time()-t0:.1f}s)")

    spec_y = y[tr_spec_mask]
    spec_oof = oof_spec[tr_spec_mask]
    argmax_bal = balanced_accuracy_score(spec_y, spec_oof.argmax(axis=1))
    raw_acc = (spec_oof.argmax(axis=1) == spec_y).mean()
    reweight_bal = balanced_accuracy_score(
        spec_y, (spec_oof / spec_prior).argmax(axis=1))
    cm = confusion_matrix(spec_y, spec_oof.argmax(axis=1), labels=[0, 1, 2])

    print(f"\n=== XGB spec-aug (w={w_orig}) on {{6,7,8}} ===")
    print(f"  spec-domain rows          : {len(spec_y)} (syn) + "
          f"{len(y_orig_spec)} (orig, weighted {w_orig})")
    print(f"  specialist argmax raw_acc : {raw_acc:.5f}")
    print(f"  specialist argmax bal_acc : {argmax_bal:.5f}")
    print(f"  specialist reweight bal   : {reweight_bal:.5f}")
    print(f"  OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART_DIR / f"oof_xgb_spec_678_aug_{suffix}.npy", oof_spec)
    np.save(ART_DIR / f"test_xgb_spec_678_aug_{suffix}.npy", test_spec)
    with open(ART_DIR / f"xgb_spec_678_aug_{suffix}_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "spec_scores": list(SPEC_SCORES),
            "orig_weight": w_orig,
            "train_rows_in_spec_syn": int(tr_spec_mask.sum()),
            "orig_rows_in_spec": int(len(y_orig_spec)),
            "test_rows_in_spec": int(te_spec_mask.sum()),
            "spec_prior": spec_prior.tolist(),
            "orig_spec_prior": orig_spec_prior.tolist(),
            "best_iters_per_fold": [int(x) for x in best_iters],
            "specialist_argmax_raw_acc": float(raw_acc),
            "specialist_argmax_bal_acc": float(argmax_bal),
            "specialist_reweight_bal_acc": float(reweight_bal),
        }, f, indent=2)
    log(f"artefacts saved with suffix _{suffix}")


if __name__ == "__main__":
    main()
