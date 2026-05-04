"""Baseline LGBM — Day-1 calibration probe.

Generic 5-fold (Stratified for classification, plain for regression),
log-bias tuning for balanced-accuracy classification.
Reads target_col/id_col/metric/task from comp-context.md.
Emits oof_/test_baseline_lgbm.npy, results.json, submission CSV.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import N_FOLDS, SEED, fast_bal_acc, folds, save_oof, tune_log_bias


def parse_comp_context() -> dict:
    """Extract YAML-ish fields from comp-context.md."""
    text = Path("comp-context.md").read_text()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s*(.+?)\s*(#.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def main():
    ctx = parse_comp_context()
    target_col = ctx.get("target_col", "target")
    id_col = ctx.get("id_col", "id")
    task = ctx.get("task", "classification")
    metric = ctx.get("metric", "bal_acc")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sub = pd.read_csv("data/sample_submission.csv")

    # Encode target if classification
    y_raw = train[target_col].values
    if task == "classification":
        classes = sorted(pd.unique(y_raw))
        cls2idx = {c: i for i, c in enumerate(classes)}
        y = np.array([cls2idx[c] for c in y_raw])
        n_class = len(classes)
        objective = "multiclass" if n_class > 2 else "binary"
    else:
        y = y_raw.astype(np.float64)
        n_class = 1
        objective = "regression"
        classes = None

    # Feature prep — drop id, encode object columns as categorical
    feat = train.drop(columns=[target_col, id_col], errors="ignore")
    test_feat = test.drop(columns=[id_col], errors="ignore")
    cat_cols = feat.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        feat[c] = feat[c].astype("category")
        test_feat[c] = test_feat[c].astype("category")

    params = dict(
        objective=objective,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_data_in_leaf=200,
        verbose=-1,
        seed=SEED,
    )
    if objective == "multiclass":
        params["num_class"] = n_class
        params["metric"] = "multi_logloss"

    oof = np.zeros((len(train), max(n_class, 1)), dtype=np.float32)
    test_proba = np.zeros((len(test), max(n_class, 1)), dtype=np.float32)
    fold_scores = []

    for k, tr, va in folds(y if task == "classification" else y.astype(int),
                           task=task):
        dtrain = lgb.Dataset(feat.iloc[tr], y[tr], categorical_feature=cat_cols)
        dval = lgb.Dataset(feat.iloc[va], y[va], categorical_feature=cat_cols)
        model = lgb.train(params, dtrain, num_boost_round=2000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        if task == "classification":
            p = model.predict(feat.iloc[va])
            if objective == "binary":
                p = np.column_stack([1 - p, p])
            oof[va] = p
            tp = model.predict(test_feat)
            if objective == "binary":
                tp = np.column_stack([1 - tp, tp])
            test_proba += tp / N_FOLDS
            fold_scores.append(fast_bal_acc(y[va], p.argmax(1), n_class))
        else:
            p = model.predict(feat.iloc[va])
            oof[va, 0] = p
            test_proba[:, 0] += model.predict(test_feat) / N_FOLDS
            fold_scores.append(float(np.sqrt(((p - y[va]) ** 2).mean())))

    fold_std = float(np.std(fold_scores))
    print(f"per-fold scores: {fold_scores}")

    # Tune log-bias for classification + bal_acc
    if task == "classification" and metric == "bal_acc":
        bias = tune_log_bias(y, oof, metric="bal_acc")
        log_p = np.log(np.clip(oof, 1e-12, None))
        oof_score = fast_bal_acc(y, (log_p + bias).argmax(1), n_class)
    else:
        bias = None
        oof_score = float(np.mean(fold_scores))

    print(f"OOF {metric}: {oof_score:.5f}  fold_std={fold_std:.5f}  bias={bias}")

    # Build submission
    if task == "classification":
        log_p = np.log(np.clip(test_proba, 1e-12, None))
        if bias is not None:
            pred_idx = (log_p + bias).argmax(1)
        else:
            pred_idx = test_proba.argmax(1)
        sub[target_col] = [classes[i] for i in pred_idx]
    else:
        sub[target_col] = test_proba[:, 0]

    Path("submissions").mkdir(exist_ok=True)
    sub.to_csv("submissions/submission_baseline_lgbm.csv", index=False)

    save_oof("baseline_lgbm", oof, test_proba, dict(
        oof_score=oof_score, fold_std=fold_std, fold_scores=fold_scores,
        bias=bias.tolist() if bias is not None else None,
        metric=metric, n_class=n_class, classes=classes,
    ))


if __name__ == "__main__":
    main()
