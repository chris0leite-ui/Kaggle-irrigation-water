#!/usr/bin/env python
"""
5-fold CV of two classical baselines on the irrigation competition:

  (a) Multinomial Logistic Regression — linear, with balanced class
      weights so it treats rare `High` correctly. Uses domain-engineered
      features from scripts/cv_ebm.engineer().
  (b) Gaussian Naive Bayes — assumes feature independence per class.
      Well-known to be a useful "independence ceiling" baseline; its
      gap vs. a non-independent model estimates how much of the signal
      is in feature interactions.

Both use the same one-hot (for LR) / label-encoded (for NB) preprocess,
the same 5 folds, and the same random seed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reuse the feature engineering from cv_ebm.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_ebm import engineer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "scripts" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def split_cols(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def make_lr_matrix(X: pd.DataFrame, enc: OneHotEncoder | None,
                   scl: StandardScaler | None, fit: bool):
    num, cat = split_cols(X)
    if fit:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        enc.fit(X[cat])
        scl = StandardScaler()
        scl.fit(X[num])
    Xc = enc.transform(X[cat])
    Xn = scl.transform(X[num])
    from scipy.sparse import hstack, csr_matrix
    return hstack([csr_matrix(Xn), Xc]).tocsr(), enc, scl


def make_nb_matrix(X: pd.DataFrame, maps: dict | None, fit: bool):
    """Label-encode categoricals (train-fold maps only), return dense float."""
    num, cat = split_cols(X)
    out = X[num].to_numpy(dtype=float, copy=True)
    if fit:
        maps = {}
        for c in cat:
            vals = X[c].astype(str).unique().tolist()
            maps[c] = {v: i for i, v in enumerate(vals)}
    cat_cols = []
    for c in cat:
        cat_cols.append(
            X[c].astype(str).map(maps[c]).fillna(-1).to_numpy(dtype=float)
        )
    if cat_cols:
        out = np.hstack([out, np.stack(cat_cols, axis=1)])
    return out, maps


def evaluate(name: str, predict_fn, X: pd.DataFrame, y: pd.Series,
             classes: list[str]) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    cum_cm = np.zeros((3, 3), dtype=int)

    for k, (tr, va) in enumerate(skf.split(X, y)):
        ft = time.time()
        pred = predict_fn(X.iloc[tr], y.iloc[tr], X.iloc[va])
        s = balanced_accuracy_score(y.iloc[va], pred)
        fold_scores.append(s)
        cum_cm += confusion_matrix(y.iloc[va], pred, labels=classes)
        print(f"  [{name}] fold {k+1}/5: bal_acc={s:.5f}  ({time.time()-ft:.1f}s)")

    fold_scores = np.array(fold_scores)
    recall = cum_cm.diagonal() / cum_cm.sum(axis=1).clip(1)
    print(f"  [{name}] mean ± std: {fold_scores.mean():.5f} ± {fold_scores.std():.5f}")
    for lbl, r in zip(classes, recall):
        print(f"    recall[{lbl}] = {r:.5f}")
    return {
        "tag": name,
        "fold_scores": fold_scores.tolist(),
        "mean": float(fold_scores.mean()),
        "std": float(fold_scores.std()),
        "classes": classes,
        "confusion_matrix": cum_cm.tolist(),
        "recall": dict(zip(classes, map(float, recall))),
    }


def predict_lr(X_tr, y_tr, X_va):
    Xtr, enc, scl = make_lr_matrix(X_tr, None, None, fit=True)
    Xva, _, _ = make_lr_matrix(X_va, enc, scl, fit=False)
    # multinomial LR with balanced weights; saga + l2 handles big sparse.
    clf = LogisticRegression(
        solver="saga", penalty="l2", C=1.0, max_iter=200, n_jobs=-1,
        class_weight="balanced", random_state=42,
    )
    clf.fit(Xtr, y_tr)
    return clf.predict(Xva)


def predict_nb(X_tr, y_tr, X_va):
    Xtr, maps = make_nb_matrix(X_tr, None, fit=True)
    Xva, _ = make_nb_matrix(X_va, maps, fit=False)
    clf = GaussianNB()
    clf.fit(Xtr, y_tr)
    return clf.predict(Xva)


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    print(f"loaded train.csv: {train.shape}")

    y = train["Irrigation_Need"]
    X = train.drop(columns=["id", "Irrigation_Need"])
    X = engineer(X)
    print(f"features (incl. FE): {X.shape[1]}")

    classes = sorted(y.unique().tolist())

    print("\n--- Multinomial Logistic Regression (class_weight=balanced) ---")
    lr_res = evaluate("lr_multinomial", predict_lr, X, y, classes)
    (ART / "cv_lr_multinomial.json").write_text(json.dumps(lr_res, indent=2))

    print("\n--- Gaussian Naive Bayes ---")
    nb_res = evaluate("gaussian_nb", predict_nb, X, y, classes)
    (ART / "cv_gaussian_nb.json").write_text(json.dumps(nb_res, indent=2))

    print(f"\ntotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
