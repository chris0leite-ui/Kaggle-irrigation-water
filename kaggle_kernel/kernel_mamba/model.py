"""Mambular tabular Mamba model + training loop.

Mambular (BASF) wraps mamba_ssm in a sklearn-style classifier with
internal preprocessing (one-hot for cats, scaling for nums). API:

    model = MambularClassifier(d_model=64, n_layers=4, dropout=0.1,
                               pooling_method='avg')
    model.fit(X_train, y_train,
              max_epochs=8, batch_size=512, lr=1e-3,
              val_size=0.0)
    proba = model.predict_proba(X_test)

X_train is a DataFrame; cat columns must be strings or pandas categorical,
num columns must be numeric. `val_size=0.0` disables mambular's
internal val split — we manage 5-fold CV ourselves at the cv.py layer.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from mambular.models import MambularClassifier


def _build_classifier(d_model: int, n_layers: int, d_state: int,
                      d_conv: int, expand: int, dropout: float):
    """Construct a MambularClassifier defensively across versions.

    Different mambular releases accept slightly different keyword sets.
    Try the modern signature first, fall back if any kwarg is unknown.
    """
    common = dict(
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        pooling_method="avg",
    )
    extras_full = dict(
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    try:
        return MambularClassifier(**common, **extras_full)
    except TypeError as e:
        print(f"[model] full kwargs rejected ({e}); retrying minimal",
              flush=True)
        return MambularClassifier(**common)


def fit_one_fold(X_tr: pd.DataFrame, y_tr: np.ndarray,
                 X_va: pd.DataFrame, X_te: pd.DataFrame,
                 *, n_epochs: int, batch_size: int, lr: float,
                 weight_decay: float, d_model: int, n_layers: int,
                 d_state: int, d_conv: int, expand: int, dropout: float,
                 num_classes: int = 3):
    """Fit one fold; return (val_proba, test_proba) as np.ndarray."""
    print(f"      build classifier d_model={d_model} n_layers={n_layers}",
          flush=True)
    clf = _build_classifier(d_model, n_layers, d_state, d_conv, expand,
                            dropout)
    fit_kwargs = dict(
        max_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        val_size=0.0,
    )
    print(f"      fit  rows={len(X_tr):,} epochs={n_epochs} "
          f"bs={batch_size}", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf.fit(X_tr, y_tr, **fit_kwargs)
        except TypeError as e:
            # Fallback: drop unknown kwargs one at a time.
            print(f"[model] fit kwargs rejected ({e}); retrying minimal",
                  flush=True)
            clf.fit(X_tr, y_tr, max_epochs=n_epochs,
                    batch_size=batch_size, lr=lr)
    print("      predict val + test", flush=True)
    p_va = clf.predict_proba(X_va)
    p_te = clf.predict_proba(X_te)
    # Defensive: some versions return list of arrays, ensure (N, C) ndarray.
    p_va = np.asarray(p_va, dtype=np.float32)
    p_te = np.asarray(p_te, dtype=np.float32)
    if p_va.ndim == 1:  # (N,) class labels — convert to one-hot
        p_va = np.eye(num_classes, dtype=np.float32)[p_va.astype(int)]
        p_te = np.eye(num_classes, dtype=np.float32)[p_te.astype(int)]
    return p_va, p_te
