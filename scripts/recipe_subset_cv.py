"""5-fold CV trainer for recipe-subset XGBs.

Supports the `no_ote` variant by skipping OrderedTE and using the
numeric-only feature set. All other variants use the same OrderedTE
pass as the full recipe but on a filtered `te_cols` set.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from recipe_ote import OrderedTE

TARGET = "Irrigation_Need"
SEED = 42


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _xgb_params(smoke: bool) -> dict:
    return dict(
        n_estimators=300 if smoke else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if smoke else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss",
        enable_categorical=False, n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if smoke else 200, verbosity=0,
    )


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           n_folds: int, smoke: bool, a_ote: float = 1.0) -> dict:
    variant = info["variant"]
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    xgb_params = _xgb_params(smoke)
    use_ote = variant != "no_ote"

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        _log(f"=== fold {fold}/{n_folds} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        if use_ote and info["te_cols"]:
            _log("  fitting OrderedTE")
            t0 = time.time()
            rng = np.random.default_rng(SEED + fold)
            perm = rng.permutation(len(X_tr))
            X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
            te = OrderedTE(a=a_ote)
            X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"],
                               target=TARGET)
            inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
            X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
            X_va = te.transform(X_va)
            X_te = te.transform(X_te)
            _log(f"    OTE done in {time.time()-t0:.1f}s")
            feat_cols = numeric_feats + te.te_col_names()
        else:
            feat_cols = numeric_feats

        sw = compute_sample_weight("balanced", y[tr_idx])

        _log(f"  training XGB on {len(feat_cols)} features")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr[feat_cols], y[tr_idx],
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500,
        )
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += (model.predict_proba(X_te[feat_cols]).astype(np.float32)
                      / n_folds)
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        _log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
             f"best_iter={model.best_iteration}")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    _log(f"=== {variant} OOF argmax bal_acc = {overall:.5f}  "
         f"(mean fold {np.mean(fold_scores):.5f} "
         f"± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols,
                n_features=len(feat_cols))
