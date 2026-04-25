"""Recipe XGB trained on train + SMOTE-NC-synthesized High rows.

The 21k High rows (3.3% prior) bottleneck the Pareto-frontier High recall
ceiling at 0.9774. SMOTE-NC synthesizes additional High rows by k-NN
interpolation in feature space, handling categorical + numeric columns
jointly.

Direct attack on class-imbalance root cause — training-data level, not
post-hoc rebalancing.

Pipeline mirrors recipe_full_te: same 443-feature matrix, same per-fold
OrderedTE, same 5-fold StratifiedKFold(seed=42), same XGB HPs. Difference:
for each train fold, SMOTE-NC oversamples High to ~42k rows (2x) BEFORE
OrderedTE fit, so the TE statistics see the augmented distribution.

Env:
  SMOTE_TARGET  — target High count per fold. Default 42000 (2x original).
  SMOTE_K       — k-NN for SMOTE-NC neighbors. Default 5.
  SMOTE_SUFFIX  — output tag. Default "smote2x".
  SMOKE=1       — 20k/2-fold quick validation.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTENC

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_full_te import (  # noqa: E402
    CLS_MAP, IDX2CLS, TARGET, load_and_engineer,
)
from recipe_ote import OrderedTE  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

SMOTE_TARGET = int(os.environ.get("SMOTE_TARGET", "42000"))
SMOTE_K = int(os.environ.get("SMOTE_K", "5"))
SMOTE_SUFFIX = os.environ.get("SMOTE_SUFFIX", "smote2x")
OUT_TAG = f"_{SMOTE_SUFFIX}"

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )

    smote_target = min(SMOTE_TARGET, 5000 if SMOKE else SMOTE_TARGET)

    cat_cols_for_smote = (info["cats"] + info["combos"] + info["digits"]
                          + info["num_as_cat"] + info["tres"])

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        y_tr = y[tr_idx]
        high_count = int((y_tr == 2).sum())
        log(f"  train class dist: L={int((y_tr == 0).sum()):,}  "
            f"M={int((y_tr == 1).sum()):,}  H={high_count:,}")

        t0 = time.time()
        raw_cols = [c for c in X_tr.columns if c != TARGET]
        cat_idx = [i for i, c in enumerate(raw_cols) if c in cat_cols_for_smote]
        smote = SMOTENC(
            categorical_features=cat_idx,
            sampling_strategy={2: smote_target},
            k_neighbors=SMOTE_K, random_state=SEED + fold,
        )
        X_tr_aug, y_tr_aug = smote.fit_resample(
            X_tr.drop(columns=[TARGET]), y_tr)
        X_tr_aug[TARGET] = y_tr_aug
        log(f"  SMOTE-NC: {len(X_tr):,} -> {len(X_tr_aug):,} rows "
            f"(H {high_count:,} -> {int((y_tr_aug == 2).sum()):,})  "
            f"wall={time.time() - t0:.1f}s")

        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr_aug))
        X_tr_shuf = X_tr_aug.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr_aug = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"  OTE done in {time.time() - t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        y_aug = X_tr_aug[TARGET].to_numpy()
        sw = compute_sample_weight("balanced", y_aug)

        log(f"  training XGB  {len(feat_cols)} feat  N_tr={len(X_tr_aug):,}")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr_aug[feat_cols], y_aug,
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500,
        )
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  best_iter={model.best_iteration}")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} +- {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    log(f"config: SMOTE_TARGET={SMOTE_TARGET}  K={SMOTE_K}  "
        f"suffix={OUT_TAG!r}  smoke={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y.astype(np.int32), prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / f"oof_recipe{OUT_TAG}.npy"
    test_path = ART / f"test_recipe{OUT_TAG}.npy"
    np.save(oof_path, result["oof"])
    np.save(test_path, result["test"])
    log(f"wrote {oof_path} + {test_path}")

    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub_path = SUB / f"submission_recipe{OUT_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}  "
        f"dist={dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, smoke=SMOKE,
        smote_target=SMOTE_TARGET, smote_k=SMOTE_K, suffix=SMOTE_SUFFIX,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned, log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
    )
    res_path = ART / f"recipe{OUT_TAG}_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
