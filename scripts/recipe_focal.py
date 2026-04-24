"""Focal-loss XGB on recipe features (High-class rare-focus variant).

Pipeline mirrors recipe_full_te.py (same 443-feature matrix, same per-fold
OrderedTE, same 5-fold StratifiedKFold seed=42) but:
  - objective = custom focal-weighted CE (gamma=2, alpha=[1,1,3] default)
  - NO sample_weight='balanced' (focal alpha already biases toward High)
  - early stopping on balanced_accuracy directly (maximize)

Hypothesis: focal loss concentrates gradient mass on (a) misclassified
rows and (b) rows of the rare High class. Different error geometry than
recipe's sample_weight='balanced'; may produce Jaccard < 0.80 blend
candidate with recipe or LB-best 3-way.

Env knobs:
  FOCAL_GAMMA   — gamma exponent (default 2.0)
  FOCAL_HIGH    — alpha for High class (default 3.0). Low/Med = 1.0.
  SMOKE=1       — 20k train, 2 folds, 300 rounds (~2-3 min)

Wall estimate: ~2h on full production (custom obj is ~2.5x slower than
native multi:softprob; soft_distill_xgb hit ~2h02m at this config).
Early-stop on bal_acc may cut a bit off.
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

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, fast_bal_acc  # noqa: E402
from focal_common import (  # noqa: E402
    make_focal_obj, make_val_bal_acc, margin_to_prob,
)
from recipe_full_te import load_and_engineer, TARGET, IDX2CLS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))
FOCAL_HIGH = float(os.environ.get("FOCAL_HIGH", "3.0"))
SUFFIX = os.environ.get("FOCAL_SUFFIX", "")
OUT_TAG = f"_{SUFFIX}" if SUFFIX else ""

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
    fold_best_iters = []

    base_params = dict(
        max_depth=4, max_leaves=30,
        eta=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        tree_method="hist",
        num_class=3,
        verbosity=0,
        disable_default_eval_metric=1,
    )
    # Cap rounds tighter than soft_distill (2000) to keep wall <2h even
    # if early stop doesn't trigger; focal usually converges faster.
    num_round = 200 if SMOKE else 2000
    esr = 40 if SMOKE else 150

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"  OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        y_tr_hard = y[tr_idx].astype(np.int32)
        y_va_hard = y[va_idx].astype(np.int32)

        dtrain = xgb.DMatrix(X_tr[feat_cols].to_numpy(dtype=np.float32),
                             label=y_tr_hard.astype(np.float32))
        dval = xgb.DMatrix(X_va[feat_cols].to_numpy(dtype=np.float32),
                           label=y_va_hard.astype(np.float32))
        dtest = xgb.DMatrix(X_te[feat_cols].to_numpy(dtype=np.float32))

        obj = make_focal_obj(y_tr_hard, gamma=FOCAL_GAMMA,
                             alpha=(1.0, 1.0, FOCAL_HIGH))
        val_metric = make_val_bal_acc(y_va_hard)

        log(f"  training XGB ({len(feat_cols)} feat, "
            f"gamma={FOCAL_GAMMA}, alpha_H={FOCAL_HIGH}, "
            f"N_tr={len(X_tr)}, N_va={len(X_va)})")
        t0 = time.time()
        booster = xgb.train(
            base_params, dtrain,
            num_boost_round=num_round,
            obj=obj, custom_metric=val_metric,
            evals=[(dval, "val")], maximize=True,
            early_stopping_rounds=esr,
            verbose_eval=200,
        )
        oof[va_idx] = margin_to_prob(booster.predict(dval, output_margin=True))
        test_pred += (
            margin_to_prob(booster.predict(dtest, output_margin=True)) / N_FOLDS
        )
        fold_bal = fast_bal_acc(y_va_hard, oof[va_idx].argmax(1))
        fold_scores.append(fold_bal)
        fold_best_iters.append(booster.best_iteration)
        log(f"  fold {fold} argmax_bal_acc = {fold_bal:.5f}  "
            f"best_iter={booster.best_iteration}  "
            f"best_score={booster.best_score:.5f}  "
            f"wall={time.time()-t0:.1f}s")

    overall = fast_bal_acc(y.astype(np.int32), oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols,
                best_iters=fold_best_iters)


def main():
    log(f"config: FOCAL_GAMMA={FOCAL_GAMMA}  FOCAL_HIGH={FOCAL_HIGH}  "
        f"suffix={OUT_TAG!r}  smoke={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y.astype(np.int32), prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / f"oof_recipe_focal{OUT_TAG}.npy"
    test_path = ART / f"test_recipe_focal{OUT_TAG}.npy"
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
    sub_path = SUB / f"submission_recipe_focal{OUT_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}  "
        f"dist={dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, smoke=SMOKE,
        focal_gamma=FOCAL_GAMMA, focal_high_alpha=FOCAL_HIGH, suffix=SUFFIX,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        fold_best_iters=[int(b) for b in result["best_iters"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned, log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
    )
    res_path = ART / f"recipe_focal{OUT_TAG}_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
