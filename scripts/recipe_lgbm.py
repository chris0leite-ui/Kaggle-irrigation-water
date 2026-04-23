"""LGBM leg of the recipe pipeline — mirror of recipe_full_te.py.

Same FE, same OrderedTE (a=1), same 5-fold StratifiedKFold(seed=42). Swaps
XGB for LightGBM. Purpose: second tree-family leg at the recipe's correct FE
level. CatBoost blended null (calibration-scale mismatch on High class);
LGBM's leaf-wise growth + different loss shape may produce orthogonal errors
WITHOUT the CAT-style softer rare-class probs — so bias mismatch is less
likely to kill the blend.

HPs mirror XGB-recipe's heavy-reg regime:
    num_leaves=30                 → matches max_leaves=30
    max_depth=4                   → matches max_depth=4
    learning_rate=0.1             → same
    reg_alpha=5, reg_lambda=5     → matches XGB reg_{alpha,lambda}=5
    feature_fraction=0.8          → matches colsample_bytree=0.8
    bagging_fraction=0.8, freq=1  → matches subsample=0.8
    max_bin=1024                  → matches XGB max_bin=1024
    min_gain_to_split=0           → default
    min_data_in_leaf=20           → default
    objective="multiclass"
    n_estimators=3000, early_stopping_rounds=200

Class-balanced sample_weight matches recipe_full_te.

SMOKE=1 → 20k train, 2 folds, 200 iter for bug-hunting.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias  # noqa: E402
from recipe_full_te import load_and_engineer, TARGET, CLS_MAP, IDX2CLS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           a_ote: float = 1.0) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    lgb_params = dict(
        objective="multiclass",
        num_class=3,
        num_leaves=30,
        max_depth=4,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=5,
        reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        min_data_in_leaf=20,
        verbose=-1,
        n_jobs=-1,
        random_state=SEED,
    )
    log(f"lgb_params: {lgb_params}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=a_ote)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        sw = compute_sample_weight("balanced", y[tr_idx])

        log(f"  training LGBM on {len(feat_cols)} features, "
            f"{len(X_tr)} train / {len(X_va)} val")
        t0 = time.time()
        model = lgb.LGBMClassifier(
            **lgb_params,
            n_estimators=300 if SMOKE else 3000,
        )
        model.fit(
            X_tr[feat_cols].to_numpy(dtype=np.float32),
            y[tr_idx],
            sample_weight=sw,
            eval_set=[(X_va[feat_cols].to_numpy(dtype=np.float32), y[va_idx])],
            callbacks=[
                lgb.early_stopping(50 if SMOKE else 200, verbose=False),
                lgb.log_evaluation(500),
            ],
        )

        oof[va_idx] = model.predict_proba(
            X_va[feat_cols].to_numpy(dtype=np.float32)
        ).astype(np.float32)
        test_pred += model.predict_proba(
            X_te[feat_cols].to_numpy(dtype=np.float32)
        ).astype(np.float32) / N_FOLDS

        bal = fast_bal_acc(y[va_idx].astype(np.int32), oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
            f"best_iter={model.best_iteration_}  "
            f"wall={time.time()-t0:.1f}s")

    overall = fast_bal_acc(y.astype(np.int32), oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y.astype(np.int32), prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / "oof_recipe_lgbm.npy"
    test_path = ART / "test_recipe_lgbm.npy"
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
    sub_path = SUB / "submission_recipe_lgbm.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
    )
    with open(ART / "recipe_lgbm_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote scripts/artifacts/recipe_lgbm_results.json")


if __name__ == "__main__":
    main()
