"""Student XGB trained with soft cross-entropy against LB-best blend teacher.

Pipeline mirrors recipe_full_te (same 443-feature matrix, same per-fold
OrderedTE, same 5-fold StratifiedKFold seed=42) but:
  - teacher = softmax(0.5 log recipe + 0.5 log pseudolabel) on OOF and test
  - objective = custom soft xent (grad = probs - y_soft)
  - no class-balanced sample_weight (teacher posterior already encodes it)
  - early stopping on hard-label mlogloss against real val labels

SMOKE=1 → 20k train, 2 folds, 300 rounds (~5 min).
Production: 504k train, 5 folds, 3000 rounds (~55 min wall).
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

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, fast_bal_acc  # noqa: E402
from recipe_full_te import load_and_engineer, TARGET, CLS_MAP, IDX2CLS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402
from soft_distill_common import (  # noqa: E402
    build_teacher_oof, build_teacher_test,
    make_soft_xent_obj, make_val_metric, margin_to_prob,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

W_RECIPE = float(os.environ.get("W_RECIPE", "0.5"))  # teacher log-blend weight
SUFFIX = os.environ.get("SOFT_SUFFIX", "")
OUT_TAG = f"_{SUFFIX}" if SUFFIX else ""
# Capacity knobs for the "capacity-reduced distillation" variant. Prior
# soft_distill at (max_depth=4, n_est=3000) overfit teacher OOF noise
# (LB -0.00148 vs OOF +0.00084). Smaller student can't memorize per-row
# teacher mistakes but can still absorb calibrated discrimination signal.
XGB_DEPTH = int(os.environ.get("XGB_DEPTH", "4"))
XGB_MAX_LEAVES = int(os.environ.get("XGB_MAX_LEAVES", "30"))
XGB_NROUND = int(os.environ.get("XGB_NROUND", "3000"))

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           teacher_oof: np.ndarray, teacher_test: np.ndarray) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    # Native xgb.train API — sklearn wrapper's custom obj plumbing is fragile
    # with multi-class. We manage the DMatrix directly.
    base_params = dict(
        max_depth=XGB_DEPTH, max_leaves=XGB_MAX_LEAVES,
        eta=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        tree_method="hist",
        num_class=3,
        verbosity=0,
    )
    num_round = 300 if SMOKE else XGB_NROUND
    esr = 50 if SMOKE else 200

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        # OrderedTE: same as recipe_full_te — fits on shuffled train,
        # reorders back, transforms val/test with full-train key stats.
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
        y_soft_tr = teacher_oof[tr_idx]
        y_va_hard = y[va_idx].astype(np.int32)

        dtrain = xgb.DMatrix(X_tr[feat_cols].to_numpy(dtype=np.float32),
                             label=y[tr_idx].astype(np.float32))
        dval = xgb.DMatrix(X_va[feat_cols].to_numpy(dtype=np.float32),
                           label=y_va_hard.astype(np.float32))
        dtest = xgb.DMatrix(X_te[feat_cols].to_numpy(dtype=np.float32))

        obj = make_soft_xent_obj(y_soft_tr)
        val_metric = make_val_metric(y_va_hard)

        log(f"  training XGB on {len(feat_cols)} features "
            f"(N_tr={len(X_tr)}, N_va={len(X_va)}, soft-xent obj)")
        t0 = time.time()
        booster = xgb.train(
            base_params, dtrain,
            num_boost_round=num_round,
            obj=obj, custom_metric=val_metric,
            evals=[(dval, "val")], maximize=False,
            early_stopping_rounds=esr,
            verbose_eval=500,
        )
        oof[va_idx] = margin_to_prob(booster.predict(dval, output_margin=True))
        test_pred += (
            margin_to_prob(booster.predict(dtest, output_margin=True)) / N_FOLDS
        )
        fold_bal = fast_bal_acc(y_va_hard, oof[va_idx].argmax(1))
        fold_scores.append(fold_bal)
        log(f"  fold {fold} argmax_bal_acc = {fold_bal:.5f}  "
            f"best_iter={booster.best_iteration}  "
            f"best_score={booster.best_score:.5f}  wall={time.time()-t0:.1f}s")

    overall = fast_bal_acc(y.astype(np.int32), oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    log(f"config: W_RECIPE={W_RECIPE}  SUFFIX={OUT_TAG!r}  smoke={SMOKE}  "
        f"depth={XGB_DEPTH}  max_leaves={XGB_MAX_LEAVES}  nround={XGB_NROUND}")
    log("building teacher from saved OOFs + test probs")
    teacher_oof = build_teacher_oof(w_recipe=W_RECIPE)
    teacher_test = build_teacher_test(w_recipe=W_RECIPE)
    ent = -(teacher_oof * np.log(np.clip(teacher_oof, 1e-9, 1.0))).sum(1).mean()
    log(f"  teacher OOF shape={teacher_oof.shape}  mean_entropy={ent:.5f}")

    train, test, info, test_ids = load_and_engineer()
    if SMOKE:
        # Smoke drops to 20k rows in load_and_engineer; we need to subset the
        # teacher OOF to match. Since subsample is fixed (seed=42) but the
        # teacher OOF was built on full 630k, we rebuild a fake smoke teacher
        # from class priors to exercise the path. Production run uses the
        # real teacher on full 630k.
        log("SMOKE: faking teacher on 20k rows from class priors")
        y = train[TARGET].to_numpy()
        rng = np.random.default_rng(SEED)
        teacher_oof = np.zeros((len(train), 3), dtype=np.float32)
        teacher_oof[np.arange(len(y)), y] = 0.95
        teacher_oof += rng.uniform(0.01, 0.03, teacher_oof.shape).astype(np.float32)
        teacher_oof /= teacher_oof.sum(1, keepdims=True)
        teacher_test = np.full((len(test), 3), 1.0/3.0, dtype=np.float32)

    result = run_cv(train, test, info, teacher_oof, teacher_test)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y.astype(np.int32), prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / f"oof_soft_distill{OUT_TAG}.npy"
    test_path = ART / f"test_soft_distill{OUT_TAG}.npy"
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
    sub_path = SUB / f"submission_soft_distill{OUT_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}  "
        f"dist={dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, smoke=SMOKE, w_recipe=W_RECIPE,
        suffix=SUFFIX,
        xgb_depth=XGB_DEPTH, xgb_max_leaves=XGB_MAX_LEAVES,
        xgb_nround=XGB_NROUND,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned, log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
    )
    res_path = ART / f"soft_distill{OUT_TAG}_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
