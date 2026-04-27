"""Leak-eliminated soft-distillation student.

Consumes the per-outer-fold teacher OOFs from leakfree_teacher_oof.py:
  oof_recipe_leakfree_outer{1..5}.npy   (630_000, 3) — zero on V_outer
  test_recipe_leakfree_outer{1..5}.npy  (270_000, 3)

Student CV: the same StratifiedKFold(seed=42) split — outer fold f's
training rows tr_idx receive teacher targets from oof_leakfree_outer{f}
(which was built ONLY on those rows via inner-CV that didn't see V_f).

Test-side teacher target = average of test_leakfree_outer{1..5}, since
each outer was trained on a different 80% of train; the student's test
output is the soft-blend across all 5.

This is the proper fix for the persistent OOF→LB gap (+0.00201 to
+0.00246 across all prior distill capacity points). Distill_small at
d=3 was OOF 0.98066 standalone vs LB 0.97865; if leak-elimination is
the right diagnosis, this student should narrow the gap below +0.0010.

Wall: ~50 min CPU at depth=3, max_leaves=15, n_round=1500 (matches
distill_small for direct comparison).
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
from recipe_full_te import load_and_engineer, TARGET, IDX2CLS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402
from soft_distill_common import (  # noqa: E402
    softmax, make_soft_xent_obj, make_val_metric, margin_to_prob,
)

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

XGB_DEPTH = int(os.environ.get("XGB_DEPTH", "3"))
XGB_MAX_LEAVES = int(os.environ.get("XGB_MAX_LEAVES", "15"))
XGB_NROUND = int(os.environ.get("XGB_NROUND", "1500"))

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_leakfree_teacher_targets(n_train: int, n_test: int,
                                    n_folds: int) -> tuple:
    """Load per-outer teacher arrays.

    Returns (per_fold_oof, test_avg) where per_fold_oof is a list of
    (n_train, 3) arrays — outer{f} has zero on V_f rows. Student uses
    per_fold_oof[f-1] as the teacher source for tr_idx of student-fold-f.
    """
    per_fold = []
    test_acc = np.zeros((n_test, 3), dtype=np.float32)
    n_test_avail = 0
    for f in range(1, n_folds + 1):
        oof_p = ART / f"oof_recipe_leakfree_outer{f}.npy"
        test_p = ART / f"test_recipe_leakfree_outer{f}.npy"
        if not oof_p.exists() or not test_p.exists():
            raise FileNotFoundError(
                f"Missing leakfree teacher for outer={f}. "
                "Run scripts/leakfree_teacher_oof.py first.")
        per_fold.append(np.load(oof_p))
        test_acc += np.load(test_p)
        n_test_avail += 1
    test_avg = test_acc / max(n_test_avail, 1)
    return per_fold, test_avg


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict,
           per_fold_teacher: list, teacher_test: np.ndarray) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

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
        # CRITICAL: use the leak-free teacher OOF whose outer-fold matches
        # the student's outer-fold. That teacher was built from inner-CV
        # restricted to (full_train \ V_f) — so neither the teacher nor
        # the student saw V_f at training.
        teacher_oof = per_fold_teacher[fold - 1]
        # Sanity: V_f rows must have zero teacher (they're not used as targets).
        assert (teacher_oof[va_idx].sum() < 1e-3), (
            f"teacher_oof has non-zero values on V_f rows for fold {fold}; "
            "leak-elimination invariant violated"
        )

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
            f"(N_tr={len(X_tr)}, N_va={len(X_va)}, leak-free teacher)")
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
        test_pred += margin_to_prob(
            booster.predict(dtest, output_margin=True)) / N_FOLDS
        fold_bal = fast_bal_acc(y_va_hard, oof[va_idx].argmax(1))
        fold_scores.append(fold_bal)
        log(f"  fold {fold} argmax bal_acc = {fold_bal:.5f}  "
            f"best_iter={booster.best_iteration}  wall={time.time()-t0:.1f}s")

    overall = fast_bal_acc(y.astype(np.int32), oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    log(f"config: depth={XGB_DEPTH} max_leaves={XGB_MAX_LEAVES} "
        f"nround={XGB_NROUND} smoke={SMOKE}")
    train, test, info, test_ids = load_and_engineer()

    log("loading per-outer leak-free teacher OOFs")
    per_fold_teacher, teacher_test = build_leakfree_teacher_targets(
        len(train), len(test), N_FOLDS,
    )
    for f, arr in enumerate(per_fold_teacher, 1):
        nz = (arr.sum(1) > 1e-3).sum()
        log(f"  outer{f}: nonzero rows = {nz:,}/{len(arr):,}")
    log(f"teacher_test mean entropy = "
        f"{-(teacher_test * np.log(np.clip(teacher_test, 1e-9, 1.0))).sum(1).mean():.5f}")

    result = run_cv(train, test, info, per_fold_teacher, teacher_test)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y.astype(np.int32), prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / "oof_leakfree_distill.npy"
    test_path = ART / "test_leakfree_distill.npy"
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
    sub_path = SUB / "submission_leakfree_distill.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}  "
        f"dist={dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, smoke=SMOKE,
        xgb_depth=XGB_DEPTH, xgb_max_leaves=XGB_MAX_LEAVES,
        xgb_nround=XGB_NROUND,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned, log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
    )
    res_path = ART / "leakfree_distill_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
