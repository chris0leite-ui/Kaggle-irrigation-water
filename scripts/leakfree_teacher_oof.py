"""Build a leak-eliminated teacher OOF for soft-distillation.

The standard teacher_oof[i] = recipe_oof[i] is leak-free for row i — that
specific recipe model didn't see row i. BUT when a STUDENT trains on
training rows in outer fold f (rows where y is provided), the soft labels
those rows receive come from recipe models that DID see other rows
in fold f (the held-out outer fold). When the student then predicts on
fold f at inference, it has implicitly fit a target shaped by fold f's
own data — that's the leak that drives the persistent +0.002 OOF→LB gap
across distill_d4 / distill_small / distill_tiny / recipeonly.

Proper fix: for each outer fold f, retrain a recipe with INNER 5-fold CV
restricted to rows in (full_train \\ V_f). That gives a teacher_oof_f for
all rows EXCEPT V_f (V_f rows simply use the standard recipe_oof[i],
since by definition fold f's recipe model didn't see them either).

This script produces 5 "outer-leak-free" teacher OOFs:
  scripts/artifacts/oof_recipe_leakfree_outer{1..5}.npy
  scripts/artifacts/test_recipe_leakfree_outer{1..5}.npy

Each outer{f}.npy has shape (630_000, 3); rows in V_f are zero-filled
(student doesn't need teacher targets for the rows it predicts on).
The student then uses outer{f}.npy as the target only for rows in tr_idx
of student's outer fold f.

Wall budget: 5 outer × 5 inner = 25 recipe trainings × ~10 min each on
CPU ≈ 4-5 hours. For SMOKE=1, runs 2 outer × 2 inner × 20k rows ≈ 4 min.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from recipe_full_te import load_and_engineer, TARGET  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
# N_INNER=3 default (down from 5) to fit ~2.5h wall budget with N_OUTER=5.
# The leak-free property holds for any n_inner ≥ 2; only teacher quality
# changes (larger inner folds → less per-fold variance, slightly cleaner
# soft labels). Override via env var if quality matters more than wall.
N_OUTER = int(os.environ.get("N_OUTER", "5"))
N_INNER = int(os.environ.get("N_INNER", "3"))
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_OUTER = 2
    N_INNER = 2

ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fit_inner_cv(X_inner: pd.DataFrame, y_inner: np.ndarray,
                  info: dict, n_inner: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Train INNER n-fold CV on X_inner, return per-row OOF probs.

    Mirrors recipe_full_te HPs and per-fold OrderedTE.
    """
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True,
                          random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    inner_oof = np.zeros((len(X_inner), 3), dtype=np.float32)
    fold_scores = []

    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss",
        enable_categorical=False, n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )

    for fold, (tr, va) in enumerate(skf.split(X_inner, y_inner), 1):
        X_tr_fold = X_inner.iloc[tr].copy().reset_index(drop=True)
        X_va_fold = X_inner.iloc[va].copy().reset_index(drop=True)
        y_tr_fold = y_inner[tr]

        rng = np.random.default_rng(SEED + fold * 7)
        perm = rng.permutation(len(X_tr_fold))
        X_tr_shuf = X_tr_fold.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr_fold = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va_fold = te.transform(X_va_fold)

        feat_cols = numeric_feats + te.te_col_names()
        sw = compute_sample_weight("balanced", y_tr_fold)
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr_fold[feat_cols], y_tr_fold,
            sample_weight=sw,
            eval_set=[(X_va_fold[feat_cols], y_inner[va])],
            verbose=False,
        )
        inner_oof[va] = model.predict_proba(X_va_fold[feat_cols]).astype(np.float32)
        bal = balanced_accuracy_score(y_inner[va], inner_oof[va].argmax(1))
        fold_scores.append(bal)
        log(f"    inner fold {fold}: argmax bal={bal:.5f}  best_iter={model.best_iteration}")
    return inner_oof, np.array(fold_scores), feat_cols


def _fit_full_for_test(X_inner: pd.DataFrame, y_inner: np.ndarray,
                       X_test: pd.DataFrame, info: dict) -> np.ndarray:
    """Fit ONE recipe on all X_inner, predict X_test (no CV).

    Used to produce the test-side teacher target. n_estimators capped
    by avg best_iter from inner CV would be more rigorous but for now
    we early-stop on a small held-out sample (5%).
    """
    rng = np.random.default_rng(SEED)
    holdout = rng.permutation(len(X_inner))[:max(1, len(X_inner) // 20)]
    train_mask = np.ones(len(X_inner), dtype=bool)
    train_mask[holdout] = False
    X_tr = X_inner.iloc[train_mask].reset_index(drop=True)
    X_va = X_inner.iloc[holdout].reset_index(drop=True)
    y_tr = y_inner[train_mask]
    y_va = y_inner[holdout]

    perm = rng.permutation(len(X_tr))
    X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
    te = OrderedTE(a=1.0)
    X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
    X_va = te.transform(X_va)
    X_te = te.transform(X_test.copy().reset_index(drop=True))

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    feat_cols = numeric_feats + te.te_col_names()
    sw = compute_sample_weight("balanced", y_tr)
    model = xgb.XGBClassifier(
        n_estimators=300 if SMOKE else 3000, max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )
    model.fit(X_tr[feat_cols], y_tr, sample_weight=sw,
              eval_set=[(X_va[feat_cols], y_va)], verbose=False)
    return model.predict_proba(X_te[feat_cols]).astype(np.float32)


def main():
    log(f"Building leak-free teacher OOFs. N_OUTER={N_OUTER}, N_INNER={N_INNER}, SMOKE={SMOKE}")
    train, test, info, _ = load_and_engineer()
    y = train[TARGET].to_numpy()
    log(f"train.shape={train.shape}, test.shape={test.shape}, y prior={np.bincount(y)/len(y)}")

    skf_outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                                random_state=SEED)
    summary_rows = []
    for outer, (tr_outer, va_outer) in enumerate(skf_outer.split(train, y), 1):
        log(f"=== outer fold {outer}/{N_OUTER}  "
            f"|tr_outer|={len(tr_outer):,}  |va_outer|={len(va_outer):,} ===")
        # Inner CV inside tr_outer ONLY → no leak from V_outer.
        X_inner = train.iloc[tr_outer].copy().reset_index(drop=True)
        y_inner = y[tr_outer]
        t0 = time.time()
        inner_oof, fold_scores, _ = _fit_inner_cv(
            X_inner, y_inner, info, N_INNER,
        )
        log(f"  inner CV done in {(time.time()-t0)/60:.1f}m  "
            f"mean fold bal={fold_scores.mean():.5f}±{fold_scores.std():.5f}")

        # Build the outer{f} teacher OOF: rows in tr_outer get inner_oof,
        # rows in va_outer get zero (student doesn't need them as targets;
        # va_outer rows go through the student as VAL of the outer fold).
        outer_teacher = np.zeros((len(train), 3), dtype=np.float32)
        outer_teacher[tr_outer] = inner_oof
        np.save(ART / f"oof_recipe_leakfree_outer{outer}.npy", outer_teacher)

        # Test-side teacher target for outer fold f. Train one recipe on
        # all of tr_outer (no CV), predict test. Average across all 5
        # outer folds when the student does its final test inference;
        # right now we just save the per-outer-fold test prediction.
        t0 = time.time()
        test_teacher = _fit_full_for_test(X_inner, y_inner, test, info)
        log(f"  full-fit + test predict in {(time.time()-t0)/60:.1f}m")
        np.save(ART / f"test_recipe_leakfree_outer{outer}.npy", test_teacher)

        summary_rows.append(dict(
            outer=outer,
            inner_fold_scores=[float(x) for x in fold_scores],
            mean_inner_bal=float(fold_scores.mean()),
            n_tr_outer=int(len(tr_outer)),
            n_va_outer=int(len(va_outer)),
        ))

    summary = dict(
        seed=SEED, n_outer=N_OUTER, n_inner=N_INNER, smoke=SMOKE,
        outer_summaries=summary_rows,
    )
    res_path = ART / "leakfree_teacher_oof_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
