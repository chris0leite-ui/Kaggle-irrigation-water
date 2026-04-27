"""Build a leak-eliminated teacher OOF for soft-distillation.

The standard teacher_oof[i] = recipe_oof[i] is leak-free for row i — that
specific recipe model didn't see row i. BUT when a STUDENT trains on
training rows in outer fold f (rows where y is provided), the soft labels
those rows receive come from recipe models that DID see other rows
in fold f (the held-out outer fold). When the student then predicts on
fold f at inference, it has implicitly fit a target shaped by fold f's
own data — that's the leak that drives the persistent +0.002 OOF→LB gap
across distill_d4 / distill_small / distill_tiny / recipeonly.

Proper fix: for each outer fold f, retrain a recipe with INNER n-fold CV
restricted to rows in (full_train \\ V_f). That gives a teacher_oof_f for
all rows EXCEPT V_f (V_f rows simply use the standard recipe_oof[i],
since by definition fold f's recipe model didn't see them either).

Outputs: 5 outer-leak-free OOF + test pairs.
  scripts/artifacts/oof_recipe_leakfree_outer{1..5}.npy   (630_000, 3)
  scripts/artifacts/test_recipe_leakfree_outer{1..5}.npy  (270_000, 3)

The test-side teacher target is the AVERAGE of test predictions from the
N_INNER inner models — comes "for free" from the inner CV (vs the prior
script's separate full-fit retrain which added 22 min/outer).

REHYDRATE-ROBUST: per-inner-fold checkpoint saves after each inner
training. A restart resumes from the next pending inner fold rather
than re-running the entire outer.

Wall budget: 5 outer × N_INNER × ~13 min. With N_INNER=3 ≈ 3.3h.
SMOKE=1: 2 outer × 2 inner × 20k rows ≈ 4 min.
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
N_OUTER = int(os.environ.get("N_OUTER", "5"))
N_INNER = int(os.environ.get("N_INNER", "3"))
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_OUTER = 2
    N_INNER = 2

ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)

# Per-inner-fold checkpoint paths. The 1d array of va_idx (within tr_outer)
# is also saved so we can reassemble inner_oof on resume without recomputing
# the StratifiedKFold split.
def _inner_ckpt_paths(outer: int, inner: int) -> tuple[Path, Path, Path]:
    return (
        ART / f"_lf_inner_outer{outer}_fold{inner}_oof.npy",
        ART / f"_lf_inner_outer{outer}_fold{inner}_test.npy",
        ART / f"_lf_inner_outer{outer}_fold{inner}_vaidx.npy",
    )


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_save(final_path: Path, arr: np.ndarray) -> None:
    """np.save to a sibling .tmp.npy file, then rename to final_path.

    np.save appends ".npy" if filename doesn't end in .npy/.npz, so the
    staging path uses a name ending in .npy to avoid double-extension.
    """
    tmp_path = final_path.with_name(final_path.stem + ".tmp.npy")
    np.save(tmp_path, arr)
    tmp_path.rename(final_path)


def _train_one_inner(X_tr_fold, y_tr_fold, X_va_fold, y_va_fold, X_te,
                     info: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit one recipe XGB on (X_tr_fold, y_tr_fold), return (val_proba,
    test_proba, best_iter). y_va_fold is used only for early-stopping
    metric; the produced val_p is leak-free regardless."""
    rng = np.random.default_rng(SEED + 13)
    perm = rng.permutation(len(X_tr_fold))
    X_tr_shuf = X_tr_fold.iloc[perm].reset_index(drop=True)
    te = OrderedTE(a=1.0)
    X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    X_tr_fold = X_tr_shuf.iloc[inv].reset_index(drop=True)
    X_va_fold = te.transform(X_va_fold)
    X_te = te.transform(X_te.copy().reset_index(drop=True))

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    feat_cols = numeric_feats + te.te_col_names()
    sw = compute_sample_weight("balanced", y_tr_fold)
    model = xgb.XGBClassifier(
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
    model.fit(
        X_tr_fold[feat_cols], y_tr_fold,
        sample_weight=sw,
        eval_set=[(X_va_fold[feat_cols], y_va_fold)],
        verbose=False,
    )
    val_p = model.predict_proba(X_va_fold[feat_cols]).astype(np.float32)
    test_p = model.predict_proba(X_te[feat_cols]).astype(np.float32)
    return val_p, test_p, int(model.best_iteration)


def _build_outer(outer: int, tr_outer: np.ndarray, va_outer: np.ndarray,
                 train: pd.DataFrame, test: pd.DataFrame, info: dict,
                 y: np.ndarray, n_inner: int) -> dict:
    """Run inner CV for outer fold `outer`, with per-inner checkpointing.

    Skips inner folds whose checkpoint files already exist.
    """
    X_inner = train.iloc[tr_outer].copy().reset_index(drop=True)
    y_inner = y[tr_outer]

    skf = StratifiedKFold(n_splits=n_inner, shuffle=True,
                          random_state=SEED)
    splits = list(skf.split(X_inner, y_inner))

    fold_scores = []
    best_iters = []
    for inner, (tr, va) in enumerate(splits, 1):
        oof_p, test_p, vaidx_p = _inner_ckpt_paths(outer, inner)
        if oof_p.exists() and test_p.exists() and vaidx_p.exists():
            saved_va = np.load(vaidx_p)
            assert np.array_equal(saved_va, va), (
                f"checkpoint va_idx mismatch for outer={outer} inner={inner}"
            )
            log(f"  outer{outer} inner{inner}: SKIPPED (checkpoint exists)")
            continue

        t0 = time.time()
        X_tr_fold = X_inner.iloc[tr].copy().reset_index(drop=True)
        X_va_fold = X_inner.iloc[va].copy().reset_index(drop=True)
        y_tr_fold = y_inner[tr]
        val_p, test_p_pred, bi = _train_one_inner(
            X_tr_fold, y_tr_fold, X_va_fold, y_inner[va], test, info,
        )
        bal = balanced_accuracy_score(y_inner[va], val_p.argmax(1))
        fold_scores.append(bal)
        best_iters.append(bi)
        # Atomic save: write tmp .npy then rename so a partial write
        # doesn't corrupt the resume signal. tmp paths still end in .npy
        # so np.save doesn't append a second extension.
        _atomic_save(oof_p, val_p)
        _atomic_save(test_p, test_p_pred)
        _atomic_save(vaidx_p, va.astype(np.int64))
        log(f"  outer{outer} inner{inner}: bal={bal:.5f}  "
            f"best_iter={bi}  wall={time.time()-t0:.1f}s")

    # Reassemble inner_oof + inner_test_avg from per-inner checkpoints.
    inner_oof = np.zeros((len(X_inner), 3), dtype=np.float32)
    inner_test_sum = np.zeros((len(test), 3), dtype=np.float32)
    n_loaded = 0
    for inner, (_, va) in enumerate(splits, 1):
        oof_p, test_p, vaidx_p = _inner_ckpt_paths(outer, inner)
        val_p = np.load(oof_p)
        inner_oof[va] = val_p
        inner_test_sum += np.load(test_p)
        n_loaded += 1
    inner_test_avg = inner_test_sum / max(n_loaded, 1)

    # Build outer{f} teacher OOF (zero on V_outer; tr_outer rows = inner OOF).
    outer_teacher = np.zeros((len(train), 3), dtype=np.float32)
    outer_teacher[tr_outer] = inner_oof
    final_oof = ART / f"oof_recipe_leakfree_outer{outer}.npy"
    final_test = ART / f"test_recipe_leakfree_outer{outer}.npy"
    _atomic_save(final_oof, outer_teacher)
    _atomic_save(final_test, inner_test_avg)

    # Cleanup per-inner checkpoints once the outer is consolidated.
    for inner in range(1, n_inner + 1):
        for p in _inner_ckpt_paths(outer, inner):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    log(f"  outer{outer} CONSOLIDATED: wrote {final_oof.name} + {final_test.name}")
    if fold_scores:
        return dict(outer=outer,
                    inner_fold_scores=[float(s) for s in fold_scores],
                    inner_best_iters=best_iters,
                    mean_inner_bal=float(np.mean(fold_scores)),
                    status="trained")
    return dict(outer=outer, status="resumed_from_checkpoints")


def main():
    log(f"Building leak-free teacher OOFs. N_OUTER={N_OUTER}, N_INNER={N_INNER}, SMOKE={SMOKE}")
    train, test, info, _ = load_and_engineer()
    y = train[TARGET].to_numpy()
    log(f"train.shape={train.shape}, test.shape={test.shape}, "
        f"y prior={np.bincount(y)/len(y)}")

    skf_outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                                random_state=SEED)
    summary_rows = []
    for outer, (tr_outer, va_outer) in enumerate(skf_outer.split(train, y), 1):
        oof_path = ART / f"oof_recipe_leakfree_outer{outer}.npy"
        test_path = ART / f"test_recipe_leakfree_outer{outer}.npy"
        if oof_path.exists() and test_path.exists():
            log(f"=== outer fold {outer}/{N_OUTER}  SKIPPED (final artifacts) ===")
            summary_rows.append(dict(outer=outer, status="skipped_final"))
            continue
        log(f"=== outer fold {outer}/{N_OUTER}  "
            f"|tr_outer|={len(tr_outer):,}  |va_outer|={len(va_outer):,} ===")
        t0 = time.time()
        info_row = _build_outer(outer, tr_outer, va_outer,
                                  train, test, info, y, N_INNER)
        info_row["outer_wall_min"] = (time.time() - t0) / 60.0
        summary_rows.append(info_row)
        log(f"  outer fold {outer}: total wall {info_row['outer_wall_min']:.1f}m")

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
