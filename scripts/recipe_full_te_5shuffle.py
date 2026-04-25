"""N3 — recipe pipeline with 5-shuffle OTE concat training augmentation.

Same recipe pipeline as `recipe_full_te.py` (FE + cats + combos + digits +
num_as_cat + tres + logits + freq + orig_stats + OrderedTE) except the
per-fold OTE step uses K=5 shuffle-concat to 5x the training pool.

Each augmented training row carries the same raw features but a different
OTE realization (per-row noise from shuffle order). Sample weights and
target are replicated K times to match.

Validation + test prediction unchanged (single OTE.transform on full-train
stats).

Per-fold checkpointing: `oof_recipe_5shuffle_fold{f}.npy` and
`test_recipe_5shuffle_fold{f}.npy` written immediately after each fold so
the partial output survives a container rehydrate (per CLAUDE.md learning).

SMOKE=1 → 20k train, 1 fold, K=2 shuffle, fewer XGB iterations.
"""
from __future__ import annotations

import gc
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
from common import tune_log_bias  # noqa: E402
from recipe_full_te import load_and_engineer, log, ART, SUB  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402
from recipe_ote_5shuffle import fit_concat_5shuffle  # noqa: E402

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
IDX2CLS = {0: "Low", 1: "Medium", 2: "High"}

SMOKE = os.environ.get("SMOKE") == "1"
# Env-var knobs to fit per-fold execution into a 10-min foreground Bash
# window (container rehydrate kills detached jobs >15min).
#   N_SHUFFLE   default 5 production / 2 smoke. Set to 2 for fast variant.
#   RUN_FOLD    default "" (run all folds). Set to 1..5 to run a single
#               fold and exit; per-fold .npy saved.
#   MAX_ROUNDS  default 3000. Set lower (1500) for fast variant.
#   ES_ROUNDS   default 200. Set lower (100) for fast variant.
N_SHUFFLE = int(os.environ.get("N_SHUFFLE", "2" if SMOKE else "5"))
RUN_FOLD = os.environ.get("RUN_FOLD", "")
RUN_FOLD_INT = int(RUN_FOLD) if RUN_FOLD else None
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "300" if SMOKE else "3000"))
ES_ROUNDS = int(os.environ.get("ES_ROUNDS", "50" if SMOKE else "200"))
if SMOKE:
    N_FOLDS = 2

VARIANT_SUFFIX = "_5shuffle" if N_SHUFFLE >= 5 else f"_{N_SHUFFLE}shuffle"


def run_cv_5shuffle(train, test, info, a_ote: float = 1.0) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"]
                     + info.get("dae_embed", [])
                     + info.get("extra_domain", [])
                     + info.get("extra_decimal", [])
                     + info.get("gby", []))
    drop_after_te = info["te_cols"]

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    xgb_params = dict(
        n_estimators=MAX_ROUNDS,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss",
        enable_categorical=False, n_jobs=-1, random_state=SEED,
        early_stopping_rounds=ES_ROUNDS, verbosity=0,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        # Single-fold mode: skip all folds except the requested one.
        if RUN_FOLD_INT is not None and fold != RUN_FOLD_INT:
            continue
        log(f"=== fold {fold}/{N_FOLDS}  K={N_SHUFFLE}-shuffle concat ===")
        t_fold = time.time()
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log(f"  fitting OrderedTE x {N_SHUFFLE} shuffles + concat")
        t0 = time.time()
        # 5-shuffle concat on training only.
        X_tr_aug, te = fit_concat_5shuffle(
            X_tr, cat_cols=info["te_cols"], target=TARGET,
            a=a_ote, n_shuffle=N_SHUFFLE, seed=SEED + fold,
        )
        log(f"    fit done in {time.time()-t0:.1f}s, X_tr_aug rows="
            f"{len(X_tr_aug):,}")
        # Single transform on val + test using last fitted (consistent stats)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE total in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        # Augmented y/sw: replicate K times in the same order as X_tr_aug
        y_tr_orig = y[tr_idx]
        y_tr_aug = np.tile(y_tr_orig, N_SHUFFLE).astype(np.int32)
        sw_orig = compute_sample_weight("balanced", y_tr_orig)
        sw_aug = np.tile(sw_orig, N_SHUFFLE).astype(np.float32)

        log(f"  training XGB on {len(feat_cols)} features, "
            f"{len(X_tr_aug):,} rows (vs {len(tr_idx):,} unaug)")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr_aug[feat_cols], y_tr_aug,
            sample_weight=sw_aug,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500,
        )
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
            f"best_iter={model.best_iteration}  "
            f"wall={time.time()-t_fold:.1f}s")

        # Per-fold checkpoint (survives container rehydrate)
        np.save(ART / f"oof_recipe{VARIANT_SUFFIX}_fold{fold}.npy", oof)
        np.save(ART / f"test_recipe{VARIANT_SUFFIX}_fold{fold}.npy", test_pred)
        log(f"  checkpoint: oof+test fold {fold} saved")

        # Aggressive cleanup before next fold
        del X_tr, X_va, X_te, X_tr_aug, model, te, y_tr_aug, sw_aug
        gc.collect()

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols)


def main():
    log(f"N3 OTE-shuffle-concat — K={N_SHUFFLE}, N_FOLDS={N_FOLDS}, "
        f"RUN_FOLD={RUN_FOLD_INT}, MAX_ROUNDS={MAX_ROUNDS}, SMOKE={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    result = run_cv_5shuffle(train, test, info, a_ote=1.0)

    y = train[TARGET].to_numpy()

    # Single-fold mode: just save the per-fold OOF/test (already done in
    # run_cv_5shuffle's per-fold checkpoint). Skip log-bias / submission /
    # full-OOF aggregation. The aggregator script handles those after all
    # folds finish.
    if RUN_FOLD_INT is not None:
        log(f"single-fold mode (RUN_FOLD={RUN_FOLD_INT}) — exiting; "
            f"per-fold .npy already saved by run_cv_5shuffle")
        return

    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / f"oof_recipe{VARIANT_SUFFIX}.npy"
    test_path = ART / f"test_recipe{VARIANT_SUFFIX}.npy"
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
    sub_path = SUB / f"submission_recipe{VARIANT_SUFFIX}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, n_shuffle=N_SHUFFLE,
        variant_suffix=VARIANT_SUFFIX,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
    )
    res_path = ART / f"recipe{VARIANT_SUFFIX}_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
