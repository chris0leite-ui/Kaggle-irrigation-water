"""Local CPU twin of kaggle_kernel/kernel_smote_recipe/recipe_smote.py.

Same per-fold raw-only SMOTE-NC + redrive architecture, no Kaggle deps,
runs against scripts/recipe_features.py + scripts/recipe_ote.py.

Env vars:
  MAX_FOLDS=N      cap fold loop (1 = quick mechanism check, ~30 min)
  SMOTE_TARGET=42000
  SMOTE_K=5
  TOTAL_KILL_SEC=3600
  SUFFIX=smote_v2  artifact filename suffix

Outputs (scripts/artifacts/):
  oof_{SUFFIX}.npy
  test_{SUFFIX}.npy
  {SUFFIX}_fold1_gate.json   (after fold 1 + gate decision)
  {SUFFIX}_results.json      (final summary)
  submissions/submission_{SUFFIX}_tuned.csv  (only if all folds completed)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa
from smote_local.cv_loop import run_cv  # noqa
from smote_local.load_engineer import (  # noqa
    load_and_engineer, TARGET, IDX2CLS, log,
)


SEED = 42
N_FOLDS = 5
MAX_FOLDS = int(os.environ.get("MAX_FOLDS", str(N_FOLDS)))
SMOTE_TARGET = int(os.environ.get("SMOTE_TARGET", "42000"))
SMOTE_K = int(os.environ.get("SMOTE_K", "5"))
TOTAL_KILL_SEC = int(os.environ.get("TOTAL_KILL_SEC", str(60 * 60)))
SUFFIX = os.environ.get("SUFFIX", "smote_v2")

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)

XGB_PARAMS = dict(
    n_estimators=3000,
    max_depth=4, max_leaves=30,
    learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=2, reg_alpha=5, reg_lambda=5,
    max_bin=1024,
    objective="multi:softprob", tree_method="hist",
    eval_metric="mlogloss",
    n_jobs=-1, random_state=SEED,
    early_stopping_rounds=200, verbosity=0,
)


def main():
    log(f"config: SMOTE_TARGET={SMOTE_TARGET} K={SMOTE_K} "
        f"MAX_FOLDS={MAX_FOLDS} suffix={SUFFIX!r}")
    train, test, raw_train, info, test_ids, maps = load_and_engineer()

    result = run_cv(
        train, test, raw_train, info, maps,
        n_folds=N_FOLDS, max_folds=MAX_FOLDS,
        smote_target=SMOTE_TARGET, smote_k=SMOTE_K,
        xgb_params=XGB_PARAMS,
        total_kill_sec=TOTAL_KILL_SEC,
        art_dir=str(ART), suffix=SUFFIX,
    )

    y = train[TARGET].to_numpy()
    folds = result["folds_completed"]
    np.save(ART / f"oof_{SUFFIX}.npy", result["oof"])
    np.save(ART / f"test_{SUFFIX}.npy", result["test"])

    tuned, bias = None, None
    if folds == N_FOLDS:
        prior = np.bincount(y, minlength=3) / len(y)
        bias, tuned = tune_log_bias(result["oof"], y.astype(np.int32), prior)
        log(f"tuned bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")
        eps = 1e-9
        test_log = np.log(np.clip(result["test"], eps, 1.0))
        pred_idx = (test_log + bias).argmax(1)
        sub = pd.DataFrame({"id": test_ids,
                            TARGET: [IDX2CLS[i] for i in pred_idx]})
        sub_path = SUB / f"submission_{SUFFIX}_tuned.csv"
        sub.to_csv(sub_path, index=False)
        log(f"wrote {sub_path}  dist={dict(sub[TARGET].value_counts())}")
    else:
        log(f"only {folds}/{N_FOLDS} folds — skipping tuned + sub")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, max_folds=MAX_FOLDS,
        smote_target=SMOTE_TARGET, smote_k=SMOTE_K, suffix=SUFFIX,
        folds_completed=folds,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=(
            float(balanced_accuracy_score(y, result["oof"].argmax(1)))
            if folds > 0 else None),
        tuned_log_bias_bal_acc=float(tuned) if tuned else None,
        log_bias=bias.tolist() if bias is not None else None,
        fold1_metrics=result["fold1_metrics"],
        gate_decision=result["gate_decision"],
    )
    (ART / f"{SUFFIX}_results.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote scripts/artifacts/{SUFFIX}_results.json")
    log(f"FINAL: gate={result['gate_decision']}  folds={folds}/{N_FOLDS}  "
        f"tuned={tuned}")


if __name__ == "__main__":
    main()
