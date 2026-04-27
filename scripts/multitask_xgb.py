"""Multi-task XGB on recipe FE: shared trees jointly predict y + aux flips.

Design (full rationale in CLAUDE.md "do all 3" entry, 2026-04-26):
  Auxiliary heads were measured at OOF AUC 0.983 (missed_high), 0.949
  (missed_med), 0.899 (flipped) — strong signal. Inserted at the
  meta-stacker level (combined v6) it overfit OOF noise. This script
  inserts the same supervision at TRAINING TIME via a custom 6-output
  XGB objective, so trees split to minimize the joint loss directly.

Pipeline mirrors recipe_full_te exactly:
  - V10 recipe FE (443 cols)
  - 5-fold StratifiedKFold(seed=42) for OOF alignment
  - Per-fold OrderedTE
  - class-balanced sample_weight on the main task (aux gets uniform)

XGB config: native xgb.train API, num_class=6, custom obj/metric.
HPs match recipe_full_te (depth=4, max_leaves=30, reg=5/5, eta=0.1,
max_bin=1024, n_round=3000, esr=200) so any delta vs recipe is
attributable to the multi-task objective, not capacity.

SMOKE=1 → 20k train, 2 folds, 300 rounds (~5 min).
Production: 504k train, 5 folds, 3000 rounds (~50-70 min CPU).

Outputs:
  scripts/artifacts/oof_multitask_xgb.npy           (main 3-class softmax)
  scripts/artifacts/test_multitask_xgb.npy
  scripts/artifacts/multitask_xgb_results.json
  submissions/submission_multitask_xgb.csv
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

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, fast_bal_acc  # noqa: E402
from multitask_common import (  # noqa: E402
    build_aux_targets, make_multitask_obj, make_multitask_metric,
    margin_to_main_prob,
)
from recipe_full_te import load_and_engineer, TARGET, IDX2CLS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

# Aux head weights (relative to main task = 1.0). Defaults are conservative —
# strong enough to bias trees toward boundary rows but not so strong that
# they dominate the y signal. Override via env if needed.
AUX_W = float(os.environ.get("AUX_W", "0.3"))
SUFFIX = os.environ.get("MT_SUFFIX", "")
OUT_TAG = f"_{SUFFIX}" if SUFFIX else ""

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_rule_pred(df: pd.DataFrame) -> np.ndarray:
    """Deterministic DGP rule from threshold flags + Stage + Mulching."""
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage = df["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage, ("Flowering", "Vegetative")), 2, 0).astype(np.int8)
    score = (2 * (df["soil_lt_25"].values + df["rain_lt_300"].values)
             + df["temp_gt_30"].values + df["wind_gt_10"].values
             + nomulch + kc).astype(np.int8)
    return np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)


def run_cv(train: pd.DataFrame, test: pd.DataFrame, info: dict) -> dict:
    y = train[TARGET].to_numpy().astype(np.int64)
    rule = compute_rule_pred(train)
    aux_full = build_aux_targets(y, rule)
    log(f"aux target prevalence: flipped={aux_full[:,0].mean():.4f} "
        f"missed_H={aux_full[:,1].mean():.4f} missed_M={aux_full[:,2].mean():.4f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    base_params = dict(
        max_depth=4, max_leaves=30,
        eta=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        tree_method="hist",
        num_class=6,             # 3 main + 3 aux outputs
        verbosity=0,
        disable_default_eval_metric=1,
    )
    num_round = 300 if SMOKE else 3000
    esr = 50 if SMOKE else 200

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        # OrderedTE shuffled-fit, transform val/test.
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
        y_tr = y[tr_idx]
        aux_tr = aux_full[tr_idx]
        sw = compute_sample_weight("balanced", y_tr).astype(np.float32)

        dtrain = xgb.DMatrix(X_tr[feat_cols].to_numpy(dtype=np.float32),
                             label=y_tr.astype(np.float32))
        dval = xgb.DMatrix(X_va[feat_cols].to_numpy(dtype=np.float32),
                           label=y[va_idx].astype(np.float32))
        dtest = xgb.DMatrix(X_te[feat_cols].to_numpy(dtype=np.float32))

        obj = make_multitask_obj(
            y_main=y_tr, aux_targets=aux_tr,
            main_weight=1.0, aux_weights=(AUX_W, AUX_W, AUX_W),
            sample_weight=sw,
        )
        val_metric = make_multitask_metric(y[va_idx])

        log(f"  training XGB on {len(feat_cols)} features "
            f"(N_tr={len(X_tr):,}, N_va={len(X_va):,}, AUX_W={AUX_W})")
        t0 = time.time()
        booster = xgb.train(
            base_params, dtrain,
            num_boost_round=num_round,
            obj=obj, custom_metric=val_metric,
            evals=[(dval, "val")], maximize=False,
            early_stopping_rounds=esr,
            verbose_eval=200,
        )
        oof[va_idx] = margin_to_main_prob(
            booster.predict(dval, output_margin=True), len(X_va))
        test_pred += margin_to_main_prob(
            booster.predict(dtest, output_margin=True), len(X_te)) / N_FOLDS
        fold_bal = fast_bal_acc(y[va_idx].astype(np.int32), oof[va_idx].argmax(1))
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
    log(f"config: AUX_W={AUX_W}  SUFFIX={OUT_TAG!r}  smoke={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / f"oof_multitask_xgb{OUT_TAG}.npy"
    test_path = ART / f"test_multitask_xgb{OUT_TAG}.npy"
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
    sub_path = SUB / f"submission_multitask_xgb{OUT_TAG}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}  "
        f"dist={dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, smoke=SMOKE, aux_w=AUX_W,
        suffix=SUFFIX,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned, log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
    )
    res_path = ART / f"multitask_xgb{OUT_TAG}_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
