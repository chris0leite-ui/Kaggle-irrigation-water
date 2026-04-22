"""Optuna HP search for xgb_nonrule (13 non-rule features, 3-class).

Inner-split: 80/20 stratified on y. Subsample train to 200k for speed.
Objective: prior-reweight bal_acc on inner val.

Artefacts:
  scripts/artifacts/hp_nonrule_best.json
  scripts/artifacts/hp_nonrule_study.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

from hp_common import (
    ART_DIR, CLS2IDX, SEED, TARGET,
    get_xgb_fixed_kwargs, log,
    reweight_bal_acc, save_hp_result, suggest_xgb_params,
)

RULE_COLS = {
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
}
DROP_COLS = {"id", "Irrigation_Need"}
SUBSAMPLE_N = 200_000
N_TRIALS_DEFAULT = 40


def build_features(tr: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list, list]:
    nonrule_cols = [c for c in tr.columns if c not in DROP_COLS and c not in RULE_COLS]
    X = tr[nonrule_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    return X, y, num_cols, cat_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--timeout-sec", type=int, default=3600)
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE_N)
    args = ap.parse_args()

    log("loading train")
    tr = pd.read_csv("data/train.csv")
    X, y, num_cols, cat_cols = build_features(tr)
    prior = np.bincount(y) / len(y)
    log(f"full train: {len(X)}  features: {X.shape[1]} "
        f"({len(num_cols)} num + {len(cat_cols)} cat)")
    log(f"prior: {prior.round(4).tolist()}")

    # Inner 80/20 split.
    idx_all = np.arange(len(X))
    tr_idx, va_idx = train_test_split(
        idx_all, train_size=0.8, stratify=y, random_state=SEED
    )
    if args.subsample < len(tr_idx):
        sub_idx, _ = train_test_split(
            tr_idx, train_size=args.subsample, stratify=y[tr_idx], random_state=SEED
        )
        tr_idx = sub_idx
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]
    log(f"inner split: train={len(y_tr)}  val={len(y_va)}  "
        f"(subsampled train to {args.subsample if args.subsample < len(idx_all) else 'full'})")

    dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dva = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)

    baseline_hp = dict(
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1e-8,
        reg_lambda=1.0,
        gamma=1e-8,
    )
    log("benchmarking baseline HPs on inner val")
    t0 = time.time()
    booster_base = xgb.train(
        {**get_xgb_fixed_kwargs(), **baseline_hp},
        dtr, num_boost_round=4000,
        evals=[(dva, "val")], early_stopping_rounds=100,
        verbose_eval=0,
    )
    bi_base = booster_base.best_iteration
    probs_base = booster_base.predict(dva, iteration_range=(0, bi_base + 1))
    baseline_val = reweight_bal_acc(probs_base, y_va, prior)
    log(f"baseline reweight bal_acc = {baseline_val:.5f}  "
        f"(best_iter={bi_base}, {time.time()-t0:.1f}s)")

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_xgb_params(trial)
        pruning_cb = optuna.integration.XGBoostPruningCallback(
            trial, observation_key="val-mlogloss"
        )
        t0 = time.time()
        booster = xgb.train(
            {**get_xgb_fixed_kwargs(), **hp},
            dtr, num_boost_round=4000,
            evals=[(dva, "val")], early_stopping_rounds=100,
            callbacks=[pruning_cb], verbose_eval=0,
        )
        bi = booster.best_iteration
        probs = booster.predict(dva, iteration_range=(0, bi + 1))
        val = reweight_bal_acc(probs, y_va, prior)
        trial.set_user_attr("best_iter", int(bi))
        trial.set_user_attr("elapsed_s", float(time.time() - t0))
        return val

    sampler = TPESampler(seed=SEED, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=100)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="hp_nonrule",
    )
    study.optimize(
        objective, n_trials=args.n_trials, timeout=args.timeout_sec,
        show_progress_bar=False, gc_after_trial=True,
    )

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    log(f"optuna done: {len(done)} complete, {len(pruned)} pruned")
    log(f"best trial: #{study.best_trial.number}  value={study.best_value:.5f}  "
        f"params={study.best_params}")
    log(f"baseline value: {baseline_val:.5f}  "
        f"delta vs best: {study.best_value - baseline_val:+.5f}")

    save_hp_result(
        "nonrule",
        study.best_params,
        study.best_value,
        baseline_val,
        extra={
            "n_trials_complete": len(done),
            "n_trials_pruned": len(pruned),
            "best_trial_best_iter": study.best_trial.user_attrs.get("best_iter"),
            "inner_train_rows": len(y_tr),
            "inner_val_rows": len(y_va),
            "n_features": X.shape[1],
        },
    )
    with open(ART_DIR / "hp_nonrule_study.pkl", "wb") as f:
        pickle.dump(study, f)
    log(f"saved hp_nonrule_best.json + study.pkl to {ART_DIR}")


if __name__ == "__main__":
    main()
