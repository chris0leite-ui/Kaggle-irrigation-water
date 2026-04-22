"""Optuna HP search for xgb_specialist_678.

Small domain (56k rows, 45k per fold). No subsample needed.
Inner-split: 80/20 stratified on y restricted to score in {6,7,8}.
Objective: prior-reweight bal_acc on inner val (spec-domain priors).

Artefacts:
  scripts/artifacts/hp_spec_678_best.json
  scripts/artifacts/hp_spec_678_study.pkl
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
    add_distance_features, argmax_bal_acc, get_xgb_fixed_kwargs, log,
    save_hp_result, suggest_xgb_params,
)

SPEC_SCORES = (6, 7, 8)
N_TRIALS_DEFAULT = 40


def build_features(tr: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list, list]:
    tr = add_distance_features(tr)
    tr_scores = tr["dgp_score"].values

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, "id")]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, "id"]]

    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    return X, y, tr_scores, num_cols, cat_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--timeout-sec", type=int, default=2400)
    args = ap.parse_args()

    log("loading train")
    tr = pd.read_csv("data/train.csv")
    X, y, tr_scores, num_cols, cat_cols = build_features(tr)
    log(f"full train: {len(X)}  features: {X.shape[1]}")

    # Restrict to spec domain.
    spec_mask = np.isin(tr_scores, SPEC_SCORES)
    X_spec = X.iloc[spec_mask].reset_index(drop=True)
    y_spec = y[spec_mask]
    log(f"spec-domain rows: {len(X_spec)}  "
        f"class counts: {np.bincount(y_spec, minlength=3).tolist()}")

    spec_prior = np.bincount(y_spec, minlength=3) / len(y_spec)
    log(f"spec prior: {spec_prior.round(4).tolist()}")

    # Inner 80/20 split on spec domain.
    tr_idx, va_idx = train_test_split(
        np.arange(len(X_spec)), train_size=0.8,
        stratify=y_spec, random_state=SEED
    )
    X_tr, y_tr = X_spec.iloc[tr_idx], y_spec[tr_idx]
    X_va, y_va = X_spec.iloc[va_idx], y_spec[va_idx]
    log(f"inner split: train={len(y_tr)}  val={len(y_va)}")

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
    # Spec domain has zero Low rows, so use argmax bal_acc (2-class macro).
    baseline_val = argmax_bal_acc(probs_base, y_va)
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
        val = argmax_bal_acc(probs, y_va)
        trial.set_user_attr("best_iter", int(bi))
        trial.set_user_attr("elapsed_s", float(time.time() - t0))
        return val

    sampler = TPESampler(seed=SEED, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=100)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="hp_spec_678",
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
        "spec_678",
        study.best_params,
        study.best_value,
        baseline_val,
        extra={
            "n_trials_complete": len(done),
            "n_trials_pruned": len(pruned),
            "best_trial_best_iter": study.best_trial.user_attrs.get("best_iter"),
            "inner_train_rows": len(y_tr),
            "inner_val_rows": len(y_va),
            "spec_scores": list(SPEC_SCORES),
            "spec_prior": spec_prior.tolist(),
        },
    )
    with open(ART_DIR / "hp_spec_678_study.pkl", "wb") as f:
        pickle.dump(study, f)
    log(f"saved hp_spec_678_best.json + study.pkl to {ART_DIR}")


if __name__ == "__main__":
    main()
