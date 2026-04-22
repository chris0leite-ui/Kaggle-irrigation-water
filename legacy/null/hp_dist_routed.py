"""Optuna HP search for xgb_dist_routed_v3.

Inner-split: 80/20 stratified on y. Subsample train to 200k for speed
(HP rankings stable at this size per 2026-04-20 LGBM Optuna sweep).
Objective: prior-reweight bal_acc on inner val (measured on FULL val,
not filtered) — matches the routed-eval metric.

Training: drops dgp_score in {0,1,2} from both train and val, mirrors
production. Predict on full val, route routed rows to rule, then
compute reweight bal_acc on full val.

Artefacts:
  scripts/artifacts/hp_dist_routed_best.json
  scripts/artifacts/hp_dist_routed_study.pkl  (Optuna study for later analysis)
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
    add_distance_features, get_xgb_fixed_kwargs, log,
    reweight_bal_acc, save_hp_result, stratified_subsample,
    suggest_xgb_params,
)

ROUTED_SCORES = (0, 1, 2)
SUBSAMPLE_N = 200_000
N_TRIALS_DEFAULT = 40


def build_features(tr: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list, list]:
    tr = add_distance_features(tr)
    tr_scores = tr["dgp_score"].values

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, "id")]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, "id"]]

    # Encode categoricals. Since we only need training data here, we don't
    # need a separate test encoding — just fit mapping on train.
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
    ap.add_argument("--timeout-sec", type=int, default=3600)
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE_N)
    args = ap.parse_args()

    log("loading train")
    tr = pd.read_csv("data/train.csv")
    X, y, tr_scores, num_cols, cat_cols = build_features(tr)
    prior = np.bincount(y) / len(y)
    log(f"train rows: {len(X)}  features: {X.shape[1]} "
        f"({len(num_cols)} num + {len(cat_cols)} cat)")
    log(f"prior: {prior.round(4).tolist()}")

    # Inner 80/20 split on FULL 630k (stratified on y). The "routed-drop"
    # is applied to the training half only; inner val is full (so we can
    # measure reweight bal_acc on routed OOF).
    idx_all = np.arange(len(X))
    tr_idx, va_idx = train_test_split(
        idx_all, train_size=0.8, stratify=y, random_state=SEED
    )

    # Apply subsampling on the training half only.
    if args.subsample < len(tr_idx):
        # stratified subsample of training idx
        sub_idx, _ = train_test_split(
            tr_idx, train_size=args.subsample, stratify=y[tr_idx], random_state=SEED
        )
        tr_idx = sub_idx
    log(f"inner split: train={len(tr_idx)}  val={len(va_idx)}  "
        f"(subsampled train to {args.subsample if args.subsample < len(idx_all) else 'full'})")

    # Drop routed scores from TRAIN (this is what production does).
    tr_scores_tr = tr_scores[tr_idx]
    tr_keep_mask = ~np.isin(tr_scores_tr, ROUTED_SCORES)
    tr_idx_kept = tr_idx[tr_keep_mask]
    n_dropped = len(tr_idx) - len(tr_idx_kept)
    log(f"dropped from train: {n_dropped} rows "
        f"(scores in {ROUTED_SCORES})")

    # Build DMatrices once — HP search only changes training params.
    # We keep val as full so we can evaluate full reweight bal_acc.
    # routed rows in val get predicted-to-Low at eval time.
    X_tr = X.iloc[tr_idx_kept]
    y_tr = y[tr_idx_kept]
    X_va = X.iloc[va_idx]
    y_va = y[va_idx]
    va_routed_mask = np.isin(tr_scores[va_idx], ROUTED_SCORES)
    log(f"inner val: {len(y_va)} rows ({va_routed_mask.sum()} routed to rule)")

    dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    # For early-stopping eval we need a matrix of NON-routed val rows only
    # (routed rows go to rule, model doesn't see them at training time).
    va_keep_mask = ~np.isin(tr_scores[va_idx], ROUTED_SCORES)
    dva_kept = xgb.DMatrix(
        X_va.iloc[va_keep_mask], label=y_va[va_keep_mask],
        enable_categorical=True,
    )
    dva_full = xgb.DMatrix(X_va, enable_categorical=True)

    # Rule prediction for routed rows (soft-clipped onehot).
    rule_prob_low = np.array([1.0 - 2e-9, 1e-9, 1e-9], dtype=np.float64)

    # Baseline run with the existing production params (for acceptance gate).
    baseline_hp = dict(
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1e-8,
        reg_lambda=1.0,  # XGB default
        gamma=1e-8,
    )
    log("benchmarking baseline HPs on inner val")
    t0 = time.time()
    booster_base = xgb.train(
        {**get_xgb_fixed_kwargs(), **baseline_hp},
        dtr, num_boost_round=4000,
        evals=[(dva_kept, "val")],
        early_stopping_rounds=100,
        verbose_eval=0,
    )
    bi_base = booster_base.best_iteration
    probs_base_full = booster_base.predict(dva_full, iteration_range=(0, bi_base + 1))
    # Route
    probs_base_full_routed = probs_base_full.copy()
    probs_base_full_routed[va_routed_mask] = rule_prob_low
    baseline_val = reweight_bal_acc(probs_base_full_routed, y_va, prior)
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
            evals=[(dva_kept, "val")],
            early_stopping_rounds=100,
            callbacks=[pruning_cb],
            verbose_eval=0,
        )
        bi = booster.best_iteration
        probs_full = booster.predict(dva_full, iteration_range=(0, bi + 1))
        probs_full_routed = probs_full.copy()
        probs_full_routed[va_routed_mask] = rule_prob_low
        val = reweight_bal_acc(probs_full_routed, y_va, prior)
        trial.set_user_attr("best_iter", int(bi))
        trial.set_user_attr("elapsed_s", float(time.time() - t0))
        return val

    sampler = TPESampler(seed=SEED, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=100)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="hp_dist_routed",
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout_sec,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    log(f"optuna done: {len(done)} complete, {len(pruned)} pruned, "
        f"{len(study.trials)} total")
    log(f"best trial: #{study.best_trial.number}  "
        f"value={study.best_value:.5f}  params={study.best_params}")
    log(f"baseline value: {baseline_val:.5f}  "
        f"delta vs best: {study.best_value - baseline_val:+.5f}")

    save_hp_result(
        "dist_routed",
        study.best_params,
        study.best_value,
        baseline_val,
        extra={
            "n_trials_complete": len(done),
            "n_trials_pruned": len(pruned),
            "best_trial_best_iter": study.best_trial.user_attrs.get("best_iter"),
            "inner_train_rows": len(y_tr),
            "inner_val_rows": len(y_va),
            "routed_scores": list(ROUTED_SCORES),
            "val_routed_rows": int(va_routed_mask.sum()),
        },
    )
    with open(ART_DIR / "hp_dist_routed_study.pkl", "wb") as f:
        pickle.dump(study, f)
    log(f"saved hp_dist_routed_best.json + study.pkl to {ART_DIR}")


if __name__ == "__main__":
    main()
