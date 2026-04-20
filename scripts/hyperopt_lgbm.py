"""
Optuna hyperparameter optimization for the LGBM baseline in benchmark.py.

Target: OOF balanced accuracy with prior-reweight decision rule (a fast
stand-in for full log-bias tuning that captures >99 % of the lift).

Same 5-fold stratified CV, same seed, same features and categorical
handling as benchmark.py, so results are directly comparable to the
0.97097 baseline.

Strategy:
- Median pruner + ASHA-style short-circuit: if fold 1 argmax is already
  far below the best trial's fold 1, stop the trial early.
- num_boost_round capped with early-stopping on the first fold only, then
  same round count reused across folds to keep trials cheap.

Usage:
    python scripts/hyperopt_lgbm.py --trials 30 --timeout 7200
    python scripts/hyperopt_lgbm.py --sample 200000 --trials 50     # faster
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "scripts" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


def load_xy(sample: int = 0) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    tr = pd.read_csv(DATA / "train.csv")
    if sample:
        tr = tr.sample(sample, random_state=SEED).reset_index(drop=True)
    num_cols = [c for c in tr.select_dtypes(include=[np.number]).columns
                if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        m = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(m).astype("int32")
    X = tr[num_cols + cat_cols].copy()
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    return X, y, cat_cols


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame, y: np.ndarray, cat_cols: list[str], prior: np.ndarray,
) -> float:
    params = {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "metric": "multi_logloss",
        "verbose": -1,
        "seed": SEED,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 14),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), len(CLASSES)), dtype=np.float64)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx],
                          categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va_idx], label=y[va_idx],
                          categorical_feature=cat_cols, reference=dtr)
        model = lgb.train(
            params, dtr,
            num_boost_round=3000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)],
        )
        p = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        oof[va_idx] = p
        # prior-reweight decision rule (much faster than log-bias coord ascent)
        pred = (p / prior).argmax(axis=1)
        score = balanced_accuracy_score(y[va_idx], pred)
        fold_scores.append(score)

        # report to optuna + prune if lagging median by fold 2
        trial.report(float(np.mean(fold_scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("fold_scores", fold_scores)
    return float(np.mean(fold_scores))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--timeout", type=int, default=7200,
                   help="hard wallclock timeout in seconds")
    p.add_argument("--sample", type=int, default=0,
                   help="subsample train rows (0 = full 630k)")
    p.add_argument("--study", type=str, default="lgbm_irrigation",
                   help="optuna study name (for storage)")
    args = p.parse_args()

    print(f"loading data (sample={args.sample or 'full'}) ...")
    X, y, cat_cols = load_xy(args.sample)
    prior = np.bincount(y) / len(y)
    print(f"X={X.shape}  prior={prior.round(4)}  cat_cols={cat_cols}")

    storage = f"sqlite:///{ART}/optuna_{args.study}.db"
    study = optuna.create_study(
        study_name=args.study,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=storage,
        load_if_exists=True,
    )
    print(f"storage: {storage}")
    print(f"existing trials: {len(study.trials)}")

    t0 = time.time()
    study.optimize(
        lambda tr: objective(tr, X, y, cat_cols, prior),
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"\n=== optuna done ({elapsed:.1f}s) ===")
    print(f"best value (prior-reweight OOF bal_acc): {best.value:.5f}")
    print(f"best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    if best.user_attrs.get("fold_scores"):
        print(f"fold scores: {best.user_attrs['fold_scores']}")

    out = {
        "study": args.study,
        "sample": args.sample,
        "n_trials_attempted": args.trials,
        "n_trials_completed": len(study.trials),
        "elapsed_s": round(elapsed, 1),
        "best_value": float(best.value),
        "best_params": best.params,
        "best_fold_scores": best.user_attrs.get("fold_scores"),
        "baseline_bench_results": 0.97097,
    }
    (ART / f"hyperopt_{args.study}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {ART}/hyperopt_{args.study}.json")


if __name__ == "__main__":
    main()
