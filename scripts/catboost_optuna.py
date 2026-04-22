"""CatBoost with native Ordered Target Statistics on raw categoricals.

Problem with prior benchmark_catboost_dist.py: ran 43 engineered features
(many pairwise products already ruled out on LGBM/XGB), with all 9 cats
passed via cat_features=. Landed at OOF tuned 0.97128, below LGBM-dist
(0.97266) and XGB-dist (0.97304).

This retry:
  1. Drops pairwise-product FE (ruled out as null on trees).
  2. Keeps minimal DGP features (dgp_score + 4 signed dists + rule_pred),
     which are the ones that actually lift LGBM by +0.00174.
  3. Passes 8 raw categorical columns via cat_features= so CatBoost's
     ordered Target Statistics does the encoding (the whole point of
     using CatBoost over LGBM/XGB).
  4. Runs Optuna on a 200k stratified subsample (same pattern as the
     earlier LGBM sweep; rankings shown to be stable 200k→630k).
  5. Refits best HPs on the full 630k and emits OOF + test probs.

Baseline refs (OOF tuned bal_acc):
  LGBM-dist              0.97266
  XGB-dist               0.97304
  CatBoost-dist (prior)  0.97128
  greedy 3-way blend     0.97375  (LB 0.97296)
  greedy + xgb-nonrule   0.97421  (LB 0.97352) <- current LB best

If CatBoost tuned > 0.972 and Jaccard vs existing OOFs is < 0.80,
it's a candidate 4th blend leg. Below 0.972 falsifies the "single
CatBoost clears 0.98" claim on our feature pipeline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
SUB.mkdir(exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_minimal_dgp_features(df: pd.DataFrame) -> pd.DataFrame:
    """DGP score + signed distances + rule_pred. No pairwise products."""
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values

    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)

    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)

    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    out["min_axis_abs"] = np.minimum.reduce([
        np.abs(out["sm_dist"].values), np.abs(out["rf_dist"].values),
        np.abs(out["tc_dist"].values), np.abs(out["ws_dist"].values),
    ]).astype(np.float32)
    return out


def tune_log_bias(p, y, prior):
    lp = np.log(np.clip(p, 1e-9, 1.0))
    b = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + b).argmax(axis=1))
    grid = np.linspace(-3, 3, 61)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = b.copy()
            sc = []
            for g in grid:
                base[k] = b[k] + g
                sc.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(sc))
            if sc[j] > best + 1e-6:
                b[k] = b[k] + grid[j]
                best = sc[j]
                imp = True
        if not imp:
            break
    return b, best


def load_features(tr: pd.DataFrame, te: pd.DataFrame):
    tr = add_minimal_dgp_features(tr)
    te = add_minimal_dgp_features(te)

    cat_cols = [
        "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
    ]
    drop_cols = {ID, TARGET}
    num_cols = [c for c in tr.columns if c not in drop_cols | set(cat_cols)]
    feat_cols = num_cols + cat_cols

    X = tr[feat_cols].copy()
    X_te = te[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
        X_te[c] = X_te[c].astype(str)

    return X, X_te, feat_cols, cat_cols


def cv_oof(X, y, cat_cols, params, folds, n_iter=2000, es=100):
    oof = np.zeros((len(X), 3), dtype=np.float64)
    best_iters = []
    fold_bals = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()
        model = CatBoostClassifier(
            iterations=n_iter,
            early_stopping_rounds=es,
            loss_function="MultiClass",
            random_seed=SEED,
            verbose=0,
            task_type="CPU",
            thread_count=-1,
            **params,
        )
        tr_pool = Pool(X.iloc[tr_idx], label=y[tr_idx], cat_features=cat_cols)
        va_pool = Pool(X.iloc[va_idx], label=y[va_idx], cat_features=cat_cols)
        model.fit(tr_pool, eval_set=va_pool, verbose=0)
        best_iters.append(int(model.tree_count_))
        oof[va_idx] = model.predict_proba(va_pool)
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        fold_bals.append(fold_bal)
        log(f"    fold {fold+1}/{len(folds)}  iter={model.tree_count_}  "
            f"bal={fold_bal:.5f}  ({time.time()-t0:.0f}s)")
    return oof, best_iters, fold_bals


def phase1_optuna(X, y, cat_cols, prior, n_trials=15, subsample=200_000):
    """Optuna on stratified subsample."""
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(X))
    # stratified subsample
    sub_idx = []
    for c in range(3):
        pool = idx[y == c]
        k = int(round(len(pool) * subsample / len(X)))
        sub_idx.append(rng.choice(pool, size=k, replace=False))
    sub_idx = np.sort(np.concatenate(sub_idx))
    Xs = X.iloc[sub_idx].reset_index(drop=True)
    ys = y[sub_idx]
    log(f"optuna phase: {len(sub_idx):,} rows ({len(sub_idx)/len(X)*100:.1f}%)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xs, ys))

    def objective(trial):
        params = dict(
            depth=trial.suggest_int("depth", 5, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.1, 3.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.5),
            border_count=trial.suggest_int("border_count", 64, 254),
            one_hot_max_size=trial.suggest_categorical("one_hot_max_size", [2, 8, 16]),
            leaf_estimation_iterations=trial.suggest_int("leaf_estimation_iterations", 1, 5),
        )
        log(f"  trial {trial.number}: {params}")
        oof, best_iters, fold_bals = cv_oof(
            Xs, ys, cat_cols, params, folds, n_iter=1500, es=75,
        )
        _, tuned = tune_log_bias(oof, ys, prior)
        log(f"  trial {trial.number}: tuned_bal={tuned:.5f} "
            f"(iters {best_iters}, fold_std={np.std(fold_bals):.5f})")
        return tuned

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log(f"best_trial={study.best_trial.number}  tuned={study.best_value:.5f}")
    log(f"best_params={study.best_params}")
    return study


def phase2_fit_full(X, y, cat_cols, prior, params, n_iter=3000):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))
    log(f"phase2: fitting best HPs on full {len(X):,} rows")
    oof, best_iters, fold_bals = cv_oof(
        X, y, cat_cols, params, folds, n_iter=n_iter, es=150,
    )
    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    return oof, best_iters, fold_bals, argmax_bal, tuned_bal, bias


def fit_test_probs(X, y, X_te, cat_cols, params, best_iters, n_iter_cap):
    """Average test probs over 5-fold refits at matched iterations."""
    test_probs = np.zeros((len(X_te), 3), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, _va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        model = CatBoostClassifier(
            iterations=min(best_iters[fold], n_iter_cap),
            loss_function="MultiClass",
            random_seed=SEED,
            verbose=0,
            task_type="CPU",
            thread_count=-1,
            **params,
        )
        tr_pool = Pool(X.iloc[tr_idx], label=y[tr_idx], cat_features=cat_cols)
        te_pool = Pool(X_te, cat_features=cat_cols)
        model.fit(tr_pool, verbose=0)
        test_probs += model.predict_proba(te_pool) / N_FOLDS
        log(f"  test refit fold {fold+1}: iter={model.tree_count_}  "
            f"({time.time()-t0:.0f}s)")
    return test_probs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--subsample", type=int, default=200_000)
    p.add_argument("--full-iter", type=int, default=3000)
    p.add_argument("--skip-optuna", action="store_true",
                   help="Skip phase 1, use --params json file instead")
    p.add_argument("--params", type=str, default="",
                   help="Path to JSON with params dict (for --skip-optuna)")
    args = p.parse_args()

    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    log(f"train={len(tr):,}  test={len(te):,}")

    X, X_te, feat_cols, cat_cols = load_features(tr, te)
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features: {len(feat_cols)} "
        f"({len(feat_cols) - len(cat_cols)} num + {len(cat_cols)} cat)")
    log(f"cat cols: {cat_cols}")
    log(f"prior: Low={prior[0]:.4f} Medium={prior[1]:.4f} High={prior[2]:.4f}")

    if args.skip_optuna:
        with open(args.params) as f:
            best_params = json.load(f)
        log(f"skipping optuna, using params={best_params}")
        study = None
    else:
        t0 = time.time()
        study = phase1_optuna(
            X, y, cat_cols, prior,
            n_trials=args.n_trials, subsample=args.subsample,
        )
        log(f"phase 1 done in {(time.time()-t0)/60:.1f} min")
        best_params = study.best_params

    oof, best_iters, fold_bals, argmax_bal, tuned_bal, bias = phase2_fit_full(
        X, y, cat_cols, prior, best_params, n_iter=args.full_iter,
    )
    log(f"full-fit OOF argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")
    log(f"fold bals: {[f'{b:.5f}' for b in fold_bals]}  "
        f"std={np.std(fold_bals):.5f}")
    log(f"bias: {bias.round(3).tolist()}")

    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    print("OOF confusion matrix (tuned):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
    per_class_recall = cm.diagonal() / cm.sum(axis=1)
    print(f"per-class recall: "
          f"Low={per_class_recall[0]:.5f} "
          f"Medium={per_class_recall[1]:.5f} "
          f"High={per_class_recall[2]:.5f}")

    log("refitting for test probs")
    test_probs = fit_test_probs(
        X, y, X_te, cat_cols, best_params, best_iters, args.full_iter,
    )

    np.save(ART / "oof_catboost_optuna.npy", oof)
    np.save(ART / "test_catboost_optuna.npy", test_probs)
    results = {
        "seed": SEED, "n_folds": N_FOLDS,
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "cat_cols": cat_cols,
        "best_params": best_params,
        "best_iters": best_iters,
        "argmax_bal": float(argmax_bal),
        "tuned_bal": float(tuned_bal),
        "fold_bals": [float(x) for x in fold_bals],
        "fold_std": float(np.std(fold_bals)),
        "log_bias": bias.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "delta_vs_lgbm_dist": float(tuned_bal - 0.97266),
        "delta_vs_xgb_dist": float(tuned_bal - 0.97304),
        "delta_vs_cat_prior": float(tuned_bal - 0.97128),
    }
    if study is not None:
        results["optuna_n_trials"] = len(study.trials)
        results["optuna_best_trial"] = study.best_trial.number
        results["optuna_trials"] = [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ]
    with open(ART / "catboost_optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)

    tuned_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({
        ID: te[ID],
        TARGET: [IDX2CLS[i] for i in tuned_idx],
    }).to_csv(SUB / "submission_catboost_optuna_tuned.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
