"""Refit the three XGB components with tuned HPs on full 5-fold outer CV.

Reads best HPs from scripts/artifacts/hp_{dist_routed,spec_678,nonrule}_best.json.
If `accepted: true` (inner-fold lift > 1 fold-std), uses tuned HPs; otherwise
falls back to baseline HPs. Saves OOFs + test preds with `_tuned` suffix.

Outputs:
  scripts/artifacts/oof_xgb_dist_routed_v3_tuned.npy
  scripts/artifacts/test_xgb_dist_routed_v3_tuned.npy
  scripts/artifacts/oof_xgb_spec_678_tuned.npy
  scripts/artifacts/test_xgb_spec_678_tuned.npy
  scripts/artifacts/oof_xgb_nonrule_tuned.npy
  scripts/artifacts/test_xgb_nonrule_tuned.npy
  scripts/artifacts/refit_best_hp_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from hp_common import (
    ART_DIR, CLS2IDX, SEED, TARGET,
    add_distance_features, get_xgb_fixed_kwargs, log,
    tune_log_bias,
)

N_FOLDS = 5
ROUTED_SCORES = (0, 1, 2)
SPEC_SCORES = (6, 7, 8)

RULE_COLS = {
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
}
DROP_COLS = {"id", TARGET}

BASELINE_HP = dict(
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
)


def load_hp(name: str, force_baseline: bool = False) -> tuple[dict, bool, dict]:
    path = ART_DIR / f"hp_{name}_best.json"
    if not path.exists() or force_baseline:
        log(f"[{name}] no tuned HPs (or forced baseline); using baseline")
        return BASELINE_HP.copy(), False, {}
    rec = json.loads(path.read_text())
    if rec.get("accepted"):
        hp = rec["best_params"]
        log(f"[{name}] tuned HPs ACCEPTED  "
            f"delta={rec['delta_vs_baseline']:+.5f}  params={hp}")
        return hp, True, rec
    log(f"[{name}] tuned delta {rec['delta_vs_baseline']:+.5f} "
        f"<= 1 fold-std; falling back to baseline")
    return BASELINE_HP.copy(), False, rec


def encode_categoricals(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")


def refit_dist_routed(tr: pd.DataFrame, te: pd.DataFrame, hp: dict) -> dict:
    """Mirror of xgb_dist_routed_v3.py but with configurable HPs."""
    log("[dist_routed] building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values
    tr_routed = np.isin(tr_scores, ROUTED_SCORES)
    te_routed = np.isin(te_scores, ROUTED_SCORES)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, "id")]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, "id"]]
    encode_categoricals(tr, te, cat_cols)
    feat_cols = num_cols + cat_cols

    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    rule_prob_low = np.array([1.0 - 2e-9, 1e-9, 1e-9], dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_pred_xgb = np.zeros((len(te), 3), dtype=np.float64)

    dte = xgb.DMatrix(X_test, enable_categorical=True)
    params = {**get_xgb_fixed_kwargs(), **hp}
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_f = tr_idx[~np.isin(tr_scores[tr_idx], ROUTED_SCORES)]
        va_f = va_idx[~np.isin(tr_scores[va_idx], ROUTED_SCORES)]
        dtr = xgb.DMatrix(X.iloc[tr_f], label=y[tr_f], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_f], label=y[va_f], enable_categorical=True)
        booster = xgb.train(params, dtr, num_boost_round=4000,
                            evals=[(dva, "val")],
                            early_stopping_rounds=100, verbose_eval=0)
        bi = booster.best_iteration
        best_iters.append(bi)

        dva_full = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        val_pred = booster.predict(dva_full, iteration_range=(0, bi + 1))
        va_mask = tr_routed[va_idx]
        oof[va_idx[~va_mask]] = val_pred[~va_mask]
        oof[va_idx[va_mask]] = rule_prob_low
        test_pred_xgb += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS

        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"[dist_routed] fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    test_pred = test_pred_xgb.copy()
    test_pred[te_routed] = rule_prob_low

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"[dist_routed] argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}  "
        f"bias={bias.round(4).tolist()}")

    np.save(ART_DIR / "oof_xgb_dist_routed_v3_tuned.npy", oof)
    np.save(ART_DIR / "test_xgb_dist_routed_v3_tuned.npy", test_pred)
    return {
        "argmax": float(argmax_bal),
        "tuned": float(tuned_bal),
        "bias": bias.tolist(),
        "best_iters": [int(x) for x in best_iters],
    }


def refit_spec_678(tr: pd.DataFrame, te: pd.DataFrame, hp: dict) -> dict:
    """Mirror of xgb_specialist_678.py but with configurable HPs."""
    log("[spec_678] building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values
    tr_spec = np.isin(tr_scores, SPEC_SCORES)
    te_spec = np.isin(te_scores, SPEC_SCORES)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, "id")]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, "id"]]
    encode_categoricals(tr, te, cat_cols)
    feat_cols = num_cols + cat_cols

    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_spec = np.zeros((len(tr), 3), dtype=np.float64)
    test_spec = np.zeros((len(te), 3), dtype=np.float64)

    dte_spec = xgb.DMatrix(X_test.iloc[te_spec], enable_categorical=True)
    params = {**get_xgb_fixed_kwargs(), **hp}
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_s = tr_idx[np.isin(tr_scores[tr_idx], SPEC_SCORES)]
        va_s = va_idx[np.isin(tr_scores[va_idx], SPEC_SCORES)]
        if len(tr_s) == 0 or len(va_s) == 0:
            continue
        dtr = xgb.DMatrix(X.iloc[tr_s], label=y[tr_s], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_s], label=y[va_s], enable_categorical=True)
        booster = xgb.train(params, dtr, num_boost_round=4000,
                            evals=[(dva, "val")],
                            early_stopping_rounds=100, verbose_eval=0)
        bi = booster.best_iteration
        best_iters.append(bi)
        val_pred = booster.predict(dva, iteration_range=(0, bi + 1))
        oof_spec[va_s] = val_pred

        spec_pos = np.where(te_spec)[0]
        test_spec_pred = booster.predict(dte_spec, iteration_range=(0, bi + 1))
        for i, pos in enumerate(spec_pos):
            test_spec[pos] += test_spec_pred[i] / N_FOLDS

        bal = balanced_accuracy_score(y[va_s], val_pred.argmax(axis=1))
        log(f"[spec_678] fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc_spec={bal:.5f}  ({time.time()-t0:.1f}s)")

    spec_y = y[tr_spec]
    spec_oof = oof_spec[tr_spec]
    spec_argmax_bal = balanced_accuracy_score(spec_y, spec_oof.argmax(axis=1))
    log(f"[spec_678] spec-domain argmax bal_acc = {spec_argmax_bal:.5f}")

    np.save(ART_DIR / "oof_xgb_spec_678_tuned.npy", oof_spec)
    np.save(ART_DIR / "test_xgb_spec_678_tuned.npy", test_spec)
    return {
        "spec_argmax_bal_acc": float(spec_argmax_bal),
        "best_iters": [int(x) for x in best_iters],
    }


def refit_nonrule(tr: pd.DataFrame, te: pd.DataFrame, hp: dict,
                  num_boost_round: int = 8000) -> dict:
    """Mirror of nonrule_features_only.py but with configurable HPs.

    Uses num_boost_round=8000 by default (vs 4000 for the other two)
    because the tuned nonrule HP config (lr=0.026, depth=4, heavy reg)
    hit the 4000-round cap during the HP search — the model was still
    improving when cut off.
    """
    log("[nonrule] building non-rule feature set")
    nonrule_cols = [c for c in tr.columns if c not in DROP_COLS and c not in RULE_COLS]

    X = tr[nonrule_cols].copy()
    X_test = te[nonrule_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te[c].map(mapping).astype("int32").astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_pred = np.zeros((len(te), 3), dtype=np.float64)

    dte = xgb.DMatrix(X_test, enable_categorical=True)
    params = {**get_xgb_fixed_kwargs(), **hp}
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        booster = xgb.train(params, dtr, num_boost_round=num_boost_round,
                            evals=[(dva, "val")],
                            early_stopping_rounds=100, verbose_eval=0)
        bi = booster.best_iteration
        best_iters.append(bi)
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS

        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"[nonrule] fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc={bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"[nonrule] argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}  "
        f"bias={bias.round(4).tolist()}")

    np.save(ART_DIR / "oof_xgb_nonrule_tuned.npy", oof)
    np.save(ART_DIR / "test_xgb_nonrule_tuned.npy", test_pred)
    return {
        "argmax": float(argmax_bal),
        "tuned": float(tuned_bal),
        "bias": bias.tolist(),
        "best_iters": [int(x) for x in best_iters],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=["dist_routed", "spec_678", "nonrule", "all"],
                    default="all")
    ap.add_argument("--force-baseline", action="store_true",
                    help="Ignore tuned HPs, use baseline (for sanity-check reproduction).")
    args = ap.parse_args()

    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    results: dict = {"components": {}}
    components = [args.component] if args.component != "all" else ["dist_routed", "spec_678", "nonrule"]

    for name in components:
        hp, accepted, rec = load_hp(name, force_baseline=args.force_baseline)
        results["components"][name] = {"hp": hp, "accepted": accepted, "hp_record": rec}

        t0 = time.time()
        if name == "dist_routed":
            metrics = refit_dist_routed(tr.copy(), te.copy(), hp)
        elif name == "spec_678":
            metrics = refit_spec_678(tr.copy(), te.copy(), hp)
        else:  # nonrule
            metrics = refit_nonrule(tr.copy(), te.copy(), hp)
        results["components"][name]["metrics"] = metrics
        results["components"][name]["wall_sec"] = float(time.time() - t0)
        log(f"[{name}] done in {time.time()-t0:.1f}s")

    with open(ART_DIR / "refit_best_hp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"saved {ART_DIR}/refit_best_hp_results.json")


if __name__ == "__main__":
    main()
