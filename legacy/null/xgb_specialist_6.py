"""Dedicated XGB specialist on dgp_score in {6, 7, 8}.

These are the worst-performing rule scores (error rates 4.0%, 9.1%,
12.3%) and carry ~31% of all rule flips (~3,239 rule-errors in 56,122
rows). A model restricted to this subset has its gradient focused
entirely on the boundary-Medium/High band, rather than diluted by the
50%+ of clean rows outside this band.

Pipeline:
  - Same 43-feature dist set
  - 5-fold stratified split ON THE FULL 630k (so val folds match the
    main XGB's val folds — needed for later stacking)
  - Train fold-specific XGB ONLY on rows with score in {6, 7, 8}
  - Predict on the full val fold (so we have specialist probs for
    ALL rows, even out-of-domain ones — caller decides gating)
  - Save OOF + test probs so downstream blend/gate scripts can
    route scores {6} to these probs

Artefacts:
    scripts/artifacts/oof_xgb_spec_6.npy        (630k × 3)
    scripts/artifacts/test_xgb_spec_6.npy       (270k × 3)
    scripts/artifacts/xgb_spec_6_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
SPEC_SCORES = (6,)
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART_DIR = Path("scripts/artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
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
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)

    out["dry"] = dry
    out["norain"] = norain
    out["hot"] = hot
    out["windy"] = windy
    out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)

    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)

    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)

    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)

    return out


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values
    tr_spec_mask = np.isin(tr_scores, SPEC_SCORES)
    te_spec_mask = np.isin(te_scores, SPEC_SCORES)
    log(f"train rows in spec scores {SPEC_SCORES}: "
        f"{tr_spec_mask.sum()} / {len(tr)} ({tr_spec_mask.mean()*100:.2f}%)")
    log(f"test  rows in spec scores {SPEC_SCORES}: "
        f"{te_spec_mask.sum()} / {len(te)} ({te_spec_mask.mean()*100:.2f}%)")

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    spec_prior = np.bincount(y[tr_spec_mask], minlength=3) / tr_spec_mask.sum()
    log(f"overall class priors: {dict(zip(CLASSES, prior.round(4)))}")
    log(f"specialist-domain priors: {dict(zip(CLASSES, spec_prior.round(4)))}")
    log(f"features: {len(feat_cols)} ({len(num_cols)} numeric + {len(cat_cols)} categorical)")

    log("running 5-fold stratified XGB specialist on scores {6}")
    # stratify on overall y so fold assignments match main XGB's folds
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # specialist OOF: only populate rows in SPEC_SCORES; others left at 0
    oof_spec = np.zeros((len(tr), 3), dtype=np.float64)
    test_spec = np.zeros((len(te), 3), dtype=np.float64)

    xgb_params = dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=SEED,
    )

    dte_spec = xgb.DMatrix(X_test.iloc[te_spec_mask], enable_categorical=True)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_spec = tr_idx[np.isin(tr_scores[tr_idx], SPEC_SCORES)]
        va_spec = va_idx[np.isin(tr_scores[va_idx], SPEC_SCORES)]

        if len(tr_spec) == 0 or len(va_spec) == 0:
            log(f"  fold {fold+1}/{N_FOLDS}  empty spec subset; skipping")
            continue

        dtr = xgb.DMatrix(X.iloc[tr_spec], label=y[tr_spec], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_spec], label=y[va_spec], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        best_iter = booster.best_iteration
        best_iters.append(best_iter)

        # predict on val spec rows (in-domain only)
        val_pred = booster.predict(dva, iteration_range=(0, best_iter + 1))
        oof_spec[va_spec] = val_pred

        # predict on test spec rows; average across folds
        test_spec_pred = booster.predict(dte_spec, iteration_range=(0, best_iter + 1))
        # accumulate only at spec-mask positions
        spec_idx = np.where(te_spec_mask)[0]
        for i, pos in enumerate(spec_idx):
            test_spec[pos] += test_spec_pred[i] / N_FOLDS

        fold_bal = balanced_accuracy_score(y[va_spec], val_pred.argmax(axis=1))
        raw_acc = (val_pred.argmax(axis=1) == y[va_spec]).mean()
        log(f"  fold {fold+1}/{N_FOLDS}  n_tr={len(tr_spec)}  n_va={len(va_spec)}  "
            f"best_iter={best_iter}  "
            f"bal_acc(spec only)={fold_bal:.5f}  raw_acc={raw_acc:.5f}  "
            f"({time.time()-t0:.1f}s)")

    # evaluate specialist on its domain only
    spec_y = y[tr_spec_mask]
    spec_oof = oof_spec[tr_spec_mask]
    argmax_bal = balanced_accuracy_score(spec_y, spec_oof.argmax(axis=1))
    raw_acc = (spec_oof.argmax(axis=1) == spec_y).mean()
    reweight_bal = balanced_accuracy_score(spec_y, (spec_oof / spec_prior).argmax(axis=1))

    # rule-only baseline on same domain (predicts High for all spec rows)
    rule_pred_on_spec = np.full(len(spec_y), 2, dtype=np.int32)
    rule_raw_acc = (rule_pred_on_spec == spec_y).mean()
    rule_bal_acc = balanced_accuracy_score(spec_y, rule_pred_on_spec)

    cm = confusion_matrix(spec_y, spec_oof.argmax(axis=1), labels=[0, 1, 2])

    print("\n=== XGB specialist on scores {6} (evaluated on spec domain only) ===")
    print(f"  n rows in spec domain     : {len(spec_y)}")
    print(f"  class distribution        : "
          f"{dict(zip(CLASSES, np.bincount(spec_y, minlength=3).tolist()))}")
    print(f"  rule raw acc (all High)   : {rule_raw_acc:.5f}")
    print(f"  rule bal_acc (all High)   : {rule_bal_acc:.5f}")
    print(f"  specialist argmax raw_acc : {raw_acc:.5f}")
    print(f"  specialist argmax bal_acc : {argmax_bal:.5f}")
    print(f"  specialist reweight bal   : {reweight_bal:.5f}")
    print(f"  Δ spec bal_acc vs rule    : {argmax_bal - rule_bal_acc:+.5f}")
    print(f"  OOF confusion matrix (spec domain):\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART_DIR / "oof_xgb_spec_6.npy", oof_spec)
    np.save(ART_DIR / "test_xgb_spec_6.npy", test_spec)
    with open(ART_DIR / "xgb_spec_6_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "spec_scores": list(SPEC_SCORES),
            "train_rows_in_spec": int(tr_spec_mask.sum()),
            "test_rows_in_spec": int(te_spec_mask.sum()),
            "spec_prior": spec_prior.tolist(),
            "best_iters_per_fold": [int(x) for x in best_iters],
            "n_features": len(feat_cols),
            "rule_raw_acc_on_spec": float(rule_raw_acc),
            "rule_bal_acc_on_spec": float(rule_bal_acc),
            "specialist_argmax_raw_acc": float(raw_acc),
            "specialist_argmax_bal_acc": float(argmax_bal),
            "specialist_reweight_bal_acc": float(reweight_bal),
        }, f, indent=2)

    log(f"spec OOF + test probs saved to {ART_DIR}/")


if __name__ == "__main__":
    main()
