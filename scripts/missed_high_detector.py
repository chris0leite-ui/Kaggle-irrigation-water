"""Missed-High detector: binary XGB trained to identify rows where
teacher_pred ∈ {L, M} but y = High.

Teacher = LB-best 3-way (recipe 0.25 + pseudo_s1 0.35 + pseudo_s7 0.40)
at fixed recipe bias [1.4324, 1.4689, 3.4008]. Teacher misses 475 of
21,009 High rows (recall 0.9774). Diagnostic shows 95% of misses live
in score ∈ {5, 6}.

Target = (y == High) AND (teacher_argmax != High).
Features = 43-dist + teacher probs/conf/argmax + recipe probs + nonrule probs.

5-fold StratifiedKFold(seed=42) on y (aligned with every OOF on disk).
scale_pos_weight = N_neg/N_pos ≈ 1300 handles the 0.075% prevalence.

Outputs: oof_missed_high.npy (630k,) P(missed-High | x) via OOF.
Gate:  OOF AUC ≥ 0.85 AND deploy sweep Δ bal_acc ≥ +0.00020 at some θ.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from common import add_distance_features
from meta_common import (
    ART, EPS, build_teacher, get_folds, load_y_and_features, recipe_bias,
)

OUT_OOF = ART / "oof_missed_high.npy"
OUT_TEST = ART / "test_missed_high.npy"
OUT_JSON = ART / "missed_high_results.json"


def build_feature_matrix(oof_teacher: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_oof, X_test, extra_train_dist_for_diagnostic)."""
    import pandas as pd
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    tr_d = add_distance_features(tr)
    te_d = add_distance_features(te)

    dist_cols = [
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "dry", "norain", "hot", "windy", "nomulch", "kc_active",
        "dgp_score", "rule_pred",
        "score_dist_low_mid", "score_dist_mid_high",
        "min_boundary_dist", "min_axis_abs",
        "sm_x_rf", "tc_x_ws", "sm_x_kc", "rf_x_kc",
    ]
    num_cols = [
        "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
        "Humidity", "Sunlight_Hours", "Soil_pH", "Organic_Carbon",
        "Electrical_Conductivity", "Field_Area_hectare", "Previous_Irrigation_mm",
    ]
    X_tr_base = np.concatenate(
        [tr_d[dist_cols].to_numpy(), tr_d[num_cols].to_numpy()], axis=1
    ).astype(np.float32)
    X_te_base = np.concatenate(
        [te_d[dist_cols].to_numpy(), te_d[num_cols].to_numpy()], axis=1
    ).astype(np.float32)

    # Teacher-derived features
    t_conf = oof_teacher.max(axis=1, keepdims=True).astype(np.float32)
    t_arg = oof_teacher.argmax(axis=1).reshape(-1, 1).astype(np.float32)
    _, test_teacher = build_teacher()
    tt_conf = test_teacher.max(axis=1, keepdims=True).astype(np.float32)
    tt_arg = test_teacher.argmax(axis=1).reshape(-1, 1).astype(np.float32)

    # Component probs for extra signal
    recipe_oof = np.load(ART / "oof_recipe_full_te.npy").astype(np.float32)
    recipe_te = np.load(ART / "test_recipe_full_te.npy").astype(np.float32)
    nr_oof = np.load(ART / "oof_xgb_nonrule.npy").astype(np.float32)
    nr_te = np.load(ART / "test_xgb_nonrule.npy").astype(np.float32)

    X_oof = np.concatenate(
        [X_tr_base, oof_teacher.astype(np.float32), t_conf, t_arg, recipe_oof, nr_oof],
        axis=1,
    ).astype(np.float32)
    X_test = np.concatenate(
        [X_te_base, test_teacher.astype(np.float32), tt_conf, tt_arg, recipe_te, nr_te],
        axis=1,
    ).astype(np.float32)
    return X_oof, X_test, tr_d["dgp_score"].to_numpy().astype(np.int16)


def run() -> dict:
    t0 = time.time()
    print("[mh] loading y + teacher ...")
    y, _, _, _, _ = load_y_and_features()
    oof_teacher, _ = build_teacher()
    bias = recipe_bias()
    t_pred = (np.log(np.clip(oof_teacher, EPS, 1.0)) + bias).argmax(1)

    target = ((y == 2) & (t_pred != 2)).astype(np.int32)
    n_pos = int(target.sum()); n_neg = int((1 - target).sum())
    print(f"[mh] target positives (missed-High) = {n_pos}  prevalence = {n_pos/(n_pos+n_neg):.5f}")
    print(f"[mh] scale_pos_weight = {n_neg / max(n_pos, 1):.1f}")

    print("[mh] building features ...")
    X_oof, X_test, tr_score = build_feature_matrix(oof_teacher)
    print(f"[mh] X_oof={X_oof.shape}  X_test={X_test.shape}")

    folds = get_folds(y)
    oof_p = np.zeros(len(y), dtype=np.float32)
    test_preds = []
    xgb_params = dict(
        objective="binary:logistic", tree_method="hist",
        max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, min_child_weight=10,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="auc", verbosity=0,
    )

    per_fold_auc = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        print(f"[mh] fold {i+1}/5 ...")
        dtr = xgb.DMatrix(X_oof[tr_idx], label=target[tr_idx])
        dva = xgb.DMatrix(X_oof[va_idx], label=target[va_idx])
        dte = xgb.DMatrix(X_test)
        bst = xgb.train(xgb_params, dtr, num_boost_round=2000,
                        evals=[(dva, "va")],
                        early_stopping_rounds=80, verbose_eval=False)
        best_it = bst.best_iteration + 1
        pv = bst.predict(dva, iteration_range=(0, best_it))
        pt = bst.predict(dte, iteration_range=(0, best_it))
        oof_p[va_idx] = pv
        test_preds.append(pt)
        auc = roc_auc_score(target[va_idx], pv)
        per_fold_auc.append(float(auc))
        print(f"[mh]   best_iter={best_it}  fold AUC={auc:.4f}  "
              f"n_val_pos={int(target[va_idx].sum())}")

    test_p = np.mean(test_preds, axis=0).astype(np.float32)
    overall_auc = roc_auc_score(target, oof_p)
    print(f"[mh] overall OOF AUC = {overall_auc:.4f}")

    # Per-score AUC
    per_score_auc = {}
    for s in range(10):
        mask = tr_score == s
        y_s = target[mask]
        if y_s.sum() < 5 or y_s.sum() == len(y_s):
            per_score_auc[f"score={s}"] = None
            continue
        auc_s = float(roc_auc_score(y_s, oof_p[mask]))
        per_score_auc[f"score={s}"] = auc_s

    np.save(OUT_OOF, oof_p)
    np.save(OUT_TEST, test_p)
    summary = dict(
        n_positives=n_pos,
        positive_prevalence=n_pos / (n_pos + n_neg),
        scale_pos_weight=n_neg / max(n_pos, 1),
        per_fold_auc=per_fold_auc,
        overall_auc=float(overall_auc),
        per_score_auc=per_score_auc,
        n_features=int(X_oof.shape[1]),
        wall_seconds=round(time.time() - t0, 1),
    )
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[mh] done in {summary['wall_seconds']}s   AUC={overall_auc:.4f}")
    return summary


if __name__ == "__main__":
    run()
