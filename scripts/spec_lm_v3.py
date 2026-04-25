"""Symmetric analog of spec_mh_v3: Low↔Medium boundary specialist.

Motivation (2026-04-24 error geometry on LB-best 3-stack):
  - dominant error bucket: score=3 Medium->Low at 4,324 rows (45% of total)
  - the stack defers to rule (Low at score=3) but 5% of score=3 rows are
    truly Medium. A targeted binary P(y=Medium | score=3) head can flag
    the specific rows to flip.

Break-even precision under macro-recall:
  P(truly-M | in override) ≥ M_count / (L_count + M_count)
                            = 239074 / (369917 + 239074)
                            = 0.393
  Higher bar than spec_mh (8.1%) but bucket is 13× bigger.

Parameters:
  SPEC_SCORE (default 3): which score band to specialise
  SMOKE (default 0): subsample for quick smoke test

Outputs: oof_spec_lm_v3_score{SPEC_SCORE}.npy (binary P(y=Medium))
         test_spec_lm_v3_score{SPEC_SCORE}.npy
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402

SEED = 42
N_FOLDS = 5
SPEC_SCORE = int(os.environ.get("SPEC_SCORE", "3"))
TARGET = "Irrigation_Need"
NONRULE_NUMS = [
    "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
    "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare",
]
SMOKE = os.environ.get("SMOKE", "0") == "1"
ART = Path("scripts/artifacts")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal_oof_test(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def build_new_teacher(y):
    """Reproduce the LB-best 3-stack OOF + test.  OOF leak-free because all
    five components were trained with fold-aligned OOFs (seed=42 5-fold)."""
    r = (_normed(np.load(ART / "oof_recipe_full_te.npy")),
         _normed(np.load(ART / "test_recipe_full_te.npy")))
    s1 = (_normed(np.load(ART / "oof_recipe_pseudolabel.npy")),
          _normed(np.load(ART / "test_recipe_pseudolabel.npy")))
    s7 = (_normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")),
          _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")))
    rm = (_normed(np.load(ART / "oof_realmlp.npy")),
          _normed(np.load(ART / "test_realmlp.npy")))
    nr = (_normed(np.load(ART / "oof_xgb_nonrule.npy")),
          _normed(np.load(ART / "test_xgb_nonrule.npy")))
    nr_iso_o, nr_iso_t = iso_cal_oof_test(nr[0], nr[1], y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r[0], s1[0], s7[0]], w3)
    lb3_t = log_blend([r[1], s1[1], s7[1]], w3)
    s1_o = log_blend([lb3_o, rm[0]], np.array([0.8, 0.2]))
    s1_t = log_blend([lb3_t, rm[1]], np.array([0.8, 0.2]))
    s2_o = log_blend([s1_o, nr_iso_o], np.array([0.925, 0.075]))
    s2_t = log_blend([s1_t, nr_iso_t], np.array([0.925, 0.075]))
    return s2_o, s2_t


def main():
    log(f"SPEC_SCORE={SPEC_SCORE} SMOKE={SMOKE}  (target: Medium-vs-Low)")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y_full = tr[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building NEW LB-best 3-stack teacher probs")
    oof_t, test_t = build_new_teacher(y_full)

    if SMOKE:
        idx = tr.sample(50_000, random_state=SEED).index.to_numpy()
        tr = tr.loc[idx].reset_index(drop=True)
        oof_t = oof_t[idx]
        y_full = y_full[idx]
        te = te.iloc[:20_000].copy().reset_index(drop=True)
        test_t = test_t[:20_000]

    tr = add_distance_features(tr)
    te = add_distance_features(te)
    s_tr = tr["dgp_score"].to_numpy()
    s_te = te["dgp_score"].to_numpy()
    tr_mask = s_tr == SPEC_SCORE
    te_mask = s_te == SPEC_SCORE

    n_tr = int(tr_mask.sum()); n_te = int(te_mask.sum())
    y_med = int((y_full[tr_mask] == CLS2IDX["Medium"]).sum())
    y_low = int((y_full[tr_mask] == CLS2IDX["Low"]).sum())
    y_high = int((y_full[tr_mask] == CLS2IDX["High"]).sum())
    log(f"train score={SPEC_SCORE}: {n_tr:,} rows  "
        f"L={y_low} ({y_low/max(n_tr,1):.1%})  "
        f"M={y_med} ({y_med/max(n_tr,1):.1%})  "
        f"H={y_high} ({y_high/max(n_tr,1):.1%})")
    log(f"test score={SPEC_SCORE}: {n_te:,}")

    # Binary target: P(y == Medium)
    y_bin = (y_full == CLS2IDX["Medium"]).astype(np.int32)

    # Teacher meta-features tailored to the L↔M boundary.
    tr["teacher_PL"] = oof_t[:, 0]
    tr["teacher_PM"] = oof_t[:, 1]
    tr["teacher_PH"] = oof_t[:, 2]
    tr["teacher_lm_margin"] = tr["teacher_PL"] - tr["teacher_PM"]
    tr["teacher_lm_ratio"] = (
        np.log(np.clip(tr["teacher_PM"], 1e-9, 1.0))
        - np.log(np.clip(tr["teacher_PL"], 1e-9, 1.0))
    )
    tr["teacher_top2_margin"] = np.sort(oof_t, axis=1)[:, -1] - np.sort(oof_t, axis=1)[:, -2]
    te["teacher_PL"] = test_t[:, 0]
    te["teacher_PM"] = test_t[:, 1]
    te["teacher_PH"] = test_t[:, 2]
    te["teacher_lm_margin"] = te["teacher_PL"] - te["teacher_PM"]
    te["teacher_lm_ratio"] = (
        np.log(np.clip(te["teacher_PM"], 1e-9, 1.0))
        - np.log(np.clip(te["teacher_PL"], 1e-9, 1.0))
    )
    te["teacher_top2_margin"] = np.sort(test_t, axis=1)[:, -1] - np.sort(test_t, axis=1)[:, -2]

    cat_cols = [c for c in tr.columns if tr[c].dtype == object and c != TARGET]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].astype(str).unique()))}
        tr[c] = tr[c].astype(str).map(mapping).astype("int32")
        te[c] = te[c].astype(str).map(mapping).fillna(-1).astype("int32")

    dist_feats = [
        "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "dry", "norain", "hot", "windy", "nomulch", "kc_active",
        "dgp_score", "rule_pred",
        "score_dist_low_mid", "score_dist_mid_high",
        "min_boundary_dist", "min_axis_abs",
        "sm_x_rf", "tc_x_ws", "sm_x_kc", "rf_x_kc",
    ]
    teacher_feats = [
        "teacher_PL", "teacher_PM", "teacher_PH",
        "teacher_lm_margin", "teacher_lm_ratio", "teacher_top2_margin",
    ]
    feat_cols = dist_feats + NONRULE_NUMS + cat_cols + teacher_feats
    feat_cols = [c for c in feat_cols if c in tr.columns]
    log(f"features: {len(feat_cols)}")

    X = tr[feat_cols].copy()
    X_te = te[feat_cols].copy()

    xgb_params = dict(
        objective="binary:logistic", eval_metric="auc",
        learning_rate=0.05, max_depth=6, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=1.0, reg_lambda=1.0,
        tree_method="hist", verbosity=0, seed=SEED,
    )
    max_rounds = 300 if SMOKE else 3000
    es_rounds = 30 if SMOKE else 150

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_ph = np.zeros(len(tr), dtype=np.float32)
    test_ph = np.zeros(len(te), dtype=np.float32)
    best_iters, fold_aucs = [], []
    dte = xgb.DMatrix(X_te[te_mask]) if te_mask.any() else None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_full)):
        t0 = time.time()
        tr_spec = tr_idx[tr_mask[tr_idx]]
        va_spec = va_idx[tr_mask[va_idx]]
        if len(tr_spec) == 0 or len(va_spec) == 0:
            log(f"  fold {fold+1}: empty spec domain")
            continue
        dtr = xgb.DMatrix(X.iloc[tr_spec], label=y_bin[tr_spec])
        dva = xgb.DMatrix(X.iloc[va_spec], label=y_bin[va_spec])
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=max_rounds,
            evals=[(dva, "val")], early_stopping_rounds=es_rounds,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        val_p = booster.predict(dva, iteration_range=(0, bi + 1))
        oof_ph[va_spec] = val_p.astype(np.float32)
        auc = roc_auc_score(y_bin[va_spec], val_p) \
            if y_bin[va_spec].sum() > 0 else float("nan")
        fold_aucs.append(auc)
        if dte is not None:
            te_p = booster.predict(dte, iteration_range=(0, bi + 1))
            test_ph[np.where(te_mask)[0]] += te_p.astype(np.float32) / N_FOLDS
        log(f"  fold {fold+1}/{N_FOLDS} n_tr={len(tr_spec):,} "
            f"n_va={len(va_spec):,} it={bi} auc={auc:.5f} "
            f"wall={time.time()-t0:.1f}s")

    if tr_mask.sum() > 0 and y_bin[tr_mask].sum() > 0:
        overall_auc = roc_auc_score(y_bin[tr_mask], oof_ph[tr_mask])
    else:
        overall_auc = float("nan")
    log(f"=== OOF AUC (L-M score={SPEC_SCORE}) = {overall_auc:.5f}")

    suffix = f"_score{SPEC_SCORE}"
    if SMOKE:
        suffix += "_smoke"
    np.save(ART / f"oof_spec_lm_v3{suffix}.npy", oof_ph)
    np.save(ART / f"test_spec_lm_v3{suffix}.npy", test_ph)
    with open(ART / f"spec_lm_v3{suffix}_results.json", "w") as f:
        json.dump({
            "spec_score": SPEC_SCORE, "seed": SEED, "n_folds": N_FOLDS,
            "smoke": SMOKE, "n_features": len(feat_cols),
            "train_rows_in_spec": n_tr,
            "truly_medium_in_spec": y_med,
            "truly_low_in_spec": y_low,
            "test_rows_in_spec": n_te,
            "best_iters": [int(b) for b in best_iters],
            "fold_aucs": [float(a) for a in fold_aucs],
            "overall_auc": float(overall_auc),
        }, f, indent=2)
    log(f"wrote oof_spec_lm_v3{suffix}.npy  auc={overall_auc:.5f}")


if __name__ == "__main__":
    main()
