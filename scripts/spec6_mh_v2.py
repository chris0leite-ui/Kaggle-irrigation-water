"""v2 specialist: spec6_mh + teacher OOF probs as meta-features.

v1 standalone AUC 0.862 on 38k score=6 rows but precision 9.8% at
theta=0.50 (break-even 8.8%), delivering only +0.00001 OOF.

v2 adds three features:
  P_L, P_M, P_H  from LB-best 3-way teacher (fold-aligned, leak-free)
  plus derived: teacher_conf = max(P_M, P_H), teacher_mh_margin = P_M - P_H.

Hypothesis: when teacher's P_M and P_H are close (margin near 0), the row
is on the boundary where the host NN flipped the label. The specialist
should use this directly rather than re-learning it from raw features.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402

SEED = 42
N_FOLDS = 5
SPEC_SCORE = 6
TARGET = "Irrigation_Need"
NONRULE_NUMS = [
    "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
    "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare",
]
W_RECIPE, W_S1, W_S7 = 0.25, 0.35, 0.40
SMOKE = os.environ.get("SMOKE", "0") == "1"

ART = Path("scripts/artifacts")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"loading data (SMOKE={SMOKE})")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    # Load LB-best 3-way teacher probs (OOF leak-free by construction)
    log("loading teacher OOF + test probs")
    oof_r = np.load(ART / "oof_recipe_full_te.npy")
    oof_s1 = np.load(ART / "oof_recipe_pseudolabel.npy")
    oof_s7 = np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")
    test_r = np.load(ART / "test_recipe_full_te.npy")
    test_s1 = np.load(ART / "test_recipe_pseudolabel.npy")
    test_s7 = np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")
    w = np.array([W_RECIPE, W_S1, W_S7])
    oof_t = log_blend([oof_r, oof_s1, oof_s7], w)   # (630k, 3)
    test_t = log_blend([test_r, test_s1, test_s7], w)

    if SMOKE:
        idx = tr.sample(50_000, random_state=SEED).index.to_numpy()
        tr = tr.loc[idx].reset_index(drop=True)
        oof_t = oof_t[idx]
        te = te.iloc[:20_000].copy().reset_index(drop=True)
        test_t = test_t[:20_000]

    tr = add_distance_features(tr)
    te = add_distance_features(te)
    s_tr = tr["dgp_score"].to_numpy()
    s_te = te["dgp_score"].to_numpy()
    tr_mask = s_tr == SPEC_SCORE
    te_mask = s_te == SPEC_SCORE
    log(f"train score={SPEC_SCORE}: {tr_mask.sum():,}  test: {te_mask.sum():,}")

    # Binary target
    y_full = tr[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    y_bin = (y_full == CLS2IDX["High"]).astype(np.int32)

    # Attach teacher probs + derived features
    tr["teacher_PL"] = oof_t[:, 0]
    tr["teacher_PM"] = oof_t[:, 1]
    tr["teacher_PH"] = oof_t[:, 2]
    tr["teacher_mh_margin"] = tr["teacher_PM"] - tr["teacher_PH"]
    tr["teacher_mh_ratio"] = np.log(np.clip(tr["teacher_PH"], 1e-9, 1.0)) - \
                              np.log(np.clip(tr["teacher_PM"], 1e-9, 1.0))
    te["teacher_PL"] = test_t[:, 0]
    te["teacher_PM"] = test_t[:, 1]
    te["teacher_PH"] = test_t[:, 2]
    te["teacher_mh_margin"] = te["teacher_PM"] - te["teacher_PH"]
    te["teacher_mh_ratio"] = np.log(np.clip(te["teacher_PH"], 1e-9, 1.0)) - \
                              np.log(np.clip(te["teacher_PM"], 1e-9, 1.0))

    cat_cols = [c for c in tr.columns if tr[c].dtype == object and c != TARGET]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).fillna(-1).astype("int32")

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
    teacher_feats = ["teacher_PL", "teacher_PM", "teacher_PH",
                     "teacher_mh_margin", "teacher_mh_ratio"]
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
    best_iters = []
    fold_aucs = []
    dte = xgb.DMatrix(X_te[te_mask]) if te_mask.any() else None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_full)):
        t0 = time.time()
        tr_spec = tr_idx[tr_mask[tr_idx]]
        va_spec = va_idx[tr_mask[va_idx]]
        if len(tr_spec) == 0 or len(va_spec) == 0:
            continue
        dtr = xgb.DMatrix(X.iloc[tr_spec], label=y_bin[tr_spec])
        dva = xgb.DMatrix(X.iloc[va_spec], label=y_bin[va_spec])
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=max_rounds,
            evals=[(dva, "val")], early_stopping_rounds=es_rounds,
            verbose_eval=0,
        )
        best_iter = booster.best_iteration
        best_iters.append(best_iter)
        val_p = booster.predict(dva, iteration_range=(0, best_iter + 1))
        oof_ph[va_spec] = val_p.astype(np.float32)
        auc = roc_auc_score(y_bin[va_spec], val_p)
        fold_aucs.append(auc)
        if dte is not None:
            te_p = booster.predict(dte, iteration_range=(0, best_iter + 1))
            test_ph[np.where(te_mask)[0]] += te_p.astype(np.float32) / N_FOLDS
        log(f"  fold {fold+1}/{N_FOLDS} n_tr={len(tr_spec):,} "
            f"n_va={len(va_spec):,} it={best_iter} auc={auc:.5f} "
            f"wall={time.time()-t0:.1f}s")

    overall_auc = roc_auc_score(y_bin[tr_mask], oof_ph[tr_mask])
    log(f"=== OOF AUC (v2) = {overall_auc:.5f}")

    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_spec6_mh_v2{suffix}.npy", oof_ph)
    np.save(ART / f"test_spec6_mh_v2{suffix}.npy", test_ph)
    top = np.sort(oof_ph[tr_mask])[::-1]
    pct = dict(p50=float(top[len(top)//2]),
               p90=float(top[len(top)//10]),
               p95=float(top[len(top)//20]),
               p99=float(top[len(top)//100]))
    with open(ART / f"spec6_mh_v2{suffix}_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS, "spec_score": SPEC_SCORE,
            "smoke": SMOKE, "n_features": len(feat_cols),
            "train_rows_in_spec": int(tr_mask.sum()),
            "test_rows_in_spec": int(te_mask.sum()),
            "best_iters": [int(b) for b in best_iters],
            "fold_aucs": [float(a) for a in fold_aucs],
            "overall_auc": float(overall_auc),
            "oof_p_high_percentiles": pct,
        }, f, indent=2)
    log(f"wrote oof_spec6_mh_v2{suffix}.npy  auc={overall_auc:.5f}")


if __name__ == "__main__":
    main()
