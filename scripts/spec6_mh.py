"""Binary Medium-vs-High specialist on dgp_score == 6 rows.

Score=6 band:
  - 38,416 rows total (rule predicts Medium for all)
  - ~33,253 true Medium (86.6%), ~4,163 true High (10.8%), 0 Low (rule never)
  - 70 % of all missed-High signal concentrates here (per 2026-04-24
    error-analysis entry). Teacher (LB-best 3-way) misses ~331 High rows
    on this band.

Pipeline:
  - 43-feature dist set + 7 non-rule continuous features
    (Humidity, Prev_Irrigation, EC, Soil_pH, Organic_Carbon, Sunlight,
     Field_Area) — EDA showed these have significant Cohen's d on flips.
  - Binary XGB P(y=High | features, score=6), trained ONLY on score=6
    train-fold rows. Val predictions on score=6 val-fold rows only.
  - 5-fold StratifiedKFold(seed=42) on full y for OOF alignment.

Output: P(High) scalar per row, shape (630k,) with zeros off-domain.
Deploy script sweeps theta and measures bal_acc delta vs teacher.
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
from common import add_distance_features  # noqa: E402

SEED = 42
N_FOLDS = 5
SPEC_SCORE = 6  # single-band specialist
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
NONRULE_NUMS = [
    "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
    "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare",
]
SMOKE = os.environ.get("SMOKE", "0") == "1"

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"loading data (SMOKE={SMOKE})")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    if SMOKE:
        tr = tr.sample(50_000, random_state=SEED).reset_index(drop=True)
        te = te.iloc[:20_000].copy().reset_index(drop=True)

    tr = add_distance_features(tr)
    te = add_distance_features(te)

    tr_score = tr["dgp_score"].to_numpy()
    te_score = te["dgp_score"].to_numpy()
    tr_mask = tr_score == SPEC_SCORE
    te_mask = te_score == SPEC_SCORE
    log(f"train score={SPEC_SCORE}: {tr_mask.sum():,} / {len(tr):,} "
        f"({100*tr_mask.mean():.2f}%)")
    log(f"test  score={SPEC_SCORE}: {te_mask.sum():,} / {len(te):,} "
        f"({100*te_mask.mean():.2f}%)")

    y_full = tr[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    y_bin = (y_full == CLS2IDX["High"]).astype(np.int32)  # 0=Med, 1=High
    dom_prior = y_bin[tr_mask].mean()
    log(f"score=6 High prevalence (train): {dom_prior*100:.3f}%")

    # Factorize cats (Mulching_Used / Crop_Growth_Stage are already encoded
    # via dist_features; the raw cat columns still live in tr/te).
    cat_cols = [c for c in tr.columns
                if tr[c].dtype == object and c != TARGET]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        # Unseen test values get -1
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
    feat_cols = dist_feats + NONRULE_NUMS + cat_cols
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
            log(f"  fold {fold+1}: empty spec subset, skipping")
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
            spec_te_idx = np.where(te_mask)[0]
            test_ph[spec_te_idx] += te_p.astype(np.float32) / N_FOLDS
        log(f"  fold {fold+1}/{N_FOLDS}  n_tr={len(tr_spec):,} "
            f"n_va={len(va_spec):,}  best_iter={best_iter}  "
            f"auc={auc:.5f}  wall={time.time()-t0:.1f}s")

    overall_auc = roc_auc_score(y_bin[tr_mask], oof_ph[tr_mask])
    log(f"=== OOF AUC on score={SPEC_SCORE}: {overall_auc:.5f}  "
        f"(mean fold {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f})")

    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_spec6_mh{suffix}.npy", oof_ph)
    np.save(ART / f"test_spec6_mh{suffix}.npy", test_ph)

    # Save metadata + rough diagnostics (deploy script does full sweep).
    n_high_dom = int(y_bin[tr_mask].sum())
    n_med_dom = int(tr_mask.sum() - n_high_dom)
    top_p = np.sort(oof_ph[tr_mask])[::-1]
    pct_p = dict(p50=float(top_p[len(top_p)//2]),
                 p90=float(top_p[len(top_p)//10]),
                 p95=float(top_p[len(top_p)//20]),
                 p99=float(top_p[len(top_p)//100]))

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, spec_score=SPEC_SCORE,
        smoke=SMOKE,
        train_rows_in_spec=int(tr_mask.sum()),
        test_rows_in_spec=int(te_mask.sum()),
        dom_high_prevalence=float(dom_prior),
        n_high_domain=n_high_dom, n_med_domain=n_med_dom,
        n_features=len(feat_cols),
        best_iters=[int(b) for b in best_iters],
        fold_aucs=[float(a) for a in fold_aucs],
        overall_auc=float(overall_auc),
        oof_p_high_percentiles=pct_p,
    )
    with open(ART / f"spec6_mh{suffix}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote oof_spec6_mh{suffix}.npy + test + json  summary={summary}")


if __name__ == "__main__":
    main()
