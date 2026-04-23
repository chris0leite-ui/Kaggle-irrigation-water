"""Pairwise OvO boundary specialists, NON-RULE-features-only variant.

First-pass OvO with the full 89-feature set produced Pearson 0.92-0.99
correlation with the main digit-XGB on in-domain rows -- a null by
architectural redundancy (the main model already internalises the
boundary sub-decisions). This variant forces the specialist onto a
strictly ORTHOGONAL feature view:

  13 non-rule features only:
    Soil_pH, Organic_Carbon, Electrical_Conductivity, Humidity,
    Sunlight_Hours, Field_Area_hectare, Previous_Irrigation_mm
    + Soil_Type, Crop_Type, Season, Irrigation_Type, Water_Source, Region

This is the same feature set that lifted OOF +0.00056 / LB +0.00056 as
an xgb_nonrule blend component. Hypothesis: on boundary-band rows where
the rule's 6 features (soil moisture, rainfall, temp, wind, mulching,
stage) are ambiguous BY CONSTRUCTION, the non-rule features carry the
NN-generator's flip signal that axis-aligned trees on rule features
cannot fully access.

Same training domain as the full-feature variant:
  Low-vs-Med spec: dgp_score in {2,3,4} AND y in {Low, Medium}
  Med-vs-High spec: dgp_score in {5,6,7} AND y in {Medium, High}

Same 5-fold StratifiedKFold(seed=42) for OOF alignment.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from dgp_formula import dgp_score as compute_dgp_score


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

# Strictly non-rule features (mirror xgb_nonrule).
NONRULE_NUMERIC = [
    "Soil_pH", "Organic_Carbon", "Electrical_Conductivity",
    "Humidity", "Sunlight_Hours",
    "Field_Area_hectare", "Previous_Irrigation_mm",
]
NONRULE_CAT = [
    "Soil_Type", "Crop_Type", "Season",
    "Irrigation_Type", "Water_Source", "Region",
]

LOWMED_SCORES = {2, 3, 4}
MEDHIGH_SCORES = {5, 6, 7}

ART = Path("scripts/artifacts")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_nonrule(tr: pd.DataFrame, te: pd.DataFrame):
    all_cols = NONRULE_NUMERIC + NONRULE_CAT
    X = tr[all_cols].copy()
    X_test = te[all_cols].copy()
    for c in NONRULE_CAT:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te[c].map(mapping).astype("int32").astype("category")
    return X, X_test


def train_ovo(X, X_test, y_full, dgp_tr, name, domain_scores, pos_idx, neg_idx, splits):
    log(f"\n=== non-rule spec: {name} (pos={CLASSES[pos_idx]}, neg={CLASSES[neg_idx]}) ===")
    in_domain_mask = (
        np.isin(dgp_tr, list(domain_scores))
        & ((y_full == pos_idx) | (y_full == neg_idx))
    )
    log(f"  domain rows={in_domain_mask.sum()} / {len(y_full)}")
    y_bin = (y_full == pos_idx).astype(np.int32)
    cc = np.bincount(y_bin[in_domain_mask])
    log(f"  domain split: {CLASSES[neg_idx]}={cc[0]}  {CLASSES[pos_idx]}={cc[1]}  "
        f"pos_rate={cc[1] / in_domain_mask.sum():.4f}")

    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
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
    oof = np.zeros(len(y_full), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)
    stats = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        t0 = time.time()
        tr_mask = in_domain_mask[tr_idx]
        va_mask = in_domain_mask[va_idx]
        tr_sub = tr_idx[tr_mask]
        va_sub = va_idx[va_mask]
        dtr = xgb.DMatrix(X.iloc[tr_sub], label=y_bin[tr_sub], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_sub], label=y_bin[va_sub], enable_categorical=True)
        booster = xgb.train(params, dtr, num_boost_round=4000,
                            evals=[(dva, "val")], early_stopping_rounds=100,
                            verbose_eval=0)
        bi = booster.best_iteration
        dva_all = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        oof[va_idx] = booster.predict(dva_all, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        try:
            auc = roc_auc_score(y_bin[va_sub],
                booster.predict(dva, iteration_range=(0, bi + 1)))
        except ValueError:
            auc = float("nan")
        stats.append({"fold": fold, "best_iter": bi, "auc": auc})
        log(f"  fold {fold+1}  best_iter={bi}  AUC={auc:.5f}  "
            f"({time.time()-t0:.1f}s)")

    try:
        full_auc = roc_auc_score(y_bin[in_domain_mask], oof[in_domain_mask])
    except ValueError:
        full_auc = float("nan")
    log(f"  OOF in-domain AUC = {full_auc:.5f}")
    return oof, test_pred, stats, full_auc


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y_full = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    X, X_test = build_nonrule(tr, te)
    log(f"non-rule features: {X.shape[1]} cols")
    dgp_tr = compute_dgp_score(tr).astype(int)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y_full))

    oof_lm, test_lm, stats_lm, auc_lm = train_ovo(
        X, X_test, y_full, dgp_tr,
        "LowMed_nonrule", LOWMED_SCORES, CLS2IDX["Low"], CLS2IDX["Medium"], splits,
    )
    oof_mh, test_mh, stats_mh, auc_mh = train_ovo(
        X, X_test, y_full, dgp_tr,
        "MedHigh_nonrule", MEDHIGH_SCORES, CLS2IDX["Medium"], CLS2IDX["High"], splits,
    )

    np.save(ART / "oof_xgb_ovo_lowmed_nonrule.npy", oof_lm)
    np.save(ART / "oof_xgb_ovo_medhigh_nonrule.npy", oof_mh)
    np.save(ART / "test_xgb_ovo_lowmed_nonrule.npy", test_lm)
    np.save(ART / "test_xgb_ovo_medhigh_nonrule.npy", test_mh)

    with open(ART / "ovo_boundary_nonrule_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "features": NONRULE_NUMERIC + NONRULE_CAT,
            "lowmed": {"auc": auc_lm, "folds": stats_lm},
            "medhigh": {"auc": auc_mh, "folds": stats_mh},
        }, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
