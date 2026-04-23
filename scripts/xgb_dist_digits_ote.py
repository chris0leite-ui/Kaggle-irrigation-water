"""XGBoost on (43 dist + 46 digit + ~48 OTE) features.

Adds Ordered Target Encoding (OTE) on top of the LB-best digit-XGB
pipeline (LB 0.97468 standalone). Hypothesis from the public-notebook
"digit + OTE" recipe: per-row K-shuffled cumulative target stats expose
within-category structure that fold-level TE flattens out, AND
non-rule-feature flips (the lever validated by xgb_nonrule at LB 0.97352)
are largely category-driven.

OTE keys mirror the prior `benchmark_te_oof.py` set so the deltas vs
that null are isolated to the per-row vs fold-level encoding choice.

Pipeline mirrors `xgb_dist_digits.py`:
  - 5-fold StratifiedKFold(seed=42), aligned with every other OOF.
  - Same XGB params as xgb_dist / xgb_dist_digits / xgb_nonrule.
  - Per-fold OTE: fit on tr_idx, apply to va_idx (no leak).
  - Test OTE: fit on full train, apply to test.
  - Saves oof_xgb_dist_digits_ote.npy + test_xgb_dist_digits_ote.npy.
  - Tuned log-bias for diagnostic standalone submission only.
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_dist import add_distance_features
from digit_features import add_digit_features, drop_zero_variance
from ote_features import OTE


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

RAW_NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
DIGITS = (-3, -2, -1, 0, 1, 2, 3)

# Variants controlled via env vars so all three runs share one code path:
#   OTE_VARIANT=default  ->  8 cats + 6 pairs + 2 rule keys, alpha=10, shuffles=8
#   OTE_VARIANT=light    ->  same keys, alpha=1, shuffles=2 (less regularisation)
#   OTE_VARIANT=digits   ->  keys = 46 surviving digit columns, alpha=10, shuffles=8
VARIANT = os.environ.get("OTE_VARIANT", "default")

# OTE keys: 8 single cats + 6 pairs + 2 rule-cell (mirror benchmark_te_oof + dgp_score / cell).
SINGLE_CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
PAIR_CATS = [
    ("Soil_Type", "Crop_Type"),
    ("Crop_Type", "Crop_Growth_Stage"),
    ("Season", "Region"),
    ("Soil_Type", "Season"),
    ("Crop_Type", "Season"),
    ("Crop_Type", "Irrigation_Type"),
]
# These two are added AFTER add_distance_features (which builds dgp_score / rule_pred).
RULE_KEYS = [
    ("dgp_score",),
    ("rule_pred", "Crop_Type"),
]

if VARIANT == "light":
    N_SHUFFLES = 2
    OTE_ALPHA = 1.0
else:
    N_SHUFFLES = 8
    OTE_ALPHA = 10.0

ART_SUFFIX = "" if VARIANT == "default" else f"_{VARIANT}"

ART = Path("scripts/artifacts")
OUT = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(
                    balanced_accuracy_score(y, (log_oof + base).argmax(axis=1))
                )
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def _all_key_specs(digit_cols: list[str] | None = None) -> list[list[str]]:
    """Return list of OTE key specs. For the 'digits' variant, override
    with per-digit-column single-key OTEs."""
    if VARIANT == "digits":
        if not digit_cols:
            raise ValueError("digits variant needs digit_cols passed in")
        return [[c] for c in digit_cols]
    return (
        [[c] for c in SINGLE_CATS]
        + [list(p) for p in PAIR_CATS]
        + [list(k) for k in RULE_KEYS]
    )


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building distance features (mirror benchmark_dist)")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    log(f"adding digit features on {len(RAW_NUMERIC)} numerics × {len(DIGITS)} digits")
    tr, new_digit_cols = add_digit_features(tr, RAW_NUMERIC, DIGITS)
    te, _ = add_digit_features(te, RAW_NUMERIC, DIGITS)
    alive_digits = drop_zero_variance(tr, te, new_digit_cols)
    log(f"  {len(new_digit_cols)} digit cols extracted, {len(alive_digits)} kept")

    # Cast SINGLE_CATS to string before OTE (factorize uses .astype(str) but we want dtype consistency).
    for c in set(SINGLE_CATS) | set([col for pair in PAIR_CATS for col in pair]):
        if c in tr.columns:
            tr[c] = tr[c].astype(str)
            te[c] = te[c].astype(str)

    drop_cols = {ID, TARGET}
    base_cols = [c for c in tr.columns if c not in drop_cols]
    num_cols = tr[base_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in base_cols if c not in num_cols]

    # Map cat strings to category codes for XGB enable_categorical.
    cat_mappings = {}
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        cat_mappings[c] = mapping
        tr[c + "__code"] = tr[c].map(mapping).astype("int32").astype("category")
        te[c + "__code"] = te[c].map(mapping).astype("int32").astype("category")

    cat_code_cols = [c + "__code" for c in cat_cols]
    feat_cols_no_ote = num_cols + cat_code_cols
    log(f"feature counts pre-OTE: total={len(feat_cols_no_ote)}  num={len(num_cols)}  cat={len(cat_code_cols)}")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    key_specs = _all_key_specs(digit_cols=alive_digits)
    log(f"OTE variant={VARIANT}  keys={len(key_specs)} × 3 classes = {3 * len(key_specs)} OTE cols")
    log(f"  shuffles={N_SHUFFLES}  alpha={OTE_ALPHA}  seed={SEED}")

    xgb_params = dict(
        objective="multi:softprob",
        num_class=len(CLASSES),
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

    log("training 5-fold XGB (dist + digits + OTE)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    # Build the test-OTE matrix ONCE outside folds, fit on FULL train.
    log("building test-side OTE on full train (one-shot)")
    t0 = time.time()
    test_ote_blocks = []
    test_ote_cols: list[str] = []
    for spec in key_specs:
        ote = OTE(spec, n_shuffles=N_SHUFFLES, alpha=OTE_ALPHA, seed=SEED)
        ote.fit_transform_train(tr, y)
        test_ote_blocks.append(ote.transform(te))
        test_ote_cols.extend(ote.feature_names())
    test_ote_block = np.hstack(test_ote_blocks)
    log(f"  test OTE built in {time.time()-t0:.1f}s, {test_ote_block.shape[1]} cols")

    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        # Build per-fold OTE: fit on tr_idx only; apply to va_idx.
        ote_blocks_tr = []
        ote_blocks_va = []
        ote_cols_fold: list[str] = []
        for spec in key_specs:
            ote = OTE(spec, n_shuffles=N_SHUFFLES, alpha=OTE_ALPHA, seed=SEED)
            block_tr = ote.fit_transform_train(tr.iloc[tr_idx], y[tr_idx])
            block_va = ote.transform(tr.iloc[va_idx])
            ote_blocks_tr.append(block_tr)
            ote_blocks_va.append(block_va)
            ote_cols_fold.extend(ote.feature_names())
        ote_tr = np.hstack(ote_blocks_tr)
        ote_va = np.hstack(ote_blocks_va)

        # Stack base feature matrix + OTE numerics. OTE cols added as a numeric DataFrame.
        ote_tr_df = pd.DataFrame(ote_tr, columns=ote_cols_fold, index=tr.index[tr_idx])
        ote_va_df = pd.DataFrame(ote_va, columns=ote_cols_fold, index=tr.index[va_idx])
        X_tr = pd.concat([tr.iloc[tr_idx][feat_cols_no_ote], ote_tr_df], axis=1)
        X_va = pd.concat([tr.iloc[va_idx][feat_cols_no_ote], ote_va_df], axis=1)

        dtr = xgb.DMatrix(X_tr, label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X_va, label=y[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters.append(bi)
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        # Test prediction with the test-OTE block (fit on full train).
        test_ote_df = pd.DataFrame(test_ote_block, columns=test_ote_cols, index=te.index)
        X_te = pd.concat([te[feat_cols_no_ote], test_ote_df], axis=1)
        dte = xgb.DMatrix(X_te, enable_categorical=True)
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS

        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)

    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"OOF confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== XGB-dist + digits + OTE (OOF bal_acc) ===")
    print(f"  argmax          : {argmax_bal:.5f}")
    print(f"  prior-reweight  : {reweight_bal:.5f}")
    print(f"  tuned log-bias  : {tuned_bal:.5f}")

    oof_path = ART / f"oof_xgb_dist_digits_ote{ART_SUFFIX}.npy"
    test_path = ART / f"test_xgb_dist_digits_ote{ART_SUFFIX}.npy"
    json_path = ART / f"xgb_dist_digits_ote{ART_SUFFIX}_results.json"
    sub_path = OUT / f"submission_xgb_dist_digits_ote{ART_SUFFIX}_tuned.csv"

    np.save(oof_path, oof)
    np.save(test_path, test_pred)
    with open(json_path, "w") as f:
        json.dump(
            {
                "variant": VARIANT,
                "seed": SEED,
                "n_folds": N_FOLDS,
                "n_features_total": len(feat_cols_no_ote) + test_ote_block.shape[1],
                "n_base_features": len(feat_cols_no_ote),
                "n_ote_cols": test_ote_block.shape[1],
                "n_ote_keys": len(key_specs),
                "ote_n_shuffles": N_SHUFFLES,
                "ote_alpha": OTE_ALPHA,
                "class_priors": prior.tolist(),
                "log_bias": bias.tolist(),
                "argmax_bal_acc": float(argmax_bal),
                "reweight_bal_acc": float(reweight_bal),
                "tuned_bal_acc": float(tuned_bal),
                "best_iters": best_iters,
            },
            f,
            indent=2,
        )

    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        sub_path, index=False
    )
    log(f"artefacts written: {oof_path}, {test_path}, {json_path}, {sub_path}")


if __name__ == "__main__":
    main()
