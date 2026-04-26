"""#6 self-supervised masked-feature pretraining.

For each top non-rule numeric, predict its value from the other 18 features
using 5-fold cross-fit on combined train+test (label-free, self-supervised).
Save (predicted - actual) and |predicted - actual| as new feature channels.

Hypothesis: rows that are 'feature-anomalous' relative to the joint
distribution may be exactly the rare-class boundary rows the recipe FE
doesn't currently distinguish. The prior DAE SwapNoise was reconstruction;
this is masked PREDICTION — different objective, different inductive bias.

Output: scripts/artifacts/{oof,test}_masked_resid.npy with shape
        (N, 2 * len(MASKED_FEATS)) for {residual, abs_residual}.

Cost: ~12-15 min CPU on 16-core (7 feats × 5 folds × ~25s/fold).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features  # noqa: E402

ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
SMOKE = __import__("os").environ.get("SMOKE", "0") == "1"

MASKED_FEATS = [
    "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
    "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"loading train + test (SMOKE={SMOKE})")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    if SMOKE:
        tr = tr.iloc[:30_000].copy().reset_index(drop=True)
        te = te.iloc[:10_000].copy().reset_index(drop=True)

    n_tr, n_te = len(tr), len(te)
    log(f"  train={n_tr:,}  test={n_te:,}")

    tr = add_distance_features(tr)
    te = add_distance_features(te)

    # Combine for self-supervised training. Drop target column.
    tgt_col = "Irrigation_Need"
    tr_no_tgt = tr.drop(columns=[tgt_col])
    combined = pd.concat([tr_no_tgt, te], ignore_index=True)
    log(f"  combined: {len(combined):,}")

    # Factorize cats globally on combined (no leak risk; this is target-free)
    cat_cols = [c for c in combined.columns
                  if not pd.api.types.is_numeric_dtype(combined[c])]
    log(f"  factorizing {len(cat_cols)} cats: {cat_cols}")
    for c in cat_cols:
        s = combined[c].astype(str)
        m = {v: i for i, v in enumerate(sorted(s.unique()))}
        combined[c] = s.map(m).fillna(-1).astype(np.int32)

    # Output arrays
    n_total = len(combined)
    n_outs = len(MASKED_FEATS) * 2
    out_arr = np.zeros((n_total, n_outs), dtype=np.float32)
    feat_names = []

    xgb_params = dict(
        objective="reg:squarederror", eval_metric="rmse",
        max_depth=6, learning_rate=0.1, min_child_weight=10,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=1.0, reg_lambda=1.0,
        tree_method="hist", verbosity=0, seed=SEED,
    )
    n_round = 60 if SMOKE else 250

    for f_idx, feat in enumerate(MASKED_FEATS):
        log(f"=== masking {feat} ({f_idx+1}/{len(MASKED_FEATS)}) ===")
        y_target = combined[feat].astype(np.float32).to_numpy()
        X = combined.drop(columns=[feat]).to_numpy().astype(np.float32)
        oof_pred = np.zeros(n_total, dtype=np.float32)

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            t0 = time.time()
            dtr = xgb.DMatrix(X[tr_idx], label=y_target[tr_idx])
            dva = xgb.DMatrix(X[va_idx], label=y_target[va_idx])
            booster = xgb.train(xgb_params, dtr, num_boost_round=n_round,
                                 evals=[(dva, "v")], early_stopping_rounds=30,
                                 verbose_eval=0)
            oof_pred[va_idx] = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
            log(f"  fold {fold+1}: it={booster.best_iteration}  wall={time.time()-t0:.1f}s")

        resid = (oof_pred - y_target).astype(np.float32)
        abs_resid = np.abs(resid).astype(np.float32)
        out_arr[:, 2 * f_idx] = resid
        out_arr[:, 2 * f_idx + 1] = abs_resid
        feat_names += [f"{feat}_resid", f"{feat}_abs_resid"]
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        log(f"  RMSE = {rmse:.4f}  resid std = {resid.std():.4f}")

    # Split out train + test
    oof = out_arr[:n_tr]
    test = out_arr[n_tr:]
    log(f"oof shape: {oof.shape}  test shape: {test.shape}")

    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_masked_resid{suffix}.npy", oof)
    np.save(ART / f"test_masked_resid{suffix}.npy", test)
    with open(ART / f"masked_resid{suffix}_results.json", "w") as f:
        json.dump({
            "smoke": SMOKE, "seed": SEED, "n_folds": N_FOLDS,
            "n_train": n_tr, "n_test": n_te,
            "masked_feats": MASKED_FEATS,
            "feat_names": feat_names,
            "n_output_cols": n_outs,
        }, f, indent=2)
    log(f"wrote oof_masked_resid{suffix}.npy + test + JSON")


if __name__ == "__main__":
    main()
