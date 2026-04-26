"""#4 polynomial / non-linear FE test.

Recipe FE has 4 pairwise products (sm_x_rf, tc_x_ws, sm_x_kc, rf_x_kc) and
4 cat-pair OTEs. It does NOT have:
  - 3-way / triple products of rule features
  - Non-linear transforms of non-rule numerics (cubes, logs, sqrts, trig)
  - Non-rule × rule cross terms beyond the pairwise products
  - Within-cell normalized features

This script tests whether 30+ new polynomial / non-linear features lift
recipe-style XGB OOF when added to the existing 43-dist feature set.

If standalone tuned OOF lifts beyond recipe's 0.97967 → run full recipe
with these added. If null → the FE space is genuinely saturated (in which
case the score=6 info-ceiling diagnosis from 2026-04-26 is reinforced).
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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import (CLS2IDX, add_distance_features, fast_bal_acc,  # noqa
                       tune_log_bias)

ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE", "0") == "1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_poly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add polynomial / non-linear features beyond what recipe FE has."""
    out = df.copy()
    sm = out["Soil_Moisture"].astype(np.float32).to_numpy()
    rf = out["Rainfall_mm"].astype(np.float32).to_numpy()
    tc = out["Temperature_C"].astype(np.float32).to_numpy()
    ws = out["Wind_Speed_kmh"].astype(np.float32).to_numpy()
    hum = out["Humidity"].astype(np.float32).to_numpy()
    pri = out["Previous_Irrigation_mm"].astype(np.float32).to_numpy()
    ec = out["Electrical_Conductivity"].astype(np.float32).to_numpy()
    ph = out["Soil_pH"].astype(np.float32).to_numpy()
    oc = out["Organic_Carbon"].astype(np.float32).to_numpy()
    sun = out["Sunlight_Hours"].astype(np.float32).to_numpy()
    fa = out["Field_Area_hectare"].astype(np.float32).to_numpy()

    # 3-way rule products
    out["sm_x_rf_x_tc"] = (sm * rf * tc).astype(np.float32) / 1000.0
    out["sm_x_rf_x_ws"] = (sm * rf * ws).astype(np.float32) / 1000.0
    out["sm_x_tc_x_ws"] = (sm * tc * ws).astype(np.float32) / 1000.0
    out["rf_x_tc_x_ws"] = (rf * tc * ws).astype(np.float32) / 1000.0

    # Non-linear transforms of rule numerics
    out["sm_squared"] = (sm ** 2).astype(np.float32) / 100.0
    out["rf_squared"] = (rf ** 2).astype(np.float32) / 10000.0
    out["sm_cubed"] = (sm ** 3).astype(np.float32) / 1e4
    out["log_sm"] = np.log(sm + 1.0).astype(np.float32)
    out["log_rf"] = np.log(rf + 1.0).astype(np.float32)
    out["sqrt_sm"] = np.sqrt(sm).astype(np.float32)
    out["sqrt_rf"] = np.sqrt(rf).astype(np.float32)

    # Periodic / trigonometric (rule features are bounded, periodic transforms
    # give smooth angular distance to thresholds)
    out["sin_sm_th"] = np.sin(np.pi * sm / 25.0).astype(np.float32)
    out["cos_sm_th"] = np.cos(np.pi * sm / 25.0).astype(np.float32)
    out["sin_rf_th"] = np.sin(np.pi * rf / 300.0).astype(np.float32)
    out["sin_tc_th"] = np.sin(np.pi * tc / 30.0).astype(np.float32)

    # Non-rule × rule crosses (we have sm_x_rf type, not these)
    out["hum_x_sm"] = (hum * sm).astype(np.float32) / 100.0
    out["hum_x_tc"] = (hum * tc).astype(np.float32) / 100.0
    out["pri_x_rf"] = (pri * rf).astype(np.float32) / 1000.0
    out["pri_x_sm"] = (pri * sm).astype(np.float32) / 1000.0
    out["ec_x_sm"] = (ec * sm).astype(np.float32)
    out["oc_x_sm"] = (oc * sm).astype(np.float32)
    out["sun_x_tc"] = (sun * tc).astype(np.float32)

    # Non-rule transforms
    out["log_hum"] = np.log(hum + 1.0).astype(np.float32)
    out["log_pri"] = np.log(pri + 1.0).astype(np.float32)
    out["sqrt_pri"] = np.sqrt(pri).astype(np.float32)
    out["hum_squared"] = (hum ** 2).astype(np.float32) / 100.0
    out["ph_minus_7"] = (ph - 7.0).astype(np.float32)
    out["ph_dev"] = np.abs(ph - 7.0).astype(np.float32)

    # Non-rule × non-rule products
    out["hum_x_pri"] = (hum * pri).astype(np.float32) / 1000.0
    out["hum_x_ec"] = (hum * ec).astype(np.float32) / 100.0
    out["fa_x_oc"] = (fa * oc).astype(np.float32)
    out["fa_x_pri"] = (fa * pri).astype(np.float32) / 100.0

    # Within-cell normalization features
    # (deviation from the mean of feature within rule-cell — high values
    # indicate the row is at the extreme of its cell)
    out["sm_pct_of_25"] = (sm / 25.0).astype(np.float32)
    out["rf_pct_of_300"] = (rf / 300.0).astype(np.float32)
    out["tc_pct_of_30"] = (tc / 30.0).astype(np.float32)
    out["ws_pct_of_10"] = (ws / 10.0).astype(np.float32)

    return out


def main() -> None:
    log(f"loading train + test (SMOKE={SMOKE})")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    if SMOKE:
        idx = np.random.RandomState(SEED).choice(len(tr), 30_000, replace=False)
        tr = tr.iloc[idx].reset_index(drop=True)
        te = te.iloc[:10_000].copy().reset_index(drop=True)

    tr = add_distance_features(tr)
    te = add_distance_features(te)
    tr = add_poly_features(tr)
    te = add_poly_features(te)

    target = "Irrigation_Need"
    y = tr[target].map(CLS2IDX).to_numpy().astype(np.int8)

    # Feature set: dist (43) + poly (38) = 81 features, no OTE
    cat_cols = [c for c in tr.columns
                  if not pd.api.types.is_numeric_dtype(tr[c]) and c != target]
    log(f"factorizing {len(cat_cols)} cats: {cat_cols}")
    for c in cat_cols:
        s_tr = tr[c].astype(str)
        s_te = te[c].astype(str)
        m = {v: i for i, v in enumerate(sorted(set(s_tr) | set(s_te)))}
        tr[c] = s_tr.map(m).astype(np.int32)
        te[c] = s_te.map(m).fillna(-1).astype(np.int32)

    drop = [target, "id"]
    feats = [c for c in tr.columns if c not in drop]
    X = tr[feats].astype(np.float32).to_numpy()
    X_te = te[feats].astype(np.float32).to_numpy()
    log(f"feature count: {X.shape[1]}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), 3), dtype=np.float32)
    test = np.zeros((len(te), 3), dtype=np.float32)
    cls_w = np.zeros(len(y), dtype=np.float32)
    bin_y = np.bincount(y, minlength=3)
    n_total = len(y)
    for k in range(3):
        cls_w[y == k] = n_total / (3.0 * max(bin_y[k], 1))

    n_round = 200 if SMOKE else 3000
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X[tr_idx], label=y[tr_idx], weight=cls_w[tr_idx])
        dva = xgb.DMatrix(X[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        params = dict(
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss",
            max_depth=4, learning_rate=0.05,
            min_child_weight=10,
            subsample=0.9, colsample_bytree=0.8,
            reg_alpha=5.0, reg_lambda=5.0,
            tree_method="hist", verbosity=0, seed=SEED,
        )
        booster = xgb.train(params, dtr, num_boost_round=n_round,
                              evals=[(dva, "v")], early_stopping_rounds=200,
                              verbose_eval=0)
        bi = booster.best_iteration
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        argmax = oof[va_idx].argmax(1)
        bal = fast_bal_acc(y[va_idx], argmax)
        log(f"  fold {fold+1}: it={bi}  argmax_bal={bal:.5f}  wall={time.time()-t0:.1f}s")

    # Tune log-bias
    prior = bin_y.astype(np.float32) / bin_y.sum()
    bias, best_tuned = tune_log_bias(oof, y, prior)
    log(f"OOF tuned bal_acc = {best_tuned:.5f}  bias = {bias.tolist()}")

    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_poly_fe{suffix}.npy", oof)
    np.save(ART / f"test_poly_fe{suffix}.npy", test)

    # Top-importance features (peek at whether new poly features matter)
    importance = booster.get_score(importance_type="gain")
    top = sorted(importance.items(), key=lambda kv: -kv[1])[:30]
    feat_idx_to_name = {f"f{i}": feats[i] for i in range(len(feats))}
    log("top-30 features by gain (last fold):")
    poly_count_in_top = 0
    poly_features = set([
        "sm_x_rf_x_tc", "sm_x_rf_x_ws", "sm_x_tc_x_ws", "rf_x_tc_x_ws",
        "sm_squared", "rf_squared", "sm_cubed", "log_sm", "log_rf",
        "sqrt_sm", "sqrt_rf", "sin_sm_th", "cos_sm_th", "sin_rf_th",
        "sin_tc_th", "hum_x_sm", "hum_x_tc", "pri_x_rf", "pri_x_sm",
        "ec_x_sm", "oc_x_sm", "sun_x_tc", "log_hum", "log_pri",
        "sqrt_pri", "hum_squared", "ph_minus_7", "ph_dev",
        "hum_x_pri", "hum_x_ec", "fa_x_oc", "fa_x_pri",
        "sm_pct_of_25", "rf_pct_of_300", "tc_pct_of_30", "ws_pct_of_10",
    ])
    for k, v in top:
        name = feat_idx_to_name.get(k, k)
        is_poly = name in poly_features
        marker = " (NEW)" if is_poly else ""
        if is_poly:
            poly_count_in_top += 1
        log(f"  {name:30s} gain={v:>10.2f}{marker}")
    log(f"poly features in top-30: {poly_count_in_top}/30")

    with open(ART / f"poly_fe{suffix}_results.json", "w") as f:
        json.dump({
            "smoke": SMOKE, "n_features": int(X.shape[1]),
            "oof_tuned_bal_acc": float(best_tuned),
            "tuned_bias": bias.tolist(),
            "poly_features_in_top_30": poly_count_in_top,
        }, f, indent=2)
    log(f"wrote oof_poly_fe{suffix}.npy + test + JSON")


if __name__ == "__main__":
    main()
