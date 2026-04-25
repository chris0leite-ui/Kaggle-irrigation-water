"""Narrow 5-input meta-stacker (cherry-picked from claude/simplify-ml-solution-Q2Zll).

Hypothesis (NEXT_STEPS #2 from the simplify branch): most of the 63-component
bank that produced LB 0.98094 contributes near-zero. A tight meta on
[recipe_oof, pseudo_oof, dgp_score, sm_dist, rf_dist] should recover most
of the +0.00086 lift from a sub-200-line script. If true, validates the
"narrow > wide" framing; if false, the wide bank is genuinely pulling weight.

Inputs (9 numeric features fed to XGB):
    recipe softprob   (3)  ← oof_recipe_full_te.npy
    pseudo softprob   (3)  ← oof_recipe_pseudolabel.npy
    dgp_score         (1)  ← integer 0..10 from common.add_distance_features
    sm_dist           (1)  ← Soil_Moisture - 25
    rf_dist           (1)  ← Rainfall_mm - 300

Same heavy-reg XGB + 5-fold StratifiedKFold(seed=42) as tier1b. Output is
isotonic-calibrated per class. Blend gate against LB-best 4-stack at fixed
recipe bias [1.4324, 1.4689, 3.4008].

Run:
    SMOKE=1 python scripts/tier1c_narrow_metastack.py   # 2 folds, ~30 s
    python scripts/tier1c_narrow_metastack.py            # 5 folds, ~3-5 min
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
from common import add_distance_features, log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import (BIAS, CLS2IDX, TARGET, bal_at_bias,  # noqa: E402
                            build_lbbest_stack, iso_cal, load_y, normed)

ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
SUFFIX = "_smoke" if SMOKE else ""


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_features():
    """Build 9-feature stacker matrix for train + test."""
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if SMOKE:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train), 20_000, replace=False)
        train = train.iloc[idx].reset_index(drop=True)
        test = test.iloc[:10_000].reset_index(drop=True)

    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    feats = ["dgp_score", "sm_dist", "rf_dist"]
    Xd_tr = tr_d[feats].to_numpy(dtype=np.float32)
    Xd_te = te_d[feats].to_numpy(dtype=np.float32)

    if SMOKE:
        # Subsample saved OOFs to match the smoke train rows.
        r_o = np.load(ART / "oof_recipe_full_te.npy").astype(np.float32)[idx]
        p_o = np.load(ART / "oof_recipe_pseudolabel.npy").astype(np.float32)[idx]
        r_t = np.load(ART / "test_recipe_full_te.npy").astype(np.float32)[:10_000]
        p_t = np.load(ART / "test_recipe_pseudolabel.npy").astype(np.float32)[:10_000]
        y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    else:
        r_o = np.load(ART / "oof_recipe_full_te.npy").astype(np.float32)
        p_o = np.load(ART / "oof_recipe_pseudolabel.npy").astype(np.float32)
        r_t = np.load(ART / "test_recipe_full_te.npy").astype(np.float32)
        p_t = np.load(ART / "test_recipe_pseudolabel.npy").astype(np.float32)
        y = load_y()

    X_tr = np.concatenate([r_o, p_o, Xd_tr], axis=1)
    X_te = np.concatenate([r_t, p_t, Xd_te], axis=1)
    return X_tr, X_te, y


def run_cv(X_tr, X_te, y):
    params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    max_rounds = 200 if SMOKE else 3000
    es = 50 if SMOKE else 200
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_preds, best_iters = [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(params, dtr, num_boost_round=max_rounds,
                            evals=[(dva, "val")], early_stopping_rounds=es,
                            verbose_eval=0)
        bi = booster.best_iteration
        best_iters.append(bi)
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_preds.append(booster.predict(dte, iteration_range=(0, bi + 1)))
        log(f"  fold {fold+1}/{N_FOLDS}: best_iter={bi}  wall={time.time()-t1:.1f}s")
    return normed(oof), normed(np.mean(test_preds, axis=0)), best_iters


def main():
    log(f"narrow meta-stacker | SMOKE={SMOKE} | n_folds={N_FOLDS}")
    X_tr, X_te, y = build_features()
    log(f"feature matrix: tr={X_tr.shape}, te={X_te.shape}")

    oof, test, best_iters = run_cv(X_tr, X_te, y)
    np.save(ART / f"oof_xgb_metastack_narrow{SUFFIX}.npy", oof)
    np.save(ART / f"test_xgb_metastack_narrow{SUFFIX}.npy", test)

    # Diagnostics: standalone tuned + iso-cal tuned + blend gate.
    raw_argmax = float(bal_at_bias(oof, y, np.zeros(3)))
    bias_raw, raw_tuned = tune_log_bias(oof, y, np.bincount(y) / len(y))
    log(f"standalone @0bias argmax = {raw_argmax:.5f}")
    log(f"standalone tuned         = {raw_tuned:.5f}  bias={[round(b,3) for b in bias_raw]}")
    oof_iso, test_iso = iso_cal(oof, test, y)
    iso_argmax = float(bal_at_bias(oof_iso, y, np.zeros(3)))
    log(f"iso  @0bias argmax       = {iso_argmax:.5f}")

    if SMOKE:
        log("SMOKE done — skipping blend gate.")
        return

    lb_oof, lb_test = build_lbbest_stack(y)
    anchor = float(bal_at_bias(lb_oof, y, BIAS))
    log(f"LB-best 4-stack @recipe_bias = {anchor:.5f}")

    sweep = []
    for alpha in [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        b = log_blend([lb_oof, oof_iso], np.array([1 - alpha, alpha]))
        sweep.append((alpha, float(bal_at_bias(b, y, BIAS))))
        log(f"  α={alpha:.3f}  blend={sweep[-1][1]:.5f}  Δ={sweep[-1][1]-anchor:+.5f}")

    out = dict(seed=SEED, n_folds=N_FOLDS, best_iters=best_iters,
               feature_dim=int(X_tr.shape[1]),
               raw_argmax=raw_argmax, raw_tuned=raw_tuned, raw_bias=bias_raw.tolist(),
               iso_argmax=iso_argmax,
               anchor_lbbest_4stack=anchor,
               sweep=sweep)
    with open(ART / f"tier1c_narrow_metastack_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"saved results -> {ART/'tier1c_narrow_metastack_results.json'}")


if __name__ == "__main__":
    main()
