"""Minimal-input meta defense per CLAUDE.md leakage rule.

After SVGP meta gives Δ ≥ +2e-4 OOF, train a 2-component meta with ONLY
[LB-3-stack + svgp_meta] as inputs. If the 2-component lands BELOW the
LB-best 4-stack (0.98084), the SVGP's apparent +Δ at full bank is
cross-component memorization, NOT orthogonal signal.

Mechanism: full-bank meta-stacker has 200+ feature dims; tree splits or
GP kernel can pick up disagreement patterns BETWEEN components that are
fold-correlated artifacts of how StratifiedKFold(seed=42) splits the data.
Minimal-input strips those interaction terms — we see only the marginal
contribution of svgp_meta over LB-3-stack alone.

Implementation: small XGB stacker (depth=4, reg_alpha=5, reg_lambda=5,
lr=0.05) on 6 features (3 lb-3-stack log-probs + 3 svgp log-probs) +
3 dist features for context = 9 dims. Same fold split for OOF alignment.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (ART, BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                             load_y, bal_at_bias)

SEED = 42
N_FOLDS = 5
SUFFIX = "_svgp"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 3-stack")
    lb3_oof, lb3_test = build_lbbest_stack(y)

    log(f"loading svgp meta: oof_xgb_metastack{SUFFIX}.npy")
    svgp_o = np.load(ART / f"oof_xgb_metastack{SUFFIX}.npy")
    svgp_t = np.load(ART / f"test_xgb_metastack{SUFFIX}.npy")

    # LB-best primary (4-stack at α=0.30 with v1 iso-meta) — anchor
    v1_o = np.load(ART / "oof_xgb_metastack.npy")
    v1_t = np.load(ART / "test_xgb_metastack.npy")
    v1_iso_o, v1_iso_t = iso_cal(v1_o, v1_t, y)
    primary_o = log_blend([lb3_oof, v1_iso_o], np.array([0.7, 0.3]))
    primary_t = log_blend([lb3_test, v1_iso_t], np.array([0.7, 0.3]))
    log(f"  LB-best PRIMARY OOF = {bal_at_bias(primary_o, y):.5f}")

    # Build minimal feature matrix: lb3 log-probs + svgp log-probs + 3 dist cols
    tr_d = add_distance_features(train)[["dgp_score", "sm_dist", "min_axis_abs"]]
    te_d = add_distance_features(test)[["dgp_score", "sm_dist", "min_axis_abs"]]
    X_tr = np.concatenate([
        np.log(np.clip(lb3_oof, 1e-9, 1)),
        np.log(np.clip(svgp_o, 1e-9, 1)),
        tr_d.to_numpy(np.float32),
    ], axis=1).astype(np.float32)
    X_te = np.concatenate([
        np.log(np.clip(lb3_test, 1e-9, 1)),
        np.log(np.clip(svgp_t, 1e-9, 1)),
        te_d.to_numpy(np.float32),
    ], axis=1).astype(np.float32)
    log(f"  minimal-input dim = {X_tr.shape[1]} (3 lb3 + 3 svgp + 3 dist)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_min = np.zeros((len(y), 3), dtype=np.float32)
    test_min_folds = []
    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t1 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(xgb_params, dtr, num_boost_round=3000,
                            evals=[(dva, "val")], early_stopping_rounds=200,
                            verbose_eval=0)
        bi = booster.best_iteration
        oof_min[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1)).astype(np.float32)
        test_min_folds.append(
            booster.predict(dte, iteration_range=(0, bi + 1)).astype(np.float32))
        log(f"  fold {fold+1} it={bi} val_argmax={balanced_accuracy_score(y[va_idx], oof_min[va_idx].argmax(1)):.5f} "
            f"wall={time.time()-t1:.1f}s")

    test_min = np.mean(test_min_folds, axis=0).astype(np.float32)
    np.save(ART / f"oof_xgb_metastack_minimal{SUFFIX}.npy", oof_min)
    np.save(ART / f"test_xgb_metastack_minimal{SUFFIX}.npy", test_min)

    # Iso-cal & blend gate
    min_iso_o, min_iso_t = iso_cal(oof_min, test_min, y)
    standalone = bal_at_bias(min_iso_o, y)
    primary_bal = bal_at_bias(primary_o, y)
    log(f"\n=== minimal-input SVGP meta ===")
    log(f"  iso standalone OOF = {standalone:.5f}")
    log(f"  LB-best PRIMARY    = {primary_bal:.5f}")
    log(f"  Δ                  = {standalone - primary_bal:+.5f}")
    if standalone < primary_bal:
        log(f"  VERDICT: minimal < primary → svgp's full-bank lift is "
            f"likely cross-component memorization (CLAUDE.md leakage rule).")
    else:
        log(f"  VERDICT: minimal ≥ primary → marginal SVGP signal looks real.")
    log(f"\nwall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
