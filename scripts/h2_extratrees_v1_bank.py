"""H2 — ExtraTrees natural meta on v1's 7-component bank.

Mechanism-preserving architectural diversification at L2:
  - Same v1 bank (7 components — DO NOT MODIFY)
  - Same `class_weight=None` (natural-cal property)
  - Same 5-fold StratifiedKFold(seed=42) for OOF alignment
  - DIFFERENT bagging-tree family: ExtraTreesClassifier
    (random splits within feature subset → more randomization
    than RF's best-split-within-subset)

Hypothesis: ET produces orthogonal errors to v1 RF. If so, geomean
blend at α≈0.50 should pick up variance reduction beyond what
seed-bagging captures (different decision-boundary geometry).

Outputs:
  oof_h2_et_natural.npy  test_h2_et_natural.npy
  h2_et_natural_results.json
  submission_h2_et_natural_standalone.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

V1_BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
]

META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]

SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum() > 0:
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = {}
    for name in V1_BANK:
        oof = _normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        tt = _normed(np.load(ART / f"test_{name}.npy").astype(np.float32))
        pool[name] = (oof, tt)
        log(f"  + {name}")

    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in names]
    log_te = [safelog(pool[n][1]) for n in names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    log(f"feature matrix: train={X_tr_s.shape}  test={X_te_s.shape}")

    n_est = 100 if SMOKE else 500
    max_depth = 8 if SMOKE else 12
    n_folds = 2 if SMOKE else 5
    et_params = dict(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=SEED,
        class_weight=None, verbose=0,
    )
    log(f"ExtraTrees natural: n_est={n_est} max_depth={max_depth} class_weight=None bootstrap=True")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_s, y), 1):
        t0 = time.time()
        et = ExtraTreesClassifier(**et_params)
        et.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = et.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = et.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold}/{n_folds} bal={bal:.5f} wall={time.time()-t0:.1f}s")

    bal = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(_normed(oof), y, prior)
    log(f"=== H2 ET overall argmax={bal:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")

    pcr = per_class_recall(y, (safelog(_normed(oof)) + bias).argmax(1))
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    np.save(ART / "oof_h2_et_natural.npy", _normed(oof))
    np.save(ART / "test_h2_et_natural.npy", _normed(test_pred))

    # Compare with v1 LB-best
    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)

    et_pred_at_v1bias = (safelog(_normed(oof)) + v1_bias).argmax(1)
    et_bal_at_v1bias = balanced_accuracy_score(y, et_pred_at_v1bias)
    log(f"v1 LB-best tuned={v1_tuned:.5f} bias={v1_bias.round(4).tolist()}")
    log(f"H2 ET at v1 bias: argmax={et_bal_at_v1bias:.5f}")

    # Diversity
    et_pred = test_pred.argmax(1)
    v1_pred = v1_test.argmax(1)
    diff = int((et_pred != v1_pred).sum())
    log(f"test argmax diff: H2 vs v1 = {diff} / {n_te} ({diff/n_te*100:.3f}%)")

    summary = dict(
        smoke=SMOKE, seed=SEED, bank=V1_BANK,
        n_estimators=n_est, max_depth=max_depth,
        fold_scores=fold_scores,
        argmax=float(bal), tuned=float(tuned), bias=bias.tolist(),
        pcr=pcr.tolist(),
        v1_tuned=float(v1_tuned), v1_bias=v1_bias.tolist(),
        et_at_v1bias=float(et_bal_at_v1bias),
        test_diff_vs_v1=diff,
    )
    with open(ART / "h2_et_natural_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"wrote {ART}/h2_et_natural_results.json")

    # Standalone submission at ET's tuned bias
    test_pred_at_bias = (safelog(_normed(test_pred)) + bias).argmax(1)
    sub_path = SUB / "submission_h2_et_natural_standalone.csv"
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_at_bias],
    })
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")


if __name__ == "__main__":
    main()
