"""T1: L3 RF-meta-of-RF natural — stacking layer ON TOP of natural-cal L2 metas.

Per LEARNINGS post-LB-0.98129 plan, the natural-cal compounding mechanism
worked at L2 (RF-meta over 7 natural-cal bases). The bank-extension trap
(a1lgbm, v2, plus_natrealmlp all LB-regressed) shows that compounding must
come from LAYERS, not WIDER L2 banks. T1 tests the L3 axis directly:

  L3 RF (this script)
   |- L2 RF natural v1     (LB 0.98129, 7-component bank)
   |- L2 RF natural a1lgbm (LB 0.98097, 10-component bank)
   |- L2 RF natural +natrealmlp (LB 0.98098, 8-component bank)
   |- rawashishsin_2600    (LB 0.98109, naturally-calibrated XGB anchor)
   |- recipe_full_te_catboost_natural (Phase 1, naturally-calibrated CB)

Mechanism: bagging at L3 over diverse L2 stackers + raw natural-cal bases
preserves natural calibration profile (per the v1 LB-result rule that
"bagging on natural-cal banks compounds monotonically"). HPs identical to
v1 RF natural so the comparison isolates the layer effect.

LEAK-SAFETY: all L2 metas + base components were trained on
StratifiedKFold(seed=42). L3 uses same fold split. Per-row OOF is
leak-free at row level (each row's L2 prediction came from a model
trained without that row); residual feature-level fold-alignment risk
is the standard stacking caveat.

Outputs:
  scripts/artifacts/oof_sklearn_rf_meta_l3_natural.npy
  scripts/artifacts/test_sklearn_rf_meta_l3_natural.npy
  scripts/artifacts/sklearn_rf_meta_l3_natural_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42

SMOKE = os.environ.get("SMOKE") == "1"
META_SUFFIX = os.environ.get("META_SUFFIX", "")  # default empty -> "_l3natural"

# L3 input bank: 3 L2 RF natural metas + 2 strongest raw natural-cal bases.
# Excludes additional L2 RF variants (a1lgbm/plus_natrealmlp) when SMOKE
# to keep wall budget tight.
L3_BANK = [
    # L2 RF natural metas (3 different banks, same architecture)
    "sklearn_rf_meta_natural_v1_lb98129",     # LB 0.98129, 7-component bank
    "sklearn_rf_meta_natural_a1lgbm",         # LB 0.98097, 10-component bank
    "sklearn_rf_meta_natural_plus_natrealmlp", # LB 0.98098, 8-component bank
    # Strongest raw natural-cal bases (orthogonal to L2 metas)
    "rawashishsin_2600",                      # LB 0.98109, narrow-FE XGB
    "recipe_full_te_catboost_natural",        # Phase 1 naturally-cal CB
]

# Distance / rule meta features (same as v1 RF natural)
META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def load_bank(n_tr, n_te):
    log(f"loading L3 natural-cal bank ({len(L3_BANK)} components)")
    pool = {}
    for name in L3_BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  SKIP {name}: missing")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != n_tr:
            log(f"  SKIP {name}: shape {o.shape}")
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    log(f"  loaded {len(pool)}/{len(L3_BANK)} components")
    return pool


def build_features(pool, train, test):
    log("constructing distance / rule meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    feature_names = list(META_COLS)
    for n in component_names:
        feature_names += [f"{n}_logL", f"{n}_logM", f"{n}_logH"]

    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]

    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  feature matrix: train={X_tr.shape}  test={X_te.shape}")
    return X_tr, X_te, feature_names


def main():
    log("=== T1 L3 RF-meta-of-RF natural ===")
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = load_bank(n_tr, n_te)
    if "sklearn_rf_meta_natural_v1_lb98129" not in pool:
        log("ERROR: v1 LB-best L2 meta missing — abort")
        return
    if "rawashishsin_2600" not in pool:
        log("ERROR: rawashishsin_2600 anchor missing — abort")
        return

    X_tr, X_te, feature_names = build_features(pool, train, test)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    n_est = 100 if SMOKE else 500
    max_depth = 8 if SMOKE else 12
    n_folds = 2 if SMOKE else 5
    rf_params = dict(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=SEED,
        class_weight=None,
        verbose=0,
    )
    log(f"L3 RF: n_est={n_est} max_depth={max_depth} class_weight=None bootstrap=True")
    log(f"input dim: {X_tr_s.shape[1]} (META {len(META_COLS)} + components × 3)")

    if SMOKE:
        sub_idx = np.arange(50_000)
        X_tr_use = X_tr_s[sub_idx]
        y_use = y[sub_idx]
    else:
        X_tr_use = X_tr_s
        y_use = y

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_use, y_use))
    oof = np.zeros((len(y_use), 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"=== fold {fold}/{n_folds}  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_use[tr_idx], y_use[tr_idx])
        p_va = rf.predict_proba(X_tr_use[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y_use[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y_use, oof.argmax(1))
    prior = np.bincount(y_use, minlength=3) / len(y_use)
    bias, tuned = tune_log_bias(oof, y_use, prior)
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    # Drift from -log(prior)
    minus_log_prior = -np.log(prior)
    drift = (bias - minus_log_prior)
    drift_max = float(np.abs(drift).max())
    log(f"  -log(prior) = {minus_log_prior.round(4).tolist()}")
    log(f"  drift       = {drift.round(4).tolist()}  |drift|_max = {drift_max:.4f}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y_use, pred_at_bias)
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    suffix = META_SUFFIX or "_l3natural"
    np.save(ART / f"oof_sklearn_rf_meta{suffix}.npy", oof)
    np.save(ART / f"test_sklearn_rf_meta{suffix}.npy", test_pred)

    # Save tuned-bias submission as a candidate (NOT for LB probe without
    # explicit user approval per CLAUDE.md rule)
    if not SMOKE:
        tuned_pred = (safelog(test_pred) + bias).argmax(1)
        sub = pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in tuned_pred],
        })
        sub_path = SUB / f"submission_sklearn_rf_meta{suffix}_standalone.csv"
        sub.to_csv(sub_path, index=False)
        log(f"  wrote {sub_path}  (NOT submitted — awaiting user approval)")

    summary = dict(
        n_folds=n_folds, smoke=SMOKE, seed=SEED,
        n_estimators=n_est, max_depth=max_depth,
        bank=L3_BANK, bank_loaded=sorted(pool.keys()),
        feature_count=len(feature_names),
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        drift_from_minus_log_prior=drift.tolist(),
        drift_max_abs=drift_max,
        per_class_recall=pcr.tolist(),
    )
    with open(ART / f"sklearn_rf_meta{suffix}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {ART}/sklearn_rf_meta{suffix}_results.json")
    log("=== T1 L3 done ===")


if __name__ == "__main__":
    main()
