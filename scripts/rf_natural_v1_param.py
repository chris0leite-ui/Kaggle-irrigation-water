"""Parameterized v1 RF natural meta-stacker for n_est sweep + fold_seed sweep.

Reproduces the LB-best v1 RF natural meta (LB 0.98129) with the ORIGINAL
7-component bank, parameterized via env vars:

  N_EST       (default 500)   number of RF estimators
  FOLD_SEED   (default 42)    StratifiedKFold seed (43=v1 LB-best alignment)
  RF_SEED     (default 42)    RF random_state (held constant across runs)

Output suffix: _n{N_EST}_fs{FOLD_SEED}

Idea 1: N_EST=1000 FOLD_SEED=42 (variance reduction at 2x the bagging)
Idea 3a: N_EST=500 FOLD_SEED=7   (fold-split sensitivity check)
Idea 3b: N_EST=500 FOLD_SEED=123 (second sensitivity check)

If LB 0.98129 holds across all three, we lock with high confidence
that v1 is robust at the structural level.
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

N_EST = int(os.environ.get("N_EST", 500))
FOLD_SEED = int(os.environ.get("FOLD_SEED", 42))
RF_SEED = int(os.environ.get("RF_SEED", 42))
SMOKE = os.environ.get("SMOKE") == "1"

SUFFIX = f"_n{N_EST}_fs{FOLD_SEED}"

# ORIGINAL v1 LB-best 7-component bank (per CLAUDE.md 2026-04-29 entry).
# DO NOT extend — bank-extension regressed LB by 0.00031 on 3 separate tests.
NATURAL_BANK_V1 = [
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


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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


def load_bank(n_tr):
    log(f"loading natural-cal v1 bank ({len(NATURAL_BANK_V1)} components)")
    pool = {}
    for name in NATURAL_BANK_V1:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  ! MISSING {name}")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != n_tr:
            log(f"  ! SKIP {name}: shape {o.shape}")
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    log(f"  loaded {len(pool)}/{len(NATURAL_BANK_V1)}")
    return pool


def build_features(pool, train, test):
    log("constructing meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    # add the boundary-distance derived cols expected by META_COLS
    for d in (tr_d, te_d):
        if "min_boundary_dist" not in d.columns:
            d["min_boundary_dist"] = d[["sm_abs", "rf_abs", "tc_abs", "ws_abs"]].min(axis=1)
        if "min_axis_abs" not in d.columns:
            d["min_axis_abs"] = d[["sm_abs", "rf_abs", "tc_abs", "ws_abs"]].min(axis=1)
        if "score_dist_low_mid" not in d.columns:
            d["score_dist_low_mid"] = (d["dgp_score"].astype(np.float32) - 3.5).abs()
        if "score_dist_mid_high" not in d.columns:
            d["score_dist_mid_high"] = (d["dgp_score"].astype(np.float32) - 6.5).abs()
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]
    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")
    return X_tr, X_te


def main():
    log(f"=== RF natural v1 — N_EST={N_EST} FOLD_SEED={FOLD_SEED} RF_SEED={RF_SEED} ===")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = load_bank(n_tr)
    if len(pool) != 7:
        log(f"FATAL: bank size {len(pool)} != 7 — abort")
        return

    X_tr, X_te = build_features(pool, train, test)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    n_est = 50 if SMOKE else N_EST
    max_depth = 8 if SMOKE else 12
    n_folds = 2 if SMOKE else 5
    rf_params = dict(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=RF_SEED,
        class_weight=None, verbose=0,
    )
    log(f"RF: n_est={n_est} max_depth={max_depth} class_weight=None")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=FOLD_SEED)
    splits = list(skf.split(X_tr_s, y))
    if SMOKE:
        sub_idx = np.arange(50_000)
        X_use, y_use = X_tr_s[sub_idx], y[sub_idx]
        skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=FOLD_SEED)
        splits = list(skf2.split(X_use, y_use))
        oof = np.zeros((len(y_use), 3), dtype=np.float32)
    else:
        X_use, y_use = X_tr_s, y
        oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"=== fold {fold}/{n_folds}  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_use[tr_idx], y_use[tr_idx])
        p_va = rf.predict_proba(X_use[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y_use[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax={bal:.5f}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y_use, oof.argmax(1))
    prior = np.bincount(y_use, minlength=3) / len(y_use)
    bias, tuned = tune_log_bias(oof, y_use, prior)
    log(f"=== overall argmax={overall:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y_use, pred_at_bias)
    log(f"  PCR = [L={pcr[0]:.4f}  M={pcr[1]:.4f}  H={pcr[2]:.4f}]")

    # Drift from -log(prior)
    natural_bias = -np.log(prior)
    drift = bias - natural_bias
    log(f"  drift from -log(prior) = {drift.round(4).tolist()}  |max|={float(np.abs(drift).max()):.3f}")

    if not SMOKE:
        np.save(ART / f"oof_rf_natural_v1{SUFFIX}.npy", oof)
        np.save(ART / f"test_rf_natural_v1{SUFFIX}.npy", test_pred)
        log(f"saved oof/test rf_natural_v1{SUFFIX}")

        # Build standalone submission at tuned bias
        test_logits = safelog(test_pred) + bias
        test_argmax = test_logits.argmax(1)
        sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in test_argmax]})
        sub_path = SUB / f"submission_rf_natural_v1{SUFFIX}_standalone.csv"
        sub.to_csv(sub_path, index=False)
        log(f"wrote {sub_path}")

        # Compare vs LB-best v1 at test side
        v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
        # Apply same bias logic to LB-best v1 (its bias was [0.43, 0.87, 3.20])
        v1_logits = safelog(v1_test) + np.array([0.43, 0.87, 3.20])
        v1_argmax = v1_logits.argmax(1)
        diff = int((test_argmax != v1_argmax).sum())
        log(f"test diff vs LB-best v1: {diff}/{n_te} ({100*diff/n_te:.3f}%)")
        # class shift
        for k in range(3):
            d = int((test_argmax == k).sum() - (v1_argmax == k).sum())
            log(f"  class {IDX2CLS[k]}: test count delta = {d:+d}")

    summary = dict(
        n_est=N_EST, fold_seed=FOLD_SEED, rf_seed=RF_SEED, smoke=SMOKE,
        bank=NATURAL_BANK_V1, bank_size=len(pool),
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall), tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        drift_from_neg_log_prior=drift.tolist(),
        max_drift_magnitude=float(np.abs(drift).max()),
        per_class_recall=pcr.tolist(),
    )
    json_p = ART / f"rf_natural_v1{SUFFIX}_results.json"
    with open(json_p, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {json_p}")


if __name__ == "__main__":
    main()
