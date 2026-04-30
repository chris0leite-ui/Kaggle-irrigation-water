"""V1 RF natural bank-extension with T2 pseudo-label component.

Adds `recipe_pseudolabel_lb98129labeler_t099` to v1's 7-component bank,
retrains the same RF natural architecture, evaluates as PRIMARY (alone +
4-stack-style architecture).

Per the bank-specificity rule (3 prior bank-extension nulls all LB-regressed
by ~-0.0003), this is structurally LIKELY to null. But T2 differs from
the prior nulls — it's pseudo-label-derived (training-data-level lever)
not just another model on recipe FE. If it has Jaccard < 0.75 vs v1 AND
fewer errors at recipe bias, it may break the bank-extension pattern.

Output:
  scripts/artifacts/oof_sklearn_rf_meta_natural_plus_t2.npy
  scripts/artifacts/test_sklearn_rf_meta_natural_plus_t2.npy
  scripts/artifacts/sklearn_rf_meta_natural_plus_t2_results.json
  submissions/submission_sklearn_rf_meta_natural_plus_t2_standalone.csv
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

# v1's 7-component bank + T2 pseudo-label
BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
    "recipe_pseudolabel_lb98129labeler_t099",  # T2 addition
]

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


def main():
    log("=== v1 RF natural + T2 bank-extension ===")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    log(f"loading {len(BANK)}-component bank")
    pool = {}
    for name in BANK:
        op = ART / f"oof_{name}.npy"
        tp = ART / f"test_{name}.npy"
        if not op.exists() or not tp.exists():
            log(f"  SKIP {name}: missing")
            continue
        o = np.load(op).astype(np.float32)
        t = np.load(tp).astype(np.float32)
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != n_tr:
            log(f"  SKIP {name}: shape {o.shape}")
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    log(f"  loaded {len(pool)}/{len(BANK)} components")

    if "recipe_pseudolabel_lb98129labeler_t099" not in pool:
        log("ERROR: T2 component missing — abort")
        return
    if "rawashishsin_2600" not in pool:
        log("ERROR: rawashishsin_2600 anchor missing — abort")
        return

    # Build features (mirror sklearn_rf_meta_natural.py exactly)
    log("constructing distance / rule meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]

    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  feature matrix: train={X_tr.shape}  test={X_te.shape}")

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    rf_params = dict(
        n_estimators=500, max_depth=12,
        min_samples_leaf=20, max_features="sqrt",
        bootstrap=True, n_jobs=-1, random_state=SEED,
        class_weight=None, verbose=0,
    )
    log(f"RF (natural): {rf_params}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y))
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"=== fold {fold}/5  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / 5.0
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / n_tr
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    minus_log_prior = -np.log(prior)
    drift = bias - minus_log_prior
    log(f"  drift = {drift.round(4).tolist()}  |max| = {abs(drift).max():.4f}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    np.save(ART / "oof_sklearn_rf_meta_natural_plus_t2.npy", oof)
    np.save(ART / "test_sklearn_rf_meta_natural_plus_t2.npy", test_pred)

    tuned_test_pred = (safelog(test_pred) + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in tuned_test_pred],
    })
    sub_path = SUB / "submission_sklearn_rf_meta_natural_plus_t2_standalone.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  (NOT submitted — awaiting user approval)")

    # Compare to v1 LB 0.98129
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_bias_arr = np.array([0.4324, 0.8689, 3.2008])
    v1_pred_test = (safelog(v1_test) + v1_bias_arr).argmax(1)
    diff_v1 = int((tuned_test_pred != v1_pred_test).sum())
    add_h = int(((tuned_test_pred == 2) & (v1_pred_test != 2)).sum())
    rem_h = int(((v1_pred_test == 2) & (tuned_test_pred != 2)).sum())
    net_h = add_h - rem_h
    churn_h = add_h + rem_h
    log(f"  vs v1 LB 0.98129: test diff = {diff_v1} / {n_te} ({100*diff_v1/n_te:.3f}%)")
    log(f"  H-flips: +{add_h} / -{rem_h}  net={net_h:+d}  ratio={abs(net_h)/max(churn_h,1):.3f}")

    summary = dict(
        bank=BANK, bank_loaded=sorted(pool.keys()),
        n_components=len(pool),
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        drift=drift.tolist(),
        drift_max=float(abs(drift).max()),
        per_class_recall=pcr.tolist(),
        test_diff_vs_v1=diff_v1,
        add_high_vs_v1=add_h,
        remove_high_vs_v1=rem_h,
        net_high_vs_v1=net_h,
        g4_ratio=float(abs(net_h)/max(churn_h, 1)),
    )
    with open(ART / "sklearn_rf_meta_natural_plus_t2_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("=== bank-extension done ===")


if __name__ == "__main__":
    main()
