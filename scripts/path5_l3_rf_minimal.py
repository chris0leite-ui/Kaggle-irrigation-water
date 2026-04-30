"""Path 5 — L3 RF natural on minimal feature set (v1 + rawashishsin + dist).

Both inputs LB-validated and naturally calibrated:
  - v1 RF natural (LB 0.98129) — saved as oof_sklearn_rf_meta_natural_v1_lb98129
  - rawashishsin v3 (LB 0.98109) — saved as oof_rawashishsin_2600

Mechanism: 20-dim minimal-feature RF with class_weight=None.
20-dim = 6 OOF logprobs + 14 dist meta features.

Different from 2026-04-28 minimal meta NULL because:
  - That used XGB on (LB-3-stack + macrorec_base) with macrorec untested.
  - Here both inputs are LB-positive AND model class is bagging not boosting.
  - Bagging preserves natural-cal vs gradient boosting compounds it.

4-gate test against v1 LB-best.

Usage:
  SMOKE=1 python scripts/path5_l3_rf_minimal.py    # 2-fold smoke
  python scripts/path5_l3_rf_minimal.py            # 5-fold production
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
from common import add_distance_features, log_blend, tune_log_bias

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42

SMOKE = os.environ.get("SMOKE") == "1"

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


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr = len(train)
    n_te = len(test)

    # Two LB-validated naturally-calibrated inputs
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)

    # Test predictions (assume v1 test exists; if not, try alternative)
    v1_test_p = ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy"
    if v1_test_p.exists():
        v1_test = np.load(v1_test_p).astype(np.float32)
    else:
        log(f"FATAL: {v1_test_p} missing")
        return
    raw_test = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)

    log(f"v1 OOF shape={v1_oof.shape} test shape={v1_test.shape}")
    log(f"raw OOF shape={raw_oof.shape} test shape={raw_test.shape}")

    v1_oof = _normed(v1_oof)
    v1_test = _normed(v1_test)
    raw_oof = _normed(raw_oof)
    raw_test = _normed(raw_test)

    # Distance features
    log("constructing distance features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    X_tr = np.concatenate([safelog(v1_oof), safelog(raw_oof), meta_tr], axis=1).astype(np.float32)
    X_te = np.concatenate([safelog(v1_test), safelog(raw_test), meta_te], axis=1).astype(np.float32)
    log(f"feature matrix: train={X_tr.shape}  test={X_te.shape}")

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    if SMOKE:
        n_folds, n_est, max_depth = 2, 100, 6
    else:
        n_folds, n_est, max_depth = 5, 300, 8
    log(f"RF: n_est={n_est} max_depth={max_depth} n_folds={n_folds}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y))
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            min_samples_leaf=20, max_features="sqrt",
            bootstrap=True, n_jobs=-1, random_state=SEED,
            class_weight=None, verbose=0,
        )
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold}/{n_folds} argmax={bal:.5f} wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"L3 OOF argmax={overall:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")
    drift = (bias - (-np.log(prior))).round(4).tolist()
    log(f"L3 drift = {drift}  (max |drift| = {max(abs(d) for d in drift):.3f})")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr_l3 = per_class_recall(y, pred_at_bias)
    log(f"L3 PCR=[L={pcr_l3[0]:.4f} M={pcr_l3[1]:.4f} H={pcr_l3[2]:.4f}]")

    np.save(ART / "oof_path5_l3_rf_minimal.npy", oof)
    np.save(ART / "test_path5_l3_rf_minimal.npy", test_pred)

    # Compare vs v1 LB-best
    v1_overall = balanced_accuracy_score(y, v1_oof.argmax(1))
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    v1_pred = (safelog(v1_oof) + v1_bias).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred)
    log(f"V1 LB-best: argmax={v1_overall:.5f} tuned={v1_tuned:.5f} bias={v1_bias.round(4).tolist()}")
    log(f"V1 PCR=[L={v1_pcr[0]:.4f} M={v1_pcr[1]:.4f} H={v1_pcr[2]:.4f}]")

    delta_tuned = float(tuned - v1_tuned)
    delta_pcr = (pcr_l3 - v1_pcr).round(5).tolist()
    log(f"L3 vs V1: Δ tuned = {delta_tuned:+.5f}  Δ PCR = {delta_pcr}")

    # 4-gate test (L3 as candidate, v1 as anchor)
    gate_results = run_4gate(oof, test_pred, v1_oof, v1_test, y, test_ids, prior)

    # Standalone L3 submission emit (always saves; user decides whether to LB-probe)
    sub_path = SUB / "submission_path5_l3_rf_minimal_standalone.csv"
    pred = (safelog(oof) + bias).argmax(1)  # standalone OOF prediction at L3's tuned bias
    pred_test = (safelog(test_pred) + bias).argmax(1)
    pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in pred_test]}).to_csv(sub_path, index=False)

    n_diff_v1 = int((pred_test != (safelog(v1_test) + v1_bias).argmax(1)).sum())
    log(f"  standalone test diff vs v1: {n_diff_v1} rows")
    log(f"  emitted {sub_path}")

    summary = dict(
        method="path5_l3_rf_minimal",
        n_folds=n_folds, n_est=n_est, max_depth=max_depth,
        l3_argmax=float(overall),
        l3_tuned=float(tuned),
        l3_bias=bias.tolist(),
        l3_drift=drift,
        l3_pcr=pcr_l3.tolist(),
        v1_tuned=float(v1_tuned),
        delta_tuned=delta_tuned,
        delta_pcr=delta_pcr,
        n_diff_v1_test=n_diff_v1,
        gate=gate_results,
    )
    out_p = ART / "path5_l3_rf_minimal_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out_p}")


def run_4gate(l3_oof, l3_test, v1_oof, v1_test, y, test_ids, prior):
    """Test L3 as candidate vs v1 PRIMARY anchor."""
    log("=== 4-gate filter: L3 as candidate, anchor = v1 LB-best ===")
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    v1_pred_oof = (safelog(v1_oof) + v1_bias).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred_oof)
    v1_test_pred = (safelog(v1_test) + v1_bias).argmax(1)

    sweep = []
    for alpha in [0.10, 0.20, 0.30, 0.40, 0.50]:
        blend_oof = log_blend([v1_oof, l3_oof], np.array([1.0 - alpha, alpha]))
        oof_pred = (safelog(blend_oof) + v1_bias).argmax(1)
        bal = balanced_accuracy_score(y, oof_pred)
        pcr = per_class_recall(y, oof_pred)
        delta = float(bal - v1_tuned)
        d_class = (pcr - v1_pcr).tolist()

        blend_test = log_blend([v1_test, l3_test], np.array([1.0 - alpha, alpha]))
        b_test_pred = (safelog(blend_test) + v1_bias).argmax(1)
        add_h = int(((b_test_pred == 2) & (v1_test_pred != 2)).sum())
        rem_h = int(((b_test_pred != 2) & (v1_test_pred == 2)).sum())
        net_h = add_h - rem_h
        churn = add_h + rem_h
        ratio = abs(net_h) / max(1, churn)

        g1 = delta >= 2e-4
        g2 = all(d >= -5e-4 for d in d_class)
        g4 = (net_h > 0) and (ratio >= 0.5)

        sweep.append(dict(
            alpha=alpha, delta=delta, pcr_delta=d_class,
            net_h=net_h, churn=churn, ratio=ratio,
            G1=bool(g1), G2=bool(g2), G4=bool(g4),
            all_pass=bool(g1 and g2 and g4),
        ))
        log(f"  α={alpha:.2f} Δ={delta:+.5f} PCR=[L{d_class[0]:+.5f} M{d_class[1]:+.5f} H{d_class[2]:+.5f}] "
            f"net_H={net_h:+d}/churn={churn} ratio={ratio:.3f}  G1={g1} G2={g2} G4={g4}")

    passing = [s for s in sweep if s["all_pass"]]
    log(f"=== {len(passing)}/{len(sweep)} alphas pass all 3 gates ===")
    if passing:
        best = max(passing, key=lambda s: s["delta"])
        log(f"BEST gate-pass α={best['alpha']:.2f} Δ={best['delta']:+.5f}")
        alpha = best["alpha"]
        blend_test = log_blend([v1_test, l3_test], np.array([1.0 - alpha, alpha]))
        b_pred = (safelog(blend_test) + v1_bias).argmax(1)
        sub_path = SUB / f"submission_path5_l3_blend_v1_a{int(alpha*100):03d}.csv"
        pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in b_pred]}).to_csv(sub_path, index=False)
        log(f"  emitted {sub_path}")
    return dict(sweep=sweep, n_passing=len(passing))


if __name__ == "__main__":
    main()
