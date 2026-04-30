"""N3 — Three-way bagging-architecture L3 mean on v1's 7-component bank.

Train 3 structurally distinct bagging metas on v1's EXACT bank:
  1. RandomForest (existing v1 LB-best — reuse on disk OOF)
  2. ExtraTrees(n=500, max_depth=12, max_features='sqrt',
     class_weight=None, bootstrap=True)
  3. BaggingClassifier(LogisticRegression(C=0.1), n_estimators=100,
     max_features=0.7, max_samples=0.8)

L3 = arithmetic mean of the 3 OOFs (NOT log-blend — preserves natural-
cal). Tune log-bias once on L3 OOF.

All three are bagging-based + natural-cal-preserving (`class_weight=None`).
RF/ET tree-based with different randomization (RF samples splits, ET
samples thresholds); BaggingLR architecturally orthogonal (linear).

Expected: ~45 min CPU. 4-gate vs v1 (LB 0.98129).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, fast_bal_acc, tune_log_bias

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}

# v1's exact 7-component bank (from sklearn_rf_meta_natural_results.json
# at commit bbddebb — the LB-validated v1 LB 0.98129 run)
BANK = [
    "rawashishsin_2600",
    "realmlp",
    "recipe_full_te",
    "recipe_full_te_catboost",
    "recipe_full_te_catboost_natural",
    "xgb_corn",
    "xgb_dist_digits",
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


def load_bank(y, n_tr, n_te):
    log(f"loading v1's bank ({len(BANK)} components)")
    pool = {}
    for name in BANK:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            raise FileNotFoundError(f"missing {name}")
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.shape != (n_tr, 3):
            raise ValueError(f"{name} OOF shape {o.shape}")
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    return pool


def build_features(pool, train, test):
    log("building feature matrix (META + per-component log-probs)")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    log_tr = [safelog(pool[n][0]) for n in component_names]
    log_te = [safelog(pool[n][1]) for n in component_names]

    X_tr = np.concatenate([meta_tr] + log_tr, axis=1).astype(np.float32)
    X_te = np.concatenate([meta_te] + log_te, axis=1).astype(np.float32)
    log(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")
    return X_tr, X_te


def train_meta(name, model_factory, X_tr_s, X_te_s, y, splits, n_te):
    """Generic 5-fold trainer for a bagging-based sklearn meta."""
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test = np.zeros((n_te, 3), dtype=np.float32)
    fold_bals = []
    for fi, (tr, va) in enumerate(splits, 1):
        t0 = time.time()
        m = model_factory()
        m.fit(X_tr_s[tr], y[tr])
        p_va = m.predict_proba(X_tr_s[va]).astype(np.float32)
        p_te = m.predict_proba(X_te_s).astype(np.float32)
        oof[va] = p_va
        test += p_te / N_FOLDS
        bal = float(balanced_accuracy_score(y[va], p_va.argmax(1)))
        fold_bals.append(bal)
        log(f"  [{name}] fold {fi}: bal={bal:.5f}  wall={time.time()-t0:.1f}s")
    return oof, test, fold_bals


def main():
    log("loading data")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int64)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)
    prior = np.bincount(y) / len(y)

    pool = load_bank(y, n_tr, n_te)
    X_tr, X_te = build_features(pool, train, test)

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    log(f"  feature matrix: {X_tr_s.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y))

    # ----- Meta 1: existing v1 RF natural (reuse on disk) -----
    log("\n=== Meta 1: RF natural (loading v1 LB-best from disk) ===")
    rf_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    rf_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    rf_bias, rf_score = tune_log_bias(rf_oof, y, prior)
    log(f"  RF tuned={rf_score:.5f}  bias={rf_bias.round(3).tolist()}")
    rf_drift = rf_bias - (-np.log(prior))
    log(f"  RF drift={rf_drift.round(3).tolist()}")

    # ----- Meta 2: ExtraTrees natural -----
    log("\n=== Meta 2: ExtraTrees natural ===")
    def make_et():
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=20,
            max_features="sqrt", bootstrap=True, n_jobs=-1,
            random_state=SEED, class_weight=None,
        )
    et_oof, et_test, et_fold_bals = train_meta("ET", make_et, X_tr_s, X_te_s,
                                                y, splits, n_te)
    et_overall = float(balanced_accuracy_score(y, et_oof.argmax(1)))
    et_bias, et_score = tune_log_bias(et_oof, y, prior)
    et_drift = et_bias - (-np.log(prior))
    log(f"  ET argmax={et_overall:.5f}  tuned={et_score:.5f}  "
        f"bias={et_bias.round(3).tolist()}  drift={et_drift.round(3).tolist()}")

    np.save(ART / "oof_extratrees_natural.npy", et_oof)
    np.save(ART / "test_extratrees_natural.npy", et_test)

    # ----- Meta 3: BaggingClassifier of LogisticRegression -----
    log("\n=== Meta 3: BaggingClassifier(LogReg) natural ===")
    def make_baglr():
        base_lr = LogisticRegression(
            C=0.1, max_iter=500, solver="lbfgs",
            class_weight=None, random_state=SEED, n_jobs=1,
        )
        return BaggingClassifier(
            estimator=base_lr,
            n_estimators=100, max_features=0.7, max_samples=0.8,
            bootstrap=True, bootstrap_features=False,
            n_jobs=-1, random_state=SEED,
        )
    baglr_oof, baglr_test, baglr_fold_bals = train_meta(
        "BagLR", make_baglr, X_tr_s, X_te_s, y, splits, n_te)
    baglr_overall = float(balanced_accuracy_score(y, baglr_oof.argmax(1)))
    baglr_bias, baglr_score = tune_log_bias(baglr_oof, y, prior)
    baglr_drift = baglr_bias - (-np.log(prior))
    log(f"  BagLR argmax={baglr_overall:.5f}  tuned={baglr_score:.5f}  "
        f"bias={baglr_bias.round(3).tolist()}  drift={baglr_drift.round(3).tolist()}")

    np.save(ART / "oof_bagginglr_natural.npy", baglr_oof)
    np.save(ART / "test_bagginglr_natural.npy", baglr_test)

    # ----- L3 arithmetic mean -----
    log("\n=== L3: arithmetic mean of 3 metas ===")
    l3_oof = (rf_oof + et_oof + baglr_oof) / 3.0
    l3_test = (rf_test + et_test + baglr_test) / 3.0
    l3_oof = _normed(l3_oof)
    l3_test = _normed(l3_test)

    l3_overall = float(balanced_accuracy_score(y, l3_oof.argmax(1)))
    l3_bias, l3_score = tune_log_bias(l3_oof, y, prior)
    l3_drift = l3_bias - (-np.log(prior))
    log(f"  L3 argmax={l3_overall:.5f}  tuned={l3_score:.5f}  "
        f"bias={l3_bias.round(3).tolist()}  drift={l3_drift.round(3).tolist()}")

    np.save(ART / "oof_n3_l3_bagging_mean.npy", l3_oof)
    np.save(ART / "test_n3_l3_bagging_mean.npy", l3_test)

    # ----- 4-gate vs v1 -----
    log("\n=== 4-gate vs v1 PRIMARY (LB 0.98129) ===")

    # v1 standalone @ v1's bias = anchor
    v1_pred = (safelog(rf_oof) + rf_bias).argmax(1)
    anchor_bal = fast_bal_acc(y, v1_pred)
    anchor_pcr = per_class_recall(y, v1_pred)
    log(f"  v1 anchor: bal={anchor_bal:.5f}  PCR=[L={anchor_pcr[0]:.5f} "
        f"M={anchor_pcr[1]:.5f} H={anchor_pcr[2]:.5f}]")

    # L3 @ L3 own tuned bias
    l3_pred = (safelog(l3_oof) + l3_bias).argmax(1)
    l3_pcr = per_class_recall(y, l3_pred)
    delta = [l3_pcr[k] - anchor_pcr[k] for k in range(3)]

    # G4 stats
    net_h = int((l3_pred == 2).sum() - (v1_pred == 2).sum())
    add_h = int(((v1_pred != 2) & (l3_pred == 2)).sum())
    rem_h = int(((v1_pred == 2) & (l3_pred != 2)).sum())
    churn = add_h + rem_h
    g4_ratio = abs(net_h) / max(churn, 1)

    delta_bal = float(l3_score - anchor_bal)
    log(f"\n  Δ bal = {delta_bal:+.5f}")
    log(f"  Δ PCR = L={delta[0]:+.5f}  M={delta[1]:+.5f}  H={delta[2]:+.5f}")
    log(f"  net_H={net_h}  add_H={add_h}  rem_H={rem_h}  ratio={g4_ratio:.3f}")

    g1 = delta_bal >= 2e-4
    g2 = all(d >= -5e-4 for d in delta)
    g3 = "n/a (no α-sweep, single L3 mean)"
    g4 = (net_h >= 0) and (g4_ratio >= 0.5)
    log(f"\n  G1 (Δ ≥ +2e-4):       {'PASS' if g1 else 'FAIL'}")
    log(f"  G2 (PCR ≥ -5e-4):      {'PASS' if g2 else 'FAIL'}")
    log(f"  G3:                    {g3}")
    log(f"  G4 (net_H≥0, asym≥0.5): {'PASS' if g4 else 'FAIL'}")
    log(f"  drift gate (|drift|≤0.30 each): "
        f"{'PASS' if max(abs(d) for d in l3_drift) <= 0.30 else 'FAIL'}  "
        f"max={max(abs(d) for d in l3_drift):.3f}")

    overall_pass = g1 and g2 and g4 and max(abs(d) for d in l3_drift) <= 0.30
    log(f"\n  OVERALL: {'PASS — submit candidate to user' if overall_pass else 'FAIL'}")

    # Emit candidate submission for user review (regardless of pass — they decide)
    inv = {0: "Low", 1: "Medium", 2: "High"}
    test_pred_labels = (safelog(l3_test) + l3_bias).argmax(1)
    sub_path = SUB / "submission_n3_l3_bagging_mean.csv"
    pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": [inv[int(c)] for c in test_pred_labels],
    }).to_csv(sub_path, index=False)
    log(f"  wrote candidate {sub_path}")

    # Diff vs v1 PRIMARY
    v1_sub = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")
    v1_lab = v1_sub["Irrigation_Need"].map(CLS_MAP).values
    diff = int((test_pred_labels != v1_lab).sum())
    log(f"  test diff vs v1 PRIMARY: {diff}")

    # ----- Save diagnostic -----
    summary = dict(
        bank=BANK,
        rf_bias=rf_bias.tolist(), rf_drift=rf_drift.tolist(), rf_score=float(rf_score),
        et_bias=et_bias.tolist(), et_drift=et_drift.tolist(), et_score=float(et_score),
        et_fold_bals=et_fold_bals,
        baglr_bias=baglr_bias.tolist(), baglr_drift=baglr_drift.tolist(),
        baglr_score=float(baglr_score), baglr_fold_bals=baglr_fold_bals,
        l3_bias=l3_bias.tolist(), l3_drift=l3_drift.tolist(),
        l3_score=float(l3_score), l3_argmax=l3_overall,
        anchor_bal=float(anchor_bal),
        anchor_pcr=anchor_pcr.tolist(),
        l3_pcr=l3_pcr.tolist(),
        delta_pcr=[float(x) for x in delta],
        delta_bal=delta_bal,
        net_h=net_h, add_h=add_h, rem_h=rem_h, g4_ratio=float(g4_ratio),
        g1_pass=bool(g1), g2_pass=bool(g2), g4_pass=bool(g4),
        drift_pass=bool(max(abs(d) for d in l3_drift) <= 0.30),
        overall_pass=bool(overall_pass),
        test_diff_vs_v1=diff,
        submission_path=str(sub_path),
    )
    out = ART / "n3_l3_bagging_results.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"\n[done] {out}")


if __name__ == "__main__":
    main()
