"""RF natural meta with natural-cal RealMLP added — 8-component bank-extension test.

Direct extension of the LB-validated sklearn_rf_meta_natural.py (LB 0.98129)
with realmlp_natural added as 8th input component. Tests whether the H3-PASS
natural-cal RealMLP (Jaccard 0.66 vs RF natural, errs +5.1% vs +27% for
rawashishsin in the same bank) contributes positive marginal signal at the
meta-stacker level — the bank-add path the calibration-transfer hypothesis
predicted.

LB-validated bank (7 components, produced LB 0.98129):
  - rawashishsin_2600
  - recipe_full_te_catboost_natural
  - recipe_full_te_catboost
  - recipe_full_te
  - realmlp                              ← baseline RealMLP n_ens=1
  - xgb_corn
  - xgb_dist_digits

This experiment adds:
  - realmlp_natural                       ← natural-cal RealMLP (this branch)

Same XGB HPs as the LB-validated run:
  - RandomForestClassifier(n_estimators=500, max_depth=12,
                           class_weight=None, bootstrap=True,
                           min_samples_leaf=20, max_features='sqrt')
  - 5-fold StratifiedKFold(seed=42)
  - 14 dist features + 8 components × 3 cls = 38-feature input

Outputs:
  scripts/artifacts/oof_sklearn_rf_meta_natural_plus_natrealmlp.npy
  scripts/artifacts/test_sklearn_rf_meta_natural_plus_natrealmlp.npy
  scripts/artifacts/sklearn_rf_meta_natural_plus_natrealmlp_results.json
  scripts/artifacts/blend_gate_rf_natural_plus_natrealmlp_results.json
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
from common import add_distance_features, log_blend, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42
SUFFIX = "_plus_natrealmlp"

SMOKE = os.environ.get("SMOKE") == "1"

# 7-component LB-validated bank + 8th natural-cal RealMLP
NATURAL_BANK = [
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
    "realmlp_natural",  # ← NEW: 8th component
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


def load_bank(y, n_tr):
    log(f"loading natural-cal bank ({len(NATURAL_BANK)} components)")
    pool = {}
    for name in NATURAL_BANK:
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
    log(f"  loaded {len(pool)}/{len(NATURAL_BANK)} components")
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
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    pool = load_bank(y, n_tr)
    if "realmlp_natural" not in pool:
        log("ERROR: realmlp_natural missing — abort")
        return
    if len(pool) < 8:
        log(f"WARN: only {len(pool)}/8 components loaded")

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
    log(f"RF: n_est={n_est} max_depth={max_depth} class_weight=None bootstrap=True "
        f"n_folds={n_folds} (LB-validated config + 8th component)")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr_s, y))
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"=== fold {fold}/{n_folds}  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr_s[tr_idx], y[tr_idx])
        p_va = rf.predict_proba(X_tr_s[va_idx]).astype(np.float32)
        p_te = rf.predict_proba(X_te_s).astype(np.float32)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    drift = bias - (-np.log(prior))
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}")
    log(f"  bias  = {bias.round(4).tolist()}")
    log(f"  drift = {drift.round(4).tolist()}  max|drift|={float(np.abs(drift).max()):.4f}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    np.save(ART / f"oof_sklearn_rf_meta_natural{SUFFIX}.npy", oof)
    np.save(ART / f"test_sklearn_rf_meta_natural{SUFFIX}.npy", test_pred)

    # Compare directly to LB-validated RF natural meta
    rf_lb_oof = np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32)
    rf_lb_test = np.load(ART / "test_sklearn_rf_meta_natural.npy").astype(np.float32)
    bias_lb, tuned_lb = tune_log_bias(rf_lb_oof, y, prior)
    pcr_lb = per_class_recall(y, (safelog(rf_lb_oof) + bias_lb).argmax(1))
    delta_tuned = tuned - tuned_lb
    log(f"  vs LB-VALIDATED RF natural (LB 0.98129):")
    log(f"    LB-validated tuned = {tuned_lb:.5f}  bias = {bias_lb.round(3).tolist()}")
    log(f"    THIS RUN     tuned = {tuned:.5f}  delta = {delta_tuned:+.5f}")
    log(f"    PCR delta vs LB-validated: L={pcr[0]-pcr_lb[0]:+.5f}  "
        f"M={pcr[1]-pcr_lb[1]:+.5f}  H={pcr[2]-pcr_lb[2]:+.5f}")

    # Test-side disagreement
    pred_test = (safelog(test_pred) + bias).argmax(1)
    pred_test_lb = (safelog(rf_lb_test) + bias_lb).argmax(1)
    n_diff = int((pred_test != pred_test_lb).sum())
    log(f"  test rows differ from LB-validated: {n_diff} / {n_te}")

    # 4-gate verdict
    g1 = bool(delta_tuned >= 2e-4)
    g2 = bool(all((pcr - pcr_lb) >= -5e-4))
    # G4 net-rare-class direction: H flips between LB-validated and this run
    add_h = int(((pred_test_lb != 2) & (pred_test == 2)).sum())
    rem_h = int(((pred_test_lb == 2) & (pred_test != 2)).sum())
    net_h = add_h - rem_h
    churn_h = add_h + rem_h
    asym_ratio = abs(net_h) / max(churn_h, 1)
    g4 = bool(net_h > 0 and asym_ratio >= 0.5)
    log(f"  4-gate vs LB-validated: G1(delta>=+2e-4)={'PASS' if g1 else 'FAIL'} "
        f"G2(pcr_min>=-5e-4)={'PASS' if g2 else 'FAIL'} "
        f"G4(asym ADD-H >=0.5)={'PASS' if g4 else 'FAIL'} "
        f"(net_H={net_h:+d}, churn_H={churn_h}, ratio={asym_ratio:.2f})")
    overall_pass = g1 and g2 and g4
    log(f"  OVERALL: {'PASS — recommend LB probe' if overall_pass else 'FAIL'}")

    # Emit submission for inspection (does NOT submit per CLAUDE.md rule)
    sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in pred_test]})
    sub_path = SUB / f"submission_sklearn_rf_meta_natural{SUFFIX}_standalone.csv"
    sub.to_csv(sub_path, index=False)
    log(f"  wrote {sub_path}")

    summary = dict(
        n_folds=n_folds, smoke=SMOKE, seed=SEED,
        n_estimators=n_est, max_depth=max_depth,
        bank=NATURAL_BANK, bank_loaded=sorted(pool.keys()),
        feature_count=len(feature_names),
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        drift=drift.tolist(),
        max_abs_drift=float(np.abs(drift).max()),
        per_class_recall=pcr.tolist(),
        # vs LB-validated comparison
        lb_validated_tuned=float(tuned_lb),
        delta_tuned_vs_lb_validated=float(delta_tuned),
        pcr_delta_vs_lb_validated=(pcr - pcr_lb).tolist(),
        test_rows_differ_from_lb_validated=n_diff,
        net_h_flip=net_h,
        churn_h=churn_h,
        asym_ratio=float(asym_ratio),
        g1_delta_pass=g1, g2_pcr_pass=g2, g4_direction_pass=g4,
        overall_pass=overall_pass,
        submission_csv=str(sub_path),
    )
    out_p = ART / f"sklearn_rf_meta_natural{SUFFIX}_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
