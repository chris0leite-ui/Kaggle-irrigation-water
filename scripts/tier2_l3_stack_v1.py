"""Tier 2 L3 stack on v1 RF natural OOF (LB 0.98129).

Train a small XGB on:
  - v1 OOF probs (3 cls)
  - 14 dist meta-features (sm_dist..ws_abs + min_axis_abs +
    score_dist_low_mid + score_dist_mid_high + dgp_score + rule_pred)
  → 17 input dims

Heavy-reg natural-cal regime (mirrors LB-best primary's bias profile):
  max_depth=3, lr=0.05, n_est=2600, no L1/L2 reg, class_weight=None,
  ORIG_ROW_WEIGHT=0.5 weighting on the 10k original rows when concatenated.

Hypothesis: the L2 RF natural meta on natural-cal bank produced LB lift
+0.00020 over the strongest base (rawashishsin). An L3 small XGB on top
of v1's per-row probs may compound monotonically because v1's predictions
are already at natural calibration ([drift -0.10, -0.10, -0.20]).

Decision: emit candidate IF
  - tuned OOF Δ ≥ +2e-4 vs v1 standalone 0.98063
  - PCR drift ≤ -5e-4 floor each class
  - net_H > 0 AND |asymmetry| ≥ 0.5
  - bias drift ≤ |0.30| each class (natural-cal property preserved)

Falls back to NULL if any gate fails — L3 distillation overfit risk
documented in CLAUDE.md (soft-distill family closed at +0.00201 LB
regression at depth=3).
"""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42

SMOKE = os.environ.get("SMOKE") == "1"

DIST_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]


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


def normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def main():
    log("loading inputs")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)
    prior = np.bincount(y, minlength=3) / len(y)

    # Load v1 RF natural OOF + test (LB-validated)
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_oof = normed(v1_oof)
    v1_test = normed(v1_test)

    # Sanity: reproduce v1 standalone metric
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    v1_pred = (safelog(v1_oof) + v1_bias).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred)
    log(f"ANCHOR v1: tuned={v1_tuned:.5f} bias={v1_bias.round(4).tolist()} "
        f"PCR=[L={v1_pcr[0]:.4f} M={v1_pcr[1]:.4f} H={v1_pcr[2]:.4f}]")

    # Build dist features
    log("constructing distance / rule meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[DIST_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[DIST_COLS].to_numpy(dtype=np.float32)

    # Concatenate: v1 log-probs (3) + dist features (14) = 17 dims
    X_tr = np.concatenate([safelog(v1_oof), meta_tr], axis=1).astype(np.float32)
    X_te = np.concatenate([safelog(v1_test), meta_te], axis=1).astype(np.float32)
    log(f"  feature matrix: train={X_tr.shape}  test={X_te.shape}")

    # XGB heavy-reg natural-cal regime
    n_iter = 200 if SMOKE else 2600
    n_folds = 2 if SMOKE else 5
    xgb_params = dict(
        objective="multi:softprob",
        num_class=3,
        eta=0.05,
        max_depth=3,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        max_bin=1100,
        tree_method="hist",
        reg_alpha=0.0,
        reg_lambda=0.0,
        seed=SEED,
        verbosity=0,
    )
    log(f"XGB L3 natural: max_iter={n_iter} max_depth=3 lr=0.05 "
        f"l1=l2=0 class_weight=None ORIG_ROW_WEIGHT=0.5 (no class_w)")

    # Build orig dataset (rule-perfect 10k) features for concat
    arch_path = Path("data/archive.zip")
    orig_X = None
    orig_y = None
    if arch_path.exists():
        with zipfile.ZipFile(arch_path) as zf:
            csv_name = next((n for n in zf.namelist()
                             if n.lower().endswith(".csv")), None)
            if csv_name:
                with zf.open(csv_name) as f:
                    orig = pd.read_csv(f)
                if TARGET in orig.columns:
                    orig_y_lab = orig[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
                    # Compute v1 PSEUDO-prediction on orig (using full-train v1)
                    # Since orig is rule-perfect, we use one-hot label as input proxy
                    orig_v1 = np.zeros((len(orig), 3), dtype=np.float32)
                    orig_v1[np.arange(len(orig)), orig_y_lab] = 0.999
                    orig_v1[orig_v1 == 0] = 0.0005
                    orig_d = add_distance_features(orig)
                    orig_meta = orig_d[DIST_COLS].to_numpy(dtype=np.float32)
                    orig_X = np.concatenate([safelog(orig_v1), orig_meta],
                                             axis=1).astype(np.float32)
                    orig_y = orig_y_lab
                    log(f"  orig: {len(orig)} rows (concat with weight 0.5)")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_tr, y))
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((n_te, 3), dtype=np.float32)
    fold_scores = []
    bias_log = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        log(f"=== fold {fold}/{n_folds}  tr={len(tr_idx):,} va={len(va_idx):,} ===")
        Xtr = X_tr[tr_idx]; ytr = y[tr_idx]
        sw_tr = np.ones(len(tr_idx), dtype=np.float32)
        if orig_X is not None:
            Xtr = np.concatenate([Xtr, orig_X], axis=0)
            ytr = np.concatenate([ytr, orig_y], axis=0)
            sw_orig = np.full(len(orig_y), 0.5, dtype=np.float32)
            sw_tr = np.concatenate([sw_tr, sw_orig], axis=0)
        dtr = xgb.DMatrix(Xtr, label=ytr, weight=sw_tr)
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(xgb_params, dtr, num_boost_round=n_iter,
                            evals=[(dva, "va")],
                            early_stopping_rounds=200,
                            verbose_eval=0)
        p_va = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        p_te = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))
        oof[va_idx] = p_va.astype(np.float32)
        test_pred += p_te.astype(np.float32) / n_folds
        bal = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_scores.append(float(bal))
        log(f"  fold {fold} argmax_bal_acc={bal:.5f} best_iter={booster.best_iteration} wall={time.time()-t0:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    bias, tuned = tune_log_bias(oof, y, prior)
    drift = (bias - (-np.log(prior))).round(3).tolist()
    log(f"=== OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias = {bias.round(4).tolist()}")
    log(f"  bias drift from -log(prior): {drift}")

    pred_at_bias = (safelog(oof) + bias).argmax(1)
    pcr = per_class_recall(y, pred_at_bias)
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # 4-gate vs v1 (anchor) at v1's bias
    pred_at_v1bias = (safelog(oof) + v1_bias).argmax(1)
    bal_at_v1bias = balanced_accuracy_score(y, pred_at_v1bias)
    pcr_at_v1bias = per_class_recall(y, pred_at_v1bias)
    pcr_delta = (pcr_at_v1bias - v1_pcr).round(5).tolist()
    delta = bal_at_v1bias - v1_tuned
    log(f"\n  AT V1 BIAS: bal={bal_at_v1bias:.5f} Δ={delta:+.5f} PCR_d={pcr_delta}")

    # Test side at v1 bias
    test_pred_v1bias = (safelog(test_pred) + v1_bias).argmax(1)
    v1_test_pred_v1bias = (safelog(v1_test) + v1_bias).argmax(1)
    net_h = int(((test_pred_v1bias == 2) & (v1_test_pred_v1bias != 2)).sum() -
                ((v1_test_pred_v1bias == 2) & (test_pred_v1bias != 2)).sum())
    churn_h = int(((test_pred_v1bias == 2) ^ (v1_test_pred_v1bias == 2)).sum())
    diff = int((test_pred_v1bias != v1_test_pred_v1bias).sum())
    g4_ratio = abs(net_h) / max(1, churn_h)
    log(f"  TEST: diff={diff} net_H={net_h:+d} churn_H={churn_h} g4_ratio={g4_ratio:.2f}")

    g1 = delta >= 2e-4
    g2 = all(d >= -5e-4 for d in pcr_delta)
    g4 = (net_h > 0) and (g4_ratio >= 0.5)
    drift_ok = all(abs(d) <= 0.30 for d in drift)
    emit = g1 and g2 and g4 and drift_ok
    log(f"\n  GATES: G1={g1} G2={g2} G4={g4} DRIFT_OK={drift_ok}  EMIT={emit}")

    np.save(ART / "oof_l3_stack_v1.npy", oof)
    np.save(ART / "test_l3_stack_v1.npy", test_pred)

    summary = dict(
        n_folds=n_folds, smoke=SMOKE, seed=SEED,
        max_iter=n_iter,
        feature_count=X_tr.shape[1],
        fold_scores_argmax=fold_scores,
        overall_argmax=float(overall),
        tuned_log_bias=float(tuned),
        log_bias=bias.tolist(),
        bias_drift=drift,
        per_class_recall=pcr.tolist(),
        anchor_v1_tuned=float(v1_tuned),
        anchor_v1_pcr=v1_pcr.tolist(),
        delta_at_v1bias=float(delta),
        pcr_at_v1bias=pcr_at_v1bias.tolist(),
        pcr_delta_at_v1bias=pcr_delta,
        net_h=net_h, churn_h=churn_h, g4_ratio=float(g4_ratio),
        test_diff=diff,
        gates={"g1": bool(g1), "g2": bool(g2), "g4": bool(g4),
               "drift_ok": bool(drift_ok), "emit": bool(emit)},
    )
    with open(ART / "tier2_l3_stack_v1_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {ART}/tier2_l3_stack_v1_results.json")

    if emit:
        sub = pd.DataFrame({"id": test_ids,
                            TARGET: [IDX2CLS[i] for i in test_pred_v1bias]})
        sub_path = SUB / "submission_tier2_l3_stack_v1_at_v1bias.csv"
        sub.to_csv(sub_path, index=False)
        log(f"  ✓ EMIT {sub_path}")
    else:
        log("  no emit (gate failed) — diagnostic only")

    # Also build a "standalone" submission at L3's own tuned bias (diagnostic)
    test_pred_idx = (safelog(test_pred) + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in test_pred_idx]})
    sub_path = SUB / "submission_l3_stack_v1_standalone.csv"
    sub.to_csv(sub_path, index=False)
    log(f"  diagnostic: wrote {sub_path} at L3 own bias")


if __name__ == "__main__":
    main()
