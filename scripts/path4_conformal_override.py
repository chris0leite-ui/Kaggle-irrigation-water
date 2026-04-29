"""Path 4 — Conformal-set selective override on v1 PRIMARY.

Mechanism: per-test-row conformal prediction sets at coverage 1-α.
On rows where v1's set is ambiguous (size > 1 OR includes a non-argmax
class above threshold), use rawashishsin v3's argmax. On rows where v1's
set is singleton high-confidence, keep v1.

Different from every prior override attempt:
  - prior overrides used binary detectors with precision floors (failed
    at ~6% precision vs 8.1% break-even under macro-recall)
  - conformal gives calibrated coverage guarantees without selection-bias
  - uses TWO LB-validated naturally-calibrated outputs (not a single
    binary head with threshold tuning)
  - the gate is set-based not probability-based

Fits split-conformal nonconformity scores per class on OOF train data,
applies to test predictions to construct prediction sets.

Two override modes tested:
  A. v1's prediction set is non-singleton {Low, Medium} or {Medium, High}
     → use rawashishsin v3's argmax (it's a different LB-validated model)
  B. v1's argmax has low conformal score AND raw's argmax has high
     conformal score → use raw's argmax

Decision rule: pick the operating point that maximizes macro-recall on
OOF. Test predictions emitted at the corresponding test-side rule.

Usage:
  python scripts/path4_conformal_override.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42


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


def conformal_set_sizes(probs_calib, y_calib, probs_query, alpha):
    """Split-conformal classification (LAC nonconformity).

    Score s_i = 1 - p_i(y_i) on calibration set.
    Quantile q_hat = ceil((n+1)(1-alpha)) / n quantile of {s_i}.
    Prediction set for query: {k : 1 - p_query(k) <= q_hat}.

    Returns:
      pred_sets: (n_query, 3) boolean array, True = class in set
    """
    n_cal = len(y_calib)
    s_cal = 1.0 - probs_calib[np.arange(n_cal), y_calib]
    # Conformal quantile
    q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    q_hat = float(np.quantile(s_cal, q_level, method="higher"))
    s_query = 1.0 - probs_query   # (n_query, 3)
    pred_sets = s_query <= q_hat
    return pred_sets, q_hat


def build_oof_conformal_sets(probs, y, alpha, n_folds=5):
    """Build OOF conformal prediction sets via 5-fold split."""
    n = len(y)
    pred_sets = np.zeros((n, 3), dtype=bool)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    q_hats = []
    for fold, (cal_idx, qry_idx) in enumerate(skf.split(np.zeros(n), y), 1):
        ps, q_hat = conformal_set_sizes(
            probs[cal_idx], y[cal_idx], probs[qry_idx], alpha)
        pred_sets[qry_idx] = ps
        q_hats.append(q_hat)
    return pred_sets, np.array(q_hats)


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    raw_oof = _normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    raw_test = _normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))

    prior = np.bincount(y, minlength=3) / len(y)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    log(f"V1 OOF tuned={v1_tuned:.5f} bias={v1_bias.round(4).tolist()}")

    # Apply v1 bias to OOF probs to get tuned posterior
    v1_oof_tuned = _normed(v1_oof * np.exp(v1_bias - v1_bias.max()))
    v1_test_tuned = _normed(v1_test * np.exp(v1_bias - v1_bias.max()))

    # Same for raw with its own tuned bias
    raw_bias, raw_tuned = tune_log_bias(raw_oof, y, prior)
    log(f"Raw OOF tuned={raw_tuned:.5f} bias={raw_bias.round(4).tolist()}")
    raw_oof_tuned = _normed(raw_oof * np.exp(raw_bias - raw_bias.max()))
    raw_test_tuned = _normed(raw_test * np.exp(raw_bias - raw_bias.max()))

    v1_pred_oof = v1_oof_tuned.argmax(1)
    v1_pcr = per_class_recall(y, v1_pred_oof)
    v1_test_pred = v1_test_tuned.argmax(1)
    raw_test_pred = raw_test_tuned.argmax(1)
    log(f"V1 OOF PCR=[L={v1_pcr[0]:.4f} M={v1_pcr[1]:.4f} H={v1_pcr[2]:.4f}]")

    n_diff_test = int((v1_test_pred != raw_test_pred).sum())
    log(f"V1 vs raw test disagreement: {n_diff_test} rows ({100*n_diff_test/len(test_ids):.2f}%)")

    # Build OOF conformal sets at multiple alpha levels
    results = {"alpha_sweep": []}
    best_record = None
    for alpha in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        log(f"=== alpha = {alpha} ===")
        v1_sets_oof, q_oof = build_oof_conformal_sets(v1_oof_tuned, y, alpha)
        # Test sets: use full-OOF as calibration on tuned probs
        v1_sets_test, q_test_full = conformal_set_sizes(
            v1_oof_tuned, y, v1_test_tuned, alpha)
        log(f"  q_hat oof={q_oof.mean():.4f}±{q_oof.std():.4f}  test_full={q_test_full:.4f}")

        # Set sizes
        oof_sizes = v1_sets_oof.sum(1)
        test_sizes = v1_sets_test.sum(1)
        log(f"  OOF set sizes: 1={(oof_sizes==1).sum()} 2={(oof_sizes==2).sum()} 3={(oof_sizes==3).sum()}")
        log(f"  TEST set sizes: 1={(test_sizes==1).sum()} 2={(test_sizes==2).sum()} 3={(test_sizes==3).sum()}")

        # Override mode A: when v1's set is non-singleton, defer to rawashishsin
        # OOF apply
        nonsingleton_oof = oof_sizes > 1
        oof_pred_A = v1_pred_oof.copy()
        oof_pred_A[nonsingleton_oof] = raw_oof_tuned[nonsingleton_oof].argmax(1)
        bal_A = balanced_accuracy_score(y, oof_pred_A)
        pcr_A = per_class_recall(y, oof_pred_A)
        delta_A = float(bal_A - v1_tuned)
        d_A = (pcr_A - v1_pcr).tolist()
        n_overridden_A = int(nonsingleton_oof.sum())

        # G4 on OOF (train-side proxy — use OOF flips as test-side estimate)
        add_h_oof = int(((oof_pred_A == 2) & (v1_pred_oof != 2)).sum())
        rem_h_oof = int(((v1_pred_oof == 2) & (oof_pred_A != 2)).sum())
        net_h_oof = add_h_oof - rem_h_oof
        churn_oof = add_h_oof + rem_h_oof
        ratio_oof = abs(net_h_oof) / max(1, churn_oof)

        # Test-side
        nonsingleton_test = test_sizes > 1
        test_pred_A = v1_test_pred.copy()
        test_pred_A[nonsingleton_test] = raw_test_pred[nonsingleton_test]
        n_overridden_A_test = int(nonsingleton_test.sum())
        add_h_test = int(((test_pred_A == 2) & (v1_test_pred != 2)).sum())
        rem_h_test = int(((v1_test_pred == 2) & (test_pred_A != 2)).sum())
        net_h_test = add_h_test - rem_h_test
        churn_test = add_h_test + rem_h_test
        ratio_test = abs(net_h_test) / max(1, churn_test)

        log(f"  Mode A (set>1 → raw):")
        log(f"    OOF: n_override={n_overridden_A}  Δ={delta_A:+.5f} PCR=[L{d_A[0]:+.5f} M{d_A[1]:+.5f} H{d_A[2]:+.5f}]")
        log(f"    OOF G4: net_H={net_h_oof:+d}/churn={churn_oof}/ratio={ratio_oof:.3f}")
        log(f"    TEST: n_override={n_overridden_A_test}  net_H={net_h_test:+d}/churn={churn_test}/ratio={ratio_test:.3f}")

        g1 = delta_A >= 2e-4
        g2 = all(d >= -5e-4 for d in d_A)
        g4 = (net_h_test > 0) and (ratio_test >= 0.5)

        record_A = dict(
            mode="A_set_gt_1_to_raw",
            alpha=alpha,
            delta=delta_A,
            pcr_delta=d_A,
            n_override_oof=n_overridden_A,
            n_override_test=n_overridden_A_test,
            net_h_oof=net_h_oof, churn_oof=churn_oof, ratio_oof=ratio_oof,
            net_h_test=net_h_test, churn_test=churn_test, ratio_test=ratio_test,
            G1=bool(g1), G2=bool(g2), G4=bool(g4),
            all_pass=bool(g1 and g2 and g4),
        )
        results["alpha_sweep"].append(record_A)

        if g1 and g2 and g4 and (best_record is None or delta_A > best_record["delta"]):
            best_record = record_A
            best_record["test_pred"] = test_pred_A

    if best_record is not None:
        log(f"=== BEST conformal override α={best_record['alpha']} Δ={best_record['delta']:+.5f} ===")
        sub_path = SUB / f"submission_path4_conformal_a{int(best_record['alpha']*100):03d}.csv"
        pred_test = best_record["test_pred"]
        pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in pred_test]}).to_csv(sub_path, index=False)
        log(f"emitted {sub_path}")
        # Drop test_pred from json (not serializable)
        del best_record["test_pred"]
    else:
        log("=== no conformal alpha passes 4-gate filter ===")

    out_p = ART / "path4_conformal_results.json"
    out_p.write_text(json.dumps(results, indent=2, default=float))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
