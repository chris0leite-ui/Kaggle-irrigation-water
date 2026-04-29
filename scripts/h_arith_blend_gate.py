"""Arithmetic-mean blend (instead of log-mean) on H1, H2, mega-bag vs v1.

Log-blend (geomean) collapses class probabilities multiplicatively;
arithmetic-mean preserves more rare-class mass when components
disagree. May reveal a different operating point than the standard
log-blend gate analyzer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def safelog(p, eps=1e-9): return np.log(np.clip(p, eps, 1.0))
def _normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum() > 0:
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_te = len(test)

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    h1_oof = _normed(np.load(ART / "oof_h1_seedbag_rf.npy").astype(np.float32))
    h1_test = _normed(np.load(ART / "test_h1_seedbag_rf.npy").astype(np.float32))
    h2_oof = _normed(np.load(ART / "oof_h2_et_natural.npy").astype(np.float32))
    h2_test = _normed(np.load(ART / "test_h2_et_natural.npy").astype(np.float32))

    prior = np.bincount(y, minlength=3) / len(y)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    print(f"v1 LB-best tuned={v1_tuned:.5f} bias={v1_bias.round(4).tolist()}")

    a_pred_oof = (safelog(v1_oof) + v1_bias).argmax(1)
    a_pred_test = (safelog(v1_test) + v1_bias).argmax(1)
    a_pcr = per_class_recall(y, a_pred_oof)
    print(f"  v1 PCR=[L={a_pcr[0]:.4f} M={a_pcr[1]:.4f} H={a_pcr[2]:.4f}]")

    print("\n=== arith-mean blend at fixed v1 bias ===")
    for cand_name, c_oof, c_test in [
        ("H1_seedbag", h1_oof, h1_test),
        ("H2_et", h2_oof, h2_test),
    ]:
        print(f"-- {cand_name} --")
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
            blend_oof = _normed((1.0 - alpha) * v1_oof + alpha * c_oof)
            blend_test = _normed((1.0 - alpha) * v1_test + alpha * c_test)
            pred_oof = (safelog(blend_oof) + v1_bias).argmax(1)
            pred_test = (safelog(blend_test) + v1_bias).argmax(1)
            bal = balanced_accuracy_score(y, pred_oof)
            d = bal - v1_tuned
            pcr = per_class_recall(y, pred_oof)
            pcr_d = (pcr - a_pcr).tolist()
            diff = int((pred_test != a_pred_test).sum())
            h_added = int(((pred_test == 2) & (a_pred_test != 2)).sum())
            h_removed = int(((a_pred_test == 2) & (pred_test != 2)).sum())
            net_h = h_added - h_removed
            churn = h_added + h_removed
            ratio = abs(net_h) / max(1, churn)
            print(f"  a={alpha:.2f}: bal={bal:.5f} d={d:+.5f} PCR=[{pcr[0]:.4f},{pcr[1]:.4f},{pcr[2]:.4f}] diff={diff} net_H={net_h:+d} ratio={ratio:.3f}")

    print("\n=== Three-way arith mean: w_v1*v1 + w_h1*h1 + w_h2*h2 ===")
    # Search over a small grid
    best = None
    for w_h1 in [0.0, 0.05, 0.10, 0.15]:
        for w_h2 in [0.0, 0.05, 0.10, 0.15, 0.20]:
            w_v1 = 1.0 - w_h1 - w_h2
            if w_v1 <= 0:
                continue
            blend_oof = _normed(w_v1 * v1_oof + w_h1 * h1_oof + w_h2 * h2_oof)
            blend_test = _normed(w_v1 * v1_test + w_h1 * h1_test + w_h2 * h2_test)
            pred_oof = (safelog(blend_oof) + v1_bias).argmax(1)
            pred_test = (safelog(blend_test) + v1_bias).argmax(1)
            bal = balanced_accuracy_score(y, pred_oof)
            d = bal - v1_tuned
            pcr = per_class_recall(y, pred_oof)
            pcr_d = (pcr - a_pcr).tolist()
            diff = int((pred_test != a_pred_test).sum())
            h_added = int(((pred_test == 2) & (a_pred_test != 2)).sum())
            h_removed = int(((a_pred_test == 2) & (pred_test != 2)).sum())
            net_h = h_added - h_removed
            ratio = abs(net_h) / max(1, h_added + h_removed)
            g1 = d >= 3e-4
            g2 = all(p >= -5e-4 for p in pcr_d)
            g4 = (net_h > 0) and (ratio >= 0.5)
            tag = ""
            if g1 and g2 and g4:
                tag = " *** PASS ***"
            elif g2 and (g4 or g1):
                tag = " (3of4 maybe)"
            print(f"  w=({w_v1:.2f},{w_h1:.2f},{w_h2:.2f}) bal={bal:.5f} d={d:+.5f} PCR_d=[{pcr_d[0]:+.4f},{pcr_d[1]:+.4f},{pcr_d[2]:+.4f}] diff={diff} net_H={net_h:+d} ratio={ratio:.3f}{tag}")
            if best is None or d > best["d"]:
                best = dict(w_v1=w_v1, w_h1=w_h1, w_h2=w_h2, d=d, bal=bal,
                            pcr=pcr.tolist(), diff=diff, net_h=net_h,
                            test_pred=pred_test)

    print(f"\nbest 3-way: w=({best['w_v1']:.2f},{best['w_h1']:.2f},{best['w_h2']:.2f}) d={best['d']:+.5f} diff={best['diff']}")

    # Emit best 3-way candidate
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in best["test_pred"]],
    })
    sub_path = SUB / "submission_arith_3way_best.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  wrote {sub_path}")


if __name__ == "__main__":
    main()
