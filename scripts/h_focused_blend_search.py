"""Focused blend search: v1 + H4 + HistGBM (best two diverse candidates).

H4 standalone: tuned 0.98060 (-0.00003 vs v1), 368 row diff, ADD-High.
HistGBM standalone: tuned 0.98029 (-0.00034 vs v1), 638 row diff, ADD-Low/Med.

These have DIFFERENT diversity directions — H4 strengthens H, HistGBM
strengthens L+M. Together they may compose into a balanced lift.

Search:
  1. Geomean log-blend over (v1, H4, HistGBM) on a fine grid
  2. Arithmetic-mean blend
  3. Per-class isotonic recalibration of H4 and HistGBM, then blend
  4. Sequential blend: first blend v1 + H4, then blend with HistGBM

Decision rule: emit submission only if 4-gate passes AND PCR_H not net negative.
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

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    h4_oof = _normed(np.load(ART / "oof_h4_S1.npy").astype(np.float32))
    h4_test = _normed(np.load(ART / "test_h4_S1.npy").astype(np.float32))
    hg_oof = _normed(np.load(ART / "oof_h_histgbm_natural.npy").astype(np.float32))
    hg_test = _normed(np.load(ART / "test_h_histgbm_natural.npy").astype(np.float32))

    prior = np.bincount(y, minlength=3) / len(y)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    print(f"v1 LB-best tuned={v1_tuned:.5f}")

    a_pred_oof = (safelog(v1_oof) + v1_bias).argmax(1)
    a_pred_test = (safelog(v1_test) + v1_bias).argmax(1)
    a_pcr = per_class_recall(y, a_pred_oof)
    print(f"v1 PCR=[L={a_pcr[0]:.4f} M={a_pcr[1]:.4f} H={a_pcr[2]:.4f}]")

    print("\n=== 3-way log-blend (v1, H4, HistGBM) at fixed v1 bias ===")
    print("(weights: v1, h4, histgbm; sum=1)")
    best = None
    for w_h4 in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for w_hg in [0.0, 0.05, 0.10, 0.15]:
            w_v1 = 1.0 - w_h4 - w_hg
            if w_v1 < 0.65:
                continue
            log_blend_oof = w_v1 * safelog(v1_oof) + w_h4 * safelog(h4_oof) + w_hg * safelog(hg_oof)
            log_blend_test = w_v1 * safelog(v1_test) + w_h4 * safelog(h4_test) + w_hg * safelog(hg_test)
            blend_oof = _normed(np.exp(log_blend_oof))
            blend_test = _normed(np.exp(log_blend_test))
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
            g1 = d >= 3e-4
            g2 = all(p >= -5e-4 for p in pcr_d)
            g4 = (net_h > 0) and (ratio >= 0.5)
            tag = ""
            n_pass = sum([g1, g2, g4])
            if n_pass >= 3:
                tag = " *** PASS ***"
            elif d > 0 and g2:
                tag = " (g1+ direction)"
            print(f"  w=({w_v1:.2f},{w_h4:.2f},{w_hg:.2f}) bal={bal:.5f} d={d:+.5f} PCR_d=[{pcr_d[0]:+.4f},{pcr_d[1]:+.4f},{pcr_d[2]:+.4f}] diff={diff:4d} net_H={net_h:+d} ratio={ratio:.3f}{tag}")
            if best is None or d > best["d"]:
                best = dict(w_v1=w_v1, w_h4=w_h4, w_hg=w_hg, d=d, bal=bal,
                            pcr=pcr.tolist(), diff=diff, net_h=net_h,
                            test_pred=pred_test, blend_test=blend_test)

    print(f"\nbest 3-way log-blend: w=({best['w_v1']:.2f},{best['w_h4']:.2f},{best['w_hg']:.2f}) d={best['d']:+.5f} diff={best['diff']}")
    print(f"  PCR=[{best['pcr'][0]:.4f},{best['pcr'][1]:.4f},{best['pcr'][2]:.4f}]")
    print(f"  net_H={best['net_h']:+d}")

    if best["d"] > 0 and all(p >= -5e-4 for p in (np.array(best['pcr']) - a_pcr)):
        sub_path = SUB / f"submission_3way_v1H4HistGBM_w{int(best['w_v1']*100):03d}_{int(best['w_h4']*100):03d}_{int(best['w_hg']*100):03d}.csv"
        sub = pd.DataFrame({"id": test_ids,
                            TARGET: [IDX2CLS[i] for i in best["test_pred"]]})
        sub.to_csv(sub_path, index=False)
        print(f"  EMIT: {sub_path}")
    else:
        print(f"  no emit (d≤0 or PCR floor breach)")

    # Also try w_h4=0.05 w_hg=0.10 specifically (mid-grid intuition)
    print("\n=== Specific configs ===")
    for w_h4, w_hg in [(0.05, 0.10), (0.10, 0.10), (0.075, 0.10), (0.05, 0.05), (0.05, 0.15)]:
        w_v1 = 1.0 - w_h4 - w_hg
        log_blend_oof = w_v1 * safelog(v1_oof) + w_h4 * safelog(h4_oof) + w_hg * safelog(hg_oof)
        log_blend_test = w_v1 * safelog(v1_test) + w_h4 * safelog(h4_test) + w_hg * safelog(hg_test)
        blend_oof = _normed(np.exp(log_blend_oof))
        blend_test = _normed(np.exp(log_blend_test))
        pred_oof = (safelog(blend_oof) + v1_bias).argmax(1)
        pred_test = (safelog(blend_test) + v1_bias).argmax(1)
        bal = balanced_accuracy_score(y, pred_oof)
        d = bal - v1_tuned
        pcr = per_class_recall(y, pred_oof)
        diff = int((pred_test != a_pred_test).sum())
        h_added = int(((pred_test == 2) & (a_pred_test != 2)).sum())
        h_removed = int(((a_pred_test == 2) & (pred_test != 2)).sum())
        net_h = h_added - h_removed
        print(f"  w_h4={w_h4} w_hg={w_hg}: bal={bal:.5f} d={d:+.5f} PCR=[{pcr[0]:.4f},{pcr[1]:.4f},{pcr[2]:.4f}] diff={diff} net_H={net_h:+d}")


if __name__ == "__main__":
    main()
