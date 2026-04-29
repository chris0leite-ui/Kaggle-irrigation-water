"""4-way blend: v1 + H4 S1 + H4 S2 + HistGBM.

Comprehensive search over weight simplex. Three diverse candidates:
  H4 S1: ADD-High (368 row diff, tuned -0.00003)
  H4 S2: ADD-Low/High (277 row diff, tuned -0.00009)
  HistGBM: ADD-Med (638 row diff, tuned -0.00034)

Each carries different per-class direction. The right combination
might balance class trades to a net-positive macro-recall lift.

Decision rule: emit submission if 4-gate passes OR strong G2 + G1 +
small ADD-direction net_H.
"""
from __future__ import annotations

import json
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
    s1_oof = _normed(np.load(ART / "oof_h4_S1.npy").astype(np.float32))
    s1_test = _normed(np.load(ART / "test_h4_S1.npy").astype(np.float32))
    s2_oof = _normed(np.load(ART / "oof_h4_S2.npy").astype(np.float32))
    s2_test = _normed(np.load(ART / "test_h4_S2.npy").astype(np.float32))
    hg_oof = _normed(np.load(ART / "oof_h_histgbm_natural.npy").astype(np.float32))
    hg_test = _normed(np.load(ART / "test_h_histgbm_natural.npy").astype(np.float32))

    prior = np.bincount(y, minlength=3) / len(y)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    print(f"v1 tuned={v1_tuned:.5f}  bias={v1_bias.round(4).tolist()}")

    a_pred_oof = (safelog(v1_oof) + v1_bias).argmax(1)
    a_pred_test = (safelog(v1_test) + v1_bias).argmax(1)
    a_pcr = per_class_recall(y, a_pred_oof)
    print(f"v1 PCR=[L={a_pcr[0]:.4f} M={a_pcr[1]:.4f} H={a_pcr[2]:.4f}]")

    print("\n=== 4-way log-blend grid ===")
    print("(v1, S1, S2, HistGBM weights, sum=1)")

    best = None
    all_passes = []
    for w_s1 in [0.0, 0.05, 0.10, 0.15]:
        for w_s2 in [0.0, 0.05, 0.10, 0.15]:
            for w_hg in [0.0, 0.05, 0.10, 0.15]:
                w_v1 = 1.0 - w_s1 - w_s2 - w_hg
                if w_v1 < 0.55:
                    continue
                log_blend_oof = (w_v1 * safelog(v1_oof) + w_s1 * safelog(s1_oof) +
                                 w_s2 * safelog(s2_oof) + w_hg * safelog(hg_oof))
                log_blend_test = (w_v1 * safelog(v1_test) + w_s1 * safelog(s1_test) +
                                  w_s2 * safelog(s2_test) + w_hg * safelog(hg_test))
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
                row = dict(w_v1=w_v1, w_s1=w_s1, w_s2=w_s2, w_hg=w_hg,
                           d=float(d), bal=float(bal), pcr=pcr.tolist(),
                           diff=diff, net_h=net_h, churn=churn, ratio=float(ratio),
                           g1=bool(g1), g2=bool(g2), g4=bool(g4),
                           pcr_d=pcr_d)
                if d > 0 and g2:
                    all_passes.append(row)
                    if best is None or d > best["d"]:
                        best = dict(row, blend_test=blend_test, pred_test=pred_test)

    print(f"\ntotal candidates with d>0 and G2 pass: {len(all_passes)}")
    if best is None:
        print("NO candidate with d>0 + G2 pass found")
        return
    print(f"best: w_v1={best['w_v1']:.2f} w_s1={best['w_s1']:.2f} w_s2={best['w_s2']:.2f} w_hg={best['w_hg']:.2f}")
    print(f"  bal={best['bal']:.5f} d={best['d']:+.5f}")
    print(f"  PCR=[{best['pcr'][0]:.4f},{best['pcr'][1]:.4f},{best['pcr'][2]:.4f}]")
    print(f"  PCR_d=[{best['pcr_d'][0]:+.4f},{best['pcr_d'][1]:+.4f},{best['pcr_d'][2]:+.4f}]")
    print(f"  diff={best['diff']}  net_H={best['net_h']:+d}  ratio={best['ratio']:.3f}")
    print(f"  G1={best['g1']}  G2={best['g2']}  G4={best['g4']}")

    # Print top 10 by OOF delta
    all_passes.sort(key=lambda r: -r["d"])
    print("\nTop 10 by OOF delta (d>0 + G2 pass):")
    for r in all_passes[:10]:
        tag = " *** G1+G4 PASS ***" if (r["g1"] and r["g4"]) else (" g1+g4_partial" if (r["g1"] or r["g4"]) else "")
        print(f"  ({r['w_v1']:.2f},{r['w_s1']:.2f},{r['w_s2']:.2f},{r['w_hg']:.2f}) d={r['d']:+.5f} pcr_d=[{r['pcr_d'][0]:+.4f},{r['pcr_d'][1]:+.4f},{r['pcr_d'][2]:+.4f}] diff={r['diff']:3d} net_H={r['net_h']:+d}{tag}")

    if best["d"] > 0:
        suffix = f"v1{int(best['w_v1']*100):02d}_s1{int(best['w_s1']*100):02d}_s2{int(best['w_s2']*100):02d}_hg{int(best['w_hg']*100):02d}"
        sub_path = SUB / f"submission_4way_best_{suffix}.csv"
        sub = pd.DataFrame({"id": test_ids,
                            TARGET: [IDX2CLS[i] for i in best["pred_test"]]})
        sub.to_csv(sub_path, index=False)
        print(f"\nEMIT: {sub_path}")

    with open(ART / "h_4way_blend_search_results.json", "w") as f:
        json.dump({"best": {k: v for k, v in best.items() if k not in ("blend_test", "pred_test")},
                   "all_passes_top20": all_passes[:20]}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
