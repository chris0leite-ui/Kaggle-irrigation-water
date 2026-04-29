"""Compare all 5 path outputs vs v1 LB-best and emit decision summary.

Loads any of the 5 candidates that are present:
  - path2: oof_rf_meta_seedbag.npy + test_rf_meta_seedbag.npy
  - path5: oof_path5_l3_rf_minimal.npy + test_path5_l3_rf_minimal.npy
  - t4:    oof_rawashishsin_pseudo.npy + test_rawashishsin_pseudo.npy

Plus:
  - v1 LB-best: oof_sklearn_rf_meta_natural_v1_lb98129.npy
  - rawashishsin v3: oof_rawashishsin_2600.npy (HEDGE)

For each candidate:
  - standalone tuned OOF
  - bias drift from -log(prior)
  - PCR delta vs v1
  - test diff vs v1, with net rare-class flip + asymmetry ratio
  - 4-gate verdict at fixed v1 bias (alpha sweep)
  - LB-probe recommendation
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias

ART = Path("scripts/artifacts")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


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


CANDIDATES = {
    "path2_seedbag": ("oof_rf_meta_seedbag.npy", "test_rf_meta_seedbag.npy"),
    "path5_l3":      ("oof_path5_l3_rf_minimal.npy", "test_path5_l3_rf_minimal.npy"),
    "t4_pseudo":     ("oof_rawashishsin_pseudo.npy", "test_rawashishsin_pseudo.npy"),
}


def main():
    print("=== Loading anchor v1 LB-best ===")
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)
    drift_anchor = -np.log(prior)

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    v1_drift = (v1_bias - drift_anchor).round(3)
    v1_pred_oof = (safelog(v1_oof) + v1_bias).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred_oof)
    v1_test_pred = (safelog(v1_test) + v1_bias).argmax(1)
    print(f"V1 PRIMARY: tuned={v1_tuned:.5f} drift={v1_drift.tolist()} "
          f"PCR=[{v1_pcr[0]:.4f},{v1_pcr[1]:.4f},{v1_pcr[2]:.4f}]")

    summary = {"v1": dict(tuned=float(v1_tuned), drift=v1_drift.tolist(), pcr=v1_pcr.tolist())}

    for name, (oof_p, test_p) in CANDIDATES.items():
        opath = ART / oof_p
        tpath = ART / test_p
        if not opath.exists():
            print(f"\n=== {name}: NOT YET AVAILABLE ===")
            summary[name] = {"available": False}
            continue
        print(f"\n=== {name} ===")
        c_oof = _normed(np.load(opath).astype(np.float32))
        c_test = _normed(np.load(tpath).astype(np.float32))

        c_bias, c_tuned = tune_log_bias(c_oof, y, prior)
        c_drift = (c_bias - drift_anchor).round(3)
        c_pred = (safelog(c_oof) + c_bias).argmax(1)
        c_pcr = per_class_recall(y, c_pred)

        # Standalone test predictions at candidate's own tuned bias
        c_test_pred = (safelog(c_test) + c_bias).argmax(1)
        n_diff = int((c_test_pred != v1_test_pred).sum())
        # Per-class shift in standalone test
        net_h_standalone = int((c_test_pred == 2).sum() - (v1_test_pred == 2).sum())
        net_m_standalone = int((c_test_pred == 1).sum() - (v1_test_pred == 1).sum())
        net_l_standalone = int((c_test_pred == 0).sum() - (v1_test_pred == 0).sum())

        # Asymmetric flip stats
        add_h = int(((c_test_pred == 2) & (v1_test_pred != 2)).sum())
        rem_h = int(((c_test_pred != 2) & (v1_test_pred == 2)).sum())
        net_h = add_h - rem_h
        churn_h = add_h + rem_h
        ratio_h = abs(net_h) / max(1, churn_h)

        delta_tuned = float(c_tuned - v1_tuned)
        delta_pcr = (c_pcr - v1_pcr).tolist()

        print(f"  standalone tuned={c_tuned:.5f}  Δ={delta_tuned:+.5f}")
        print(f"  drift={c_drift.tolist()}  max_drift={float(np.abs(c_drift).max()):.2f}")
        print(f"  PCR=[{c_pcr[0]:.4f},{c_pcr[1]:.4f},{c_pcr[2]:.4f}]")
        print(f"  Δ PCR=[{delta_pcr[0]:+.5f},{delta_pcr[1]:+.5f},{delta_pcr[2]:+.5f}]")
        print(f"  test diff vs v1: {n_diff} rows ({100*n_diff/len(v1_test_pred):.3f}%)")
        print(f"  net flips: L={net_l_standalone:+d} M={net_m_standalone:+d} H={net_h_standalone:+d}")
        print(f"  asymmetric H: add={add_h} rem={rem_h} net={net_h:+d} churn={churn_h} ratio={ratio_h:.3f}")

        # Standalone 4-gate (drift + per-class + direction)
        g_drift = float(np.abs(c_drift).max()) <= 0.40
        g_g1 = delta_tuned >= 2e-4 or (delta_tuned >= -1e-5 and net_h_standalone > 0)
        g_g2 = all(d >= -5e-4 for d in delta_pcr)
        g_g4_dir = (net_h_standalone > 0) and (ratio_h >= 0.5)
        all_pass = g_drift and g_g1 and g_g2 and g_g4_dir
        print(f"  GATES: drift={g_drift} G1={g_g1} G2={g_g2} G4={g_g4_dir}  ALL={all_pass}")

        # Blended sweep at v1 bias for completeness
        sweep = []
        for alpha in [0.10, 0.20, 0.30, 0.40, 0.50]:
            b_oof = log_blend([v1_oof, c_oof], np.array([1 - alpha, alpha]))
            b_pred = (safelog(b_oof) + v1_bias).argmax(1)
            b_bal = balanced_accuracy_score(y, b_pred)
            b_pcr = per_class_recall(y, b_pred)
            b_d = float(b_bal - v1_tuned)
            b_dpcr = (b_pcr - v1_pcr).tolist()
            b_test = log_blend([v1_test, c_test], np.array([1 - alpha, alpha]))
            b_test_pred = (safelog(b_test) + v1_bias).argmax(1)
            ah = int(((b_test_pred == 2) & (v1_test_pred != 2)).sum())
            rh = int(((b_test_pred != 2) & (v1_test_pred == 2)).sum())
            sweep.append({
                "alpha": alpha, "delta": b_d,
                "pcr_delta": b_dpcr,
                "net_h": ah - rh, "churn": ah + rh,
                "ratio": abs(ah - rh) / max(1, ah + rh),
            })

        summary[name] = dict(
            available=True, tuned=float(c_tuned), drift=c_drift.tolist(),
            max_drift=float(np.abs(c_drift).max()),
            pcr=c_pcr.tolist(), delta_pcr=delta_pcr, delta_tuned=delta_tuned,
            n_diff_test=n_diff,
            standalone_net_h=net_h_standalone,
            asym_h_ratio=ratio_h, asym_h_add=add_h, asym_h_rem=rem_h,
            standalone_pass_all=all_pass,
            blend_sweep=sweep,
        )

    print("\n=== DECISION SUMMARY ===")
    for k, v in summary.items():
        if k == "v1" or not v.get("available"):
            print(f"  {k}: {'(anchor)' if k == 'v1' else 'NOT AVAILABLE'}")
            continue
        verdict = "READY-LB-PROBE" if v["standalone_pass_all"] else "GATE-FAIL"
        print(f"  {k}: standalone_pass={v['standalone_pass_all']} "
              f"Δ={v['delta_tuned']:+.5f} max_drift={v['max_drift']:.2f} "
              f"net_H={v['standalone_net_h']:+d} → {verdict}")

    out_p = ART / "path_compare_all_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
