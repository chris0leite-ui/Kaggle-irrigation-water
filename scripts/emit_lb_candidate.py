"""Emit LB-probe candidate from a completed path.

For each candidate that passes ALL 4 gates (drift, G1, G2, G4) at standalone
or blended level, build the submission CSV at the candidate's optimal
configuration. Print the recommended LB submit command per CLAUDE.md rule
(one shot only, requires user approval).

Usage:
  python scripts/emit_lb_candidate.py path5_l3
  python scripts/emit_lb_candidate.py path2_seedbag
  python scripts/emit_lb_candidate.py t4_pseudo
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
SUB = Path("submissions")
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


def main(name):
    if name not in CANDIDATES:
        print(f"unknown: {name}; one of {list(CANDIDATES.keys())}")
        return
    oof_p, test_p = CANDIDATES[name]
    if not (ART / oof_p).exists():
        print(f"NOT AVAILABLE: {oof_p}")
        return

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    v1_pred_oof = (safelog(v1_oof) + v1_bias).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred_oof)
    v1_test_pred = (safelog(v1_test) + v1_bias).argmax(1)

    c_oof = _normed(np.load(ART / oof_p).astype(np.float32))
    c_test = _normed(np.load(ART / test_p).astype(np.float32))
    c_bias, c_tuned = tune_log_bias(c_oof, y, prior)
    c_drift_max = float(np.abs(c_bias - (-np.log(prior))).max())

    print(f"=== {name} ===")
    print(f"  v1 tuned     = {v1_tuned:.5f}")
    print(f"  candidate     = {c_tuned:.5f}  Δ = {c_tuned-v1_tuned:+.5f}  max_drift={c_drift_max:.2f}")

    # Find best blend or standalone candidate that passes 4-gate
    best = None  # (config_name, alpha, oof_delta, oof_blend, c_test_pred_to_save)

    # Standalone candidate (alpha=1.0 effective)
    c_pred_oof = (safelog(c_oof) + c_bias).argmax(1)
    c_pcr = per_class_recall(y, c_pred_oof)
    c_test_pred = (safelog(c_test) + c_bias).argmax(1)
    n_diff = int((c_test_pred != v1_test_pred).sum())
    add_h = int(((c_test_pred == 2) & (v1_test_pred != 2)).sum())
    rem_h = int(((c_test_pred != 2) & (v1_test_pred == 2)).sum())
    net_h = add_h - rem_h
    churn_h = add_h + rem_h
    ratio_h = abs(net_h) / max(1, churn_h)

    delta_pcr = (c_pcr - v1_pcr).tolist()
    g_drift = c_drift_max <= 0.40
    g_g1_strict = (c_tuned - v1_tuned) >= 2e-4
    g_g1_relax = (c_tuned - v1_tuned) >= -1e-5 and net_h > 0
    g_g2 = all(d >= -5e-4 for d in delta_pcr)
    g_g4 = (net_h > 0) and (ratio_h >= 0.5)

    print(f"\n  STANDALONE (alpha=1):")
    print(f"    drift_max={c_drift_max:.2f} ({'PASS' if g_drift else 'FAIL'})")
    print(f"    delta_tuned={c_tuned-v1_tuned:+.5f} (G1 strict={g_g1_strict} relaxed={g_g1_relax})")
    print(f"    delta_PCR={delta_pcr} (G2 {'PASS' if g_g2 else 'FAIL'})")
    print(f"    test n_diff={n_diff} net_H={net_h:+d} churn_H={churn_h} ratio={ratio_h:.3f} "
          f"(G4 {'PASS' if g_g4 else 'FAIL'})")

    if g_drift and g_g2 and g_g4 and (g_g1_strict or g_g1_relax):
        config = ("standalone", 1.0, c_tuned - v1_tuned, c_tuned, c_test_pred)
        best = ("standalone", config)
        print(f"    => STANDALONE PASSES ALL GATES")

    # Blended sweep at v1 bias
    print(f"\n  BLEND SWEEP (at v1 bias):")
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
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
        n_h = ah - rh
        ch = ah + rh
        rt = abs(n_h) / max(1, ch)

        g1 = b_d >= 2e-4
        g2 = all(d >= -5e-4 for d in b_dpcr)
        g4 = (n_h > 0) and (rt >= 0.5)
        all_pass = g1 and g2 and g4
        n_diff_b = int((b_test_pred != v1_test_pred).sum())

        marker = "  ALL-GATE PASS" if all_pass else ""
        print(f"    α={alpha:.2f}: Δ={b_d:+.5f} PCR=[{b_dpcr[0]:+.5f},{b_dpcr[1]:+.5f},{b_dpcr[2]:+.5f}] "
              f"net_H={n_h:+d} ratio={rt:.3f} ({n_diff_b} rows){marker}")

        if all_pass:
            if best is None or b_d > best[1][2]:
                best = (f"blend_a{int(alpha*100):03d}", (f"blend_a{int(alpha*100):03d}",
                                                        alpha, b_d, b_bal, b_test_pred))

    if best is None:
        print(f"\n=== {name}: NO CONFIG PASSES ALL GATES — DO NOT LB-PROBE ===")
        return

    config_name, (cn, alpha, delta, oof, test_pred) = best
    print(f"\n=== {name}: BEST = {config_name}  Δ_OOF={delta:+.5f}  OOF={oof:.5f} ===")
    sub_path = SUB / f"submission_{name}_{config_name}.csv"
    pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in test_pred]}).to_csv(sub_path, index=False)
    print(f"emitted {sub_path}")

    # CLAUDE.md rule (2026-04-30): verify candidate not already LB-tested.
    # Single-line guard via lb_status.py.
    import subprocess
    try:
        guard = subprocess.run(
            ["python", str(Path(__file__).parent / "lb_status.py"), sub_path.name],
            capture_output=True, text=True, timeout=60,
        )
        if guard.returncode == 0:
            # Already submitted — print actual LB result instead of recommending submit
            print(f"\n*** WARNING: this candidate has ALREADY been LB-probed ***")
            print(f"  {guard.stdout.strip()}")
            print(f"  DO NOT re-submit. Surface the actual LB result above.")
            return
        elif guard.returncode == 1:
            # Not yet probed — safe to recommend
            print(f"\nLB-status check: {guard.stdout.strip()}")
        else:
            print(f"\nWARNING: lb_status.py guard errored ({guard.returncode}); "
                  f"manually verify before LB-probing.")
    except Exception as e:
        print(f"\nWARNING: lb_status.py guard failed ({e}); "
              f"manually verify before LB-probing.")

    print(f"\nLB-probe command (REQUIRES USER APPROVAL):")
    print(f"  kaggle competitions submit -f {sub_path} -m 'path={name} {config_name} OOF={oof:.5f}' -c playground-series-s6e4")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <candidate>")
        print(f"candidates: {list(CANDIDATES.keys())}")
    else:
        main(sys.argv[1])
