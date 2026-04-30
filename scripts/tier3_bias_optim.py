"""Tier 3 — fine-grained per-class bias optimization on v1 OOF.

Tests whether v1's tuned bias [0.4324, 0.8689, 3.2008] is locally optimal
or whether finer-grained search finds a better operating point on the
"bias-ridge" identified 2026-04-28 (where multiple bias settings give
LB-equivalent test predictions).

Methods:
  M1. Coord-ascent at FINE grid (step 0.005, 0.01) — vs default 0.05/0.1
  M2. Random search 10000 perturbations within ±0.50 of v1 bias
  M3. Per-class scan with finer resolution on each axis individually

Decision: emit submission ONLY if a config has tuned OOF Δ ≥ +2e-4
AND PCR within -5e-4 of v1 PCR each class.

Note: prior 2026-04-28 audit established the bias-ridge is structurally
LB-equivalent (variant A at OOF-optimal bias landed LB 0.98093 vs PRIMARY
0.98094 = -0.00001 = noise). For v1, ridge magnitude likely similar:
~+0.00010 OOF lift achievable but with ~0% LB transfer due to ridge
flatness. This run confirms or falsifies that for v1 specifically.
"""
from __future__ import annotations

import json
import sys
import time
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


def fine_coord_ascent(log_oof, y, init_bias, step=0.005, ranges=None,
                      max_iter=30):
    """Coord-ascent at fine step. ranges: per-class search half-width."""
    if ranges is None:
        ranges = [0.30, 0.30, 0.50]  # Low/Med/High
    bias = init_bias.copy()
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(1))
    for _ in range(max_iter):
        improved = False
        for k in range(3):
            r = ranges[k]
            grid = np.arange(-r, r + step / 2, step)
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-7:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def main():
    log("loading inputs")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)

    # Reproduce v1 standalone metric
    v1_bias_std, v1_tuned_std = tune_log_bias(v1_oof, y, prior)
    v1_pred = (safelog(v1_oof) + v1_bias_std).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred)
    log(f"ANCHOR v1 (default tune): tuned={v1_tuned_std:.5f} bias={v1_bias_std.round(4).tolist()} "
        f"PCR=[L={v1_pcr[0]:.4f} M={v1_pcr[1]:.4f} H={v1_pcr[2]:.4f}]")

    log_oof = safelog(v1_oof)

    # M1. Fine coord-ascent (step 0.005)
    log("\n=== M1. Fine coord-ascent step=0.005 ===")
    bias_m1, best_m1 = fine_coord_ascent(log_oof, y, v1_bias_std, step=0.005)
    pred_m1 = (log_oof + bias_m1).argmax(1)
    pcr_m1 = per_class_recall(y, pred_m1)
    drift_m1 = (bias_m1 + np.log(prior)).round(3).tolist()
    log(f"  bias_m1={bias_m1.round(4).tolist()} bal={best_m1:.5f} "
        f"Δ={best_m1 - v1_tuned_std:+.5f} drift={drift_m1}")
    log(f"  PCR_m1=[{pcr_m1[0]:.4f}, {pcr_m1[1]:.4f}, {pcr_m1[2]:.4f}] "
        f"PCR_d={[round(d, 5) for d in (pcr_m1 - v1_pcr).tolist()]}")

    # M2. Random search
    log("\n=== M2. Random search (10000 perturbations within ±0.50) ===")
    rng = np.random.RandomState(42)
    n_trials = 10000
    best_m2 = v1_tuned_std
    bias_m2 = v1_bias_std.copy()
    for _ in range(n_trials):
        delta = rng.uniform(-0.50, 0.50, size=3)
        b = v1_bias_std + delta
        bal = balanced_accuracy_score(y, (log_oof + b).argmax(1))
        if bal > best_m2:
            best_m2 = bal
            bias_m2 = b.copy()
    pred_m2 = (log_oof + bias_m2).argmax(1)
    pcr_m2 = per_class_recall(y, pred_m2)
    drift_m2 = (bias_m2 + np.log(prior)).round(3).tolist()
    log(f"  bias_m2={bias_m2.round(4).tolist()} bal={best_m2:.5f} "
        f"Δ={best_m2 - v1_tuned_std:+.5f} drift={drift_m2}")
    log(f"  PCR_m2=[{pcr_m2[0]:.4f}, {pcr_m2[1]:.4f}, {pcr_m2[2]:.4f}] "
        f"PCR_d={[round(d, 5) for d in (pcr_m2 - v1_pcr).tolist()]}")

    # M3. Per-class fine scan (one axis at a time, step 0.001)
    log("\n=== M3. Per-class single-axis fine scan (step=0.001) ===")
    bias_m3 = v1_bias_std.copy()
    best_m3 = v1_tuned_std
    for k in range(3):
        grid = np.arange(-0.20, 0.20, 0.001)
        base = bias_m3.copy()
        scores = []
        for g in grid:
            base[k] = bias_m3[k] + g
            scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(1)))
        j = int(np.argmax(scores))
        if scores[j] > best_m3:
            bias_m3[k] = bias_m3[k] + grid[j]
            best_m3 = scores[j]
            log(f"  axis {k}: shift {grid[j]:+.3f} -> bal={best_m3:.5f}")
    pred_m3 = (log_oof + bias_m3).argmax(1)
    pcr_m3 = per_class_recall(y, pred_m3)
    drift_m3 = (bias_m3 + np.log(prior)).round(3).tolist()
    log(f"  bias_m3={bias_m3.round(4).tolist()} bal={best_m3:.5f} "
        f"Δ={best_m3 - v1_tuned_std:+.5f} drift={drift_m3}")

    # Pick the best across methods that satisfy the gates
    log("\n=== GATE EVAL (Δ ≥ +2e-4 AND PCR each class within -5e-4 of v1 PCR) ===")

    candidates = [("M1", bias_m1, best_m1, pcr_m1),
                  ("M2", bias_m2, best_m2, pcr_m2),
                  ("M3", bias_m3, best_m3, pcr_m3)]
    best_gate_pass = None
    for name, b, bal, pcr in candidates:
        delta = bal - v1_tuned_std
        pcr_delta = (pcr - v1_pcr).tolist()
        g1 = delta >= 2e-4
        g2 = all(d >= -5e-4 for d in pcr_delta)
        # Test-side diff
        test_pred = (safelog(v1_test) + b).argmax(1)
        v1_test_pred = (safelog(v1_test) + v1_bias_std).argmax(1)
        diff = int((test_pred != v1_test_pred).sum())
        net_h = int(((test_pred == 2) & (v1_test_pred != 2)).sum() -
                    ((v1_test_pred == 2) & (test_pred != 2)).sum())
        churn_h = int(((test_pred == 2) ^ (v1_test_pred == 2)).sum())
        ratio = abs(net_h) / max(1, churn_h) if churn_h > 0 else 0
        log(f"  {name}: Δ={delta:+.5f} G1={g1} G2={g2} test_diff={diff} "
            f"net_H={net_h:+d} churn={churn_h} ratio={ratio:.2f}")
        if g1 and g2 and best_gate_pass is None:
            best_gate_pass = (name, b, bal, pcr, test_pred)

    summary = dict(
        v1_anchor_tuned=float(v1_tuned_std),
        v1_anchor_bias=v1_bias_std.tolist(),
        v1_anchor_pcr=v1_pcr.tolist(),
        m1=dict(bias=bias_m1.tolist(), bal=float(best_m1),
                delta=float(best_m1 - v1_tuned_std),
                drift=drift_m1, pcr=pcr_m1.tolist()),
        m2=dict(bias=bias_m2.tolist(), bal=float(best_m2),
                delta=float(best_m2 - v1_tuned_std),
                drift=drift_m2, pcr=pcr_m2.tolist()),
        m3=dict(bias=bias_m3.tolist(), bal=float(best_m3),
                delta=float(best_m3 - v1_tuned_std),
                drift=drift_m3, pcr=pcr_m3.tolist()),
    )

    out_p = ART / "tier3_bias_optim_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"\nwrote {out_p}")

    if best_gate_pass:
        name, b, bal, pcr, test_pred = best_gate_pass
        sub = pd.DataFrame({"id": test_ids,
                            TARGET: [IDX2CLS[i] for i in test_pred]})
        sub_path = SUB / f"submission_tier3_bias_optim_{name}.csv"
        sub.to_csv(sub_path, index=False)
        log(f"  ✓ EMIT {sub_path} (best gate-pass)")
    else:
        log("  no method passed all gates; no submission emitted")


if __name__ == "__main__":
    main()
