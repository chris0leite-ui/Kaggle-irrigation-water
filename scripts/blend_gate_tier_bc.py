"""Tier B/C blend-gate: B1+B2 alone + B-best × RF v1 ensemble (C1).

Tests:
  1. B2 HistGBM standalone (already known: tuned 0.98050)
  2. C1: RF v1 × B2 HistGBM geomean ensemble at α-sweep
  3. C1': RF v1 × B1 ExtraTrees (lower priority since B1 standalone NULL)

Architecture: log_blend([rf_v1_oof, b2_oof], [1-α, α]) — geomean in
prob space, then tune log-bias on the blend OOF, then 4-gate analysis
vs anchors:
  - rawashishsin (hedge anchor)
  - primary (LB-best 4-stack 0.98094)

Decision: emit candidate IF
  - Δ vs RF v1 ≥ +2e-4 OOF
  - PCR drift ≤ -5e-4 floor each class
  - net_H > 0 AND |asymmetry| ≥ 0.5 on test
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
from common import log_blend, tune_log_bias  # noqa: E402

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
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def blend_gate(name_a, oof_a, test_a, name_b, oof_b, test_b, y, prior, test_ids):
    """4-gate sweep for log_blend(a, b) at α ∈ {0.05..0.50}."""
    bias_a, tuned_a = tune_log_bias(oof_a, y, prior)
    pred_a = (safelog(oof_a) + bias_a).argmax(1)
    pcr_a = per_class_recall(y, pred_a)
    log(f"  ANCHOR {name_a}: tuned={tuned_a:.5f} PCR={[round(x,4) for x in pcr_a]}")

    sweep = []
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        b_oof = log_blend([oof_a, oof_b], np.array([1.0 - alpha, alpha]))
        b_bias, b_tuned = tune_log_bias(b_oof, y, prior)
        b_pred = (safelog(b_oof) + b_bias).argmax(1)
        b_pcr = per_class_recall(y, b_pred)
        d_class = (b_pcr - pcr_a).tolist()
        d_total = float(b_tuned - tuned_a)
        errs = int((b_pred != y).sum())
        # Test side
        b_test = log_blend([test_a, test_b], np.array([1.0 - alpha, alpha]))
        b_test_pred = (safelog(b_test) + b_bias).argmax(1)
        a_test_pred = (safelog(test_a) + bias_a).argmax(1)
        net_h = int(((b_test_pred == 2) & (a_test_pred != 2)).sum() -
                    ((a_test_pred == 2) & (b_test_pred != 2)).sum())
        churn_h = int(((b_test_pred == 2) ^ (a_test_pred == 2)).sum())
        diff = int((b_test_pred != a_test_pred).sum())
        # G4 ratio
        g4_ratio = abs(net_h) / max(1, churn_h) if churn_h > 0 else 0
        sweep.append({
            "alpha": alpha,
            "tuned": float(b_tuned),
            "delta": d_total,
            "pcr_delta": d_class,
            "errs": errs,
            "net_H": net_h,
            "churn_H": churn_h,
            "test_diff": diff,
            "g4_ratio": float(g4_ratio),
            "g4_pass": net_h > 0 and g4_ratio >= 0.5,
        })
        passed = (d_total >= 2e-4) and all(d >= -5e-4 for d in d_class)
        mark = "✓" if passed else " "
        log(f"  α={alpha:.2f} {mark} tuned={b_tuned:.5f} Δ={d_total:+.5f} "
            f"PCR_d=[{d_class[0]:+.4f} {d_class[1]:+.4f} {d_class[2]:+.4f}] "
            f"net_H={net_h:+d} ratio={g4_ratio:.2f} diff={diff}")

    # Find best gate-pass
    passing_g123 = [s for s in sweep if s["delta"] >= 2e-4
                    and all(d >= -5e-4 for d in s["pcr_delta"])]
    passing_g4 = [s for s in passing_g123 if s["g4_pass"]]

    log(f"  G1+G2 pass: {len(passing_g123)} alphas | G4 pass: {len(passing_g4)} alphas")

    best_g4 = max(passing_g4, key=lambda s: s["delta"]) if passing_g4 else None
    best_g123 = max(passing_g123, key=lambda s: s["delta"]) if passing_g123 else None

    # Emit candidate at best G4-pass alpha
    if best_g4:
        alpha = best_g4["alpha"]
        b_test = log_blend([test_a, test_b], np.array([1.0 - alpha, alpha]))
        b_oof = log_blend([oof_a, oof_b], np.array([1.0 - alpha, alpha]))
        b_bias, _ = tune_log_bias(b_oof, y, prior)
        b_test_pred = (safelog(b_test) + b_bias).argmax(1)
        sub_path = SUB / f"submission_blend_{name_a}_x_{name_b}_a{int(alpha*100):03d}_g4pass.csv"
        sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in b_test_pred]})
        sub.to_csv(sub_path, index=False)
        log(f"  ✓✓ G4-PASS candidate emitted: {sub_path}")

    return dict(
        anchor_tuned=float(tuned_a),
        sweep=sweep,
        best_g4=best_g4,
        best_g123=best_g123,
    )


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    log("loading components: RF v1 (LB-validated) + B1 ExtraTrees + B2 HistGBM")
    rf_v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    rf_v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    et_oof = np.load(ART / "oof_sklearn_extratrees_natural_v1bank.npy").astype(np.float32)
    et_test = np.load(ART / "test_sklearn_extratrees_natural_v1bank.npy").astype(np.float32)
    hg_oof = np.load(ART / "oof_sklearn_histgbm_natural_v1bank.npy").astype(np.float32)
    hg_test = np.load(ART / "test_sklearn_histgbm_natural_v1bank.npy").astype(np.float32)

    results = {}

    log("\n=== C1a: blend RF v1 × B2 HistGBM (highest priority — H+0.0027) ===")
    results["rf_v1_x_b2_histgbm"] = blend_gate(
        "rf_v1_lbbest", rf_v1_oof, rf_v1_test,
        "b2_histgbm", hg_oof, hg_test,
        y, prior, test_ids,
    )

    log("\n=== C1b: blend RF v1 × B1 ExtraTrees (lower priority — B1 standalone NULL) ===")
    results["rf_v1_x_b1_extratrees"] = blend_gate(
        "rf_v1_lbbest", rf_v1_oof, rf_v1_test,
        "b1_extratrees", et_oof, et_test,
        y, prior, test_ids,
    )

    log("\n=== B3: 3-way ensemble RF v1 + B2 + B1 (geomean log-blend) ===")
    for w_b1 in [0.10, 0.15, 0.20]:
        for w_b2 in [0.20, 0.30, 0.40]:
            w_rf = 1.0 - w_b1 - w_b2
            if w_rf <= 0:
                continue
            blend_oof = log_blend([rf_v1_oof, hg_oof, et_oof],
                                    np.array([w_rf, w_b2, w_b1]))
            bias, tuned = tune_log_bias(blend_oof, y, prior)
            pred = (safelog(blend_oof) + bias).argmax(1)
            pcr = per_class_recall(y, pred)
            log(f"  w_rf={w_rf:.2f} w_b2={w_b2:.2f} w_b1={w_b1:.2f} | "
                f"tuned={tuned:.5f} PCR=[{pcr[0]:.4f} {pcr[1]:.4f} {pcr[2]:.4f}]")

    out_p = ART / "blend_gate_tier_bc_results.json"
    out_p.write_text(json.dumps(results, indent=2, default=float))
    log(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
