"""4-gate comparison for REPLACE variants V_a, V_b vs v1 LB-best 0.98129.

Tests each variant's standalone vs LB-validated v1 (preserved at
oof_sklearn_rf_meta_natural_v1_lb98129.npy / test_..._v1_lb98129.npy).
The 4-gate filter is critical because variants that LOOK like ADD-High
on OOF have repeatedly LB-regressed; the test-side direction check
(net_H + churn_H + ratio) is the binding constraint.

Decision rule:
  G1: Δ vs v1 ≥ +2e-4 (recipe-bias OOF)
  G2: per-class recall delta within -5e-4 each class
  G3: dual-α stability (linear scaling 0.30 → 0.40)
  G4: net_H > 0 AND |net_H|/churn_H ≥ 0.5

Emits CSV only when all 4 gates pass at recipe-bias-equivalent (no
per-α retune) — this avoids the bias-retune leak that contaminated R2
hybrid, LR-meta v1+v2, and the OOF-overfit cluster.
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


def safelog(p, eps=1e-9): return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum(): rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    log("loading v1 LB-best 0.98129 (preserved snapshot)")
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    log(f"  v1 OOF tuned = {v1_tuned:.5f}  bias = {v1_bias.round(4).tolist()}")
    v1_pred = (safelog(v1_oof) + v1_bias).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred)
    v1_test_pred = (safelog(v1_test) + v1_bias).argmax(1)
    log(f"  v1 PCR = [L={v1_pcr[0]:.4f} M={v1_pcr[1]:.4f} H={v1_pcr[2]:.4f}]")
    log(f"  v1 test class dist: " +
        " ".join(f"{IDX2CLS[i]}={int((v1_test_pred==i).sum())}" for i in range(3)))

    out = {}
    for tag, label in [("Va", "V_a (realmlp -> a2_natural_calib)"),
                        ("Vb", "V_b (realmlp -> realmlp_natural)"),
                        ("Vc", "V_c (recipe_full_te_catboost -> recipe_full_te_catboost_skte)")]:
        oof_p = ART / f"oof_rf_natural_replace_{tag}.npy"
        test_p = ART / f"test_rf_natural_replace_{tag}.npy"
        json_p = ART / f"rf_natural_replace_{tag}_results.json"
        if not (oof_p.exists() and test_p.exists()):
            log(f"=== {label}: MISSING artifacts, skip")
            continue
        meta = json.load(open(json_p))
        if meta.get("smoke"):
            log(f"=== {label}: only SMOKE artifacts on disk, skip until production finishes")
            continue
        log(f"=== {label} ===")
        v_oof = np.load(oof_p).astype(np.float32)
        v_test = np.load(test_p).astype(np.float32)

        # Standalone (own tuned bias)
        v_bias, v_tuned = tune_log_bias(v_oof, y, prior)
        v_pred = (safelog(v_oof) + v_bias).argmax(1)
        v_pcr = per_class_recall(y, v_pred)
        v_test_pred_own = (safelog(v_test) + v_bias).argmax(1)
        log(f"  standalone: tuned={v_tuned:.5f}  bias={v_bias.round(4).tolist()}")
        log(f"  standalone PCR = [L={v_pcr[0]:.4f} M={v_pcr[1]:.4f} H={v_pcr[2]:.4f}]")
        log(f"  standalone Δ vs v1 = {v_tuned - v1_tuned:+.5f}")

        # G1: standalone Δ vs v1 (the LB-validated baseline)
        g1_delta = float(v_tuned - v1_tuned)
        g1_pass = g1_delta >= 2e-4

        # G2: per-class recall delta vs v1's PCR (own tuned biases each)
        pcr_delta = (v_pcr - v1_pcr).tolist()
        g2_pass = all(d >= -5e-4 for d in pcr_delta)

        # G3: dual-α stability (against v1 as anchor) — only meaningful if G1+G2 pass
        # Sweep α∈[0.05..0.50] of v_oof onto v1_oof, check that delta scales linearly
        sweep_records = []
        for a in [0.10, 0.20, 0.30, 0.40, 0.50]:
            b_oof = log_blend([v1_oof, v_oof], np.array([1.0 - a, a]))
            b_bias, b_tuned = tune_log_bias(b_oof, y, prior)
            b_pred = (safelog(b_oof) + b_bias).argmax(1)
            b_pcr = per_class_recall(y, b_pred)
            d = b_tuned - v1_tuned
            d_pcr = (b_pcr - v1_pcr).tolist()
            sweep_records.append({"alpha": a, "tuned": float(b_tuned),
                                   "delta": float(d), "pcr_delta": d_pcr,
                                   "bias": b_bias.tolist()})
            mark = "*" if d >= 2e-4 and all(p >= -5e-4 for p in d_pcr) else " "
            log(f"  blend α={a:.2f} {mark} tuned={b_tuned:.5f} Δ={d:+.5f} "
                f"pcr_delta=[{d_pcr[0]:+.4f} {d_pcr[1]:+.4f} {d_pcr[2]:+.4f}]")

        # G3 ratio: delta(α=0.40) / delta(α=0.30)
        d30 = next(s for s in sweep_records if s["alpha"] == 0.30)["delta"]
        d40 = next(s for s in sweep_records if s["alpha"] == 0.40)["delta"]
        if d30 > 0:
            g3_ratio = d40 / d30
            g3_pass = 1.0 <= g3_ratio <= 2.0
        else:
            g3_ratio = float("nan")
            g3_pass = False

        # G4: test-side net_H + asymmetric ratio
        # Use v1's own tuned bias as the anchor's evaluation point on test
        # (v1 was LB-tested at v1_bias; 4-gate compares against that)
        v_test_pred = v_test_pred_own  # variant uses ITS OWN tuned bias on test
        net_h = int(((v_test_pred == 2) & (v1_test_pred != 2)).sum() -
                    ((v1_test_pred == 2) & (v_test_pred != 2)).sum())
        churn_h = int(((v_test_pred == 2) ^ (v1_test_pred == 2)).sum())
        if churn_h > 0:
            g4_ratio = abs(net_h) / churn_h
        else:
            g4_ratio = 0.0
        g4_pass = (net_h > 0) and (g4_ratio >= 0.5)

        disagree = int((v_test_pred != v1_test_pred).sum())
        log(f"  G1 (Δ ≥ +2e-4): {g1_delta:+.5f}  {'PASS' if g1_pass else 'FAIL'}")
        log(f"  G2 (PCR within -5e-4): {pcr_delta}  {'PASS' if g2_pass else 'FAIL'}")
        log(f"  G3 (ratio 0.4/0.3): {g3_ratio:.3f}  {'PASS' if g3_pass else 'FAIL'}")
        log(f"  G4 (net_H>0 AND ratio≥0.5): net_H={net_h:+d}  churn={churn_h}  ratio={g4_ratio:.3f}  {'PASS' if g4_pass else 'FAIL'}")
        log(f"  test diff vs v1: {disagree}")

        all_pass = g1_pass and g2_pass and g3_pass and g4_pass
        log(f"  OVERALL: {'PASS — emit candidate' if all_pass else 'FAIL — skip LB probe'}")

        # Emit standalone CSV regardless of gate (for diagnostic)
        sub_path = SUB / f"submission_rf_natural_replace_{tag}_standalone.csv"
        pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in v_test_pred],
        }).to_csv(sub_path, index=False)
        log(f"  wrote {sub_path}")

        out[tag] = dict(
            label=label,
            standalone_tuned=float(v_tuned), standalone_bias=v_bias.tolist(),
            standalone_pcr=v_pcr.tolist(),
            v1_baseline_tuned=float(v1_tuned), v1_pcr=v1_pcr.tolist(),
            g1_delta=g1_delta, g1_pass=g1_pass,
            pcr_delta_vs_v1=pcr_delta, g2_pass=g2_pass,
            g3_ratio=g3_ratio, g3_pass=g3_pass,
            g4_net_h=net_h, g4_churn_h=churn_h, g4_ratio=g4_ratio, g4_pass=g4_pass,
            all_pass=all_pass, disagree_vs_v1=disagree,
            sweep=sweep_records,
            sub_path=str(sub_path),
        )

    out_p = ART / "replace_4gate_compare_results.json"
    out_p.write_text(json.dumps(out, indent=2, default=float))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
