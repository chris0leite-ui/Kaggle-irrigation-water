"""Tier 1 cheap probes on top of LB-best RF natural v1 (LB 0.98129).

Two mechanism-distinct probes against v1 + a1lgbm + primary:

  Probe 1 — Soft 3-way log-blend (geomean):
    log_blend([v1, a1lgbm, primary], [w_v1, w_a1, w_pr])
    Different from 2026-04-29 hard-vote 3-way gate (which was structurally
    biased toward v1's already-correct calls). Geomean averages probability
    masses; calibration profile carries through smoothly.

  Probe 2 — Confidence-weighted gating (v1 ↔ a1lgbm via primary.conf):
    per-row weight w(x) = primary.max_prob(x)
    pred(x) = w(x) * v1(x) + (1 - w(x)) * a1lgbm(x)
    Probabilistic per-row blend conditioned on primary's confidence —
    different from hard argmax-vote.

Anchor: v1 (LB-best 0.98129, tuned bias [0.4324, 0.8689, 3.2008]).
All blends evaluated at FIXED v1 bias (no retune — defends against the
binhigh OOF-tune trap).

4-gate filter:
  G1: ΔOOF ≥ +2e-4 vs v1 standalone 0.98063
  G2: PCR each class ≥ v1 PCR − 5e-4 (Pareto guardrail)
  G3: dual-α stability (1.0-2.0× linear scaling) — N/A for single picks
  G4: net rare-class flip > 0 AND |asymmetry| ratio ≥ 0.5

Outputs (per probe-config that gate-passes):
  scripts/artifacts/tier1_soft_blend_probes_results.json
  submissions/submission_tier1_<config>.csv  (only if all gates pass)
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
from tier1b_helpers import build_lbbest_stack  # noqa: E402

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


def gate_eval(name, blend_oof, blend_test, y, anchor_oof, anchor_test,
              anchor_bias, anchor_tuned, anchor_pcr, test_ids):
    """Apply v1's fixed bias, run 4-gate filter, decide on emit."""
    pred = (safelog(blend_oof) + anchor_bias).argmax(1)
    bal = balanced_accuracy_score(y, pred)
    pcr = per_class_recall(y, pred)
    delta = bal - anchor_tuned
    pcr_delta = (pcr - anchor_pcr).round(5).tolist()

    a_pred = (safelog(anchor_oof) + anchor_bias).argmax(1)
    b_pred_test = (safelog(blend_test) + anchor_bias).argmax(1)
    a_pred_test = (safelog(anchor_test) + anchor_bias).argmax(1)
    net_h = int(((b_pred_test == 2) & (a_pred_test != 2)).sum() -
                ((a_pred_test == 2) & (b_pred_test != 2)).sum())
    churn_h = int(((b_pred_test == 2) ^ (a_pred_test == 2)).sum())
    diff = int((b_pred_test != a_pred_test).sum())
    g4_ratio = abs(net_h) / max(1, churn_h)

    g1 = delta >= 2e-4
    g2 = all(d >= -5e-4 for d in pcr_delta)
    g4 = (net_h > 0) and (g4_ratio >= 0.5)

    mark = "*" if (g1 and g2 and g4) else " "
    log(f"  {mark} {name}: tuned={bal:.5f} Δ={delta:+.5f} "
        f"PCR_d={pcr_delta} net_H={net_h:+d} ratio={g4_ratio:.2f} diff={diff}")

    return dict(name=name, tuned=float(bal), delta=float(delta),
                pcr=pcr.tolist(), pcr_delta=pcr_delta,
                net_h=net_h, churn_h=churn_h, g4_ratio=float(g4_ratio),
                test_diff=diff, g1=bool(g1), g2=bool(g2), g4=bool(g4),
                emit=bool(g1 and g2 and g4),
                blend_test_pred=b_pred_test if (g1 and g2 and g4) else None)


def main():
    log("loading inputs")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    # v1 RF natural (LB 0.98129) — anchor
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)

    # a1lgbm RF natural (LB 0.98097, REGRESSION but with +H direction)
    a1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_a1lgbm.npy").astype(np.float32)
    a1_test = np.load(ART / "test_sklearn_rf_meta_natural_a1lgbm.npy").astype(np.float32)

    # Primary 4-stack (LB 0.98094, recipe family) — independent third opinion
    pr_oof, pr_test = build_lbbest_stack(y)

    # Anchor calibration
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    v1_pred = (safelog(v1_oof) + v1_bias).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred)
    log(f"ANCHOR v1: tuned={v1_tuned:.5f} bias={v1_bias.round(4).tolist()} "
        f"PCR={v1_pcr.round(4).tolist()}")
    log(f"  prior={prior.round(4).tolist()}  "
        f"-log(prior)={(-np.log(prior)).round(4).tolist()}  "
        f"drift={(v1_bias + np.log(prior)).round(3).tolist()}")

    results = {"anchor_v1_tuned": float(v1_tuned),
               "anchor_v1_bias": v1_bias.tolist(),
               "anchor_v1_pcr": v1_pcr.tolist(),
               "configs": []}
    emit_ready = []

    log("\n=== Probe 1 — Soft 3-way geomean log-blend (fixed v1 bias) ===")
    # Sweep configurations: keep v1 weight high (anchor), small contributions
    # from a1lgbm and primary. Symmetric & asymmetric tries.
    configs_p1 = [
        ("p1_w90_5_5",   [0.90, 0.05, 0.05]),
        ("p1_w85_10_5",  [0.85, 0.10, 0.05]),
        ("p1_w85_5_10",  [0.85, 0.05, 0.10]),
        ("p1_w85_75_75", [0.85, 0.075, 0.075]),
        ("p1_w80_10_10", [0.80, 0.10, 0.10]),
        ("p1_w80_15_5",  [0.80, 0.15, 0.05]),
        ("p1_w80_5_15",  [0.80, 0.05, 0.15]),
        ("p1_w70_15_15", [0.70, 0.15, 0.15]),
        ("p1_w50_25_25", [0.50, 0.25, 0.25]),
        # No-primary variants
        ("p1_v1_a1_95_5",  [0.95, 0.05, 0.0]),
        ("p1_v1_a1_90_10", [0.90, 0.10, 0.0]),
        ("p1_v1_a1_85_15", [0.85, 0.15, 0.0]),
        ("p1_v1_a1_80_20", [0.80, 0.20, 0.0]),
        # No-a1lgbm variants
        ("p1_v1_pr_95_5",  [0.95, 0.0, 0.05]),
        ("p1_v1_pr_90_10", [0.90, 0.0, 0.10]),
        ("p1_v1_pr_85_15", [0.85, 0.0, 0.15]),
    ]
    for name, w in configs_p1:
        w = np.array(w, dtype=np.float64)
        # Only include components with non-zero weight
        oofs, tests, weights = [], [], []
        for src_o, src_t, wi in [(v1_oof, v1_test, w[0]),
                                  (a1_oof, a1_test, w[1]),
                                  (pr_oof, pr_test, w[2])]:
            if wi > 0:
                oofs.append(src_o); tests.append(src_t); weights.append(wi)
        weights = np.array(weights, dtype=np.float64)
        b_oof = log_blend(oofs, weights)
        b_test = log_blend(tests, weights)
        r = gate_eval(name, b_oof, b_test, y, v1_oof, v1_test,
                      v1_bias, v1_tuned, v1_pcr, test_ids)
        results["configs"].append(r)
        if r["emit"]:
            emit_ready.append((name, r["blend_test_pred"]))

    log("\n=== Probe 2 — Confidence-weighted gating (primary.conf × v1 + (1-conf) × a1lgbm) ===")
    pr_conf_oof = pr_oof.max(axis=1)
    pr_conf_test = pr_test.max(axis=1)
    log(f"  primary OOF max_prob: p25={np.percentile(pr_conf_oof,25):.3f} "
        f"p50={np.percentile(pr_conf_oof,50):.3f} "
        f"p75={np.percentile(pr_conf_oof,75):.3f} "
        f"p99={np.percentile(pr_conf_oof,99):.3f}")

    configs_p2 = [
        ("p2_raw_conf",       1.0, 0.0),   # weight = primary.conf
        ("p2_conf_floor95",   0.95, 0.0),  # weight = max(primary.conf, 0.95)
        ("p2_conf_floor90",   0.90, 0.0),
        ("p2_conf_pow2",      None, 2.0),  # weight = primary.conf^2 (sharpen toward v1)
        ("p2_inv_conf",       None, -1.0), # weight = 1 - primary.conf (low conf -> v1)
    ]
    for name, floor, power in configs_p2:
        if power is not None and power > 0:
            wo = pr_conf_oof ** power
            wt = pr_conf_test ** power
        elif power is not None and power == -1.0:
            wo = 1 - pr_conf_oof
            wt = 1 - pr_conf_test
        elif floor is not None:
            wo = np.maximum(pr_conf_oof, floor)
            wt = np.maximum(pr_conf_test, floor)
        else:
            wo = pr_conf_oof
            wt = pr_conf_test
        # Convex blend in PROB space (not log-space, to preserve per-row weighting)
        b_oof = (wo[:, None] * v1_oof + (1 - wo)[:, None] * a1_oof)
        b_test = (wt[:, None] * v1_test + (1 - wt)[:, None] * a1_test)
        r = gate_eval(name, b_oof, b_test, y, v1_oof, v1_test,
                      v1_bias, v1_tuned, v1_pcr, test_ids)
        results["configs"].append(r)
        if r["emit"]:
            emit_ready.append((name, r["blend_test_pred"]))

    out_p = ART / "tier1_soft_blend_probes_results.json"
    out_p.write_text(json.dumps({k: v for k, v in results.items()
                                 if k != "configs"} |
                                {"configs": [{kk: vv for kk, vv in c.items()
                                              if kk != "blend_test_pred"}
                                             for c in results["configs"]]},
                                indent=2, default=float))
    log(f"\nwrote {out_p}")

    if emit_ready:
        for name, pred in emit_ready:
            sub_path = SUB / f"submission_tier1_{name}.csv"
            sub = pd.DataFrame({"id": test_ids,
                                TARGET: [IDX2CLS[i] for i in pred]})
            sub.to_csv(sub_path, index=False)
            log(f"  ✓ EMIT {sub_path}")
    else:
        log("  no config passed all 4 gates; no submission emitted")

    # Summary
    log("\n=== SUMMARY ===")
    pass_g1 = [c for c in results["configs"] if c["g1"]]
    pass_g4 = [c for c in results["configs"] if c["g1"] and c["g2"] and c["g4"]]
    log(f"  configs G1+G2+G4 PASS:  {len(pass_g4)}")
    log(f"  configs G1 PASS:        {len(pass_g1)}")
    if pass_g1:
        best = max(pass_g1, key=lambda c: c["delta"])
        log(f"  best G1: {best['name']} Δ={best['delta']:+.5f} "
            f"net_H={best['net_h']:+d} g4_ratio={best['g4_ratio']:.2f}")


if __name__ == "__main__":
    main()
