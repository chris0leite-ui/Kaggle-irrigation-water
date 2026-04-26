"""Blend-gate analysis for 3-way OTE + NN-distance candidates.

Pipeline:
  1. Build LB-best 4-stack reference: lb3 + RealMLP@0.20 +
     xgb_nonrule_iso@0.075 + xgb_metastack_iso@0.30. OOF 0.98084 / LB 0.98094.
  2. Load each candidate OOF (3way / nndist) at fixed recipe bias.
  3. Compute the 4-axis gate vs anchor:
     - tuned bal_acc ≥ 0.974
     - Jaccard < 0.80 vs anchor argmax
     - errs ≤ 1.05 × anchor errs
     - per-class recall ≥ anchor floors (Low/Med/High − 5e-4)
  4. Sweep α-blend onto the LB-best 4-stack at fixed bias; report
     Δ peak + emit submission only when peak Δ ≥ +2e-4 AND blend
     guardrail PASS at peak α.

Designed to surface FE-layer levers that satisfy the binding
magnitude rule + per-class direction (the structural reason 13+
saturation confirmations all failed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, SUB, build_lbbest_stack, iso_cal, load_y,
    log, normed,
)

GATE_TUNED_FLOOR = 0.974
GATE_JACCARD_MAX = 0.80
GATE_ERR_RATIO_MAX = 1.05
GATE_PCR_TOLERANCE = 5e-4
EMIT_DELTA_FLOOR = 2e-4

CANDIDATES = ["recipe_full_te_3way", "recipe_full_te_nndist"]


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    diag = np.diag(cm).astype(float)
    return diag / cm.sum(axis=1).clip(1)


def jaccard_argmax(p1, p2, b=BIAS):
    a1 = (np.log(np.clip(p1, 1e-12, 1)) + b).argmax(1)
    a2 = (np.log(np.clip(p2, 1e-12, 1)) + b).argmax(1)
    inter = (a1 != a2).sum()  # disagreements — but we want error overlap
    return inter, a1, a2


def err_jaccard(y, p1, p2, b=BIAS):
    a1 = (np.log(np.clip(p1, 1e-12, 1)) + b).argmax(1)
    a2 = (np.log(np.clip(p2, 1e-12, 1)) + b).argmax(1)
    e1 = a1 != y
    e2 = a2 != y
    inter = (e1 & e2).sum()
    union = (e1 | e2).sum()
    return inter / max(union, 1)


def gate_one(name, y, anchor_o, anchor_t, cand_o, cand_t):
    """Run the 4-axis gate + α-sweep + emit for one candidate."""
    res = {"name": name}
    # Standalone @ recipe bias.
    pred_cand = (np.log(np.clip(cand_o, 1e-12, 1)) + BIAS).argmax(1)
    pred_anc = (np.log(np.clip(anchor_o, 1e-12, 1)) + BIAS).argmax(1)
    bal_cand_atbias = balanced_accuracy_score(y, pred_cand)
    bal_anc_atbias = balanced_accuracy_score(y, pred_anc)
    pcr_cand = per_class_recall(y, pred_cand)
    pcr_anc = per_class_recall(y, pred_anc)
    errs_cand = (pred_cand != y).sum()
    errs_anc = (pred_anc != y).sum()
    err_jac = err_jaccard(y, cand_o, anchor_o)
    # Standalone tuned (for reference).
    prior = np.bincount(y, minlength=3) / len(y)
    bias_cand, tuned_cand = tune_log_bias(cand_o, y, prior)
    log(f"[{name}] standalone @ recipe bias = {bal_cand_atbias:.5f}  "
        f"tuned = {tuned_cand:.5f}  bias={bias_cand.round(3).tolist()}")
    log(f"  errs = {errs_cand} (anchor {errs_anc}, ratio {errs_cand/max(errs_anc,1):.3f})")
    log(f"  PCR  = L={pcr_cand[0]:.4f} M={pcr_cand[1]:.4f} H={pcr_cand[2]:.4f}")
    log(f"  PCR Δ vs anchor = "
        f"L={pcr_cand[0]-pcr_anc[0]:+.4f} "
        f"M={pcr_cand[1]-pcr_anc[1]:+.4f} "
        f"H={pcr_cand[2]-pcr_anc[2]:+.4f}")
    log(f"  Err Jaccard vs anchor = {err_jac:.4f}")
    res["standalone_atbias"] = float(bal_cand_atbias)
    res["standalone_tuned"] = float(tuned_cand)
    res["tuned_bias"] = bias_cand.round(4).tolist()
    res["errs"] = int(errs_cand)
    res["err_ratio_vs_anchor"] = float(errs_cand / max(errs_anc, 1))
    res["err_jaccard_vs_anchor"] = float(err_jac)
    res["pcr"] = pcr_cand.tolist()
    res["pcr_delta_vs_anchor"] = (pcr_cand - pcr_anc).tolist()
    # Standalone gate.
    g_tuned = tuned_cand >= GATE_TUNED_FLOOR
    g_err = res["err_ratio_vs_anchor"] <= GATE_ERR_RATIO_MAX
    g_jac = err_jac < GATE_JACCARD_MAX
    g_pcr = all((pcr_cand[c] >= pcr_anc[c] - GATE_PCR_TOLERANCE) for c in range(3))
    res["gate_tuned"] = bool(g_tuned)
    res["gate_err_ratio"] = bool(g_err)
    res["gate_jaccard"] = bool(g_jac)
    res["gate_pcr"] = bool(g_pcr)
    log(f"  GATE: tuned={g_tuned} err={g_err} jac={g_jac} pcr={g_pcr}")

    # α-sweep against LB-best 4-stack at fixed recipe bias.
    alphas = np.array([0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
                       0.30, 0.35, 0.40, 0.50, 0.65])
    sweep = []
    bal_anc = balanced_accuracy_score(
        y, (np.log(np.clip(anchor_o, 1e-12, 1)) + BIAS).argmax(1))
    for a in alphas:
        if a == 0:
            blend_o = anchor_o
        else:
            blend_o = log_blend([anchor_o, cand_o],
                                np.array([1 - a, a], dtype=np.float64))
        pred_b = (np.log(np.clip(blend_o, 1e-12, 1)) + BIAS).argmax(1)
        pcr_b = per_class_recall(y, pred_b)
        errs_b = (pred_b != y).sum()
        bal_b = balanced_accuracy_score(y, pred_b)
        guardrail = all(pcr_b[c] >= pcr_anc[c] - GATE_PCR_TOLERANCE for c in range(3))
        sweep.append(dict(alpha=float(a), bal=float(bal_b),
                          delta=float(bal_b - bal_anc),
                          errs=int(errs_b), pcr=pcr_b.tolist(),
                          guardrail_pass=bool(guardrail)))
    res["sweep"] = sweep
    # Best α among guardrail-passing variants.
    cand_pass = [s for s in sweep if s["guardrail_pass"]]
    best = max(cand_pass, key=lambda s: s["delta"]) if cand_pass else None
    res["peak"] = best
    if best is not None:
        log(f"  PEAK GATE-PASS: α={best['alpha']:.3f}  Δ={best['delta']:+.5f}  "
            f"errs={best['errs']}")
    else:
        log("  PEAK GATE-PASS: NONE (every α fails per-class guardrail)")
    return res


def emit_submission(name, anchor_t, cand_t, alpha, ids, idx2cls):
    """Write submission CSV for the gate-pass best α."""
    blend_t = log_blend([anchor_t, cand_t],
                        np.array([1 - alpha, alpha], dtype=np.float64))
    pred = (np.log(np.clip(blend_t, 1e-12, 1)) + BIAS).argmax(1)
    sub = pd.DataFrame({"id": ids,
                        "Irrigation_Need": [idx2cls[i] for i in pred]})
    p = SUB / f"submission_lbbest_plus_{name}_a{int(round(alpha*1000)):03d}.csv"
    sub.to_csv(p, index=False)
    log(f"emitted {p}  pred dist: {dict(sub['Irrigation_Need'].value_counts())}")
    return str(p)


def main():
    log("loading y + LB-best 4-stack reference")
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)
    # Add the meta-stacker iso layer to reach LB-best 4-stack (LB 0.98094).
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    anchor_o = log_blend([lb3_o, meta_o_iso], np.array([0.70, 0.30]))
    anchor_t = log_blend([lb3_t, meta_t_iso], np.array([0.70, 0.30]))
    bal_anc = balanced_accuracy_score(
        y, (np.log(np.clip(anchor_o, 1e-12, 1)) + BIAS).argmax(1))
    log(f"LB-best 4-stack OOF @ recipe bias = {bal_anc:.5f} (target 0.98084)")

    # Test ids for submission writing.
    sub_path = "submissions/submission_recipe_full_te.csv"
    ids = pd.read_csv(sub_path)["id"].values
    idx2cls = {0: "Low", 1: "Medium", 2: "High"}

    summary = {"anchor_oof": float(bal_anc), "candidates": []}
    for name in CANDIDATES:
        log(f"=== {name} ===")
        oof_path = ART / f"oof_{name}.npy"
        test_path = ART / f"test_{name}.npy"
        if not oof_path.exists():
            log(f"  SKIP: {oof_path} missing")
            summary["candidates"].append({"name": name, "missing": True})
            continue
        cand_o = normed(np.load(oof_path).astype(np.float32))
        cand_t = normed(np.load(test_path).astype(np.float32))
        # Iso-calibrate for blend (improves prob-scale alignment with
        # anchor's recipe-bias operating point; matches Tier-1b protocol).
        cand_o_iso, cand_t_iso = iso_cal(cand_o, cand_t, y)
        log(f"  iso-cal applied to {name}")
        res = gate_one(name, y, anchor_o, anchor_t, cand_o_iso, cand_t_iso)
        # Emit submission if peak Δ ≥ +2e-4 AND guardrail PASS.
        if res.get("peak") and res["peak"]["delta"] >= EMIT_DELTA_FLOOR:
            res["submission"] = emit_submission(
                name, anchor_t, cand_t_iso, res["peak"]["alpha"], ids, idx2cls)
        else:
            res["submission"] = None
            log(f"  NO-EMIT: peak Δ below +{EMIT_DELTA_FLOOR}, no submission")
        summary["candidates"].append(res)

    out = ART / "blend_gate_3way_nn_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
