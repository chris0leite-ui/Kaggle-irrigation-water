"""P3 instability blend-gate: recipe + INSTAB vs LB-best 4-stack.

Mirrors v6_compare_variants.py's 4-criteria gate, calibrated against the
recipe baseline + LB-best 4-stack reference.

Gates (ALL must pass for an LB probe to be warranted):
  G1. standalone iso OOF ≥ recipe vanilla iso OOF (no regression on standalone)
  G2. errs at recipe bias ≤ recipe vanilla errs (no error-magnitude
      regression)
  G3. peak-α blend per-class recall ≥ LB-best 4-stack PCR − 5e-4 each class
  G4. peak-α blend Δ vs LB-best 4-stack ≥ +2e-4 AND Jaccard < 0.97

Outputs:
  scripts/artifacts/p3_blend_gate_results.json
  submissions/submission_p3_instab_a{α}.csv (only if all 4 gates pass)
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
from common import log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, DATA, SUB, BIAS, build_lbbest_stack, iso_cal, log, bal_at_bias,
)

CLASSES = ["Low", "Medium", "High"]
GATE_LB_DELTA = 2e-4
GATE_JACC = 0.97
PCR_FLOOR_DELTA = 5e-4  # subtract from anchor PCR to get per-class floor


def predict(p, bias=BIAS):
    return (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return cm.diagonal() / cm.sum(axis=1).clip(min=1)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    # Anchors
    log("loading anchors")
    lb3_oof, lb3_test = build_lbbest_stack(y)
    mv1 = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    mv1_te = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    mv1_iso, mv1_iso_te = iso_cal(mv1, mv1_te, y)
    lb4_oof = log_blend([lb3_oof, mv1_iso], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, mv1_iso_te], np.array([0.7, 0.3]))
    lb3_bal = bal_at_bias(lb3_oof, y)
    lb4_bal = bal_at_bias(lb4_oof, y)
    log(f"  LB-best 3-stack OOF = {lb3_bal:.5f}")
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f}")

    # Recipe vanilla baseline
    rv_oof = np.load(ART / "oof_recipe_full_te.npy").astype(np.float32)
    rv_te = np.load(ART / "test_recipe_full_te.npy").astype(np.float32)
    rv_iso, rv_iso_te = iso_cal(rv_oof, rv_te, y)
    rv_argmax = balanced_accuracy_score(y, rv_oof.argmax(1))
    rv_iso_tuned = bal_at_bias(rv_iso, y)
    rv_pred_iso = predict(rv_iso)
    rv_errs = int((rv_pred_iso != y).sum())
    rv_pcr = per_class_recall(y, rv_pred_iso)
    log(f"\nrecipe vanilla (anchor for G1/G2):")
    log(f"  iso tuned OOF = {rv_iso_tuned:.5f}  errs={rv_errs}  "
        f"PCR={rv_pcr.round(5).tolist()}")

    # P3 instability candidate
    p3_oof = np.load(ART / "oof_recipe_full_te_instab.npy").astype(np.float32)
    p3_te = np.load(ART / "test_recipe_full_te_instab.npy").astype(np.float32)
    p3_iso, p3_iso_te = iso_cal(p3_oof, p3_te, y)

    p3_argmax = balanced_accuracy_score(y, p3_oof.argmax(1))
    p3_raw_tuned = bal_at_bias(p3_oof, y)
    p3_iso_tuned = bal_at_bias(p3_iso, y)
    p3_pred_iso = predict(p3_iso)
    p3_errs = int((p3_pred_iso != y).sum())
    p3_pcr = per_class_recall(y, p3_pred_iso)

    log(f"\n=== P3 recipe + INSTAB standalone ===")
    log(f"  argmax OOF       {p3_argmax:.5f}")
    log(f"  raw @recipe-bias {p3_raw_tuned:.5f}")
    log(f"  iso @recipe-bias {p3_iso_tuned:.5f}")
    log(f"  errs (iso)       {p3_errs}")
    log(f"  PCR [L,M,H]      {p3_pcr.round(5).tolist()}")

    g1 = p3_iso_tuned >= rv_iso_tuned - 1e-5  # tiny noise tolerance
    g2 = p3_errs <= rv_errs
    log(f"  G1 (iso ≥ recipe vanilla iso {rv_iso_tuned:.5f}): {g1}")
    log(f"  G2 (errs ≤ recipe vanilla errs {rv_errs}):       {g2}")

    # Blend sweep into LB-best 4-stack
    log(f"\n=== sweep — P3_iso × LB-best 4-stack (anchor {lb4_bal:.5f}) ===")
    rows = []
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    pred_lb4 = predict(lb4_oof)
    pcr_lb4 = per_class_recall(y, pred_lb4)
    pcr_floor = pcr_lb4 - PCR_FLOOR_DELTA
    log(f"  PCR floor at peak-α [L,M,H] = {pcr_floor.round(5).tolist()}")
    for a in alphas:
        b = log_blend([lb4_oof, p3_iso], np.array([1 - a, a]))
        bb = bal_at_bias(b, y)
        pb = predict(b)
        pcr_b = per_class_recall(y, pb)
        e1 = pred_lb4 != y; e2 = pb != y
        jacc = (e1 & e2).sum() / max((e1 | e2).sum(), 1)
        rows.append({"alpha": a, "oof": float(bb),
                     "delta_vs_lb4": float(bb - lb4_bal),
                     "errs": int((pb != y).sum()),
                     "pcr": pcr_b.round(5).tolist(),
                     "jaccard": float(jacc)})
        log(f"  α={a:5.3f}  OOF={bb:.5f}  Δ={bb-lb4_bal:+.5f}  "
            f"errs={int((pb!=y).sum())}  J={jacc:.4f}  "
            f"PCR={[round(p,4) for p in pcr_b]}")
    best = max(rows, key=lambda r: r["delta_vs_lb4"])

    pcr_peak = np.array(best["pcr"])
    g3 = bool(np.all(pcr_peak >= pcr_floor))
    g4 = (best["delta_vs_lb4"] >= GATE_LB_DELTA and best["jaccard"] < GATE_JACC)
    log(f"\n  G3 (PCR ≥ floor at peak):  {g3}  PCR={pcr_peak.tolist()}")
    log(f"  G4 (Δ ≥ {GATE_LB_DELTA} AND Jacc < {GATE_JACC}):  {g4}  "
        f"Δ={best['delta_vs_lb4']:+.5f}  J={best['jaccard']:.4f}  "
        f"α={best['alpha']:.3f}")

    emit = bool(g1 and g2 and g3 and g4)
    log(f"\n=== EMIT DECISION: {emit} (g1={g1} g2={g2} g3={g3} g4={g4}) ===")
    if emit:
        a = best["alpha"]
        tb = log_blend([lb4_test, p3_iso_te], np.array([1 - a, a]))
        pred_t = predict(tb)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub["Irrigation_Need"] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_p3_instab_a{int(a*1000):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"WROTE {path}  (still requires explicit user approval before submit)")

    out = dict(
        recipe_iso_tuned=float(rv_iso_tuned), recipe_errs=int(rv_errs),
        recipe_pcr=rv_pcr.tolist(),
        p3_argmax=float(p3_argmax), p3_raw_tuned=float(p3_raw_tuned),
        p3_iso_tuned=float(p3_iso_tuned), p3_errs=int(p3_errs),
        p3_pcr=p3_pcr.tolist(),
        lb3_bal=float(lb3_bal), lb4_bal=float(lb4_bal),
        lb4_pcr=pcr_lb4.tolist(), pcr_floor=pcr_floor.tolist(),
        sweep=rows, best=best,
        gates=dict(g1=bool(g1), g2=bool(g2), g3=bool(g3),
                   g4=bool(g4), emit=emit),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "p3_blend_gate_results.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote {ART / 'p3_blend_gate_results.json'}")


if __name__ == "__main__":
    main()
