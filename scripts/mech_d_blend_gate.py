"""Mech D per-row attention — 4-criteria gate vs LB-best 4-stack.

Mech D output is a 3-class prob per row (already softmaxed). Treat it
as a candidate meta-stacker output: iso-cal it, blend into LB-best
3-stack at α (mirroring tier1b v1 LB-best meta-stacker pattern).

Gates:
  G1. iso OOF tuned ≥ tier1b v1 reference (0.98059)
  G2. errs at recipe bias ≤ 9100 (≤ v1 + noise margin)
  G3. peak-α blend per-class recall ≥ LB-best 4-stack PCR − 5e-4
  G4. peak-α blend Δ vs LB-best 4-stack ≥ +2e-4 AND Jaccard < 0.97
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
GATE_ISO = 0.98059
GATE_ERRS = 9100
PCR_FLOOR_DELTA = 5e-4
GATE_LB_DELTA = 2e-4
GATE_JACC = 0.97


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

    log("loading anchors")
    lb3_oof, lb3_test = build_lbbest_stack(y)
    mv1 = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    mv1_te = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    mv1_iso, mv1_iso_te = iso_cal(mv1, mv1_te, y)
    lb4_oof = log_blend([lb3_oof, mv1_iso], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, mv1_iso_te], np.array([0.7, 0.3]))
    lb3_bal = bal_at_bias(lb3_oof, y)
    lb4_bal = bal_at_bias(lb4_oof, y)
    pred_lb4 = predict(lb4_oof)
    pcr_lb4 = per_class_recall(y, pred_lb4)
    pcr_floor = pcr_lb4 - PCR_FLOOR_DELTA
    log(f"  LB-best 3-stack OOF = {lb3_bal:.5f}")
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f}")
    log(f"  PCR floor [L,M,H] = {pcr_floor.round(5).tolist()}")

    # Mech D output
    md_oof = np.load(ART / "oof_mech_d.npy").astype(np.float32)
    md_te = np.load(ART / "test_mech_d.npy").astype(np.float32)
    md_iso, md_iso_te = iso_cal(md_oof, md_te, y)

    md_argmax = balanced_accuracy_score(y, md_oof.argmax(1))
    md_raw_tuned = bal_at_bias(md_oof, y)
    md_iso_tuned = bal_at_bias(md_iso, y)
    pred_md = predict(md_iso)
    md_errs = int((pred_md != y).sum())
    md_pcr = per_class_recall(y, pred_md)

    log(f"\n=== Mech D standalone ===")
    log(f"  argmax OOF        {md_argmax:.5f}")
    log(f"  raw @recipe-bias  {md_raw_tuned:.5f}")
    log(f"  iso @recipe-bias  {md_iso_tuned:.5f}")
    log(f"  errs (iso)        {md_errs}")
    log(f"  PCR [L,M,H]       {md_pcr.round(5).tolist()}")

    g1 = md_iso_tuned >= GATE_ISO
    g2 = md_errs <= GATE_ERRS
    log(f"  G1 (iso ≥ {GATE_ISO}):     {g1}")
    log(f"  G2 (errs ≤ {GATE_ERRS}):       {g2}")

    # Sweep into LB-best 3-stack (mirror v1 meta-stacker α=0.30 pattern)
    log(f"\n=== sweep — md_iso × LB-best 3-stack (anchor {lb3_bal:.5f}) ===")
    rows_3 = []
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for a in alphas:
        b = log_blend([lb3_oof, md_iso], np.array([1 - a, a]))
        bb = bal_at_bias(b, y)
        rows_3.append({"alpha": a, "oof": float(bb), "delta": float(bb - lb3_bal)})
        log(f"  α={a:5.3f}  OOF={bb:.5f}  Δ={bb-lb3_bal:+.5f}")
    best_3 = max(rows_3, key=lambda r: r["delta"])

    # Sweep into LB-best 4-stack (the actual primary)
    log(f"\n=== sweep — md_iso × LB-best 4-stack (anchor {lb4_bal:.5f}) ===")
    rows_4 = []
    for a in alphas:
        b = log_blend([lb4_oof, md_iso], np.array([1 - a, a]))
        bb = bal_at_bias(b, y)
        pb = predict(b)
        pcr_b = per_class_recall(y, pb)
        e1 = pred_lb4 != y; e2 = pb != y
        jacc = (e1 & e2).sum() / max((e1 | e2).sum(), 1)
        rows_4.append({"alpha": a, "oof": float(bb),
                       "delta": float(bb - lb4_bal),
                       "errs": int((pb != y).sum()),
                       "pcr": pcr_b.round(5).tolist(),
                       "jaccard": float(jacc)})
        log(f"  α={a:5.3f}  OOF={bb:.5f}  Δ={bb-lb4_bal:+.5f}  "
            f"errs={int((pb!=y).sum())}  J={jacc:.4f}  "
            f"PCR={[round(p,4) for p in pcr_b]}")
    best_4 = max(rows_4, key=lambda r: r["delta"])

    pcr_peak = np.array(best_4["pcr"])
    g3 = bool(np.all(pcr_peak >= pcr_floor))
    g4 = (best_4["delta"] >= GATE_LB_DELTA and best_4["jaccard"] < GATE_JACC)
    log(f"\n  G3 (PCR ≥ floor at peak):  {g3}  PCR={pcr_peak.tolist()}")
    log(f"  G4 (Δ ≥ {GATE_LB_DELTA} AND Jacc < {GATE_JACC}):  {g4}  "
        f"Δ={best_4['delta']:+.5f}  J={best_4['jaccard']:.4f}  α={best_4['alpha']:.3f}")

    emit = bool(g1 and g2 and g3 and g4)
    log(f"\n=== EMIT: {emit} (g1={g1} g2={g2} g3={g3} g4={g4}) ===")

    if emit:
        a = best_4["alpha"]
        tb = log_blend([lb4_test, md_iso_te], np.array([1 - a, a]))
        pred_t = predict(tb)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub["Irrigation_Need"] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_mech_d_a{int(a*1000):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"WROTE {path}  (still requires explicit user approval before submit)")

    out = dict(
        md_argmax_oof=float(md_argmax), md_raw_tuned=float(md_raw_tuned),
        md_iso_tuned=float(md_iso_tuned), md_errs=int(md_errs),
        md_pcr=md_pcr.tolist(),
        lb3_bal=float(lb3_bal), lb4_bal=float(lb4_bal),
        pcr_floor=pcr_floor.tolist(),
        sweep_lb3=rows_3, best_lb3=best_3,
        sweep_lb4=rows_4, best_lb4=best_4,
        gates=dict(g1=g1, g2=g2, g3=g3, g4=g4, emit=emit),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "mech_d_blend_gate_results.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote {ART / 'mech_d_blend_gate_results.json'}")


if __name__ == "__main__":
    main()
