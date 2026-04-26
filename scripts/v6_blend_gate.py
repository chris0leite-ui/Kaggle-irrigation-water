"""Blend-gate analysis for v6 aggregate-stats meta-stacker.

Three gate criteria (ALL must pass for an LB probe to be warranted):
  1. Standalone iso OOF ≥ 0.98080  (within fold-noise of v1 0.98059)
  2. Errs ≤ 9550 at fixed recipe bias  (≤ 1.05× v1 anchor)
  3. Per-class recall ≥ [0.9950, 0.9690, 0.9770] at peak α blend
     into LB-best 3-stack  (each class ≥ anchor − 5e-4)
  4. LB-probe gate: peak α Δ vs LB-best 4-stack ≥ +2e-4 AND Jaccard < 0.97

Outputs:
  scripts/artifacts/v6_blend_gate_results.json
  submissions/submission_v6_a{α}.csv  (only if all gates pass)
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
GATE_STANDALONE_ISO = 0.98080
GATE_ERRS = 9550
GATE_PCR = np.array([0.9950, 0.9690, 0.9770])
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

    # Anchors: LB-best 3-stack (meta target) and 4-stack (LB-best primary)
    lb3_oof, lb3_test = build_lbbest_stack(y)
    # 4-stack: 3-stack + xgb_metastack__iso α=0.30
    meta_v1 = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    meta_v1_te = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    meta_v1_iso, meta_v1_iso_te = iso_cal(meta_v1, meta_v1_te, y)
    lb4_oof = log_blend([lb3_oof, meta_v1_iso], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, meta_v1_iso_te], np.array([0.7, 0.3]))

    lb3_bal = bal_at_bias(lb3_oof, y)
    lb4_bal = bal_at_bias(lb4_oof, y)
    log(f"LB-best 3-stack OOF = {lb3_bal:.5f}")
    log(f"LB-best 4-stack OOF = {lb4_bal:.5f}")

    # Load v6 standalone meta and iso-cal
    v6 = np.load(ART / "oof_xgb_metastack_v6.npy").astype(np.float32)
    v6_te = np.load(ART / "test_xgb_metastack_v6.npy").astype(np.float32)
    v6_iso, v6_iso_te = iso_cal(v6, v6_te, y)

    v6_argmax = balanced_accuracy_score(y, v6.argmax(1))
    v6_tuned = bal_at_bias(v6, y)
    v6_iso_tuned = bal_at_bias(v6_iso, y)
    pred_v6 = predict(v6_iso)
    v6_errs = (pred_v6 != y).sum()
    pcr_v6 = per_class_recall(y, pred_v6)

    log(f"\n=== v6 META standalone ===")
    log(f"  argmax OOF        {v6_argmax:.5f}")
    log(f"  @recipe-bias      {v6_tuned:.5f}")
    log(f"  iso @recipe-bias  {v6_iso_tuned:.5f}")
    log(f"  errs (iso bias)   {v6_errs}")
    log(f"  PCR [L,M,H]       {pcr_v6.round(5).tolist()}")

    # Three gate criteria check
    g1 = v6_iso_tuned >= GATE_STANDALONE_ISO
    g2 = v6_errs <= GATE_ERRS
    log(f"\nGate 1 (iso OOF ≥ {GATE_STANDALONE_ISO}):  {g1}  ({v6_iso_tuned:.5f})")
    log(f"Gate 2 (errs ≤ {GATE_ERRS}):              {g2}  ({v6_errs})")

    # Blend sweep into LB-best 3-stack (the meta target)
    log(f"\n=== fixed-bias blend sweep — v6_iso × LB-best 3-stack (anchor 0.98061) ===")
    rows_3 = []
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for a in alphas:
        b = log_blend([lb3_oof, v6_iso], np.array([1 - a, a]))
        bal_oof = bal_at_bias(b, y)
        pred_b = predict(b)
        pcr_b = per_class_recall(y, pred_b)
        errs_b = (pred_b != y).sum()
        rows_3.append({"alpha": a, "oof": float(bal_oof),
                       "delta_vs_lb3": float(bal_oof - lb3_bal),
                       "errs": int(errs_b), "pcr": pcr_b.round(5).tolist()})
        log(f"  α={a:5.3f}  OOF={bal_oof:.5f}  Δ={bal_oof-lb3_bal:+.5f}  "
            f"errs={errs_b}  PCR={pcr_b.round(4).tolist()}")
    best_3 = max(rows_3, key=lambda r: r["delta_vs_lb3"])

    # Gate 3: per-class recall at peak-α
    pcr_peak = np.array(best_3["pcr"])
    g3 = bool(np.all(pcr_peak >= GATE_PCR))
    log(f"\nGate 3 (PCR ≥ {GATE_PCR.tolist()} at peak-α={best_3['alpha']:.3f}):  {g3}  "
        f"PCR={pcr_peak.tolist()}")

    # Blend sweep into LB-best 4-stack (the LB-validated primary)
    log(f"\n=== fixed-bias blend sweep — v6_iso × LB-best 4-stack (anchor {lb4_bal:.5f}) ===")
    rows_4 = []
    for a in alphas:
        b = log_blend([lb4_oof, v6_iso], np.array([1 - a, a]))
        bal_oof = bal_at_bias(b, y)
        pred_b = predict(b)
        pcr_b = per_class_recall(y, pred_b)
        errs_b = (pred_b != y).sum()
        # Jaccard vs LB-best 4-stack (errors)
        pred_lb4 = predict(lb4_oof)
        e1 = pred_lb4 != y
        e2 = pred_b != y
        jacc = (e1 & e2).sum() / max((e1 | e2).sum(), 1)
        rows_4.append({"alpha": a, "oof": float(bal_oof),
                       "delta_vs_lb4": float(bal_oof - lb4_bal),
                       "errs": int(errs_b), "pcr": pcr_b.round(5).tolist(),
                       "jaccard": float(jacc)})
        log(f"  α={a:5.3f}  OOF={bal_oof:.5f}  Δ={bal_oof-lb4_bal:+.5f}  "
            f"errs={errs_b}  PCR={pcr_b.round(4).tolist()}  J={jacc:.4f}")
    best_4 = max(rows_4, key=lambda r: r["delta_vs_lb4"])

    g4 = (best_4["delta_vs_lb4"] >= GATE_LB_DELTA and
          best_4["jaccard"] < GATE_JACC and
          all(p >= q for p, q in zip(best_4["pcr"], GATE_PCR.tolist())))
    log(f"\nGate 4 (Δ vs LB4 ≥ {GATE_LB_DELTA}, Jacc < {GATE_JACC}, PCR ok):  {g4}")
    log(f"  best vs LB4: α={best_4['alpha']:.3f}  Δ={best_4['delta_vs_lb4']:+.5f}  "
        f"Jacc={best_4['jaccard']:.4f}")

    # Emit submission only if ALL gates pass
    emit = g1 and g2 and g3 and g4
    log(f"\n=== EMIT DECISION: {emit} (g1={g1} g2={g2} g3={g3} g4={g4}) ===")
    if emit:
        a = best_4["alpha"]
        test_blend = log_blend([lb4_test, v6_iso_te], np.array([1 - a, a]))
        pred_t = predict(test_blend)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub["Irrigation_Need"] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_v6_a{int(a*1000):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"WROTE {path}")

    out = dict(
        v6_argmax_oof=float(v6_argmax), v6_tuned_oof=float(v6_tuned),
        v6_iso_tuned_oof=float(v6_iso_tuned), v6_errs_at_bias=int(v6_errs),
        v6_pcr=pcr_v6.tolist(),
        lb3_bal=float(lb3_bal), lb4_bal=float(lb4_bal),
        sweep_vs_lb3=rows_3, best_vs_lb3=best_3,
        sweep_vs_lb4=rows_4, best_vs_lb4=best_4,
        gates=dict(g1=bool(g1), g2=bool(g2), g3=bool(g3), g4=bool(g4),
                   emit=bool(emit)),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "v6_blend_gate_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {ART / 'v6_blend_gate_results.json'}")


if __name__ == "__main__":
    main()
