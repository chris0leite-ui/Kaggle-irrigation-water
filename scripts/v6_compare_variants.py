"""Compare v6 (108-pool) vs v6_lbpool (62-pool) vs LB-best v1.

Decision rule (mirroring CLAUDE.md gate doctrine + linear-projection):
  Variant must satisfy ALL 4 gates to warrant LB probe:
    G1. iso OOF ≥ 0.98080
    G2. errs at recipe bias ≤ 9550
    G3. peak-α blend per-class recall ≥ [0.9950, 0.9690, 0.9770]
    G4. peak-α blend Δ vs LB-best 4-stack ≥ +2e-4 AND Jaccard < 0.97

Linear-projection rule (per 2026-04-25 LR closure):
  If a variant's α=peak Δ < +2e-4 OR PCR fails, do NOT probe smaller α.
  The OOF→LB gap inflation is roughly linear in α; conservative dilution
  manufactures smaller positive Δ but does NOT clear the projected LB.

Outputs:
  scripts/artifacts/v6_compare_results.json
  submission_v6{full|lb}_a{α}.csv (only for variants passing all 4 gates)
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
# Calibrated against tier1b v1 LB-best (the reference meta-stacker that landed
# LB 0.98094): standalone iso OOF 0.98059, errs 9044, PCR [0.99556, 0.97124,
# 0.97496]; LB-best 4-stack PCR [0.99553, 0.96951, 0.97749].
GATE_ISO = 0.98059                                  # ≥ v1 reference
GATE_ERRS = 9100                                    # ≤ v1 + 50 noise margin
GATE_PCR = np.array([0.99503, 0.96901, 0.97699])    # 4-stack PCR − 5e-4 floor
GATE_LB_DELTA = 2e-4
GATE_JACC = 0.97


def predict(p, bias=BIAS):
    return (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return cm.diagonal() / cm.sum(axis=1).clip(min=1)


def evaluate_variant(name, oof, test_arr, lb3_oof, lb3_test, lb4_oof, lb4_test, y):
    log(f"\n=========== {name} ===========")
    iso_oof, iso_test = iso_cal(oof, test_arr, y)

    argmax_b = balanced_accuracy_score(y, oof.argmax(1))
    raw_tuned = bal_at_bias(oof, y)
    iso_tuned = bal_at_bias(iso_oof, y)
    pred_iso = predict(iso_oof)
    errs = int((pred_iso != y).sum())
    pcr = per_class_recall(y, pred_iso)

    log(f"  argmax OOF        {argmax_b:.5f}")
    log(f"  raw @recipe-bias  {raw_tuned:.5f}")
    log(f"  iso @recipe-bias  {iso_tuned:.5f}")
    log(f"  errs (iso)        {errs}")
    log(f"  PCR [L,M,H]       {pcr.round(5).tolist()}")

    g1 = iso_tuned >= GATE_ISO
    g2 = errs <= GATE_ERRS
    log(f"  G1 (iso ≥ {GATE_ISO}):  {g1}")
    log(f"  G2 (errs ≤ {GATE_ERRS}):    {g2}")

    # Sweep into LB-best 3-stack
    log(f"  --- sweep vs LB-best 3-stack (anchor {bal_at_bias(lb3_oof, y):.5f}) ---")
    rows_3 = []
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for a in alphas:
        b = log_blend([lb3_oof, iso_oof], np.array([1 - a, a]))
        bb = bal_at_bias(b, y); pb = predict(b)
        rows_3.append({"alpha": a, "oof": float(bb),
                       "errs": int((pb != y).sum()),
                       "pcr": per_class_recall(y, pb).round(5).tolist()})
    base3 = bal_at_bias(lb3_oof, y)
    best3 = max(rows_3, key=lambda r: r["oof"])
    best3["delta"] = best3["oof"] - base3
    for r in rows_3:
        marker = " ← peak" if r is best3 else ""
        log(f"    α={r['alpha']:5.3f}  OOF={r['oof']:.5f}  Δ={r['oof']-base3:+.5f}  "
            f"errs={r['errs']}  PCR={[round(p,4) for p in r['pcr']]}{marker}")

    # Sweep into LB-best 4-stack
    log(f"  --- sweep vs LB-best 4-stack (anchor {bal_at_bias(lb4_oof, y):.5f}) ---")
    base4 = bal_at_bias(lb4_oof, y)
    pred_lb4 = predict(lb4_oof)
    rows_4 = []
    for a in alphas:
        b = log_blend([lb4_oof, iso_oof], np.array([1 - a, a]))
        bb = bal_at_bias(b, y); pb = predict(b)
        e1 = pred_lb4 != y; e2 = pb != y
        jacc = (e1 & e2).sum() / max((e1 | e2).sum(), 1)
        rows_4.append({"alpha": a, "oof": float(bb),
                       "errs": int((pb != y).sum()),
                       "pcr": per_class_recall(y, pb).round(5).tolist(),
                       "jaccard": float(jacc)})
    best4 = max(rows_4, key=lambda r: r["oof"])
    best4["delta"] = best4["oof"] - base4
    for r in rows_4:
        marker = " ← peak" if r is best4 else ""
        log(f"    α={r['alpha']:5.3f}  OOF={r['oof']:.5f}  Δ={r['oof']-base4:+.5f}  "
            f"errs={r['errs']}  J={r['jaccard']:.4f}  PCR={[round(p,4) for p in r['pcr']]}"
            f"{marker}")

    pcr_peak = np.array(best4["pcr"])
    g3 = bool(np.all(pcr_peak >= GATE_PCR))
    g4 = (best4["delta"] >= GATE_LB_DELTA and best4["jaccard"] < GATE_JACC)
    log(f"  G3 (PCR ≥ floor at peak):  {g3}  PCR={pcr_peak.tolist()}")
    log(f"  G4 (Δ ≥ {GATE_LB_DELTA} AND Jacc < {GATE_JACC}):  {g4}  "
        f"Δ={best4['delta']:+.5f}  J={best4['jaccard']:.4f}")

    emit = g1 and g2 and g3 and g4
    log(f"  EMIT: {emit}  (g1={g1} g2={g2} g3={g3} g4={g4})")

    return dict(
        name=name,
        argmax_oof=float(argmax_b),
        raw_tuned=float(raw_tuned),
        iso_tuned=float(iso_tuned),
        errs=errs, pcr=pcr.tolist(),
        sweep_lb3=rows_3, best_lb3=best3,
        sweep_lb4=rows_4, best_lb4=best4,
        gates=dict(g1=bool(g1), g2=bool(g2), g3=bool(g3),
                   g4=bool(g4), emit=bool(emit)),
        iso_oof=iso_oof, iso_test=iso_test,
    )


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    lb3_oof, lb3_test = build_lbbest_stack(y)
    meta_v1 = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    meta_v1_te = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    meta_v1_iso, meta_v1_iso_te = iso_cal(meta_v1, meta_v1_te, y)
    lb4_oof = log_blend([lb3_oof, meta_v1_iso], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, meta_v1_iso_te], np.array([0.7, 0.3]))

    log(f"LB-best 3-stack OOF = {bal_at_bias(lb3_oof, y):.5f}")
    log(f"LB-best 4-stack OOF = {bal_at_bias(lb4_oof, y):.5f}")

    variants = []
    for name, suffix in [("v6_full (108-pool + agg)", "v6"),
                          ("v6_lbpool (62-pool + agg)", "v6lb")]:
        oof_p = ART / f"oof_xgb_metastack_{suffix}.npy"
        test_p = ART / f"test_xgb_metastack_{suffix}.npy"
        if not oof_p.exists():
            log(f"\n[skip] {name} — {oof_p} not yet produced")
            continue
        oof = np.load(oof_p).astype(np.float32)
        tst = np.load(test_p).astype(np.float32)
        if (oof.sum(1) < 1e-3).any():
            log(f"\n[skip] {name} — partial-fold artefact")
            continue
        v = evaluate_variant(name, oof, tst, lb3_oof, lb3_test, lb4_oof, lb4_test, y)
        v["suffix"] = suffix
        variants.append(v)

    # Emit submission for any variant that passes
    sample = pd.read_csv(DATA / "sample_submission.csv")
    for v in variants:
        if v["gates"]["emit"]:
            a = v["best_lb4"]["alpha"]
            tb = log_blend([lb4_test, v["iso_test"]], np.array([1 - a, a]))
            pred_t = predict(tb)
            sub = sample.copy()
            sub["Irrigation_Need"] = [CLASSES[i] for i in pred_t]
            tag = f"{v['suffix']}_a{int(a*1000):03d}"
            path = SUB / f"submission_{tag}.csv"
            sub.to_csv(path, index=False)
            log(f"\nWROTE {path}")

    out = dict(
        lb3_bal=float(bal_at_bias(lb3_oof, y)),
        lb4_bal=float(bal_at_bias(lb4_oof, y)),
        variants=[{k: v for k, v in vv.items()
                   if k not in ("iso_oof", "iso_test")}
                  for vv in variants],
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "v6_compare_results.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote {ART / 'v6_compare_results.json'}")


if __name__ == "__main__":
    main()
