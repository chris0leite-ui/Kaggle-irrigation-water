"""Mech A boundary-confined TTA — 4-criteria gate vs LB-best 4-stack.

Mirrors the v6 / mech_b / mech_d gate pattern. Loads
oof/test_recipe_full_te_btta095k10s005.npy and tests:
  - standalone iso vs vanilla recipe (G1, G2)
  - substitution into LB-best 3-stack (replace recipe leg)
  - soft-blend into LB-best 4-stack
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, DATA, SUB, BIAS, build_lbbest_stack, iso_cal, log, bal_at_bias, normed,
)

CLASSES = ["Low", "Medium", "High"]
GATE_LB_DELTA = 2e-4
GATE_JACC = 0.97
PCR_FLOOR_DELTA = 5e-4

SUFFIX = os.environ.get("TTA_SUFFIX", "btta095k10s005")


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
    lb4_bal = bal_at_bias(lb4_oof, y)
    pred_lb4 = predict(lb4_oof)
    pcr_lb4 = per_class_recall(y, pred_lb4)
    pcr_floor = pcr_lb4 - PCR_FLOOR_DELTA
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f}")

    rv_oof = normed(np.load(ART / "oof_recipe_full_te.npy").astype(np.float32))
    rv_te = normed(np.load(ART / "test_recipe_full_te.npy").astype(np.float32))
    rv_iso, rv_iso_te = iso_cal(rv_oof, rv_te, y)
    rv_iso_tuned = bal_at_bias(rv_iso, y)
    pred_rv = predict(rv_iso)
    rv_errs = int((pred_rv != y).sum())
    log(f"\nrecipe vanilla:  iso={rv_iso_tuned:.5f}  errs={rv_errs}")

    ma_path_oof = ART / f"oof_recipe_full_te_{SUFFIX}.npy"
    ma_path_te = ART / f"test_recipe_full_te_{SUFFIX}.npy"
    if not (ma_path_oof.exists() and ma_path_te.exists()):
        raise SystemExit(f"missing {ma_path_oof}")
    ma_oof = normed(np.load(ma_path_oof).astype(np.float32))
    ma_te = normed(np.load(ma_path_te).astype(np.float32))
    ma_iso, ma_iso_te = iso_cal(ma_oof, ma_te, y)
    ma_argmax = balanced_accuracy_score(y, ma_oof.argmax(1))
    ma_iso_tuned = bal_at_bias(ma_iso, y)
    pred_ma = predict(ma_iso)
    ma_errs = int((pred_ma != y).sum())
    ma_pcr = per_class_recall(y, pred_ma)

    log(f"\n=== Mech A standalone ({SUFFIX}) ===")
    log(f"  argmax OOF        {ma_argmax:.5f}")
    log(f"  iso @recipe-bias  {ma_iso_tuned:.5f}  (Δ vs vanilla {ma_iso_tuned - rv_iso_tuned:+.5f})")
    log(f"  errs (iso)        {ma_errs}  (Δ {ma_errs - rv_errs:+d})")
    log(f"  PCR [L,M,H]       {ma_pcr.round(5).tolist()}")

    g1 = ma_iso_tuned >= rv_iso_tuned - 1e-5
    g2 = ma_errs <= rv_errs

    # Substitution test (replace recipe in lb3)
    s1 = normed(np.load(ART / "oof_recipe_pseudolabel.npy").astype(np.float32))
    s1_te = normed(np.load(ART / "test_recipe_pseudolabel.npy").astype(np.float32))
    s7 = normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy").astype(np.float32))
    s7_te = normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy").astype(np.float32))
    rm = normed(np.load(ART / "oof_realmlp.npy").astype(np.float32))
    rm_te = normed(np.load(ART / "test_realmlp.npy").astype(np.float32))
    nr = normed(np.load(ART / "oof_xgb_nonrule.npy").astype(np.float32))
    nr_te = normed(np.load(ART / "test_xgb_nonrule.npy").astype(np.float32))
    nr_iso, nr_iso_te = iso_cal(nr, nr_te, y)

    new_lb3 = log_blend([ma_oof, s1, s7], np.array([0.25, 0.35, 0.40]))
    new_lb3_te = log_blend([ma_te, s1_te, s7_te], np.array([0.25, 0.35, 0.40]))
    new_s2 = log_blend([new_lb3, rm], np.array([0.8, 0.2]))
    new_s2_te = log_blend([new_lb3_te, rm_te], np.array([0.8, 0.2]))
    new_s3 = log_blend([new_s2, nr_iso], np.array([0.925, 0.075]))
    new_s3_te = log_blend([new_s2_te, nr_iso_te], np.array([0.925, 0.075]))
    new_lb4 = log_blend([new_s3, mv1_iso], np.array([0.7, 0.3]))
    new_lb4_te = log_blend([new_s3_te, mv1_iso_te], np.array([0.7, 0.3]))
    sub_bal = bal_at_bias(new_lb4, y)
    pred_sub = predict(new_lb4)
    sub_pcr = per_class_recall(y, pred_sub)
    sub_errs = int((pred_sub != y).sum())
    log(f"\n=== substitution test ===")
    log(f"  new 4-stack OOF = {sub_bal:.5f}  Δ vs lb4 = {sub_bal - lb4_bal:+.5f}")
    log(f"  errs={sub_errs}  PCR={sub_pcr.round(5).tolist()}")

    # Soft-blend sweep
    log(f"\n=== sweep — ma_iso × LB-best 4-stack ===")
    rows = []
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for a in alphas:
        b = log_blend([lb4_oof, ma_iso], np.array([1 - a, a]))
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
    sub_g3 = bool(np.all(sub_pcr >= pcr_floor))
    sub_g4 = (sub_bal - lb4_bal) >= GATE_LB_DELTA

    log(f"\nG1 standalone iso ≥ vanilla iso:  {g1}")
    log(f"G2 errs ≤ vanilla errs:            {g2}")
    log(f"G3 (PCR floor at peak):            {g3}  PCR={pcr_peak.tolist()}")
    log(f"G4 (Δ ≥ {GATE_LB_DELTA} AND J < {GATE_JACC}):  {g4}  Δ={best['delta_vs_lb4']:+.5f} J={best['jaccard']:.4f} α={best['alpha']:.3f}")
    log(f"sub_G3:  {sub_g3}  PCR={sub_pcr.round(5).tolist()}")
    log(f"sub_G4 (sub Δ ≥ {GATE_LB_DELTA}):  {sub_g4}  Δ_sub={sub_bal - lb4_bal:+.5f}")

    emit_blend = bool(g1 and g2 and g3 and g4)
    emit_sub = bool(g1 and g2 and sub_g3 and sub_g4)
    log(f"\n=== EMIT BLEND: {emit_blend}  EMIT SUBSTITUTION: {emit_sub} ===")

    if emit_blend or emit_sub:
        sample = pd.read_csv(DATA / "sample_submission.csv")
    if emit_blend:
        a = best["alpha"]
        tb = log_blend([lb4_test, ma_iso_te], np.array([1 - a, a]))
        pred_t = predict(tb)
        sub = sample.copy()
        sub["Irrigation_Need"] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_mech_a_blend_{SUFFIX}_a{int(a*1000):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"WROTE {path}  (still requires explicit user approval before submit)")
    if emit_sub:
        pred_t = predict(new_lb4_te)
        sub = sample.copy()
        sub["Irrigation_Need"] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_mech_a_sub_{SUFFIX}.csv"
        sub.to_csv(path, index=False)
        log(f"WROTE {path}  (still requires explicit user approval before submit)")

    out = dict(
        suffix=SUFFIX,
        recipe_iso_tuned=float(rv_iso_tuned), recipe_errs=rv_errs,
        ma_iso_tuned=float(ma_iso_tuned), ma_errs=ma_errs,
        ma_pcr=ma_pcr.tolist(), ma_argmax=float(ma_argmax),
        sub_bal=float(sub_bal), sub_errs=sub_errs, sub_pcr=sub_pcr.tolist(),
        sub_delta=float(sub_bal - lb4_bal),
        lb4_bal=float(lb4_bal), pcr_floor=pcr_floor.tolist(),
        sweep=rows, best_blend=best,
        gates=dict(g1=g1, g2=g2, g3=g3, g4=g4,
                   sub_g3=sub_g3, sub_g4=sub_g4,
                   emit_blend=emit_blend, emit_sub=emit_sub),
        elapsed_sec=float(time.time() - t0),
    )
    out_path = ART / f"mech_a_blend_gate_{SUFFIX}_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
