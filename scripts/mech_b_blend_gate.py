"""Mech B anchor-uncertainty-weighted recipe — blend gate analysis.

Loads `oof/test_recipe_full_te_anchw{α}.npy` and tests it both as a
standalone replacement for `recipe_full_te` (substituted INTO the LB-best
3-stack reconstruction) AND as a soft-blend leg vs LB-best 4-stack.

4-criteria gate:
  G1. standalone iso OOF ≥ recipe vanilla iso OOF (no regression)
  G2. errs at recipe bias ≤ recipe vanilla errs
  G3. peak-α blend per-class recall ≥ LB-best 4-stack PCR − 5e-4
  G4. peak-α blend Δ vs LB-best 4-stack ≥ +2e-4 AND Jaccard < 0.97

Substitution test: build LB-best 3-stack with recipe_anchw replacing
recipe in the (recipe, pseudo_s1, pseudo_s7) log_blend at weights
(0.25, 0.35, 0.40); pseudo legs unchanged.

Outputs:
  scripts/artifacts/mech_b_blend_gate_alpha{α}_results.json
  submissions/submission_mech_b_alpha{α}_a{α'}.csv  (only if all gates pass)
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

# Pull suffix from env (default "anchw20" for ALPHA=2.0)
ALPHA = float(os.environ.get("ANCHOR_WEIGHT_ALPHA", "2.0"))
SUFFIX = f"anchw{int(abs(ALPHA)*10):02d}"


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
    lb4_bal = bal_at_bias(lb4_oof, y)
    pred_lb4 = predict(lb4_oof)
    pcr_lb4 = per_class_recall(y, pred_lb4)
    pcr_floor = pcr_lb4 - PCR_FLOOR_DELTA
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f}")
    log(f"  PCR  [L,M,H] = {pcr_lb4.round(5).tolist()}")
    log(f"  floor [L,M,H] = {pcr_floor.round(5).tolist()}")

    # Recipe vanilla baseline (G1/G2 anchor)
    rv_oof = normed(np.load(ART / "oof_recipe_full_te.npy").astype(np.float32))
    rv_te = normed(np.load(ART / "test_recipe_full_te.npy").astype(np.float32))
    rv_iso, rv_iso_te = iso_cal(rv_oof, rv_te, y)
    rv_iso_tuned = bal_at_bias(rv_iso, y)
    pred_rv = predict(rv_iso)
    rv_errs = int((pred_rv != y).sum())
    log(f"\nrecipe vanilla (anchor for G1/G2):")
    log(f"  iso tuned OOF = {rv_iso_tuned:.5f}  errs={rv_errs}")

    # Mech B candidate
    mb_path_oof = ART / f"oof_recipe_full_te_{SUFFIX}.npy"
    mb_path_te = ART / f"test_recipe_full_te_{SUFFIX}.npy"
    if not (mb_path_oof.exists() and mb_path_te.exists()):
        raise SystemExit(f"missing {mb_path_oof} or {mb_path_te}")
    mb_oof = normed(np.load(mb_path_oof).astype(np.float32))
    mb_te = normed(np.load(mb_path_te).astype(np.float32))
    mb_iso, mb_iso_te = iso_cal(mb_oof, mb_te, y)
    mb_argmax = balanced_accuracy_score(y, mb_oof.argmax(1))
    mb_iso_tuned = bal_at_bias(mb_iso, y)
    pred_mb = predict(mb_iso)
    mb_errs = int((pred_mb != y).sum())
    mb_pcr = per_class_recall(y, pred_mb)

    log(f"\n=== Mech B recipe + anchor-weight α={ALPHA} standalone ===")
    log(f"  argmax OOF       {mb_argmax:.5f}")
    log(f"  iso @recipe-bias {mb_iso_tuned:.5f}")
    log(f"  errs (iso)       {mb_errs}")
    log(f"  PCR [L,M,H]      {mb_pcr.round(5).tolist()}")

    g1 = mb_iso_tuned >= rv_iso_tuned - 1e-5
    g2 = mb_errs <= rv_errs
    log(f"  G1 (iso ≥ recipe vanilla {rv_iso_tuned:.5f}):  {g1}")
    log(f"  G2 (errs ≤ recipe vanilla {rv_errs}):          {g2}")

    # Substitution test: replace recipe in lb3 with mb
    log("\n=== substitution test (recipe → recipe_anchw in lb3) ===")
    s1 = normed(np.load(ART / "oof_recipe_pseudolabel.npy").astype(np.float32))
    s1_te = normed(np.load(ART / "test_recipe_pseudolabel.npy").astype(np.float32))
    s7 = normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy").astype(np.float32))
    s7_te = normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy").astype(np.float32))
    rm = normed(np.load(ART / "oof_realmlp.npy").astype(np.float32))
    rm_te = normed(np.load(ART / "test_realmlp.npy").astype(np.float32))
    nr = normed(np.load(ART / "oof_xgb_nonrule.npy").astype(np.float32))
    nr_te = normed(np.load(ART / "test_xgb_nonrule.npy").astype(np.float32))
    nr_iso, nr_iso_te = iso_cal(nr, nr_te, y)

    # New 3-stack: replace recipe with mb (use raw mb, not iso, to mirror anchor)
    new_lb3 = log_blend([mb_oof, s1, s7], np.array([0.25, 0.35, 0.40]))
    new_lb3_te = log_blend([mb_te, s1_te, s7_te], np.array([0.25, 0.35, 0.40]))
    new_s2 = log_blend([new_lb3, rm], np.array([0.8, 0.2]))
    new_s2_te = log_blend([new_lb3_te, rm_te], np.array([0.8, 0.2]))
    new_s3 = log_blend([new_s2, nr_iso], np.array([0.925, 0.075]))
    new_s3_te = log_blend([new_s2_te, nr_iso_te], np.array([0.925, 0.075]))
    new_lb4 = log_blend([new_s3, mv1_iso], np.array([0.7, 0.3]))
    new_lb4_te = log_blend([new_s3_te, mv1_iso_te], np.array([0.7, 0.3]))
    sub_bal = bal_at_bias(new_lb4, y)
    pred_sub = predict(new_lb4)
    sub_errs = int((pred_sub != y).sum())
    sub_pcr = per_class_recall(y, pred_sub)
    log(f"  substituted 4-stack OOF = {sub_bal:.5f}  Δ vs lb4 = {sub_bal - lb4_bal:+.5f}")
    log(f"  errs={sub_errs}  PCR={sub_pcr.round(5).tolist()}")

    # Soft-blend mb_iso × LB-best 4-stack (alpha sweep)
    log(f"\n=== soft-blend sweep — mb_iso × LB-best 4-stack (anchor {lb4_bal:.5f}) ===")
    rows = []
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for a in alphas:
        b = log_blend([lb4_oof, mb_iso], np.array([1 - a, a]))
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

    # Substitution gate: similar to G4 but on the substituted stack
    sub_g4 = (sub_bal - lb4_bal) >= GATE_LB_DELTA
    sub_g3 = bool(np.all(sub_pcr >= pcr_floor))
    log(f"\nsubstitution gate: G3={sub_g3} G4={sub_g4} "
        f"Δ_sub={sub_bal - lb4_bal:+.5f}")

    emit_blend = bool(g1 and g2 and g3 and g4)
    emit_sub = bool(g1 and g2 and sub_g3 and sub_g4)
    log(f"\n=== EMIT BLEND: {emit_blend} (g1={g1} g2={g2} g3={g3} g4={g4}) ===")
    log(f"=== EMIT SUBSTITUTION: {emit_sub} (g1={g1} g2={g2} sub_g3={sub_g3} sub_g4={sub_g4}) ===")

    if emit_blend or emit_sub:
        sample = pd.read_csv(DATA / "sample_submission.csv")
    if emit_blend:
        a = best["alpha"]
        tb = log_blend([lb4_test, mb_iso_te], np.array([1 - a, a]))
        pred_t = predict(tb)
        sub = sample.copy()
        sub["Irrigation_Need"] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_mech_b_blend_{SUFFIX}_a{int(a*1000):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"WROTE {path}  (still requires explicit user approval before submit)")
    if emit_sub:
        pred_t = predict(new_lb4_te)
        sub = sample.copy()
        sub["Irrigation_Need"] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_mech_b_sub_{SUFFIX}.csv"
        sub.to_csv(path, index=False)
        log(f"WROTE {path}  (still requires explicit user approval before submit)")

    out = dict(
        alpha=ALPHA, suffix=SUFFIX,
        recipe_iso_tuned=float(rv_iso_tuned), recipe_errs=int(rv_errs),
        mb_iso_tuned=float(mb_iso_tuned), mb_errs=int(mb_errs),
        mb_pcr=mb_pcr.tolist(), mb_argmax=float(mb_argmax),
        sub_bal=float(sub_bal), sub_errs=int(sub_errs), sub_pcr=sub_pcr.tolist(),
        sub_delta=float(sub_bal - lb4_bal),
        lb4_bal=float(lb4_bal), pcr_floor=pcr_floor.tolist(),
        sweep=rows, best_blend=best,
        gates=dict(g1=g1, g2=g2, g3=g3, g4=g4,
                   sub_g3=sub_g3, sub_g4=sub_g4,
                   emit_blend=emit_blend, emit_sub=emit_sub),
        elapsed_sec=float(time.time() - t0),
    )
    out_path = ART / f"mech_b_blend_gate_{SUFFIX}_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
