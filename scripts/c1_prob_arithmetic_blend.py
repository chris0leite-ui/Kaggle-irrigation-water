"""C1: Replace log-blend with prob-space arithmetic that decouples classes.

Mechanism: instead of P_blend = softmax(α·log(P_a) + (1-α)·log(P_b)),
use prob-space arithmetic with per-class weights:

    P_blend_k = (1-α_k)·P_primary_k + α_k·P_b2_k    for k in {L, M, H}
    then renormalize so rows sum to 1

Setting α_H=0 means: P_blend_H = P_primary_H exactly. After
renormalization the H probability magnitude shifts slightly because
L and M change (their sum changes), but the relative-to-LM ordering
of H is preserved.

Three variants tested:
  V1: per-class prob-arith blend with α_H=0, sweep α_L, α_M
  V2: conditional override — where primary argmax IS High, keep primary
      else, log-blend(primary, B2) at α
  V3: mask-blend — where primary P_H > threshold, keep primary
      else, log-blend at α

All against the LB-best 4-stack PRIMARY (with full-OOF iso).

Outputs:
  scripts/artifacts/c1_prob_arithmetic_results.json
  submissions/submission_c1_*.csv (only if 4-gate PASS)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] C1: {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_full(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32); tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return normed(oo), normed(tt)


def per_class_recall(y, pred):
    return np.array([(pred[y == k] == k).mean() for k in range(3)])


def four_gate(prim_o, blend_o, y):
    pred_p = (np.log(np.clip(prim_o,  1e-12, 1)) + RECIPE_BIAS).argmax(1)
    pred_b = (np.log(np.clip(blend_o, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
    bal_p = balanced_accuracy_score(y, pred_p)
    bal_b = balanced_accuracy_score(y, pred_b)
    pcr_p = per_class_recall(y, pred_p)
    pcr_b = per_class_recall(y, pred_b)
    pcr_d = pcr_b - pcr_p

    add_h = int(((pred_b == 2) & (pred_p != 2)).sum())
    rem_h = int(((pred_p == 2) & (pred_b != 2)).sum())
    net_h = add_h - rem_h
    churn = add_h + rem_h
    g4r = abs(net_h) / max(churn, 1)

    g1 = (bal_b - bal_p) >= 2e-4
    g2 = bool((pcr_d >= -5e-4).all())
    g4 = (net_h > 0) and (g4r >= 0.5)

    return {
        "delta_oof": float(bal_b - bal_p),
        "pcr_delta": [float(x) for x in pcr_d],
        "errs_blend": int((pred_b != y).sum()),
        "errs_prim":  int((pred_p != y).sum()),
        "net_h": net_h, "churn_h": churn, "g4_ratio": float(g4r),
        "g1": bool(g1), "g2": bool(g2), "g4": bool(g4),
        "n_pass_no_g3": int(g1) + int(g2) + int(g4),
    }


def four_gate_test_diff(prim_t, blend_t, primary_pred):
    """Helper: test-side row diff vs current PRIMARY submission."""
    pred = (np.log(np.clip(blend_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
    return int((pred != primary_pred).sum())


def emit(blend_t, name, gate, mark="★"):
    if gate["n_pass_no_g3"] != 3:
        return False
    pred = (np.log(np.clip(blend_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred]
    path = SUB / f"submission_{name}.csv"
    sub.to_csv(path, index=False)
    log(f"  {mark} 4-gate PASS — wrote {path}")
    return True


def build_4stack_and_primary(y):
    """4-stack base + LB-best PRIMARY (with full-OOF iso on metastack)."""
    r  = normed(np.load(ART / "oof_recipe_full_te.npy"))
    rt = normed(np.load(ART / "test_recipe_full_te.npy"))
    s1 = normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1t= normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7 = normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7t= normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm = normed(np.load(ART / "oof_realmlp.npy"))
    rmt= normed(np.load(ART / "test_realmlp.npy"))
    nr = normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nrt= normed(np.load(ART / "test_xgb_nonrule.npy"))
    nr_iso, nrt_iso = iso_full(nr, nrt, y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    st1_o = log_blend([lb3_o, rm], np.array([0.80, 0.20]))
    st1_t = log_blend([lb3_t, rmt], np.array([0.80, 0.20]))
    base_o = log_blend([st1_o, nr_iso], np.array([0.925, 0.075]))
    base_t = log_blend([st1_t, nrt_iso], np.array([0.925, 0.075]))

    ms = normed(np.load(ART / "oof_xgb_metastack.npy"))
    mst= normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_iso, mst_iso = iso_full(ms, mst, y)
    prim_o = log_blend([base_o, ms_iso], np.array([0.70, 0.30]))
    prim_t = log_blend([base_t, mst_iso], np.array([0.70, 0.30]))
    return base_o, base_t, prim_o, prim_t


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("loading components")
    base_o, base_t, prim_o, prim_t = build_4stack_and_primary(y)

    # B2 raw + iso
    b2_o = normed(np.load(ART / "oof_xgb_metastack_perfoldiso_inputs.npy"))
    b2_t = normed(np.load(ART / "test_xgb_metastack_perfoldiso_inputs.npy"))
    b2_iso_o, b2_iso_t = iso_full(b2_o, b2_t, y)

    # B2 as a primary itself (4-stack + B2_iso @ 0.30)
    b2_prim_o = log_blend([base_o, b2_iso_o], np.array([0.70, 0.30]))
    b2_prim_t = log_blend([base_t, b2_iso_t], np.array([0.70, 0.30]))

    primary_csv = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
    primary_pred_csv = primary_csv[TARGET].map(CLS2IDX).to_numpy()

    log(f"PRIMARY OOF @ recipe = {balanced_accuracy_score(y, (np.log(np.clip(prim_o, 1e-12, 1)) + RECIPE_BIAS).argmax(1)):.5f}")
    log(f"B2_prim OOF @ recipe = {balanced_accuracy_score(y, (np.log(np.clip(b2_prim_o, 1e-12, 1)) + RECIPE_BIAS).argmax(1)):.5f}")

    results = {"v1_perclass_arith": [], "v2_conditional": [], "v3_mask": []}

    # ─────────────────────────────────────────────────────────────────────
    log("\n=== V1 — Per-class prob-arithmetic blend (decoupled, α_H=0) ===")
    log("    BLEND: P_b_k = (1-α_k)·prim_k + α_k·b2_iso_k, then renormalize")
    log("    α_H pinned to 0 → preserves PRIMARY's H prob exactly before normalization")

    def perclass_arith_blend(prim, cand, alphas):
        ALPHA = np.array(alphas)
        # P_blend_k = (1-α_k)*prim_k + α_k*cand_k
        blend = (1 - ALPHA)[None, :] * prim + ALPHA[None, :] * cand
        return normed(blend)

    for aL in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        for aM in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
            alphas = [aL, aM, 0.0]
            blend_o = perclass_arith_blend(prim_o, b2_iso_o, alphas)
            blend_t = perclass_arith_blend(prim_t, b2_iso_t, alphas)
            gate = four_gate(prim_o, blend_o, y)
            diff = four_gate_test_diff(prim_t, blend_t, primary_pred_csv)
            ok = gate["n_pass_no_g3"] == 3
            if gate["delta_oof"] > 1e-5 or ok:
                v = "PASS" if ok else f"{gate['n_pass_no_g3']}/3"
                log(f"  αL={aL:.2f} αM={aM:.2f} αH=0  Δ={gate['delta_oof']:+.5f}  "
                    f"PCR=[{gate['pcr_delta'][0]:+.5f}, {gate['pcr_delta'][1]:+.5f}, {gate['pcr_delta'][2]:+.5f}]  "
                    f"net_H={gate['net_h']:+d}  diff={diff}  G124={v}")
            emit(blend_t, f"c1_v1_aL{int(aL*100):03d}_aM{int(aM*100):03d}", gate)
            results["v1_perclass_arith"].append({"aL": aL, "aM": aM, "aH": 0.0, "diff": diff, **gate})

    # ─────────────────────────────────────────────────────────────────────
    log("\n=== V2 — Conditional override: primary-H rows → primary; else log-blend ===")
    log("    Where primary argmax IS High, take PRIMARY's prediction.")
    log("    Else, use log-blend(prim, B2_iso) at α.")

    pred_prim_o = (np.log(np.clip(prim_o, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
    pred_prim_t = (np.log(np.clip(prim_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
    h_mask_o = pred_prim_o == 2
    h_mask_t = pred_prim_t == 2

    for a in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        # Compute log-blend for non-H rows
        blend_full_o = log_blend([prim_o, b2_iso_o], np.array([1 - a, a]))
        blend_full_t = log_blend([prim_t, b2_iso_t], np.array([1 - a, a]))
        # Override: where prim argmax was H, use prim; else use blend
        blend_o = np.where(h_mask_o[:, None], prim_o, blend_full_o)
        blend_t = np.where(h_mask_t[:, None], prim_t, blend_full_t)
        # Renormalize after stitch (no-op if both sides already normalized)
        blend_o = normed(blend_o); blend_t = normed(blend_t)
        gate = four_gate(prim_o, blend_o, y)
        diff = four_gate_test_diff(prim_t, blend_t, primary_pred_csv)
        ok = gate["n_pass_no_g3"] == 3
        v = "PASS" if ok else f"{gate['n_pass_no_g3']}/3"
        log(f"  α={a:.2f}  Δ={gate['delta_oof']:+.5f}  "
            f"PCR=[{gate['pcr_delta'][0]:+.5f}, {gate['pcr_delta'][1]:+.5f}, {gate['pcr_delta'][2]:+.5f}]  "
            f"net_H={gate['net_h']:+d}  diff={diff}  G124={v}")
        emit(blend_t, f"c1_v2_a{int(a*100):03d}", gate)
        results["v2_conditional"].append({"alpha": a, "diff": diff, **gate})

    # ─────────────────────────────────────────────────────────────────────
    log("\n=== V3 — Mask-blend: primary-H-confident rows → primary; else log-blend ===")
    log("    Where primary's max prob IS H AND >= τ, take PRIMARY's prediction.")
    log("    Else, use log-blend at α.")

    for tau in [0.50, 0.70, 0.85, 0.95]:
        # H-confident mask: primary's argmax is H AND prob > τ
        primary_max_o = prim_o.max(1)
        primary_max_t = prim_t.max(1)
        h_conf_mask_o = h_mask_o & (primary_max_o >= tau)
        h_conf_mask_t = h_mask_t & (primary_max_t >= tau)
        for a in [0.20, 0.30, 0.40]:
            blend_full_o = log_blend([prim_o, b2_iso_o], np.array([1 - a, a]))
            blend_full_t = log_blend([prim_t, b2_iso_t], np.array([1 - a, a]))
            blend_o = np.where(h_conf_mask_o[:, None], prim_o, blend_full_o)
            blend_t = np.where(h_conf_mask_t[:, None], prim_t, blend_full_t)
            blend_o = normed(blend_o); blend_t = normed(blend_t)
            gate = four_gate(prim_o, blend_o, y)
            diff = four_gate_test_diff(prim_t, blend_t, primary_pred_csv)
            ok = gate["n_pass_no_g3"] == 3
            v = "PASS" if ok else f"{gate['n_pass_no_g3']}/3"
            log(f"  τ={tau:.2f} α={a:.2f}  mask={h_conf_mask_o.sum()}/630k  "
                f"Δ={gate['delta_oof']:+.5f}  "
                f"PCR=[{gate['pcr_delta'][0]:+.5f}, {gate['pcr_delta'][1]:+.5f}, {gate['pcr_delta'][2]:+.5f}]  "
                f"net_H={gate['net_h']:+d}  diff={diff}  G124={v}")
            emit(blend_t, f"c1_v3_tau{int(tau*100):03d}_a{int(a*100):03d}", gate)
            results["v3_mask"].append({"tau": tau, "alpha": a,
                                       "mask_size": int(h_conf_mask_o.sum()), "diff": diff, **gate})

    out_path = ART / "c1_prob_arithmetic_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
