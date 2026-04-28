"""Three follow-up experiments to B2's G4 RESHUFFLE failure.

#1: Ensemble v1 + B2 metas at meta level (50/50, then α=0.30 into 4-stack)
#2: α-sweep B2_full meta to find G4-PASS region
#3: Per-class α blending of B2 with α_H=0 (preserve High exactly)

All run on existing artifacts. ~5 min CPU total.

Outputs:
  scripts/artifacts/b3_followup_results.json
  submissions/submission_b3_*.csv (only if 4-gate PASS)
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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_full(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32); tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return normed(oo), normed(tt)


def bal_at_bias(p, y, bias):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1))


def per_class_recall(y, pred):
    return np.array([(pred[y == k] == k).mean() for k in range(3)])


def jaccard(y, pa, pb):
    ea = pa != y; eb = pb != y
    return float((ea & eb).sum() / max((ea | eb).sum(), 1))


def four_gate_check(prim_o, blend_o, y, primary_pred):
    """4-gate against current PRIMARY (computed at recipe bias)."""
    pred_blend = (np.log(np.clip(blend_o, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
    pred_prim  = (np.log(np.clip(prim_o,  1e-12, 1)) + RECIPE_BIAS).argmax(1)

    bal_prim  = balanced_accuracy_score(y, pred_prim)
    bal_blend = balanced_accuracy_score(y, pred_blend)

    pcr_prim  = per_class_recall(y, pred_prim)
    pcr_blend = per_class_recall(y, pred_blend)
    pcr_delta = pcr_blend - pcr_prim

    add_h = int(((pred_blend == 2) & (pred_prim != 2)).sum())
    rem_h = int(((pred_prim == 2) & (pred_blend != 2)).sum())
    net_h = add_h - rem_h
    churn = add_h + rem_h
    g4_ratio = abs(net_h) / max(churn, 1)

    g1 = (bal_blend - bal_prim) >= 2e-4
    g2 = bool((pcr_delta >= -5e-4).all())
    g4 = (net_h > 0) and (g4_ratio >= 0.5)

    return {
        "delta_oof": float(bal_blend - bal_prim),
        "pcr_delta": [float(x) for x in pcr_delta],
        "errs_blend": int((pred_blend != y).sum()),
        "errs_prim":  int((pred_prim  != y).sum()),
        "net_h": int(net_h),
        "churn_h": int(churn),
        "g4_ratio": float(g4_ratio),
        "g1": bool(g1), "g2": bool(g2), "g4": bool(g4),
        "n_pass_no_g3": int(g1) + int(g2) + int(g4),
    }


def build_4stack(y):
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
    st2_o = log_blend([st1_o, nr_iso], np.array([0.925, 0.075]))
    st2_t = log_blend([st1_t, nrt_iso], np.array([0.925, 0.075]))
    return st2_o, st2_t


def emit_if_pass(blend_t, name, gate_dict):
    """Emit submission CSV if 4-gate (G1+G2+G4) all PASS."""
    if gate_dict["n_pass_no_g3"] != 3:
        return False
    pred = (np.log(np.clip(blend_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred]
    path = SUB / f"submission_{name}.csv"
    sub.to_csv(path, index=False)
    log(f"  ★ 4-gate PASS — wrote {path}")
    return True


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("loading 4-stack base + v1/B2 raw metas")
    base_o, base_t = build_4stack(y)
    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    b2_o = normed(np.load(ART / "oof_xgb_metastack_perfoldiso_inputs.npy"))
    b2_t = normed(np.load(ART / "test_xgb_metastack_perfoldiso_inputs.npy"))

    log("applying full-OOF iso to both metas")
    v1_iso_o, v1_iso_t = iso_full(v1_o, v1_t, y)
    b2_iso_o, b2_iso_t = iso_full(b2_o, b2_t, y)

    # Reference: current PRIMARY (= 4-stack + v1_iso @ 0.30)
    primary_o = log_blend([base_o, v1_iso_o], np.array([0.70, 0.30]))
    primary_t = log_blend([base_t, v1_iso_t], np.array([0.70, 0.30]))
    primary_csv = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
    primary_pred_csv = primary_csv[TARGET].map(CLS2IDX).to_numpy()
    log(f"  PRIMARY OOF @ recipe = {bal_at_bias(primary_o, y, RECIPE_BIAS):.5f}")

    results = {}

    # ─────────────────────────────────────────────────────────────────────
    log("\n=== #1 — Ensemble v1 + B2 metas at meta level (50/50) ===")
    # Try multiple meta-blend ratios
    for ratio in [0.30, 0.40, 0.50, 0.60, 0.70]:
        meta_ens_o = log_blend([v1_iso_o, b2_iso_o], np.array([1 - ratio, ratio]))
        meta_ens_t = log_blend([v1_iso_t, b2_iso_t], np.array([1 - ratio, ratio]))
        # Then standard α=0.30 into 4-stack
        blend_o = log_blend([base_o, meta_ens_o], np.array([0.70, 0.30]))
        blend_t = log_blend([base_t, meta_ens_t], np.array([0.70, 0.30]))

        gate = four_gate_check(primary_o, blend_o, y, primary_pred_csv)
        # Test diff vs current PRIMARY
        pred_test = (np.log(np.clip(blend_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
        diff = int((pred_test != primary_pred_csv).sum())
        verdict = "PASS" if gate["n_pass_no_g3"] == 3 else f"FAIL({gate['n_pass_no_g3']}/3)"
        log(f"  B2_share={ratio:.2f}  Δ={gate['delta_oof']:+.5f}  "
            f"PCR=[{gate['pcr_delta'][0]:+.5f}, {gate['pcr_delta'][1]:+.5f}, {gate['pcr_delta'][2]:+.5f}]  "
            f"net_H={gate['net_h']:+d}  G4r={gate['g4_ratio']:.2f}  diff={diff}  G124={verdict}")
        emit_if_pass(blend_t, f"b3_ensemble_meta_b2share{int(ratio*100):03d}", gate)
        results.setdefault("ensemble_meta", []).append({
            "b2_share": ratio, "diff_vs_primary_test": diff, **gate
        })

    # ─────────────────────────────────────────────────────────────────────
    log("\n=== #2 — α-sweep B2_full meta (find G4-PASS region) ===")
    for a in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        blend_o = log_blend([base_o, b2_iso_o], np.array([1 - a, a]))
        blend_t = log_blend([base_t, b2_iso_t], np.array([1 - a, a]))
        gate = four_gate_check(primary_o, blend_o, y, primary_pred_csv)
        pred_test = (np.log(np.clip(blend_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
        diff = int((pred_test != primary_pred_csv).sum())
        verdict = "PASS" if gate["n_pass_no_g3"] == 3 else f"FAIL({gate['n_pass_no_g3']}/3)"
        log(f"  α={a:.3f}  Δ={gate['delta_oof']:+.5f}  "
            f"PCR=[{gate['pcr_delta'][0]:+.5f}, {gate['pcr_delta'][1]:+.5f}, {gate['pcr_delta'][2]:+.5f}]  "
            f"net_H={gate['net_h']:+d}  G4r={gate['g4_ratio']:.2f}  diff={diff}  G124={verdict}")
        emit_if_pass(blend_t, f"b3_b2_only_a{int(a*1000):03d}", gate)
        results.setdefault("b2_alpha_sweep", []).append({"alpha": a, "diff": diff, **gate})

    # ─────────────────────────────────────────────────────────────────────
    log("\n=== #3 — Per-class α blending of B2 (α_H=0, sweep α_L, α_M) ===")
    # Build the B2 meta primary at mixed weights: blend_p_k = (1-α_k)*prim_p_k + α_k*b2_iso_p_k
    # Equivalent in log space to: log(blend_p_k) ≈ (1-α_k)*log(prim_p_k) + α_k*log(b2_iso_p_k) (un-normalized, then renorm)

    log_prim_o = np.log(np.clip(primary_o, 1e-12, 1.0))
    log_prim_t = np.log(np.clip(primary_t, 1e-12, 1.0))
    log_b2_o   = np.log(np.clip(b2_iso_o,  1e-12, 1.0))
    log_b2_t   = np.log(np.clip(b2_iso_t,  1e-12, 1.0))

    def per_class_blend(log_prim, log_b2, alphas):
        # alphas = [α_L, α_M, α_H]
        ALPHA = np.array(alphas)
        log_blend = (1 - ALPHA)[None, :] * log_prim + ALPHA[None, :] * log_b2
        return normed(np.exp(log_blend))

    # α_H pinned to 0 to defeat REMOVE-High failure. Sweep α_L, α_M.
    for alpha_L in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
        for alpha_M in [0.10, 0.20, 0.30, 0.40, 0.50]:
            alphas = [alpha_L, alpha_M, 0.0]
            blend_o = per_class_blend(log_prim_o, log_b2_o, alphas)
            blend_t = per_class_blend(log_prim_t, log_b2_t, alphas)
            gate = four_gate_check(primary_o, blend_o, y, primary_pred_csv)
            pred_test = (np.log(np.clip(blend_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
            diff = int((pred_test != primary_pred_csv).sum())
            if gate["delta_oof"] > 0 or gate["n_pass_no_g3"] == 3:
                verdict = "PASS" if gate["n_pass_no_g3"] == 3 else f"FAIL({gate['n_pass_no_g3']}/3)"
                log(f"  α_L={alpha_L:.2f} α_M={alpha_M:.2f} α_H=0.00  Δ={gate['delta_oof']:+.5f}  "
                    f"PCR=[{gate['pcr_delta'][0]:+.5f}, {gate['pcr_delta'][1]:+.5f}, {gate['pcr_delta'][2]:+.5f}]  "
                    f"net_H={gate['net_h']:+d}  diff={diff}  G124={verdict}")
            emit_if_pass(blend_t,
                         f"b3_perclass_aL{int(alpha_L*100):03d}_aM{int(alpha_M*100):03d}_aH000",
                         gate)
            results.setdefault("per_class_blend", []).append({
                "alpha_L": alpha_L, "alpha_M": alpha_M, "alpha_H": 0.0,
                "diff": diff, **gate
            })

    # Also sweep α_H > 0 with α_L=0, α_M=0 (just modify primary High prob)
    log("\n=== #3b — only H movement (α_L=α_M=0, sweep α_H) ===")
    for alpha_H in [0.05, 0.10, 0.15, 0.20, 0.30]:
        alphas = [0.0, 0.0, alpha_H]
        blend_o = per_class_blend(log_prim_o, log_b2_o, alphas)
        blend_t = per_class_blend(log_prim_t, log_b2_t, alphas)
        gate = four_gate_check(primary_o, blend_o, y, primary_pred_csv)
        pred_test = (np.log(np.clip(blend_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
        diff = int((pred_test != primary_pred_csv).sum())
        verdict = "PASS" if gate["n_pass_no_g3"] == 3 else f"FAIL({gate['n_pass_no_g3']}/3)"
        log(f"  α_L=0 α_M=0 α_H={alpha_H:.2f}  Δ={gate['delta_oof']:+.5f}  "
            f"PCR=[{gate['pcr_delta'][0]:+.5f}, {gate['pcr_delta'][1]:+.5f}, {gate['pcr_delta'][2]:+.5f}]  "
            f"net_H={gate['net_h']:+d}  diff={diff}  G124={verdict}")
        emit_if_pass(blend_t, f"b3_h_only_aH{int(alpha_H*100):03d}", gate)
        results.setdefault("h_only_blend", []).append({
            "alpha_H": alpha_H, "diff": diff, **gate
        })

    # ─────────────────────────────────────────────────────────────────────
    out_path = ART / "b3_followup_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
