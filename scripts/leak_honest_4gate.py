"""Leak-honest 4-gate diagnostic.

Builds the exact architecture-matched leak-honest primary:
  primary' = log_blend(LB-3stack + RealMLP α=0.20 + xgb_nonrule_iso(perfold) α=0.075,
                       xgb_metastack_iso(perfold) α=0.30)
  with PER-FOLD isotonic instead of full-OOF.

Then re-runs 4-gate filter for candidates that previously failed against
the full-OOF-iso primary (OOF 0.98084):

    sklearn_rf_meta            (B': RF meta-stacker — gap was +0.00010)
    mlp_metastack              (MLP-meta v1 — gap was +0.00027)
    recipe_full_te_macrorec_T1_lam03   (macrorec standalone — first G4 PASS)
    recipe_full_te_dropdet     (DROP_DETERMINISTIC — REMOVE-High diagnosed)

Reports for each candidate at fixed α-sweep:
    OOF Δ vs leak-honest primary
    errs vs anchor (G2)
    per-class recall delta (G2 & G3)
    Jaccard
    net rare-class flip & churn (G4)
    overall PASS / FAIL

Writes:
    scripts/artifacts/leak_honest_primary_oof.npy
    scripts/artifacts/leak_honest_primary_test.npy
    scripts/artifacts/leak_honest_4gate_results.json
    submissions/submission_leak_honest_primary.csv
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
from common import log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"
BIAS = np.array([1.4324, 1.4689, 3.4008])
SEED = 42
N_FOLDS = 5

CANDIDATES = [
    "sklearn_rf_meta",
    "mlp_metastack",
    "recipe_full_te_macrorec_T1_lam03",
    "recipe_full_te_dropdet",
]
ALPHAS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_perfold(oof, test, y):
    """Per-fold leak-safe iso. Uses the same StratifiedKFold split that
    produced oof. For each row i in fold k, fit iso on (oof[!=k], y[!=k]).
    Test uses full-OOF iso (test rows aren't in OOF training)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oo = np.zeros_like(oof, dtype=np.float32)
    for tr_idx, va_idx in skf.split(oof, y):
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip",
                                    y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof[tr_idx, c], (y[tr_idx] == c).astype(np.float32))
            oo[va_idx, c] = ir.predict(oof[va_idx, c])
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        tt[:, c] = ir.predict(test[:, c])
    return normed(oo), normed(tt)


def predict(p):
    """Apply fixed bias and argmax."""
    return (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)


def bal(p, y): return balanced_accuracy_score(y, predict(p))


def per_class_recall(y, pred):
    return np.array([(pred[y == k] == k).mean() for k in range(3)])


def jaccard_err(y, pred_a, pred_b):
    e_a = pred_a != y
    e_b = pred_b != y
    inter = (e_a & e_b).sum()
    union = (e_a | e_b).sum()
    return float(inter / max(union, 1))


def build_leak_honest_primary(y):
    """Reconstruct primary with per-fold iso swapped for full-OOF iso.

    Architecture (matches LB-best primary's tier1b_greedy_meta):
        st1  = log_blend(LB-3stack, RealMLP;       0.80 / 0.20)
        st2  = log_blend(st1, xgb_nonrule_iso;     0.925 / 0.075)
        prim = log_blend(st2, xgb_metastack_iso;   0.70 / 0.30)
    Where LB-3stack = log_blend(recipe, pseudo_s1, pseudo_s7; 0.25/0.35/0.40)
    """
    log("loading bank components")
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
    ms = normed(np.load(ART / "oof_xgb_metastack.npy"))
    mst= normed(np.load(ART / "test_xgb_metastack.npy"))

    log("per-fold iso on xgb_nonrule")
    nr_iso, nrt_iso = iso_perfold(nr, nrt, y)
    log("per-fold iso on xgb_metastack")
    ms_iso, mst_iso = iso_perfold(ms, mst, y)

    log("building 3-stack + RealMLP + nonrule_iso 4-stack base")
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    st1_o = log_blend([lb3_o, rm], np.array([0.80, 0.20]))
    st1_t = log_blend([lb3_t, rmt], np.array([0.80, 0.20]))
    st2_o = log_blend([st1_o, nr_iso], np.array([0.925, 0.075]))
    st2_t = log_blend([st1_t, nrt_iso], np.array([0.925, 0.075]))
    log(f"  4-stack OOF (per-fold iso) = {bal(st2_o, y):.5f}")

    log("adding xgb_metastack_iso(per-fold) at α=0.30 (architecture-matched)")
    prim_o = log_blend([st2_o, ms_iso], np.array([0.70, 0.30]))
    prim_t = log_blend([st2_t, mst_iso], np.array([0.70, 0.30]))
    log(f"  leak-honest primary OOF = {bal(prim_o, y):.5f}")

    return prim_o, prim_t


def four_gate(anchor_o, cand_o, cand_t, anchor_t, y, alpha,
              g1_thresh=2e-4, g2_pcr_floor=-5e-4, g3_lo=1.0, g3_hi=2.0,
              g4_ratio=0.5):
    """Apply 4-gate filter to cand at given α blended into anchor.

    Returns dict with all metrics + per-gate verdicts + overall PASS/FAIL.
    """
    blend_o = log_blend([anchor_o, cand_o], np.array([1 - alpha, alpha]))
    blend_t = log_blend([anchor_t, cand_t], np.array([1 - alpha, alpha]))

    bal_anchor = bal(anchor_o, y)
    bal_blend  = bal(blend_o, y)
    delta_oof  = bal_blend - bal_anchor

    pred_anchor = predict(anchor_o)
    pred_blend  = predict(blend_o)

    errs_anchor = int((pred_anchor != y).sum())
    errs_blend  = int((pred_blend  != y).sum())

    pcr_anchor = per_class_recall(y, pred_anchor)
    pcr_blend  = per_class_recall(y, pred_blend)
    pcr_delta  = pcr_blend - pcr_anchor

    jac = jaccard_err(y, pred_blend, pred_anchor)

    # G4: rare-class (High = idx 2) net flip + churn.
    blend_high = pred_blend == 2
    anchor_high = pred_anchor == 2
    add_high = int((blend_high & ~anchor_high).sum())
    rem_high = int((anchor_high & ~blend_high).sum())
    net_high = add_high - rem_high
    churn = add_high + rem_high
    g4_ratio_actual = abs(net_high) / max(churn, 1)
    g4_direction_add = net_high > 0

    g1 = delta_oof >= g1_thresh
    g2 = bool((pcr_delta >= g2_pcr_floor).all())
    # G3 stability requires another α point; we'll just record peak alpha for now.
    g3 = None  # requires dual-α; deferred to summary
    g4 = g4_direction_add and g4_ratio_actual >= g4_ratio

    return {
        "alpha": float(alpha),
        "bal_anchor": float(bal_anchor),
        "bal_blend":  float(bal_blend),
        "delta_oof":  float(delta_oof),
        "errs_anchor": errs_anchor,
        "errs_blend":  errs_blend,
        "errs_delta":  errs_blend - errs_anchor,
        "pcr_anchor": [float(x) for x in pcr_anchor],
        "pcr_blend":  [float(x) for x in pcr_blend],
        "pcr_delta":  [float(x) for x in pcr_delta],
        "jaccard_vs_anchor": jac,
        "net_high":  int(net_high),
        "add_high":  add_high,
        "rem_high":  rem_high,
        "churn_high": int(churn),
        "g4_ratio": float(g4_ratio_actual),
        "g1_pass": bool(g1),
        "g2_pass": bool(g2),
        "g4_pass": bool(g4),
        "n_gates_pass_no_g3": int(g1) + int(g2) + int(g4),
    }


def main():
    t0 = time.time()
    log("=== leak-honest 4-gate diagnostic ===")

    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    prim_o, prim_t = build_leak_honest_primary(y)
    bal_lh = bal(prim_o, y)
    log(f"\nLEAK-HONEST PRIMARY OOF = {bal_lh:.5f}")
    log(f"  vs full-OOF-iso primary OOF (0.98084) = {bal_lh - 0.98084:+.5f}")
    log(f"  inflation magnitude = {0.98084 - bal_lh:+.5f}")

    np.save(ART / "leak_honest_primary_oof.npy",  prim_o.astype(np.float32))
    np.save(ART / "leak_honest_primary_test.npy", prim_t.astype(np.float32))
    log(f"saved leak-honest primary OOF + test arrays")

    # Build leak-honest primary submission (always — caller chooses to LB-probe)
    pred_lh = predict(prim_t)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred_lh]
    sub_path = SUB / "submission_leak_honest_primary.csv"
    sub.to_csv(sub_path, index=False)
    log(f"  submission emitted: {sub_path}")

    log(f"\n=== gate sweep against leak-honest primary ===")
    results = {
        "leak_honest_primary_oof": float(bal_lh),
        "primary_full_oof_iso_reference": 0.98084,
        "inflation_magnitude": float(0.98084 - bal_lh),
        "candidates": {},
    }

    for cand_name in CANDIDATES:
        oof_p  = ART / f"oof_{cand_name}.npy"
        test_p = ART / f"test_{cand_name}.npy"
        if not (oof_p.exists() and test_p.exists()):
            log(f"  SKIP {cand_name}: missing OOF/test")
            continue

        log(f"\n--- {cand_name} ---")
        cand_o = normed(np.load(oof_p).astype(np.float32))
        cand_t = normed(np.load(test_p).astype(np.float32))

        # Apply per-fold iso to candidate (so its calibration is leak-free too)
        cand_o_iso, cand_t_iso = iso_perfold(cand_o, cand_t, y)

        cand_results = []
        for a in ALPHAS:
            row = four_gate(prim_o, cand_o_iso, cand_t_iso, prim_t, y, a)
            cand_results.append(row)
            n_pass = row['n_gates_pass_no_g3']
            verdict = 'PASS' if n_pass == 3 else f'FAIL({n_pass}/3)'
            pcr0, pcr1, pcr2 = row['pcr_delta']
            log(f"  α={a:.3f}  Δ={row['delta_oof']:+.5f}  errs={row['errs_blend']}({row['errs_delta']:+d})  "
                f"PCR=[{pcr0:+.5f}, {pcr1:+.5f}, {pcr2:+.5f}]  "
                f"Jac={row['jaccard_vs_anchor']:.3f}  net_H={row['net_high']:+d}  G124={verdict}")

        # Find best gate-pass alpha
        gate_pass = [r for r in cand_results if r["n_gates_pass_no_g3"] == 3]
        if gate_pass:
            best = max(gate_pass, key=lambda r: r["delta_oof"])
            log(f"  BEST GATE-PASS α={best['alpha']:.3f} Δ={best['delta_oof']:+.5f}  "
                f"errs={best['errs_blend']}  Jac={best['jaccard_vs_anchor']:.3f}")
        else:
            best = max(cand_results, key=lambda r: r["delta_oof"])
            log(f"  NO gate-pass; best peak α={best['alpha']:.3f} Δ={best['delta_oof']:+.5f}  "
                f"failing G1={not best['g1_pass']} G2={not best['g2_pass']} G4={not best['g4_pass']}")

        results["candidates"][cand_name] = {
            "sweep": cand_results,
            "best_gate_pass": best if gate_pass else None,
            "best_peak_no_gate": None if gate_pass else best,
        }

    out_path = ART / "leak_honest_4gate_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
