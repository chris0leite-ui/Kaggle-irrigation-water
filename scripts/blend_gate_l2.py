"""4-gate blend-gate analyzer for L2 SupCon-NCM candidate.

Anchor = v1 RF natural-cal meta (`oof_sklearn_rf_meta_natural.npy`),
the OOF surface behind the current LB-0.98134 unanimous-override
submission. The override is test-side only, so OOF gates target the
v1 RF natural directly.

NCM has NO tuned-bias retune by construction, so the gate uses the
NCM's argmax directly for decision metrics. As a sanity comparison
the analyzer ALSO reports v1 at its tuned bias [0.43, 0.87, 3.20].

Gates (post-22-LB-saturation framework + 4-gate refinements):
  G1: blend OOF Δ ≥ +0.0003 vs v1 anchor at v1's tuned bias
  G2: per-class recall ≥ anchor − 5e-4 each class
  G3: dual-α stability — Δ at α=0.40 / Δ at α=0.30 ∈ [1.0, 2.0]
  G4: net-rare-class direction — net_high_flip > 0 AND
       |net_high|/total_high_churn ≥ 0.5 (asymmetric ADD-High)

Plus diagnostic: standalone NCM at its OWN argmax (no bias retune,
which IS the load-bearing structural property).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import load_y, normed  # noqa: E402

ART = Path("scripts/artifacts")


def per_class_recall(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    for c in range(3):
        m = y == c
        out[c] = (pred[m] == c).mean() if m.any() else 0.0
    return out


def evaluate_blend(cand_o, cand_t, anchor_o, anchor_t, y, alpha, anchor_bias):
    """Log-blend candidate at weight α into anchor; argmax at anchor's bias."""
    w = np.array([1.0 - alpha, alpha])
    blend_o = log_blend([anchor_o, cand_o], w)
    blend_t = log_blend([anchor_t, cand_t], w)
    p_o = (np.log(np.clip(blend_o, 1e-12, 1)) + anchor_bias).argmax(1)
    bal = balanced_accuracy_score(y, p_o)
    return blend_o, blend_t, bal, p_o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-oof", default="oof_l2_supcon_ncm.npy")
    ap.add_argument("--cand-test", default="test_l2_supcon_ncm.npy")
    ap.add_argument("--anchor-oof", default="oof_sklearn_rf_meta_natural.npy")
    ap.add_argument("--anchor-test", default="test_sklearn_rf_meta_natural.npy")
    ap.add_argument("--label", default="l2_supcon_ncm")
    args = ap.parse_args()

    cand_o = normed(np.load(ART / args.cand_oof).astype(np.float32))
    cand_t = normed(np.load(ART / args.cand_test).astype(np.float32))
    anc_o = normed(np.load(ART / args.anchor_oof).astype(np.float32))
    anc_t = normed(np.load(ART / args.anchor_test).astype(np.float32))
    assert cand_o.shape == (630_000, 3), cand_o.shape
    assert anc_o.shape == (630_000, 3), anc_o.shape

    y = load_y()

    # Anchor at its tuned bias (v1 RF natural's documented tuned bias).
    prior = np.bincount(y, minlength=3) / len(y)
    anc_bias, anc_tuned = tune_log_bias(anc_o, y, prior)
    anc_p = (np.log(np.clip(anc_o, 1e-12, 1)) + anc_bias).argmax(1)
    anc_bal = balanced_accuracy_score(y, anc_p)
    anc_pcr = per_class_recall(y, anc_p)
    print(f"anchor (v1 RF natural) tuned bias = {[round(b,3) for b in anc_bias]}")
    print(f"  OOF tuned = {anc_bal:.5f}  errs = {(anc_p != y).sum()}")
    print(f"  PCR L={anc_pcr[0]:.5f} M={anc_pcr[1]:.5f} H={anc_pcr[2]:.5f}")

    # Standalone NCM diagnostics (no bias retune — this is the mechanism).
    cand_argmax = cand_o.argmax(1)
    cand_bal_argmax = balanced_accuracy_score(y, cand_argmax)
    cand_pcr_argmax = per_class_recall(y, cand_argmax)
    cand_errs_argmax = int((cand_argmax != y).sum())
    print(f"\ncandidate {args.label} STANDALONE (NCM argmax, no bias retune):")
    print(f"  OOF = {cand_bal_argmax:.5f}  errs = {cand_errs_argmax}")
    print(f"  PCR L={cand_pcr_argmax[0]:.5f} M={cand_pcr_argmax[1]:.5f} "
          f"H={cand_pcr_argmax[2]:.5f}")

    # Cross-check via tuned-bias (diagnostic only, NOT the real decision rule).
    cand_bias_diag, cand_tuned_diag = tune_log_bias(cand_o, y, prior)
    print(f"  diagnostic tuned (would-be retune): {cand_tuned_diag:.5f} "
          f"bias_diag={[round(b,3) for b in cand_bias_diag]}")

    # Jaccard of error sets vs anchor (at anchor's tuned bias).
    err_a = anc_p != y
    err_c = cand_argmax != y
    inter = int((err_a & err_c).sum())
    union = int((err_a | err_c).sum())
    jacc = inter / max(union, 1)
    print(f"  Jaccard(errs cand-vs-anchor) = {jacc:.4f}")

    # Blend sweep at anchor's tuned bias.
    print("\nblend sweep (fixed anchor bias):")
    print(f"  {'α':>5}  {'OOF':>8}  {'Δ vs anc':>9}  {'errs':>6}  "
          f"{'recL':>7} {'recM':>7} {'recH':>7}")
    sweep = {}
    blend_test_at: dict[float, np.ndarray] = {}
    for alpha in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        bo, bt, bal, p = evaluate_blend(
            cand_o, cand_t, anc_o, anc_t, y, alpha, anc_bias)
        pcr = per_class_recall(y, p)
        delta = bal - anc_bal
        errs = int((p != y).sum())
        sweep[alpha] = dict(oof=float(bal), delta=float(delta), errs=errs,
                            pcr=pcr.tolist())
        blend_test_at[alpha] = bt
        print(f"  {alpha:>5.2f}  {bal:>.5f}  {delta:>+.5f}  {errs:>6}  "
              f"{pcr[0]:.5f} {pcr[1]:.5f} {pcr[2]:.5f}")

    # 4-gate evaluation at α=0.30 (primary blend slot).
    a30 = sweep[0.30]; a40 = sweep[0.40]
    g1 = a30["delta"] >= 3e-4
    pcr30 = np.array(a30["pcr"])
    g2 = bool(np.all(pcr30 >= anc_pcr - 5e-4))
    if a30["delta"] > 1e-9:
        ratio = a40["delta"] / a30["delta"]
        g3 = 1.0 <= ratio <= 2.0
    else:
        ratio = float("nan"); g3 = False

    # G4: rare-class direction on TEST predictions vs anchor.
    anc_t_pred = (np.log(np.clip(anc_t, 1e-12, 1)) + anc_bias).argmax(1)
    blend_t_pred = (np.log(np.clip(blend_test_at[0.30], 1e-12, 1)) + anc_bias).argmax(1)
    n_high_anc = int((anc_t_pred == 2).sum())
    n_high_blend = int((blend_t_pred == 2).sum())
    net_high = n_high_blend - n_high_anc
    churn_high = int(((anc_t_pred == 2) != (blend_t_pred == 2)).sum())
    g4_dir = net_high > 0
    g4_asym = (abs(net_high) / max(churn_high, 1)) >= 0.5
    g4 = g4_dir and g4_asym

    print(f"\n=== 4-GATE @ α=0.30 (anchor=v1 RF natural) ===")
    print(f"  G1 (Δ ≥ +3e-4):           {a30['delta']:+.5f}  {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 (PCR ≥ anchor-5e-4):    {pcr30 - anc_pcr}  {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 (α=0.4/α=0.3 ratio):    {ratio:.3f}  {'PASS' if g3 else 'FAIL'}")
    print(f"  G4 (net_H>0 & ratio≥0.5):  net={net_high} churn={churn_high}  "
          f"{'PASS' if g4 else 'FAIL'}")
    overall = g1 and g2 and g3 and g4
    print(f"\nOVERALL: {'PASS — LB-probe candidate' if overall else 'FAIL — do not LB-probe'}")

    out = ART / f"blend_gate_{args.label}_results.json"
    with open(out, "w") as f:
        json.dump(dict(
            candidate=args.label,
            anchor_oof=str(args.anchor_oof),
            anchor_tuned=float(anc_bal),
            anchor_bias=[float(b) for b in anc_bias],
            anchor_pcr=anc_pcr.tolist(),
            cand_standalone_argmax=float(cand_bal_argmax),
            cand_standalone_pcr=cand_pcr_argmax.tolist(),
            cand_standalone_errs=cand_errs_argmax,
            cand_diag_tuned=float(cand_tuned_diag),
            cand_diag_bias=[float(b) for b in cand_bias_diag],
            jaccard_errs=float(jacc),
            sweep={f"{a:.2f}": s for a, s in sweep.items()},
            gates=dict(G1=bool(g1), G2=bool(g2), G3=bool(g3), G4=bool(g4),
                       overall=bool(overall)),
            net_high_flip=net_high, total_high_churn=churn_high,
        ), f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
