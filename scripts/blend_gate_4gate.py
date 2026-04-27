"""4-gate blend-gate analyzer for Phase A / Phase B candidates.

Loads candidate OOF + test, reconstructs LB-best 4-stack anchor, applies fixed
recipe bias [1.4324, 1.4689, 3.4008] (no per-α retune — defends against the
binhigh OOF-inflation trap), evaluates the 4 post-22-LB-saturation gates:

  G1: blend OOF Δ ≥ +0.0003 vs LB-best 4-stack (0.98084)
  G2: per-class recall ≥ anchor − 5e-4 (each class)
  G3: dual-α stability — Δ at α=0.40 / Δ at α=0.30 in [1.0, 2.0]
       (linear scaling = real signal; sublinear or super-linear = OOF noise)
  G4: net-rare-class direction — net_high_flip > 0 AND
       |net_high|/total_high_churn ≥ 0.5 (asymmetric ADD-High, not RESHUFFLE
       or REMOVE-High)

ALL FOUR must pass before LB submission is recommended. G4 added 2026-04-27
after D 3-meta and R2/R5 a045 nulled despite passing G1+G2+G3.

Usage:
  python scripts/blend_gate_4gate.py --candidate residte
  python scripts/blend_gate_4gate.py --candidate basemargin_K40

Reads:
  scripts/artifacts/oof_recipe_full_te_<candidate>.npy
  scripts/artifacts/test_recipe_full_te_<candidate>.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                             load_y, normed)

ART = Path("scripts/artifacts")
ANCHOR_OOF = 0.98084  # documented LB-best 4-stack OOF


def per_class_recall(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    for c in range(3):
        m = y == c
        out[c] = (pred[m] == c).mean() if m.any() else 0.0
    return out


def evaluate_blend(cand_o, cand_t, lb4_o, lb4_t, y, alpha):
    """Log-blend candidate at weight alpha into LB4 anchor; return diagnostics."""
    w = np.array([1.0 - alpha, alpha])
    blend_o = log_blend([lb4_o, cand_o], w)
    blend_t = log_blend([lb4_t, cand_t], w)
    p = (np.log(np.clip(blend_o, 1e-12, 1)) + BIAS).argmax(1)
    bal = balanced_accuracy_score(y, p)
    return blend_o, blend_t, bal, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="suffix after recipe_full_te_, e.g. 'residte' or 'basemargin_K40'")
    ap.add_argument("--use-iso", action="store_true",
                    help="iso-calibrate candidate before blend")
    args = ap.parse_args()

    cand_name = args.candidate
    oof_path = ART / f"oof_recipe_full_te_{cand_name}.npy"
    test_path = ART / f"test_recipe_full_te_{cand_name}.npy"
    if not oof_path.exists() or not test_path.exists():
        # Fallback: try without recipe_full_te_ prefix.
        oof_path = ART / f"oof_{cand_name}.npy"
        test_path = ART / f"test_{cand_name}.npy"
    assert oof_path.exists(), f"missing {oof_path}"
    assert test_path.exists(), f"missing {test_path}"

    print(f"loading candidate from {oof_path.name}")
    cand_o = normed(np.load(oof_path).astype(np.float32))
    cand_t = normed(np.load(test_path).astype(np.float32))
    assert cand_o.shape == (630_000, 3), cand_o.shape

    y = load_y()
    print("reconstructing LB-best 4-stack anchor")
    # LB-best 4-stack = LB-best 3-stack + xgb_metastack__iso α=0.30
    lb3_o, lb3_t = build_lbbest_stack(y)
    mv_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    mv_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    mv_o_iso, mv_t_iso = iso_cal(mv_o, mv_t, y)
    lb4_o = log_blend([lb3_o, mv_o_iso], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, mv_t_iso], np.array([0.7, 0.3]))

    anchor_p = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)
    anchor_bal = balanced_accuracy_score(y, anchor_p)
    anchor_pcr = per_class_recall(y, anchor_p)
    print(f"\nLB-best 4-stack OOF @ recipe bias = {anchor_bal:.5f}")
    print(f"  PCR L={anchor_pcr[0]:.5f} M={anchor_pcr[1]:.5f} H={anchor_pcr[2]:.5f}")
    print(f"  errs = {(anchor_p != y).sum()}")

    if args.use_iso:
        print("iso-calibrating candidate")
        cand_o, cand_t = iso_cal(cand_o, cand_t, y)

    # Standalone candidate diagnostics.
    cand_p = (np.log(np.clip(cand_o, 1e-12, 1)) + BIAS).argmax(1)
    cand_bal = balanced_accuracy_score(y, cand_p)
    print(f"\ncandidate {cand_name} standalone @ recipe bias = {cand_bal:.5f}")
    print(f"  errs = {(cand_p != y).sum()}")

    # Sweep α ∈ {0, 0.10, 0.20, 0.30, 0.40, 0.50}.
    print("\nblend sweep (fixed recipe bias):")
    print(f"  {'α':>5}  {'OOF':>8}  {'Δ':>+9}  {'errs':>6}  {'recL':>7} {'recM':>7} {'recH':>7}")
    sweep = {}
    blend_test_at: dict[float, np.ndarray] = {}
    for alpha in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        bo, bt, bal, p = evaluate_blend(cand_o, cand_t, lb4_o, lb4_t, y, alpha)
        pcr = per_class_recall(y, p)
        delta = bal - anchor_bal
        errs = int((p != y).sum())
        sweep[alpha] = dict(oof=float(bal), delta=float(delta), errs=errs,
                            pcr=pcr.tolist())
        blend_test_at[alpha] = bt
        print(f"  {alpha:>5.2f}  {bal:>.5f}  {delta:>+.5f}  {errs:>6}  "
              f"{pcr[0]:.5f} {pcr[1]:.5f} {pcr[2]:.5f}")

    # Gate evaluation at α=0.30 (primary blend slot).
    a30 = sweep[0.30]; a40 = sweep[0.40]
    g1 = a30["delta"] >= 0.0003
    pcr30 = np.array(a30["pcr"])
    g2 = bool(np.all(pcr30 >= anchor_pcr - 5e-4))
    if a30["delta"] > 1e-9:
        ratio = a40["delta"] / a30["delta"]
        g3 = 1.0 <= ratio <= 2.0
    else:
        ratio = float("nan"); g3 = False

    # G4: rare-class flip direction at α=0.30 vs anchor on TEST predictions.
    anchor_t = (np.log(np.clip(lb4_t, 1e-12, 1)) + BIAS).argmax(1)
    blend_t_pred = (np.log(np.clip(blend_test_at[0.30], 1e-12, 1)) + BIAS).argmax(1)
    n_high_anchor = int((anchor_t == 2).sum())
    n_high_blend = int((blend_t_pred == 2).sum())
    net_high = n_high_blend - n_high_anchor
    churn_high = int(((anchor_t == 2) != (blend_t_pred == 2)).sum())
    g4_dir = net_high > 0
    g4_asym = (abs(net_high) / max(churn_high, 1)) >= 0.5
    g4 = g4_dir and g4_asym

    print(f"\n=== 4-GATE @ α=0.30 ===")
    print(f"  G1 (Δ ≥ +3e-4):        {a30['delta']:+.5f}  {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 (PCR ≥ anchor-5e-4): {pcr30 - anchor_pcr}  {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 (α=0.4/α=0.3 ratio): {ratio:.3f}  {'PASS' if g3 else 'FAIL'}")
    print(f"  G4 (net_H>0 & ratio≥0.5): net={net_high} churn={churn_high}  {'PASS' if g4 else 'FAIL'}")
    overall = g1 and g2 and g3 and g4
    print(f"\nOVERALL: {'PASS — LB-probe candidate' if overall else 'FAIL — do not LB-probe'}")

    out = ART / f"blend_gate_4gate_{cand_name}{'_iso' if args.use_iso else ''}_results.json"
    with open(out, "w") as f:
        json.dump(dict(
            candidate=cand_name, use_iso=args.use_iso,
            anchor_bal=float(anchor_bal),
            anchor_pcr=anchor_pcr.tolist(),
            sweep={f"{a:.2f}": s for a, s in sweep.items()},
            gates=dict(G1=bool(g1), G2=bool(g2), G3=bool(g3), G4=bool(g4),
                       overall=bool(overall)),
            net_high_flip=net_high, total_high_churn=churn_high,
        ), f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
