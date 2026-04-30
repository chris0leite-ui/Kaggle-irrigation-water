"""4-gate analysis for LGBM-GOSS recipe variant against LB-best PRIMARY.

Loads the GOSS OOF/test produced by LGBM_BOOSTING=goss recipe_full_te_lgbm.py,
applies the standard fixed-bias (recipe) blend gate vs LB-best 4-stack,
and reports whether the GOSS variant clears the +0.0002 threshold + per-class
recall guardrail + magnitude rule + asymmetric direction rule.

Same framework as scripts/blend_gate_4gate.py. No LB probe is emitted by this
script — user must approve any submission per CLAUDE.md.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, fast_bal_acc  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, normed, iso_cal, build_lbbest_stack, load_y, log,
)

OUT = ART / "blend_gate_4gate_lgbm_goss_results.json"
GATE_G1_MIN = 2e-4
GATE_G2_FLOOR = -5e-4
GATE_G3_MAX_RATIO = 1.05
GATE_G4_MIN_ASYM = 0.5


def score_at_alpha(stack_o, stack_t, cand_o, cand_t, alpha, anchor_pred, y, cc):
    blend_o = log_blend([stack_o, cand_o], np.array([1 - alpha, alpha]))
    blend_t = log_blend([stack_t, cand_t], np.array([1 - alpha, alpha]))
    pred = (np.log(np.clip(blend_o, 1e-12, 1)) + BIAS).argmax(1)
    bal = fast_bal_acc(y, pred, class_counts=cc)
    rec = np.array([(pred[y == k] == k).mean() for k in range(3)])
    rec_anchor = np.array([(anchor_pred[y == k] == k).mean() for k in range(3)])
    add_h = int(((pred == 2) & (anchor_pred != 2)).sum())
    rem_h = int(((pred != 2) & (anchor_pred == 2)).sum())
    net_h = add_h - rem_h
    asym = net_h / max(add_h + rem_h, 1)
    return {
        "alpha": alpha, "bal_acc": float(bal),
        "rec_L": float(rec[0]), "rec_M": float(rec[1]), "rec_H": float(rec[2]),
        "delta_rec_L": float(rec[0] - rec_anchor[0]),
        "delta_rec_M": float(rec[1] - rec_anchor[1]),
        "delta_rec_H": float(rec[2] - rec_anchor[2]),
        "errs": int((pred != y).sum()),
        "add_H": add_h, "rem_H": rem_h, "net_H": net_h, "asym": float(asym),
        "blend_test_pred": pred,  # discarded outside scoring; useful for debug
    }


def main():
    log("=== LGBM-GOSS 4-gate analysis vs LB-best 4-stack ===")
    y = load_y()
    cc = np.bincount(y, minlength=3)

    # Reconstruct LB-best 3-stack (s2_o = anchor for meta_iso step)
    s2_o, s2_t = build_lbbest_stack(y)

    # Add xgb_metastack_iso at α=0.30 → PRIMARY
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    primary_o = log_blend([s2_o, meta_o_iso], np.array([0.7, 0.3]))
    primary_t = log_blend([s2_t, meta_t_iso], np.array([0.7, 0.3]))

    anchor_pred = (np.log(np.clip(primary_o, 1e-12, 1)) + BIAS).argmax(1)
    anchor_bal = fast_bal_acc(y, anchor_pred, class_counts=cc)
    anchor_errs = int((anchor_pred != y).sum())
    log(f"  PRIMARY anchor:  bal={anchor_bal:.5f}  errs={anchor_errs}")

    # Load LGBM-GOSS candidate
    cand_o = normed(np.load(ART / "oof_recipe_full_te_lgbm_goss.npy").astype(np.float32))
    cand_t = normed(np.load(ART / "test_recipe_full_te_lgbm_goss.npy").astype(np.float32))
    cand_pred_at_anchor_bias = (np.log(np.clip(cand_o, 1e-12, 1)) + BIAS).argmax(1)
    cand_bal = fast_bal_acc(y, cand_pred_at_anchor_bias, class_counts=cc)
    cand_errs = int((cand_pred_at_anchor_bias != y).sum())
    err_a = (anchor_pred != y); err_c = (cand_pred_at_anchor_bias != y)
    jacc = float((err_a & err_c).sum() / max(1, (err_a | err_c).sum()))
    log(f"  LGBM-GOSS standalone @ recipe-bias:  bal={cand_bal:.5f}  errs={cand_errs}  J={jacc:.4f}")

    # iso-cal'd variant
    cand_o_iso, cand_t_iso = iso_cal(cand_o, cand_t, y)
    cand_pred_iso = (np.log(np.clip(cand_o_iso, 1e-12, 1)) + BIAS).argmax(1)
    cand_iso_bal = fast_bal_acc(y, cand_pred_iso, class_counts=cc)
    cand_iso_errs = int((cand_pred_iso != y).sum())
    err_ci = (cand_pred_iso != y)
    jacc_iso = float((err_a & err_ci).sum() / max(1, (err_a | err_ci).sum()))
    log(f"  LGBM-GOSS iso-cal @ recipe-bias:     bal={cand_iso_bal:.5f}  errs={cand_iso_errs}  J={jacc_iso:.4f}")

    # α-sweep blend gate (raw + iso)
    log("\n  α-sweep on PRIMARY blend (fixed recipe bias):")
    print(f"    {'α':>5s} {'bal_acc':>9s} {'Δ':>9s} {'errs':>6s} "
          f"{'recL':>9s} {'recM':>9s} {'recH':>9s} {'net_H':>6s} {'asym':>6s}  gates")
    sweep_raw, sweep_iso = [], []
    for alpha in [0.00, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        s_raw = score_at_alpha(primary_o, primary_t, cand_o, cand_t, alpha, anchor_pred, y, cc)
        s_iso = score_at_alpha(primary_o, primary_t, cand_o_iso, cand_t_iso, alpha, anchor_pred, y, cc)
        for label, s in [("raw", s_raw), ("iso", s_iso)]:
            d = s["bal_acc"] - anchor_bal
            g1 = d >= GATE_G1_MIN
            g2 = (s["delta_rec_L"] >= GATE_G2_FLOOR and
                  s["delta_rec_M"] >= GATE_G2_FLOOR and
                  s["delta_rec_H"] >= GATE_G2_FLOOR)
            g3 = s["errs"] <= GATE_G3_MAX_RATIO * anchor_errs
            g4 = (s["net_H"] >= 0 and s["asym"] >= GATE_G4_MIN_ASYM)
            gs = "".join("✓" if g else "✗" for g in (g1, g2, g3, g4))
            print(f"    {alpha:>5.3f} {label:>3s} {s['bal_acc']:>9.5f} {d:+9.5f} "
                  f"{s['errs']:>6d} {s['rec_L']:>9.5f} {s['rec_M']:>9.5f} "
                  f"{s['rec_H']:>9.5f} {s['net_H']:>+6d} {s['asym']:>+6.2f}  {gs}")
            s["delta_bal"] = float(d)
            s["gates"] = {"G1": g1, "G2": g2, "G3": g3, "G4": g4,
                          "all_pass": g1 and g2 and g3 and g4}
            del s["blend_test_pred"]
            (sweep_raw if label == "raw" else sweep_iso).append(s)

    out = {
        "anchor_bal": float(anchor_bal), "anchor_errs": anchor_errs,
        "standalone_raw": {"bal": float(cand_bal), "errs": cand_errs, "jaccard": jacc},
        "standalone_iso": {"bal": float(cand_iso_bal), "errs": cand_iso_errs, "jaccard": jacc_iso},
        "sweep_raw": sweep_raw, "sweep_iso": sweep_iso,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\n  saved {OUT}")


if __name__ == "__main__":
    main()
