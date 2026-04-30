"""4-gate analysis for A3 stacked-regression vs LB-best PRIMARY.

Loads the Gaussian-decoded 3-class probs produced by recipe_y_regression.py
(σ tuned on OOF), applies the standard fixed-bias (recipe) blend gate vs
LB-best PRIMARY, and reports whether the regression formulation clears the
+0.0002 threshold + per-class recall guardrail + magnitude rule + asymmetric
direction rule.
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

OUT = ART / "blend_gate_4gate_y_regression_results.json"
GATE_G1_MIN = 2e-4
GATE_G2_FLOOR = -5e-4
GATE_G3_MAX_RATIO = 1.05
GATE_G4_MIN_ASYM = 0.5


def gate(s, anchor_bal, anchor_errs):
    d = s["bal_acc"] - anchor_bal
    g1 = d >= GATE_G1_MIN
    g2 = (s["delta_rec_L"] >= GATE_G2_FLOOR and
          s["delta_rec_M"] >= GATE_G2_FLOOR and
          s["delta_rec_H"] >= GATE_G2_FLOOR)
    g3 = s["errs"] <= GATE_G3_MAX_RATIO * anchor_errs
    g4 = (s["net_H"] >= 0 and s["asym"] >= GATE_G4_MIN_ASYM)
    return g1, g2, g3, g4, d


def score_at_alpha(stack_o, stack_t, cand_o, cand_t, alpha, anchor_pred, y, cc):
    if alpha == 0:
        blend_o = stack_o
    else:
        blend_o = log_blend([stack_o, cand_o], np.array([1 - alpha, alpha]))
    pred = (np.log(np.clip(blend_o, 1e-12, 1)) + BIAS).argmax(1)
    bal = fast_bal_acc(y, pred, class_counts=cc)
    rec = np.array([(pred[y == k] == k).mean() for k in range(3)])
    rec_a = np.array([(anchor_pred[y == k] == k).mean() for k in range(3)])
    add_h = int(((pred == 2) & (anchor_pred != 2)).sum())
    rem_h = int(((pred != 2) & (anchor_pred == 2)).sum())
    net_h = add_h - rem_h
    asym = net_h / max(add_h + rem_h, 1)
    return {
        "alpha": float(alpha), "bal_acc": float(bal), "errs": int((pred != y).sum()),
        "rec_L": float(rec[0]), "rec_M": float(rec[1]), "rec_H": float(rec[2]),
        "delta_rec_L": float(rec[0] - rec_a[0]),
        "delta_rec_M": float(rec[1] - rec_a[1]),
        "delta_rec_H": float(rec[2] - rec_a[2]),
        "add_H": add_h, "rem_H": rem_h, "net_H": net_h, "asym": float(asym),
    }


def main():
    log("=== A3 stacked-regression 4-gate analysis vs LB-best PRIMARY ===")
    y = load_y()
    cc = np.bincount(y, minlength=3)

    # PRIMARY anchor (= LB-best 3-stack + xgb_metastack_iso α=0.30)
    s2_o, s2_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    primary_o = log_blend([s2_o, meta_o_iso], np.array([0.7, 0.3]))
    primary_t = log_blend([s2_t, meta_t_iso], np.array([0.7, 0.3]))
    anchor_pred = (np.log(np.clip(primary_o, 1e-12, 1)) + BIAS).argmax(1)
    anchor_bal = fast_bal_acc(y, anchor_pred, class_counts=cc)
    anchor_errs = int((anchor_pred != y).sum())
    log(f"  PRIMARY anchor:  bal={anchor_bal:.5f}  errs={anchor_errs}")

    # A3 candidate (Gaussian-decoded 3-class probs)
    cand_o = normed(np.load(ART / "oof_recipe_y_regression_3cls.npy").astype(np.float32))
    cand_t = normed(np.load(ART / "test_recipe_y_regression_3cls.npy").astype(np.float32))

    # Continuous regressor output (for the threshold-decoded variant)
    yhat_oof = np.load(ART / "oof_recipe_y_regression.npy").astype(np.float32)
    yhat_test = np.load(ART / "test_recipe_y_regression.npy").astype(np.float32)
    log(f"  yhat OOF range = [{yhat_oof.min():.3f}, {yhat_oof.max():.3f}]  "
        f"mean = {yhat_oof.mean():.3f}")

    # standalone @ recipe bias (3cls Gaussian decoding)
    cand_pred = (np.log(np.clip(cand_o, 1e-12, 1)) + BIAS).argmax(1)
    cand_bal = fast_bal_acc(y, cand_pred, class_counts=cc)
    cand_errs = int((cand_pred != y).sum())
    err_a = (anchor_pred != y); err_c = (cand_pred != y)
    jacc = float((err_a & err_c).sum() / max(1, (err_a | err_c).sum()))
    log(f"  A3-3cls standalone @ recipe-bias:  bal={cand_bal:.5f}  errs={cand_errs}  J={jacc:.4f}")

    # iso-cal
    cand_o_iso, cand_t_iso = iso_cal(cand_o, cand_t, y)
    cand_pred_iso = (np.log(np.clip(cand_o_iso, 1e-12, 1)) + BIAS).argmax(1)
    cand_bal_iso = fast_bal_acc(y, cand_pred_iso, class_counts=cc)
    cand_errs_iso = int((cand_pred_iso != y).sum())
    err_ci = (cand_pred_iso != y)
    jacc_iso = float((err_a & err_ci).sum() / max(1, (err_a | err_ci).sum()))
    log(f"  A3-3cls iso-cal @ recipe-bias:     bal={cand_bal_iso:.5f}  errs={cand_errs_iso}  J={jacc_iso:.4f}")

    log("\n  α-sweep on PRIMARY blend (fixed recipe bias, raw + iso):")
    print(f"    {'α':>5s} {'var':>3s} {'bal_acc':>9s} {'Δ':>9s} {'errs':>6s} "
          f"{'recL':>9s} {'recM':>9s} {'recH':>9s} {'net_H':>6s} {'asym':>6s}  gates")
    sweep_raw, sweep_iso = [], []
    for alpha in [0.000, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        s_raw = score_at_alpha(primary_o, primary_t, cand_o, cand_t, alpha, anchor_pred, y, cc)
        s_iso = score_at_alpha(primary_o, primary_t, cand_o_iso, cand_t_iso, alpha, anchor_pred, y, cc)
        for label, s, dst in [("raw", s_raw, sweep_raw), ("iso", s_iso, sweep_iso)]:
            g1, g2, g3, g4, d = gate(s, anchor_bal, anchor_errs)
            gs = "".join("✓" if g else "✗" for g in (g1, g2, g3, g4))
            print(f"    {alpha:>5.3f} {label:>3s} {s['bal_acc']:>9.5f} {d:+9.5f} "
                  f"{s['errs']:>6d} {s['rec_L']:>9.5f} {s['rec_M']:>9.5f} "
                  f"{s['rec_H']:>9.5f} {s['net_H']:>+6d} {s['asym']:>+6.2f}  {gs}")
            s["delta_bal"] = float(d)
            s["gates"] = {"G1": g1, "G2": g2, "G3": g3, "G4": g4,
                          "all_pass": g1 and g2 and g3 and g4}
            dst.append(s)

    out = {
        "anchor_bal": float(anchor_bal), "anchor_errs": anchor_errs,
        "yhat_oof_stats": {"min": float(yhat_oof.min()), "max": float(yhat_oof.max()),
                           "mean": float(yhat_oof.mean()), "std": float(yhat_oof.std())},
        "standalone_raw": {"bal": float(cand_bal), "errs": cand_errs, "jaccard": jacc},
        "standalone_iso": {"bal": float(cand_bal_iso), "errs": cand_errs_iso, "jaccard": jacc_iso},
        "sweep_raw": sweep_raw, "sweep_iso": sweep_iso,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\n  saved {OUT}")


if __name__ == "__main__":
    main()
