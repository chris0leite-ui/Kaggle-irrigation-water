"""Combined blend: LB-best 4-stack + α_A * adv_s050_iso + α_B * sklearn_rf_iso.

Tests whether A (adversarial recipe XGB) and B' (sklearn RF meta) errors
are jointly orthogonal enough to compound, even though each alone fails
the +0.0002 gate. Grid search over (α_A, α_B) in {0, 0.025, 0.05, 0.10, 0.15}.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                             load_y, normed)

ART = Path("scripts/artifacts")


def _bal(p, y):
    pred = (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)
    pcr = np.array([(pred[y == c] == c).mean() for c in range(3)])
    bal = float(pcr.mean())
    errs = (pred != y).sum()
    return bal, errs, pcr, pred


def main():
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)
    mv_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    mv_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    mv_o_iso, mv_t_iso = iso_cal(mv_o, mv_t, y)
    lb4_o = log_blend([lb3_o, mv_o_iso], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, mv_t_iso], np.array([0.7, 0.3]))
    lb4_bal, lb4_errs, lb4_pcr, lb4_pred = _bal(lb4_o, y)
    print(f"LB-best 4-stack: bal={lb4_bal:.5f} errs={lb4_errs} pcr={lb4_pcr.round(5).tolist()}")

    a_o = normed(np.load(ART / "oof_recipe_adv_s050.npy"))
    a_t = normed(np.load(ART / "test_recipe_adv_s050.npy"))
    a_o_iso, a_t_iso = iso_cal(a_o, a_t, y)

    b_o = normed(np.load(ART / "oof_sklearn_rf_meta.npy"))
    b_t = normed(np.load(ART / "test_sklearn_rf_meta.npy"))
    b_o_iso, b_t_iso = iso_cal(b_o, b_t, y)

    # 3-way log-blend grid
    grid = [0.000, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
    results = []
    print(f"\n{'α_A':<6}{'α_B':<6}{'Δ':<10}{'errs':<7}{'asym':<7}{'g2':<3}{'g3':<3}{'g4':<3}{'pcr'}")
    print("-" * 70)
    for aA in grid:
        for aB in grid:
            if aA + aB >= 0.5:
                continue
            w_lb4 = 1.0 - aA - aB
            blend = log_blend([lb4_o, a_o_iso, b_o_iso], np.array([w_lb4, aA, aB]))
            bal, errs, pcr, pred = _bal(blend, y)
            delta = bal - lb4_bal
            new_high = (pred == 2).sum()
            anchor_high = (lb4_pred == 2).sum()
            net_change = int(new_high - anchor_high)
            churn = int(((pred == 2) ^ (lb4_pred == 2)).sum())
            asym = abs(net_change) / max(churn, 1)
            g2 = errs <= lb4_errs + 5
            g3 = all(pcr[c] >= lb4_pcr[c] - 5e-4 for c in range(3))
            g4 = asym >= 0.5
            results.append({
                "aA": aA, "aB": aB,
                "bal": float(bal), "delta": float(delta),
                "errs": int(errs), "pcr": pcr.tolist(),
                "asym": float(asym),
                "g2": bool(g2), "g3": bool(g3), "g4": bool(g4),
                "pass_all": bool(g2 and g3 and g4 and delta > 0),
            })
            mark = "←pass" if (g2 and g3 and g4 and delta > 1e-5) else ""
            print(f"{aA:<6}{aB:<6}{delta:+.5f}  {errs:<7}{asym:.3f}  "
                  f"{str(g2)[0]:<3}{str(g3)[0]:<3}{str(g4)[0]:<3}"
                  f"{[round(p,4) for p in pcr]} {mark}")

    # Best gate-pass entry
    passing = [r for r in results if r["pass_all"]]
    if passing:
        best = max(passing, key=lambda r: r["delta"])
        print(f"\nBEST GATE-PASS: aA={best['aA']} aB={best['aB']} Δ={best['delta']:+.5f}")
    else:
        best_any = max(results, key=lambda r: r["delta"])
        print(f"\nNO GATE-PASS; best Δ={best_any['delta']:+.5f} at aA={best_any['aA']} aB={best_any['aB']} "
              f"(g2={best_any['g2']}, g3={best_any['g3']}, g4={best_any['g4']})")

    with open(ART / "blend_gate_AB_combined_results.json", "w") as f:
        json.dump({"results": results,
                   "anchor_lb4": {"bal": lb4_bal, "errs": int(lb4_errs), "pcr": lb4_pcr.tolist()}},
                  f, indent=2, default=float)


if __name__ == "__main__":
    main()
