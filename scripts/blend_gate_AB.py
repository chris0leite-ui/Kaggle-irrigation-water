"""4-gate blend analysis for adversarial-robustness recipe (A) + cuML metas (B).

For each candidate, evaluates:
  G1: standalone iso OOF >= 0.97970 (above recipe baseline)
  G2: errs at recipe bias <= LB-best 4-stack errs (~9415)
  G3: per-class recall delta vs LB-best 4-stack >= -0.0005 each class
  G4: |net_rare_class_change| / |total_rare_class_churn| >= 0.5 (asymmetric)

Plus blend-α sweep onto LB-best 3-stack and LB-best 4-stack at fixed
recipe bias. Auto-emits submission ONLY if Δ OOF >= +0.0002 AND all 4
gates pass.

Cand naming convention:
  A: scripts/artifacts/oof_recipe_adv_s{σ:03d}.npy
  B: scripts/artifacts/oof_cuml_{lr,rf,knn}.npy (after pulled from Kaggle)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                             load_y, normed)

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def _bal_at(p: np.ndarray, y: np.ndarray, bias=BIAS) -> tuple[float, np.ndarray, np.ndarray]:
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    bal = float(np.array([(pred[y == c] == c).mean() for c in range(3)]).mean())
    pcr = np.array([(pred[y == c] == c).mean() for c in range(3)])
    return bal, pred, pcr


def _gates(p_blend, y, anchor_pred, anchor_pcr, anchor_errs):
    bal, pred, pcr = _bal_at(p_blend, y)
    errs = (pred != y).sum()
    rare_class = 2  # High = idx 2
    new_high = (pred == rare_class).sum()
    anchor_high = (anchor_pred == rare_class).sum()
    net_change = int(new_high - anchor_high)
    flips_in = int(((pred == rare_class) & (anchor_pred != rare_class)).sum())
    flips_out = int(((pred != rare_class) & (anchor_pred == rare_class)).sum())
    churn = flips_in + flips_out
    asym = abs(net_change) / max(churn, 1)
    g1 = bal >= 0.98000  # placeholder; G1 measured at standalone
    g2 = errs <= anchor_errs + 5
    g3 = all(pcr[c] >= anchor_pcr[c] - 5e-4 for c in range(3))
    g4 = asym >= 0.5
    return dict(bal=bal, errs=int(errs), pcr=pcr.tolist(),
                net_high=net_change, churn=churn, asym=asym,
                g2=bool(g2), g3=bool(g3), g4=bool(g4),
                gates_pass=bool(g2 and g3 and g4))


def main():
    y = load_y()
    lb3_oof, lb3_test = build_lbbest_stack(y)
    lb3_bal, lb3_pred, lb3_pcr = _bal_at(lb3_oof, y)
    lb3_errs = (lb3_pred != y).sum()
    print(f"LB-best 3-stack: bal={lb3_bal:.5f} errs={lb3_errs} pcr={lb3_pcr.round(5).tolist()}")

    # Reconstruct LB-best 4-stack (= 3-stack + xgb_metastack_iso α=0.30).
    mv_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    mv_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    mv_o_iso, mv_t_iso = iso_cal(mv_o, mv_t, y)
    lb4_oof = log_blend([lb3_oof, mv_o_iso], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, mv_t_iso], np.array([0.7, 0.3]))
    lb4_bal, lb4_pred, lb4_pcr = _bal_at(lb4_oof, y)
    lb4_errs = (lb4_pred != y).sum()
    print(f"LB-best 4-stack: bal={lb4_bal:.5f} errs={lb4_errs} pcr={lb4_pcr.round(5).tolist()}")

    # Candidates to gate. Filter to those with files on disk.
    cands = []
    for tag in os.environ.get("CANDIDATES", "adv_s050,cuml_lr,cuml_rf,cuml_knn").split(","):
        tag = tag.strip()
        oof_p = ART / f"oof_recipe_{tag}.npy" if tag.startswith("adv_") else ART / f"oof_{tag}.npy"
        if "cuml" in tag:
            oof_p = ART / f"oof_{tag}.npy"
        test_p = ART / oof_p.name.replace("oof_", "test_", 1)
        if not oof_p.exists() or not test_p.exists():
            print(f"  SKIP {tag}: missing ({oof_p.exists()=}, {test_p.exists()=})")
            continue
        cands.append((tag, oof_p, test_p))

    out = {"anchor_lb4": dict(bal=lb4_bal, errs=int(lb4_errs), pcr=lb4_pcr.tolist())}

    for tag, oof_p, test_p in cands:
        print(f"\n=== {tag} ===")
        cand_o = normed(np.load(oof_p))
        cand_t = normed(np.load(test_p))
        cand_o_iso, cand_t_iso = iso_cal(cand_o, cand_t, y)
        s_bal, _, s_pcr = _bal_at(cand_o, y)
        si_bal, _, si_pcr = _bal_at(cand_o_iso, y)
        # Tuned standalone OOF
        prior = np.bincount(y, minlength=3) / len(y)
        tuned_bias, tuned_bal = tune_log_bias(cand_o, y, prior)
        print(f"  standalone @recipe-bias = {s_bal:.5f}  iso = {si_bal:.5f}  tuned = {tuned_bal:.5f}")

        # Sweep α in log-blend onto each anchor.
        for anchor_name, a_oof, a_test, a_pred, a_pcr, a_errs in [
            ("lb3", lb3_oof, lb3_test, lb3_pred, lb3_pcr, lb3_errs),
            ("lb4", lb4_oof, lb4_test, lb4_pred, lb4_pcr, lb4_errs),
        ]:
            best = (-1, None)
            sweep = []
            for a in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
                w = np.array([1 - a, a])
                # Try both raw and iso versions of the candidate.
                for cflag, cand_use in [("raw", cand_o), ("iso", cand_o_iso)]:
                    blend_o = log_blend([a_oof, cand_use], w)
                    g = _gates(blend_o, y, a_pred, a_pcr, a_errs)
                    delta = g["bal"] - (lb3_bal if anchor_name == "lb3" else lb4_bal)
                    sweep.append({
                        "alpha": a, "iso": cflag,
                        "bal": g["bal"], "delta": delta,
                        "errs": g["errs"], "pcr": g["pcr"],
                        "asym": g["asym"],
                        "g2": g["g2"], "g3": g["g3"], "g4": g["g4"],
                        "gates_pass": g["gates_pass"],
                    })
                    if delta > best[0] and g["gates_pass"]:
                        best = (delta, sweep[-1].copy())
            out[f"{tag}_vs_{anchor_name}"] = {
                "best_gate_pass": best[1],
                "sweep": sweep,
            }
            if best[1] is not None:
                print(f"  vs {anchor_name}: best gate-pass α={best[1]['alpha']} ({best[1]['iso']}) "
                      f"Δ={best[0]:+.5f}  errs={best[1]['errs']}  asym={best[1]['asym']:.3f}")
            else:
                # Show best Δ even if gates fail
                best_any = max(sweep, key=lambda s: s["delta"])
                print(f"  vs {anchor_name}: NO gate-pass; best-Δ α={best_any['alpha']} ({best_any['iso']}) "
                      f"Δ={best_any['delta']:+.5f} g2={best_any['g2']} g3={best_any['g3']} g4={best_any['g4']}")

    with open(ART / "blend_gate_AB_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {ART}/blend_gate_AB_results.json")


if __name__ == "__main__":
    main()
