"""Blend-gate diagnostic for MoE gated blend (#1).

Runs fixed-bias log-blend sweep of MoE onto two anchors:
  - LB-best 3-stack  (anchor for the original meta_iso α=0.30 path)
  - LB-best 4-stack  (anchor = current LB best 0.98094 primary)

Reports Jaccard, magnitude (errs), per-class recall + alpha sweep with
the standard +5e-4 emit gate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                            load_y, normed)


ART = Path("scripts/artifacts")


def L(name):
    return (normed(np.load(ART / f"oof_{name}.npy").astype(np.float32)),
            normed(np.load(ART / f"test_{name}.npy").astype(np.float32)))


def per_class_recall(y, pred):
    out = []
    for c in range(3):
        mask = y == c
        out.append(float((pred[mask] == c).mean()))
    return out


def main():
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o, meta_t = L("xgb_metastack")
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    lb4_o = log_blend([lb3_o, meta_iso_o], np.array([0.70, 0.30]))
    lb4_t = log_blend([lb3_t, meta_iso_t], np.array([0.70, 0.30]))

    moe_o = np.load(ART / "oof_moe_gated.npy")
    moe_t = np.load(ART / "test_moe_gated.npy")

    def bal(p):
        return balanced_accuracy_score(
            y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))

    def errs(p):
        return int((y != (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)).sum())

    def pred(p):
        return (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)

    print(f"Anchors @ recipe bias:")
    print(f"  LB-best 3-stack: {bal(lb3_o):.6f}  errs={errs(lb3_o)}")
    print(f"  LB-best 4-stack (PRIMARY): {bal(lb4_o):.6f}  errs={errs(lb4_o)}  "
          f"PCR={[round(x,4) for x in per_class_recall(y, pred(lb4_o))]}")
    print(f"  MoE standalone: {bal(moe_o):.6f}  errs={errs(moe_o)}  "
          f"PCR={[round(x,4) for x in per_class_recall(y, pred(moe_o))]}")
    print()

    p_lb3 = pred(lb3_o)
    p_lb4 = pred(lb4_o)
    p_moe = pred(moe_o)
    j_lb3 = (np.logical_and(p_moe != y, p_lb3 != y).sum()
             / max(np.logical_or(p_moe != y, p_lb3 != y).sum(), 1))
    j_lb4 = (np.logical_and(p_moe != y, p_lb4 != y).sum()
             / max(np.logical_or(p_moe != y, p_lb4 != y).sum(), 1))
    print(f"Error-Jaccards:  vs LB-best 3-stack {j_lb3:.4f}   "
          f"vs LB-best 4-stack {j_lb4:.4f}")
    print()

    alphas = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    res = {"vs_lb3": [], "vs_lb4": []}
    print("Sweep vs LB-best 3-stack (anchor 0.98061):")
    print(f"  {'α':>6}  {'OOF':>8}  {'Δ':>9}  {'errs':>6}  PCR")
    for a in alphas:
        if a == 0.0:
            blend = lb3_o
        else:
            blend = log_blend([lb3_o, moe_o], np.array([1 - a, a]))
        b = bal(blend)
        delta = b - bal(lb3_o)
        e = errs(blend)
        pcr = per_class_recall(y, pred(blend))
        print(f"  {a:>6.3f}  {b:.6f}  {delta:+.5f}  {e:>6}  "
              f"{[round(x,4) for x in pcr]}")
        res["vs_lb3"].append({"alpha": a, "oof": b, "delta": delta,
                              "errs": e, "pcr": pcr})

    print()
    print("Sweep vs LB-best 4-stack (anchor 0.98084 = PRIMARY):")
    print(f"  {'α':>6}  {'OOF':>8}  {'Δ':>9}  {'errs':>6}  PCR")
    for a in alphas:
        if a == 0.0:
            blend = lb4_o
        else:
            blend = log_blend([lb4_o, moe_o], np.array([1 - a, a]))
        b = bal(blend)
        delta = b - bal(lb4_o)
        e = errs(blend)
        pcr = per_class_recall(y, pred(blend))
        print(f"  {a:>6.3f}  {b:.6f}  {delta:+.5f}  {e:>6}  "
              f"{[round(x,4) for x in pcr]}")
        res["vs_lb4"].append({"alpha": a, "oof": b, "delta": delta,
                              "errs": e, "pcr": pcr})

    res["jaccard_vs_lb3"] = j_lb3
    res["jaccard_vs_lb4"] = j_lb4
    res["lb_best_3stack_oof"] = bal(lb3_o)
    res["lb_best_4stack_oof"] = bal(lb4_o)
    res["moe_standalone_oof"] = bal(moe_o)
    res["pcr_lb4"] = per_class_recall(y, pred(lb4_o))
    res["pcr_moe"] = per_class_recall(y, pred(moe_o))
    (ART / "moe_blend_gate_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved moe_blend_gate_results.json")


if __name__ == "__main__":
    main()
