"""Blend gate for joint weights+bias optimization (#4) and CMA-ES (#2).

Both produce constant-weight blend OOF/test outputs; same gate logic applies.
Reports standalone OOF + Jaccard + per-class recall + α-sweep against
LB-best 4-stack at the LB-VALIDATED bias [1.4324, 1.4689, 3.4008].
"""
from __future__ import annotations

import json
import sys
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


def gate(name_or_oof, label, save_json=None):
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o, meta_t = L("xgb_metastack")
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    lb4_o = log_blend([lb3_o, meta_iso_o], np.array([0.70, 0.30]))
    lb4_t = log_blend([lb3_t, meta_iso_t], np.array([0.70, 0.30]))

    if isinstance(name_or_oof, str):
        cand_o = np.load(ART / f"oof_{name_or_oof}.npy").astype(np.float32)
    else:
        cand_o = name_or_oof.astype(np.float32)
    cand_o = normed(cand_o)

    def bal(p):
        return balanced_accuracy_score(
            y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))

    def errs(p):
        return int((y != (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)).sum())

    def pred(p):
        return (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)

    def pcr(p):
        out = []
        pp = pred(p)
        for c in range(3):
            mask = y == c
            out.append(float((pp[mask] == c).mean()))
        return out

    p_lb4 = pred(lb4_o)
    p_cand = pred(cand_o)
    j = (np.logical_and(p_cand != y, p_lb4 != y).sum()
         / max(np.logical_or(p_cand != y, p_lb4 != y).sum(), 1))

    print(f"=== {label} ===")
    print(f"  Anchor LB-4 OOF: {bal(lb4_o):.6f}  errs={errs(lb4_o)}  PCR={[round(x,4) for x in pcr(lb4_o)]}")
    print(f"  {label} OOF:       {bal(cand_o):.6f}  errs={errs(cand_o)}  PCR={[round(x,4) for x in pcr(cand_o)]}")
    print(f"  Jaccard vs LB-4:   {j:.4f}")
    print(f"  α-sweep vs LB-4 (anchor 0.98084):")
    print(f"    {'α':>6}  {'OOF':>8}  {'Δ':>9}  {'errs':>6}  PCR")
    rows = []
    base = bal(lb4_o)
    for a in [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        if a == 0.0:
            blend = lb4_o
        else:
            blend = log_blend([lb4_o, cand_o], np.array([1 - a, a]))
        b = bal(blend)
        delta = b - base
        e = errs(blend)
        cls_recall = pcr(blend)
        print(f"    {a:>6.3f}  {b:.6f}  {delta:+.5f}  {e:>6}  {[round(x,4) for x in cls_recall]}")
        rows.append({"alpha": a, "oof": b, "delta": delta, "errs": e, "pcr": cls_recall})
    if save_json:
        out = {"label": label, "jaccard_vs_lb4": j,
               "anchor_oof": bal(lb4_o), "candidate_oof": bal(cand_o),
               "anchor_errs": errs(lb4_o), "candidate_errs": errs(cand_o),
               "alpha_sweep": rows}
        (ART / save_json).write_text(json.dumps(out, indent=2))
        print(f"  Saved {save_json}")
    print()


def main():
    if len(sys.argv) < 3:
        print("Usage: joint_blend_gate.py <name> <label> [<save_json>]")
        sys.exit(0)
    name = sys.argv[1]
    label = sys.argv[2]
    save = sys.argv[3] if len(sys.argv) > 3 else None
    gate(name, label, save)


if __name__ == "__main__":
    main()
