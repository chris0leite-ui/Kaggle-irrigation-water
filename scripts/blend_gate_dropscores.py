"""Blend-gate analyzer for the DROP_SCORES recipe variant.

Compares scripts/artifacts/oof_recipe_full_te_ds012.npy against three anchors:
  - recipe_full_te          (immediate baseline this would replace)
  - LB-best 3-stack          (lb3 at OOF 0.98061)
  - LB-best 4-stack          (3-stack + xgb_metastack_iso α=0.30, OOF 0.98084 / LB 0.98094)

For each anchor, reports standalone metrics + fixed-recipe-bias α-sweep.
Emit gate: Δ ≥ +2e-4 AND per-class recall ≥ anchor − 5e-4 each class.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    BIAS, ART, SUB, bal_at_bias, build_lbbest_stack, iso_cal, load_y, normed,
)

CAND_NAME = "recipe_full_te_ds012"
ALPHAS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
          0.50, 0.65, 0.80]
EMIT_GATE_DELTA = 2e-4
PCR_FLOOR_DROP = 5e-4
EPS = 1e-12


def per_class_recall(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, EPS, 1.0)) + bias).argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    return cm.diagonal() / np.maximum(cm.sum(1), 1)


def jaccard_err(p_a, p_b, y, bias=BIAS):
    err_a = (np.log(np.clip(p_a, EPS, 1.0)) + bias).argmax(1) != y
    err_b = (np.log(np.clip(p_b, EPS, 1.0)) + bias).argmax(1) != y
    inter = (err_a & err_b).sum()
    union = (err_a | err_b).sum()
    return inter / max(union, 1)


def main():
    print(f"=== blend gate: {CAND_NAME} ===")
    y = load_y()

    # --- LOAD candidate
    oof_c = normed(np.load(ART / f"oof_{CAND_NAME}.npy").astype(np.float32))
    test_c = normed(np.load(ART / f"test_{CAND_NAME}.npy").astype(np.float32))

    cand_at_anchor = bal_at_bias(oof_c, y)
    prior = np.bincount(y, minlength=3) / len(y)
    own_bias, cand_tuned = tune_log_bias(oof_c, y, prior)
    own_bias = np.array(own_bias)
    print(f"\nstandalone CANDIDATE")
    print(f"  @recipe-bias bal_acc = {cand_at_anchor:.5f}")
    print(f"  own tuned bal_acc    = {cand_tuned:.5f}  bias={own_bias.round(3).tolist()}")

    # --- ANCHORS
    print("\nbuilding anchors...")
    # recipe baseline (no DROP)
    o_r = normed(np.load(ART / "oof_recipe_full_te.npy").astype(np.float32))
    t_r = normed(np.load(ART / "test_recipe_full_te.npy").astype(np.float32))
    # LB-best 3-stack
    lb3_o, lb3_t = build_lbbest_stack(y)
    # LB-best 4-stack: 3-stack + xgb_metastack_iso α=0.30
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_oi, meta_ti = iso_cal(meta_o, meta_t, y)
    lb4_o = log_blend([lb3_o, meta_oi], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, meta_ti], np.array([0.7, 0.3]))

    anchors = [
        ("recipe_full_te",  o_r,   t_r),
        ("lb_best_3stack",  lb3_o, lb3_t),
        ("lb_best_4stack",  lb4_o, lb4_t),
    ]
    for name, oo, _ in anchors:
        print(f"  {name:<20} @recipe-bias = {bal_at_bias(oo, y):.5f}")

    # --- candidate per-class recall and Jaccard vs each anchor
    print("\nDIAGNOSTICS @ fixed recipe bias [1.4324, 1.4689, 3.4008]")
    pcr_c = per_class_recall(oof_c, y)
    err_c = ((np.log(np.clip(oof_c, EPS, 1.0)) + BIAS).argmax(1) != y).sum()
    print(f"  candidate     errs={int(err_c):,}  PCR={pcr_c.round(4).tolist()}")
    for name, oo, _ in anchors:
        pcr = per_class_recall(oo, y)
        errs = ((np.log(np.clip(oo, EPS, 1.0)) + BIAS).argmax(1) != y).sum()
        jac = jaccard_err(oof_c, oo, y)
        print(f"  {name:<20} errs={int(errs):,}  PCR={pcr.round(4).tolist()}  "
              f"Jaccard(cand,anchor)={jac:.4f}")

    # --- α-sweep blend vs each anchor at fixed recipe bias
    summary = {"candidate": CAND_NAME, "standalone_at_anchor": float(cand_at_anchor),
               "standalone_tuned": float(cand_tuned),
               "own_bias": own_bias.tolist(), "anchors": {}}

    print("\nα-SWEEP (log-blend onto anchor at fixed recipe bias)")
    for name, oo, _ in anchors:
        anchor_score = bal_at_bias(oo, y)
        anchor_pcr = per_class_recall(oo, y)
        sweep = []
        for alpha in ALPHAS:
            mix = log_blend([oo, oof_c], np.array([1 - alpha, alpha]))
            sc = bal_at_bias(mix, y)
            pcr = per_class_recall(mix, y)
            pcr_pass = all(pcr[k] >= anchor_pcr[k] - PCR_FLOOR_DROP for k in range(3))
            sweep.append({"alpha": alpha, "bal_acc": float(sc),
                          "delta": float(sc - anchor_score),
                          "pcr": pcr.round(5).tolist(),
                          "pcr_pass": bool(pcr_pass)})
        peak = max(sweep, key=lambda r: r["bal_acc"])
        gate_pass = (peak["delta"] >= EMIT_GATE_DELTA) and peak["pcr_pass"]
        print(f"\n  vs {name} (anchor {anchor_score:.5f}, PCR {anchor_pcr.round(4).tolist()})")
        for r in sweep:
            tag = ""
            if r is peak:
                tag = "  <-- peak"
                if gate_pass:
                    tag += "  EMIT"
            print(f"    α={r['alpha']:.3f}  bal={r['bal_acc']:.5f}  Δ={r['delta']:+.5f}  "
                  f"pcr={r['pcr']}  pass={r['pcr_pass']}{tag}")
        summary["anchors"][name] = {
            "anchor_bal_acc": float(anchor_score),
            "anchor_pcr": anchor_pcr.round(5).tolist(),
            "sweep": sweep,
            "peak_alpha": float(peak["alpha"]),
            "peak_delta": float(peak["delta"]),
            "gate_pass": bool(gate_pass),
        }

    out = ART / "blend_gate_dropscores_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
