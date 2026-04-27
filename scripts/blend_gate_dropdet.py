"""Blend-gate analyzer for the DROP_DETERMINISTIC recipe variant.

Mirrors blend_gate_dropscores.py but adds the 4th gate (G4: net rare-class
flip direction) introduced 2026-04-27. Evaluates oof_recipe_full_te_dropdet
against three anchors:
  - recipe_full_te          (immediate baseline this would replace)
  - LB-best 3-stack          (lb3 at OOF 0.98061)
  - LB-best 4-stack          (3-stack + xgb_metastack_iso α=0.30, OOF 0.98084 / LB 0.98094)

For each anchor, reports standalone metrics + fixed-recipe-bias α-sweep.

4-gate emit criterion (per CLAUDE.md 2026-04-27 R2/R5 closure):
  G1: blend OOF Δ ≥ +2e-4 vs anchor
  G2: errs ≤ 1.05× anchor (no magnitude trap)
  G3: per-class recall ≥ anchor − 5e-4 each class
  G4: |net_rare_class_flip| / |total_rare_class_churn| ≥ 0.5
      AND net direction is ADD-High (not REMOVE-High)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, bal_at_bias, build_lbbest_stack, iso_cal, load_y, normed,
)

CAND_NAME = "recipe_full_te_dropdet"
ALPHAS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
          0.50, 0.65, 0.80]
EMIT_GATE_DELTA = 2e-4
PCR_FLOOR_DROP = 5e-4
MAGNITUDE_LIMIT = 1.05
G4_RATIO_FLOOR = 0.5
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


def err_count(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, EPS, 1.0)) + bias).argmax(1)
    return int((pred != y).sum())


def rare_flip_diag(p_anchor, p_blend, bias=BIAS):
    """G4: net rare-class flip + churn ratio + direction.

    Returns dict with:
      net_high_change   = (#new High predictions) - (#removed High predictions)
      churn             = total rows where High prediction differs
      ratio             = |net| / churn (should be >= 0.5)
      direction         = "add" if net > 0, "remove" if net < 0, "noop" if 0
    """
    pa = (np.log(np.clip(p_anchor, EPS, 1.0)) + bias).argmax(1)
    pb = (np.log(np.clip(p_blend, EPS, 1.0)) + bias).argmax(1)
    add = int(((pa != 2) & (pb == 2)).sum())
    remove = int(((pa == 2) & (pb != 2)).sum())
    churn = add + remove
    net = add - remove
    if churn == 0:
        return {"add": add, "remove": remove, "net": net, "churn": 0,
                "ratio": 0.0, "direction": "noop"}
    direction = "add" if net > 0 else ("remove" if net < 0 else "noop")
    return {"add": add, "remove": remove, "net": net, "churn": churn,
            "ratio": abs(net) / churn, "direction": direction}


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

    # --- standalone diagnostics
    print("\nDIAGNOSTICS @ fixed recipe bias [1.4324, 1.4689, 3.4008]")
    pcr_c = per_class_recall(oof_c, y)
    err_c = err_count(oof_c, y)
    print(f"  candidate     errs={err_c:,}  PCR={pcr_c.round(4).tolist()}")
    for name, oo, _ in anchors:
        pcr = per_class_recall(oo, y)
        errs = err_count(oo, y)
        jac = jaccard_err(oof_c, oo, y)
        print(f"  {name:<20} errs={errs:,}  PCR={pcr.round(4).tolist()}  "
              f"Jaccard(cand,anchor)={jac:.4f}")

    # --- α-sweep blend vs each anchor at fixed recipe bias, with 4-gate filter
    summary = {"candidate": CAND_NAME, "standalone_at_anchor": float(cand_at_anchor),
               "standalone_tuned": float(cand_tuned),
               "own_bias": own_bias.tolist(), "anchors": {}}

    print("\nα-SWEEP (log-blend onto anchor at fixed recipe bias)")
    print("  4-gate filter: G1 Δ≥+2e-4 / G2 errs≤1.05×anchor / G3 PCR≥anchor−5e-4 / G4 net-rare ratio≥0.5 ADD direction")
    for name, oo, _ in anchors:
        anchor_score = bal_at_bias(oo, y)
        anchor_pcr = per_class_recall(oo, y)
        anchor_errs = err_count(oo, y)
        sweep = []
        for alpha in ALPHAS:
            mix = log_blend([oo, oof_c], np.array([1 - alpha, alpha]))
            sc = bal_at_bias(mix, y)
            pcr = per_class_recall(mix, y)
            mix_errs = err_count(mix, y)
            rare = rare_flip_diag(oo, mix)
            g1 = (sc - anchor_score) >= EMIT_GATE_DELTA
            g2 = mix_errs <= MAGNITUDE_LIMIT * anchor_errs
            g3 = all(pcr[k] >= anchor_pcr[k] - PCR_FLOOR_DROP for k in range(3))
            g4 = (rare["ratio"] >= G4_RATIO_FLOOR and rare["direction"] == "add")
            sweep.append({"alpha": alpha, "bal_acc": float(sc),
                          "delta": float(sc - anchor_score),
                          "pcr": pcr.round(5).tolist(),
                          "errs": mix_errs,
                          "rare": rare,
                          "g1": bool(g1), "g2": bool(g2),
                          "g3": bool(g3), "g4": bool(g4)})
        peak = max(sweep, key=lambda r: r["bal_acc"])
        gate_4_pass = peak["g1"] and peak["g2"] and peak["g3"] and peak["g4"]
        print(f"\n  vs {name} (anchor bal={anchor_score:.5f}, errs={anchor_errs:,}, PCR {anchor_pcr.round(4).tolist()})")
        for r in sweep:
            flags = "".join(["G1✓" if r["g1"] else "G1✗",
                             " G2✓" if r["g2"] else " G2✗",
                             " G3✓" if r["g3"] else " G3✗",
                             " G4✓" if r["g4"] else " G4✗"])
            tag = ""
            if r is peak:
                tag = "  <-- peak"
                if gate_4_pass:
                    tag += "  EMIT (all 4 gates pass)"
            print(f"    α={r['alpha']:.3f}  bal={r['bal_acc']:.5f}  Δ={r['delta']:+.5f}  "
                  f"errs={r['errs']:,}  net_H={r['rare']['net']:+d} "
                  f"churn={r['rare']['churn']} dir={r['rare']['direction']:>6} "
                  f"[{flags}]{tag}")
        summary["anchors"][name] = {
            "anchor_bal_acc": float(anchor_score),
            "anchor_pcr": anchor_pcr.round(5).tolist(),
            "anchor_errs": anchor_errs,
            "sweep": sweep,
            "peak_alpha": float(peak["alpha"]),
            "peak_delta": float(peak["delta"]),
            "gate_4_pass": bool(gate_4_pass),
        }

    out = ART / "blend_gate_dropdet_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
