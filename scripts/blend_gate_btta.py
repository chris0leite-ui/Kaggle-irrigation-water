"""Blend-gate analyzer for the Mech A boundary-confined TTA recipe variant.

Compares scripts/artifacts/oof_recipe_full_te_btta095k10s005.npy against:
  - recipe_full_te          (immediate baseline this would replace)
  - LB-best 3-stack          (lb3 at OOF 0.98061)
  - LB-best 4-stack PRIMARY   (3-stack + xgb_metastack_iso α=0.30, OOF 0.98084 / LB 0.98094)

Two-mode analysis:
  Mode A — α-sweep BLEND of candidate onto each anchor (additive lever).
  Mode B — SUBSTITUTION: rebuild LB-best primary with TTA-recipe replacing
           the vanilla recipe inside lb3 = log_blend([recipe, s1, s7], ...).
           This is Mech A's natural deployment mode since the TTA tweaks
           recipe rows in-place at boundary rows only.

Emit gate: Δ ≥ +2e-4 vs LB-best primary AND per-class recall ≥ anchor − 5e-4.
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

# Discover the candidate file dynamically (suffix encodes K + σ + threshold).
# Filter out per-fold checkpoints (`_foldN.npy`) which are partial-shape.
import re
_btta = [p for p in sorted(ART.glob("oof_recipe_full_te_btta*.npy"))
         if not re.search(r"_fold\d+\.npy$", p.name)]
if not _btta:
    raise SystemExit("no btta candidate found in scripts/artifacts/")
CAND_PATH = _btta[-1]  # latest
CAND_NAME = CAND_PATH.stem.replace("oof_", "", 1)
print(f"candidate: {CAND_NAME}")

ALPHAS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
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


def build_lbbest_with_recipe(recipe_oof, recipe_test, y):
    """Reconstruct LB-best 3-stack but with arbitrary recipe arrays.

    Mirrors tier1b_helpers.build_lbbest_stack except the recipe component
    is supplied externally (so we can substitute TTA-recipe in).
    """
    def L(name):
        return (normed(np.load(ART / f"oof_{name}.npy")),
                normed(np.load(ART / f"test_{name}.npy")))
    s1 = L("recipe_pseudolabel")
    s7 = L("recipe_pseudolabel_seed7labeler")
    rm = L("realmlp")
    nr = L("xgb_nonrule")
    nr_o, nr_t = iso_cal(nr[0], nr[1], y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([recipe_oof, s1[0], s7[0]], w3)
    lb3_t = log_blend([recipe_test, s1[1], s7[1]], w3)
    s1_o = log_blend([lb3_o, rm[0]], np.array([0.8, 0.2]))
    s1_t = log_blend([lb3_t, rm[1]], np.array([0.8, 0.2]))
    s2_o = log_blend([s1_o, nr_o], np.array([0.925, 0.075]))
    s2_t = log_blend([s1_t, nr_t], np.array([0.925, 0.075]))
    return s2_o, s2_t


def main():
    print(f"=== blend gate: {CAND_NAME} ===")
    y = load_y()

    oof_c = normed(np.load(ART / f"oof_{CAND_NAME}.npy").astype(np.float32))
    test_c = normed(np.load(ART / f"test_{CAND_NAME}.npy").astype(np.float32))

    cand_at_anchor = bal_at_bias(oof_c, y)
    prior = np.bincount(y, minlength=3) / len(y)
    own_bias, cand_tuned = tune_log_bias(oof_c, y, prior)
    own_bias = np.array(own_bias)
    print(f"\nstandalone CANDIDATE (TTA recipe)")
    print(f"  @recipe-bias bal_acc = {cand_at_anchor:.5f}")
    print(f"  own tuned bal_acc    = {cand_tuned:.5f}  bias={own_bias.round(3).tolist()}")

    print("\nbuilding anchors...")
    o_r = normed(np.load(ART / "oof_recipe_full_te.npy").astype(np.float32))
    t_r = normed(np.load(ART / "test_recipe_full_te.npy").astype(np.float32))
    lb3_o, lb3_t = build_lbbest_stack(y)
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

    summary = {"candidate": CAND_NAME, "standalone_at_anchor": float(cand_at_anchor),
               "standalone_tuned": float(cand_tuned),
               "own_bias": own_bias.tolist(), "anchors": {}}

    # MODE A: α-sweep blend onto each anchor
    print("\n=== MODE A: α-SWEEP BLEND (log-blend onto anchor at fixed recipe bias) ===")
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

    # MODE B: SUBSTITUTION — rebuild primary with TTA recipe in lb3
    print("\n=== MODE B: SUBSTITUTION (rebuild LB-best primary with TTA recipe) ===")
    lb3_tta_o, lb3_tta_t = build_lbbest_with_recipe(oof_c, test_c, y)
    lb4_tta_o = log_blend([lb3_tta_o, meta_oi], np.array([0.7, 0.3]))
    lb4_tta_t = log_blend([lb3_tta_t, meta_ti], np.array([0.7, 0.3]))

    lb4_score = bal_at_bias(lb4_o, y)
    lb4_pcr = per_class_recall(lb4_o, y)
    lb4_tta_score = bal_at_bias(lb4_tta_o, y)
    lb4_tta_pcr = per_class_recall(lb4_tta_o, y)
    lb4_tta_errs = ((np.log(np.clip(lb4_tta_o, EPS, 1.0)) + BIAS).argmax(1) != y).sum()
    lb4_errs = ((np.log(np.clip(lb4_o, EPS, 1.0)) + BIAS).argmax(1) != y).sum()
    jac_sub = jaccard_err(lb4_tta_o, lb4_o, y)
    delta = lb4_tta_score - lb4_score
    pcr_pass_sub = all(lb4_tta_pcr[k] >= lb4_pcr[k] - PCR_FLOOR_DROP for k in range(3))
    gate_pass_sub = (delta >= EMIT_GATE_DELTA) and pcr_pass_sub

    print(f"\n  PRIMARY (LB-best 4-stack with VANILLA recipe):")
    print(f"    bal_acc = {lb4_score:.5f}  errs={int(lb4_errs):,}  "
          f"PCR={lb4_pcr.round(4).tolist()}")
    print(f"  PRIMARY' (LB-best 4-stack with TTA recipe):")
    print(f"    bal_acc = {lb4_tta_score:.5f}  errs={int(lb4_tta_errs):,}  "
          f"PCR={lb4_tta_pcr.round(4).tolist()}")
    print(f"    Δ vs vanilla = {delta:+.5f}  pcr_pass={pcr_pass_sub}  "
          f"Jaccard(tta,vanilla)={jac_sub:.4f}")
    print(f"    GATE: {'EMIT' if gate_pass_sub else 'no-emit'} "
          f"(Δ ≥ +2e-4: {delta >= EMIT_GATE_DELTA}, pcr_pass: {pcr_pass_sub})")

    # Test-side disagreement count
    pred_test_van = (np.log(np.clip(lb4_t, EPS, 1.0)) + BIAS).argmax(1)
    pred_test_tta = (np.log(np.clip(lb4_tta_t, EPS, 1.0)) + BIAS).argmax(1)
    n_diff = int((pred_test_van != pred_test_tta).sum())
    print(f"  test-side disagreement: {n_diff:,} / {len(pred_test_van):,} rows "
          f"({100*n_diff/len(pred_test_van):.3f}%)")

    summary["substitution"] = {
        "vanilla_primary_bal": float(lb4_score),
        "vanilla_primary_pcr": lb4_pcr.round(5).tolist(),
        "vanilla_primary_errs": int(lb4_errs),
        "tta_primary_bal": float(lb4_tta_score),
        "tta_primary_pcr": lb4_tta_pcr.round(5).tolist(),
        "tta_primary_errs": int(lb4_tta_errs),
        "delta": float(delta),
        "pcr_pass": bool(pcr_pass_sub),
        "gate_pass": bool(gate_pass_sub),
        "jaccard_with_vanilla_primary": float(jac_sub),
        "test_disagreement_count": n_diff,
        "test_disagreement_pct": float(100 * n_diff / len(pred_test_van)),
    }

    # Save TTA primary for cross-branch reuse if gate passed
    if gate_pass_sub:
        np.save(ART / "oof_lbbest_primary_btta.npy", lb4_tta_o.astype(np.float32))
        np.save(ART / "test_lbbest_primary_btta.npy", lb4_tta_t.astype(np.float32))
        # Build candidate submission CSV at fixed recipe bias
        test_ids = pd.read_csv("data/test.csv")["id"].values
        test_log = np.log(np.clip(lb4_tta_t, EPS, 1.0))
        pred_idx = (test_log + BIAS).argmax(1)
        cls = ["Low", "Medium", "High"]
        sub = pd.DataFrame({"id": test_ids,
                            "Irrigation_Need": [cls[i] for i in pred_idx]})
        sub_path = SUB / "submission_lbbest_primary_btta.csv"
        sub.to_csv(sub_path, index=False)
        print(f"  EMIT: wrote {sub_path}")
        print(f"        OOF {lb4_tta_score:.5f} (Δ vs primary {delta:+.5f})")

    # Filename matches gitignore whitelist convention.
    out = ART / f"mech_a_blend_gate_{CAND_NAME.replace('recipe_full_te_', '')}_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
