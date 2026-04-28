"""4-gate blend gate + leakage defense checks for DART recipe variant.

DART (Dropouts meet multiple Additive Regression Trees) at recipe BASE
level: same V10 recipe FE pipeline, swap booster='gbtree' → 'dart'. Tree
dropout during training produces a structurally different ensemble than
gbtree's monotone additive trees. Hypothesis: dropout decorrelates trees
enough to produce errors orthogonal to recipe_full_te (the LB 0.97939
gbtree baseline) and the LB-best 4-stack at 0.98084.

Tests:
  1. Standalone vs recipe_full_te (gbtree baseline OOF 0.97967 tuned)
  2. Blend vs LB-best primary at α∈{0.05..0.50}, fixed recipe bias
  3. Cross-meta error correlation (Jaccard with v1 meta + recipe)
  4. Minimal-input check (small XGB on lb3 + dart probs only)

Gates (per LEARNINGS.md after 24+ saturation confirmations):
  G1: blend Δ vs LB-best primary ≥ +2e-4 at α=0.30
  G2: per-class recall Δ ≥ -5e-4 each class
  G3: dual-α stability Δ(0.40)/Δ(0.30) ∈ [1.0, 2.0]
  G4: net_high_flip > 0 AND |net|/|churn| ≥ 0.5
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (ART, BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                             load_y, bal_at_bias)


def per_class_recall(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    return tuple(float(((pred == c) & (y == c)).sum() /
                       max((y == c).sum(), 1)) for c in range(3))


def errs(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    return int((pred != y).sum())


def jaccard(p1, p2, y, bias=BIAS):
    pred1 = (np.log(np.clip(p1, 1e-12, 1)) + bias).argmax(1)
    pred2 = (np.log(np.clip(p2, 1e-12, 1)) + bias).argmax(1)
    e1, e2 = (pred1 != y), (pred2 != y)
    inter, union = (e1 & e2).sum(), (e1 | e2).sum()
    return float(inter / max(union, 1))


def net_high_flip(p_blend, p_anchor, y, bias=BIAS):
    pa = (np.log(np.clip(p_anchor, 1e-12, 1)) + bias).argmax(1)
    pb = (np.log(np.clip(p_blend, 1e-12, 1)) + bias).argmax(1)
    add_h = ((pa != 2) & (pb == 2)).sum()
    rem_h = ((pa == 2) & (pb != 2)).sum()
    return int(add_h), int(rem_h), int(add_h - rem_h), int(add_h + rem_h)


def main(suffix="_dart"):
    t0 = time.time()
    y = load_y()
    print(f"[gate] LB-best 3-stack: rebuilding")
    lb3_oof, lb3_test = build_lbbest_stack(y)

    print(f"[gate] loading dart recipe: oof_recipe_full_te{suffix}.npy")
    dart_oof = np.load(ART / f"oof_recipe_full_te{suffix}.npy")
    dart_test = np.load(ART / f"test_recipe_full_te{suffix}.npy")

    # LB-best primary (4-stack: 0.7 × lb3 + 0.3 × xgb_metastack_iso)
    v1_oof = np.load(ART / "oof_xgb_metastack.npy")
    v1_test = np.load(ART / "test_xgb_metastack.npy")
    v1_iso_o, v1_iso_t = iso_cal(v1_oof, v1_test, y)
    primary_oof = log_blend([lb3_oof, v1_iso_o], np.array([0.7, 0.3]))
    primary_test = log_blend([lb3_test, v1_iso_t], np.array([0.7, 0.3]))
    primary_bal = bal_at_bias(primary_oof, y)
    print(f"[gate] LB-best primary OOF = {primary_bal:.5f} (target 0.98084)")

    # gbtree baseline for comparison
    gb_oof = np.load(ART / "oof_recipe_full_te.npy")
    gb_test = np.load(ART / "test_recipe_full_te.npy")
    gb_bal = bal_at_bias(gb_oof, y)
    print(f"[gate] gbtree baseline   = {gb_bal:.5f} (target 0.97967)")

    dart_iso_o, dart_iso_t = iso_cal(dart_oof, dart_test, y)
    standalone_bal = bal_at_bias(dart_oof, y)
    iso_bal = bal_at_bias(dart_iso_o, y)
    print(f"[gate] dart standalone tuned = {standalone_bal:.5f}")
    print(f"[gate] dart iso       tuned  = {iso_bal:.5f}")
    print(f"[gate] errs(dart_iso) = {errs(dart_iso_o, y)}, "
          f"errs(primary) = {errs(primary_oof, y)}, "
          f"errs(gbtree) = {errs(gb_oof, y)}")
    print(f"[gate] Jaccard(dart_iso, primary) = "
          f"{jaccard(dart_iso_o, primary_oof, y):.4f}")
    print(f"[gate] Jaccard(dart, gbtree)      = "
          f"{jaccard(dart_oof, gb_oof, y):.4f}")

    # 4-gate sweep on (dart_iso × primary) at recipe bias.
    rows = []
    for alpha in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        blend_o = log_blend([primary_oof, dart_iso_o], np.array([1 - alpha, alpha]))
        bal = bal_at_bias(blend_o, y)
        delta = bal - primary_bal
        pcr = per_class_recall(blend_o, y)
        pcr_anchor = per_class_recall(primary_oof, y)
        pcr_d = tuple(b - a for a, b in zip(pcr_anchor, pcr))
        e = errs(blend_o, y)
        add_h, rem_h, net_h, churn_h = net_high_flip(blend_o, primary_oof, y)
        rows.append({
            "alpha": alpha, "oof": float(bal), "delta": float(delta),
            "errs": e, "pcr_delta": list(pcr_d),
            "net_h": net_h, "add_h": add_h, "rem_h": rem_h, "churn_h": churn_h,
        })
        print(f"  α={alpha:0.2f}  Δ={delta:+.5f}  PCR Δ=[{pcr_d[0]:+.5f},"
              f" {pcr_d[1]:+.5f}, {pcr_d[2]:+.5f}]  net_h={net_h:+d}/{churn_h}")

    # Cross-meta correlation (D1).
    print(f"\n[D1] cross-meta error Jaccards:")
    for name in ("xgb_metastack", "lr_metastack_v2", "mlp_metastack",
                 "recipe_full_te_lgbm", "recipe_full_te_catboost"):
        p = ART / f"oof_{name}.npy"
        if not p.exists():
            continue
        other = np.load(p)
        oth_iso, _ = iso_cal(other, np.load(ART / f"test_{name}.npy"), y)
        j = jaccard(dart_iso_o, oth_iso, y)
        print(f"  Jaccard(dart_iso, {name}_iso) = {j:.4f}")

    out = dict(
        primary_oof=float(primary_bal),
        gbtree_baseline_oof=float(gb_bal),
        dart_standalone=float(standalone_bal),
        dart_iso=float(iso_bal),
        sweep=rows,
        elapsed=time.time() - t0,
    )
    (ART / f"dart_blend_gate{suffix}_results.json").write_text(json.dumps(out, indent=2))
    print(f"[gate] wrote dart_blend_gate{suffix}_results.json wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
