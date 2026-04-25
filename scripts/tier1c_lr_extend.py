"""Extended diagnostics for the Tier-1c LR meta-stacker:
  1. Reload LR_iso meta-stacker output saved by tier1c_lr_metastack.py
  2. Blend at α ∈ [0..1] (extended) into BOTH:
       - LB-best 3-stack (OOF 0.98061) — the meta-stacker anchor
       - LB-best 4-stack (OOF 0.98084) — primary submission, includes XGB meta
  3. Per-class recall delta on all blend candidates
  4. Per-class recall guardrail (-0.0005 on rare class) + +2e-4 emit gate
  5. Cross-component diagnostic: blend LR + XGB meta-stackers (since they
     have Jaccard 0.717, may stack on top of the 4-stack better than either
     alone)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, SUB, TARGET,
    bal_at_bias as bal, build_lbbest_stack, iso_cal, log,
)


def per_class_recall(p, y):
    pred = (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)
    out = np.zeros(3)
    for c in range(3):
        m = y == c
        out[c] = (pred[m] == c).mean() if m.sum() else 0.0
    return out


def errors(p, y):
    return int(((np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1) != y).sum())


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    # Anchors
    log("building LB-best 3-stack (anchor for meta-stackers)")
    lb3_o, lb3_t = build_lbbest_stack(y)
    lb3_bal = bal(lb3_o, y)
    lb3_pcr = per_class_recall(lb3_o, y)
    lb3_err = errors(lb3_o, y)
    log(f"  3-stack OOF = {lb3_bal:.5f}  errs = {lb3_err}  "
        f"PCR = [L {lb3_pcr[0]:.4f} M {lb3_pcr[1]:.4f} H {lb3_pcr[2]:.4f}]")

    # Build 4-stack: 3-stack + xgb_metastack_iso at α=0.30 (LB-best 0.98094)
    log("building LB-best 4-stack (primary, OOF 0.98084 / LB 0.98094)")
    xgb_oof = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    xgb_test = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    xgb_iso_o, xgb_iso_t = iso_cal(xgb_oof, xgb_test, y)
    lb4_o = log_blend([lb3_o, xgb_iso_o], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, xgb_iso_t], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_o, y)
    lb4_pcr = per_class_recall(lb4_o, y)
    lb4_err = errors(lb4_o, y)
    log(f"  4-stack OOF = {lb4_bal:.5f}  errs = {lb4_err}  "
        f"PCR = [L {lb4_pcr[0]:.4f} M {lb4_pcr[1]:.4f} H {lb4_pcr[2]:.4f}]")

    # Load LR meta-stacker output, iso-cal it
    log("loading LR meta-stacker + iso-cal")
    lr_oof = np.load(ART / "oof_lr_metastack.npy").astype(np.float32)
    lr_test = np.load(ART / "test_lr_metastack.npy").astype(np.float32)
    lr_iso_o, lr_iso_t = iso_cal(lr_oof, lr_test, y)
    lr_bal = bal(lr_iso_o, y)
    lr_pcr = per_class_recall(lr_iso_o, y)
    lr_err = errors(lr_iso_o, y)
    log(f"  LR_iso OOF  = {lr_bal:.5f}  errs = {lr_err}  "
        f"PCR = [L {lr_pcr[0]:.4f} M {lr_pcr[1]:.4f} H {lr_pcr[2]:.4f}]")

    # Extended blend sweeps. Skip submission emission here — diagnostic only.
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

    def sweep(label, a_oof, a_test, a_bal):
        log(f"\n=== {label} ===")
        log(f"{'alpha':>6} {'OOF':>9} {'Δ':>9} {'errs':>6} {'recL':>7} {'recM':>7} {'recH':>7}")
        rows = []
        best = {"alpha": 0.0, "delta": 0.0, "oof": float(a_bal),
                "errs": errors(a_oof, y), "pcr": per_class_recall(a_oof, y).tolist()}
        for a in alphas:
            blend = log_blend([a_oof, lr_iso_o], np.array([1 - a, a]))
            b = bal(blend, y)
            d = b - a_bal
            er = errors(blend, y)
            pcr = per_class_recall(blend, y)
            rows.append({"alpha": a, "oof": float(b), "delta": float(d),
                         "errs": er, "pcr": pcr.tolist()})
            tag = " *" if d > best["delta"] else ""
            log(f"{a:>6.2f} {b:>9.5f} {d:>+9.5f} {er:>6} "
                f"{pcr[0]:>7.4f} {pcr[1]:>7.4f} {pcr[2]:>7.4f}{tag}")
            if d > best["delta"]:
                best = {"alpha": float(a), "delta": float(d), "oof": float(b),
                        "errs": er, "pcr": pcr.tolist()}
        # Also test-side blend at the best α
        a = best["alpha"]
        if a > 0:
            test_blend = log_blend([a_test, lr_iso_t], np.array([1 - a, a]))
            best["test_blend_built"] = True
            best["test_blend_argmax_classcounts"] = np.bincount(
                (np.log(np.clip(test_blend, 1e-12, 1)) + BIAS).argmax(1),
                minlength=3,
            ).tolist()
        return rows, best, (test_blend if a > 0 else None)

    # vs LB-best 3-stack (anchor for meta-stackers)
    rows3, best3, test_blend3 = sweep("LR_iso × LB-3-stack (OOF 0.98061)",
                                      lb3_o, lb3_t, lb3_bal)

    # vs LB-best 4-stack (primary, LB 0.98094)
    rows4, best4, test_blend4 = sweep("LR_iso × LB-4-stack (OOF 0.98084 / LB 0.98094)",
                                      lb4_o, lb4_t, lb4_bal)

    # Cross-stacker compound: blend BOTH meta-stackers into the 3-stack
    log("\n=== compound: 3-stack + α_xgb · XGB_iso + α_lr · LR_iso ===")
    log(f"{'a_xgb':>6} {'a_lr':>6} {'OOF':>9} {'Δ':>9} {'errs':>6} {'recH':>7}")
    compound = []
    best_cmp = {"a_xgb": 0.0, "a_lr": 0.0, "oof": float(lb3_bal), "delta": 0.0,
                "errs": lb3_err}
    for a_xgb in (0.0, 0.10, 0.20, 0.30, 0.40):
        for a_lr in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
            w_anchor = 1 - a_xgb - a_lr
            if w_anchor <= 0:
                continue
            w = np.array([w_anchor, a_xgb, a_lr])
            blend = log_blend([lb3_o, xgb_iso_o, lr_iso_o], w)
            b = bal(blend, y)
            d = b - lb3_bal
            er = errors(blend, y)
            pcr = per_class_recall(blend, y)
            compound.append({"a_xgb": a_xgb, "a_lr": a_lr,
                             "oof": float(b), "delta": float(d), "errs": er,
                             "pcr": pcr.tolist()})
            tag = " *" if d > best_cmp["delta"] else ""
            log(f"{a_xgb:>6.2f} {a_lr:>6.2f} {b:>9.5f} {d:>+9.5f} {er:>6} "
                f"{pcr[2]:>7.4f}{tag}")
            if d > best_cmp["delta"]:
                best_cmp = {"a_xgb": float(a_xgb), "a_lr": float(a_lr),
                            "oof": float(b), "delta": float(d), "errs": er,
                            "pcr": pcr.tolist()}

    # Per-class guardrail: drop ≤ 0.0005 on the worst-hurt class
    def passes_guardrail(pcr_blend, pcr_anchor, tol=5e-4):
        return all(pcr_blend[c] >= pcr_anchor[c] - tol for c in range(3))

    log("\n=== guardrail check on best blend candidates ===")
    candidates = [
        ("LR×3-stack", best3, lb3_pcr),
        ("LR×4-stack", best4, lb4_pcr),
    ]
    for name, b, pcr_anchor in candidates:
        ok = passes_guardrail(b["pcr"], pcr_anchor)
        log(f"  {name}: α={b['alpha']:.2f} Δ={b['delta']:+.5f} "
            f"PCR={b['pcr']} guardrail_pass={ok}")

    out = {
        "lb3_baseline": {"oof": float(lb3_bal), "errs": lb3_err, "pcr": lb3_pcr.tolist()},
        "lb4_baseline": {"oof": float(lb4_bal), "errs": lb4_err, "pcr": lb4_pcr.tolist()},
        "lr_iso_standalone": {"oof": float(lr_bal), "errs": lr_err, "pcr": lr_pcr.tolist()},
        "sweep_vs_lb3": rows3, "best_vs_lb3": best3,
        "sweep_vs_lb4": rows4, "best_vs_lb4": best4,
        "compound_grid": compound, "best_compound": best_cmp,
        "elapsed_sec": float(time.time() - t0),
    }
    (ART / "tier1c_lr_extend_results.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote scripts/artifacts/tier1c_lr_extend_results.json")
    log(f"TOTAL elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
