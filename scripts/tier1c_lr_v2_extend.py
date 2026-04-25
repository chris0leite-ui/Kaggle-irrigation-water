"""LR v2 follow-up: extend blend analysis to 4-stack anchor + emit
guardrail-passing candidates correctly.

The original v2 script's emit gate has a max-then-guard ordering bug —
it picks max-Δ entry first then drops if guardrail fails.  Correct logic:
among guardrail-PASS entries, pick max-Δ.  Manually emit candidates for
LB-probe consideration.
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
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, SUB, TARGET,
    bal_at_bias as bal, build_lbbest_stack, iso_cal, log,
    normed,
)


def per_class_recall(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    return [float((pred[y == k] == k).mean()) for k in range(3)]


def emit(blend_test, name):
    pred = (np.log(np.clip(blend_test, 1e-12, 1)) + BIAS).argmax(1)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred]
    p = SUB / f"submission_{name}.csv"
    sub.to_csv(p, index=False)
    log(f"  → {p}")
    return str(p)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("loading LR v2 OOF + test")
    lr2_oof_raw = np.load(ART / "oof_lr_metastack_v2.npy").astype(np.float32)
    lr2_test_raw = np.load(ART / "test_lr_metastack_v2.npy").astype(np.float32)
    lr2_oof, lr2_test = iso_cal(lr2_oof_raw, lr2_test_raw, y)

    log("rebuilding LB-best 3-stack + 4-stack anchors")
    lb3_oof, lb3_test = build_lbbest_stack(y)
    meta_oof = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    meta_test = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    meta_iso_oof, meta_iso_test = iso_cal(meta_oof, meta_test, y)
    lb4_oof = log_blend([lb3_oof, meta_iso_oof], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, meta_iso_test], np.array([0.7, 0.3]))

    lb3_bal = bal(lb3_oof, y)
    lb4_bal = bal(lb4_oof, y)
    log(f"  3-stack OOF = {lb3_bal:.5f}")
    log(f"  4-stack OOF = {lb4_bal:.5f}")

    pcr_lb3 = per_class_recall(lb3_oof, y)
    pcr_lb4 = per_class_recall(lb4_oof, y)
    log(f"  3-stack PCR: L={pcr_lb3[0]:.4f} M={pcr_lb3[1]:.4f} H={pcr_lb3[2]:.4f}")
    log(f"  4-stack PCR: L={pcr_lb4[0]:.4f} M={pcr_lb4[1]:.4f} H={pcr_lb4[2]:.4f}")

    # Two sweeps: vs 3-stack (deeper alphas) and vs 4-stack (smaller alphas)
    results = {}
    for label, anchor_oof, anchor_test, anchor_bal, pcr_anchor in [
        ("3stack", lb3_oof, lb3_test, lb3_bal, pcr_lb3),
        ("4stack", lb4_oof, lb4_test, lb4_bal, pcr_lb4),
    ]:
        log(f"\n=== sweep: LR_v2_iso × LB-best {label} ===")
        log(f"{'alpha':>8} {'OOF':>9} {'Δ':>9} {'recL':>7} {'recM':>7} {'recH':>7}  guard")
        rows = []
        for a in [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.35, 0.40]:
            blend = log_blend([anchor_oof, lr2_oof], np.array([1 - a, a]))
            b = bal(blend, y)
            d = b - anchor_bal
            pcr = per_class_recall(blend, y)
            passes = all(pcr[k] >= pcr_anchor[k] - 5e-4 for k in range(3))
            rows.append({"alpha": a, "oof": float(b), "delta": float(d),
                         "pcr": pcr, "guardrail_pass": bool(passes)})
            log(f"{a:>8.3f} {b:>9.5f} {d:>+9.5f} {pcr[0]:>7.4f} {pcr[1]:>7.4f} {pcr[2]:>7.4f}  {'PASS' if passes else 'FAIL'}")

        passing = [r for r in rows if r["guardrail_pass"] and r["delta"] >= 2e-4]
        if passing:
            best_pass = max(passing, key=lambda r: r["delta"])
            a = best_pass["alpha"]
            blend_test = log_blend([anchor_test, lr2_test], np.array([1 - a, a]))
            tag = f"lr_v2_iso_{label}_a{int(a*1000):03d}"
            sub_path = emit(blend_test, tag)
            log(f"  best guardrail-PASS: α={a:.3f} Δ=+{best_pass['delta']:.5f} → {sub_path}")
            best_pass["submission"] = sub_path
        else:
            best_pass = None
            log("  no α passes both gates")

        results[label] = {
            "anchor_oof": float(anchor_bal),
            "sweep": rows,
            "best_pass": best_pass,
        }

    # Diagnostic: errors + Jaccard
    pred_lb3 = (np.log(np.clip(lb3_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_lb4 = (np.log(np.clip(lb4_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_lr2 = (np.log(np.clip(lr2_oof, 1e-12, 1)) + BIAS).argmax(1)
    log(f"\nerrs LB-3stack={int((pred_lb3 != y).sum())}  "
        f"LB-4stack={int((pred_lb4 != y).sum())}  "
        f"LR_v2_iso={int((pred_lr2 != y).sum())}")
    for tag, a in [("3stack", pred_lb3), ("4stack", pred_lb4)]:
        i = ((a != y) & (pred_lr2 != y)).sum()
        u = ((a != y) | (pred_lr2 != y)).sum()
        log(f"Jaccard(LR_v2_iso, {tag}) = {i / max(u, 1):.4f}")

    out = dict(results=results, elapsed_sec=float(time.time() - t0))
    (ART / "tier1c_lr_v2_extend_results.json").write_text(json.dumps(out, indent=2))
    log(f"\nelapsed: {time.time()-t0:.1f}s")
    log("SUBMISSIONS EMITTED ARE CANDIDATES — DO NOT UPLOAD without user confirmation.")


if __name__ == "__main__":
    main()
