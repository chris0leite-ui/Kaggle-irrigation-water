"""LR v2 + per-class isotonic re-fit AFTER blend (Option B from the
2026-04-25 LR v2 LB null closure).

Mechanism:
  1. iso_cal LR v2 standalone (already on disk: oof_lr_metastack_v2.npy)
  2. log-blend(LB-best 3-stack, LR_v2_iso) at α
  3. PER-FOLD per-class isotonic on the BLENDED OOF (using
     StratifiedKFold(5, seed=42)); apply each fold's iso to its
     held-out rows. For test: fit per-class iso on full blended OOF,
     apply to test.
  4. Normalize, then evaluate at fixed recipe bias.

Why this might unlock LR v2 where dilution can't:
  - Iso-after-blend re-aligns the ensemble's per-class distribution
    with macro-recall optimum WITHOUT changing any decision-rule
    parameter (no log-bias retune → no binhigh-style overfit).
  - LR's convex-log-loss optimum produces probs at a different
    operating point than macro-recall wants; per-class iso on the
    blend output corrects exactly this mismatch.
  - Per-fold iso prevents leak (each fold's iso fit doesn't see
    held-out blended probs).

Gate: Δ ≥ +2e-4 OOF over LB-best 3-stack anchor AND per-class
recall guardrail PASS (each class ≥ anchor − 5e-4).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, N_FOLDS, SEED, SUB, TARGET,
    bal_at_bias as bal, build_lbbest_stack, iso_cal, log,
    normed,
)


def per_class_recall(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    return [float((pred[y == k] == k).mean()) for k in range(3)]


def per_fold_iso_on_blend(blend_oof, blend_test, y, n_folds=N_FOLDS, seed=SEED):
    """Per-fold per-class isotonic on the blend OOF.
    Held-out fold rows get iso predictions from the OTHER 4 folds.
    Test rows get full-OOF-fitted iso.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    iso_oof = np.zeros_like(blend_oof, dtype=np.float32)
    for tr_idx, va_idx in skf.split(blend_oof, y):
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip",
                                    y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(blend_oof[tr_idx, c], (y[tr_idx] == c).astype(np.float32))
            iso_oof[va_idx, c] = ir.predict(blend_oof[va_idx, c])
    iso_test = np.zeros_like(blend_test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(blend_oof[:, c], (y == c).astype(np.float32))
        iso_test[:, c] = ir.predict(blend_test[:, c])
    return normed(iso_oof), normed(iso_test)


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

    log("loading LR v2 + LB-best 3-stack")
    lr2_raw = np.load(ART / "oof_lr_metastack_v2.npy").astype(np.float32)
    lr2_test_raw = np.load(ART / "test_lr_metastack_v2.npy").astype(np.float32)
    lr2_oof, lr2_test = iso_cal(lr2_raw, lr2_test_raw, y)
    lb3_oof, lb3_test = build_lbbest_stack(y)

    lb3_bal = bal(lb3_oof, y)
    pcr_lb3 = per_class_recall(lb3_oof, y)
    log(f"  3-stack OOF = {lb3_bal:.5f}")
    log(f"  3-stack PCR: L={pcr_lb3[0]:.4f} M={pcr_lb3[1]:.4f} H={pcr_lb3[2]:.4f}")

    # Pre-iso baseline (no iso-after-blend) for comparison
    log("\n=== baseline: pre-iso-after-blend ===")
    log(f"{'alpha':>8} {'OOF':>9} {'Δ':>9} {'recL':>7} {'recM':>7} {'recH':>7}  guard")
    pre_rows = []
    for a in [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        b_oof = log_blend([lb3_oof, lr2_oof], np.array([1 - a, a]))
        bv = bal(b_oof, y)
        d = bv - lb3_bal
        pcr = per_class_recall(b_oof, y)
        passes = all(pcr[k] >= pcr_lb3[k] - 5e-4 for k in range(3))
        pre_rows.append({"alpha": a, "oof": float(bv), "delta": float(d),
                         "pcr": pcr, "guardrail_pass": bool(passes)})
        log(f"{a:>8.3f} {bv:>9.5f} {d:>+9.5f} {pcr[0]:>7.4f} {pcr[1]:>7.4f} {pcr[2]:>7.4f}  "
            f"{'PASS' if passes else 'FAIL'}")

    # WITH iso-after-blend (the actual experiment)
    log("\n=== WITH per-fold per-class iso re-fit AFTER blend ===")
    log(f"{'alpha':>8} {'OOF':>9} {'Δ':>9} {'recL':>7} {'recM':>7} {'recH':>7}  guard")
    iso_rows = []
    iso_test_for_emit = {}
    for a in [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        b_oof = log_blend([lb3_oof, lr2_oof], np.array([1 - a, a]))
        b_test = log_blend([lb3_test, lr2_test], np.array([1 - a, a]))
        iso_b_oof, iso_b_test = per_fold_iso_on_blend(b_oof, b_test, y)
        bv = bal(iso_b_oof, y)
        d = bv - lb3_bal
        pcr = per_class_recall(iso_b_oof, y)
        passes = all(pcr[k] >= pcr_lb3[k] - 5e-4 for k in range(3))
        iso_rows.append({"alpha": a, "oof": float(bv), "delta": float(d),
                         "pcr": pcr, "guardrail_pass": bool(passes)})
        iso_test_for_emit[a] = iso_b_test
        log(f"{a:>8.3f} {bv:>9.5f} {d:>+9.5f} {pcr[0]:>7.4f} {pcr[1]:>7.4f} {pcr[2]:>7.4f}  "
            f"{'PASS' if passes else 'FAIL'}")

    # Identify best guardrail-PASS candidate
    passing = [r for r in iso_rows if r["guardrail_pass"] and r["delta"] >= 2e-4]
    sub_path = None
    if passing:
        best = max(passing, key=lambda r: r["delta"])
        a = best["alpha"]
        tag = f"lr_v2_isoafter_3stack_a{int(a*1000):03d}"
        sub_path = emit(iso_test_for_emit[a], tag)
        best["submission"] = sub_path
        log(f"\nBEST guardrail-PASS: α={a:.3f} Δ=+{best['delta']:.5f} → {sub_path}")
    else:
        log("\nno α passes both gates after iso-after-blend")
        # fall back: emit the best-Δ guardrail-PASS even if Δ < 2e-4 for inspection
        guard_passing = [r for r in iso_rows if r["guardrail_pass"]]
        if guard_passing:
            best = max(guard_passing, key=lambda r: r["delta"])
            a = best["alpha"]
            log(f"  best guardrail-only: α={a:.3f} Δ=+{best['delta']:.5f} (below +2e-4)")

    # Diagnostic Jaccards
    pred_lb3 = (np.log(np.clip(lb3_oof, 1e-12, 1)) + BIAS).argmax(1)
    if passing:
        a = best["alpha"]
        b_oof_emit = log_blend([lb3_oof, lr2_oof], np.array([1 - a, a]))
        iso_b_oof_emit, _ = per_fold_iso_on_blend(b_oof_emit, b_oof_emit, y)  # only need OOF
        pred_iso = (np.log(np.clip(iso_b_oof_emit, 1e-12, 1)) + BIAS).argmax(1)
        i = ((pred_lb3 != y) & (pred_iso != y)).sum()
        u = ((pred_lb3 != y) | (pred_iso != y)).sum()
        log(f"\nerrs LB-3stack={int((pred_lb3 != y).sum())}  "
            f"iso-after-blend={int((pred_iso != y).sum())}  "
            f"Jaccard={i / max(u, 1):.4f}")

    out = dict(
        anchor_3stack_oof=float(lb3_bal),
        pre_iso_sweep=pre_rows,
        post_iso_sweep=iso_rows,
        best_pass=passing[0] if passing else None,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "tier1c_lr_v2_isoafter_results.json").write_text(json.dumps(out, indent=2))
    log(f"\nelapsed: {time.time()-t0:.1f}s")
    log("LB submission requires explicit user confirmation.")


if __name__ == "__main__":
    main()
