"""Blend-gate analysis for path A PROBE (fold 1 only).

Computes:
  - standalone tuned OOF on the 126k filled rows
  - errors + Jaccard vs LB-best 4-stack at recipe bias (fold-1 rows only)
  - log-blend sweep vs LB-best 3-stack and 4-stack on the filled subset
  - per-class recall guardrail check

Path A is the FIRST NN to reach OOF ~0.976 at fold-1 on this competition.
The blend gate determines whether to push 5-fold production.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, bal_at_bias, build_lbbest_stack, iso_cal, load_y, normed,
)


def per_class_recall(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    return [float((pred[y == c] == c).mean()) for c in range(3)]


def main():
    y = load_y()
    print(f"y prior: {np.bincount(y) / len(y)}\n")

    # Load path A PROBE (fold 1 only — other folds are zeros).
    pA_o = np.load(ART / "oof_path_a_recipe_mlp_probe.npy").astype(np.float32)
    pA_t = np.load(ART / "test_path_a_recipe_mlp_probe.npy").astype(np.float32)
    filled = pA_o.sum(1) > 1e-6
    print(f"path_A filled rows: {filled.sum():,} / {len(pA_o):,} (fold-1 only)\n")

    # Reconstruct LB-best 3-stack and 4-stack ON THE SAME ROW SUBSET.
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    w4 = np.array([0.70, 0.30])
    lb4_o = normed(log_blend([lb3_o, meta_o_iso], w4))
    lb4_t = normed(log_blend([lb3_t, meta_t_iso], w4))

    # Restrict to fold-1 filled rows.
    pA_o_f = normed(pA_o[filled])
    lb3_o_f = lb3_o[filled]
    lb4_o_f = lb4_o[filled]
    y_f = y[filled]

    # Standalone diagnostics.
    print(f"path_A argmax (fold-1 rows) = {balanced_accuracy_score(y_f, pA_o_f.argmax(1)):.5f}")
    prior = np.bincount(y, minlength=3) / len(y)
    bias_a, bal_a = tune_log_bias(pA_o_f, y_f, prior)
    print(f"path_A tuned (own bias)     = {bal_a:.5f}  bias={bias_a.round(4).tolist()}")
    print(f"path_A tuned (recipe bias)  = {bal_at_bias(pA_o_f, y_f):.5f}\n")

    # iso-cal (matches LB-best calibration pipeline). Iso must be fit on
    # FULL OOF rows for the y-class-fraction model; we have fold-1 only.
    # Use fold-1 oof + corresponding y for iso fit.
    pA_o_iso, pA_t_iso = iso_cal(pA_o_f, pA_t, y_f)
    print(f"path_A iso  (recipe bias) = {bal_at_bias(pA_o_iso, y_f):.5f}\n")

    # Anchor metrics on the same subset.
    bal_lb3 = bal_at_bias(lb3_o_f, y_f)
    bal_lb4 = bal_at_bias(lb4_o_f, y_f)
    print(f"LB-3 stack on fold-1 rows @recipe-bias = {bal_lb3:.5f}")
    print(f"LB-4 stack on fold-1 rows @recipe-bias = {bal_lb4:.5f}\n")

    # Errors + Jaccard.
    pred_a = (np.log(np.clip(pA_o_iso, 1e-12, 1)) + BIAS).argmax(1)
    pred_lb3 = (np.log(np.clip(lb3_o_f, 1e-12, 1)) + BIAS).argmax(1)
    pred_lb4 = (np.log(np.clip(lb4_o_f, 1e-12, 1)) + BIAS).argmax(1)
    err_a = pred_a != y_f
    err_lb3 = pred_lb3 != y_f
    err_lb4 = pred_lb4 != y_f
    j_lb3 = (err_a & err_lb3).sum() / max((err_a | err_lb3).sum(), 1)
    j_lb4 = (err_a & err_lb4).sum() / max((err_a | err_lb4).sum(), 1)
    print(f"errs    path_A={err_a.sum()}  LB3={err_lb3.sum()}  LB4={err_lb4.sum()}")
    print(f"  ratio path_A/LB4 = {err_a.sum() / max(err_lb4.sum(), 1):.3f}")
    print(f"Jaccard(path_A, LB3) = {j_lb3:.4f}")
    print(f"Jaccard(path_A, LB4) = {j_lb4:.4f}\n")

    # Blend sweep on fold-1 rows.
    print(f"{'α':>6} | {'vs LB3 OOF':>12} {'Δ_LB3':>10} | {'vs LB4 OOF':>12} {'Δ_LB4':>10}")
    print("-" * 70)
    for a in (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        b3 = normed(log_blend([lb3_o_f, pA_o_iso], np.array([1 - a, a])))
        b4 = normed(log_blend([lb4_o_f, pA_o_iso], np.array([1 - a, a])))
        s3 = bal_at_bias(b3, y_f); s4 = bal_at_bias(b4, y_f)
        print(f"{a:>6.3f} | {s3:>12.5f} {s3 - bal_lb3:>+10.5f} | "
              f"{s4:>12.5f} {s4 - bal_lb4:>+10.5f}")
    print()

    # Per-class recall trade vs LB4.
    base_pcr = per_class_recall(lb4_o_f, y_f)
    print(f"LB4 base PCR = {[f'{v:.4f}' for v in base_pcr]}")
    for a in (0.05, 0.10, 0.15, 0.20):
        b4 = normed(log_blend([lb4_o_f, pA_o_iso], np.array([1 - a, a])))
        pcr = per_class_recall(b4, y_f)
        d = [pcr[i] - base_pcr[i] for i in range(3)]
        print(f"α={a}: PCR={[f'{v:.4f}' for v in pcr]} (Δ={[f'{v:+.4f}' for v in d]})")


if __name__ == "__main__":
    main()
