"""Blend-gate analysis for path B per-cell MLP.

Computes:
  - standalone tuned OOF
  - errors + Jaccard vs LB-best 4-stack at recipe bias
  - log-blend sweep vs LB-best 3-stack and 4-stack
  - per-class recall guardrail check
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


def errs_at_bias(p, y, bias=BIAS):
    return int((np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1) != y).sum() if False else \
        (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1) != y


def jaccard(a_err, b_err):
    inter = (a_err & b_err).sum()
    union = (a_err | b_err).sum()
    return float(inter) / max(int(union), 1)


def per_class_recall(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    out = []
    for c in range(3):
        m = (y == c)
        out.append(float((pred[m] == c).mean()))
    return out


def main():
    y = load_y()
    print(f"y prior: {np.bincount(y) / len(y)}\n")

    # Load path B + LB-best 4-stack (= 3-stack + xgb_metastack_iso @α=0.30).
    pB_o = normed(np.load(ART / "oof_path_b_cell_mlp.npy").astype(np.float32))
    pB_t = normed(np.load(ART / "test_path_b_cell_mlp.npy").astype(np.float32))

    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    w4 = np.array([0.70, 0.30])
    lb4_o = normed(log_blend([lb3_o, meta_o_iso], w4))
    lb4_t = normed(log_blend([lb3_t, meta_t_iso], w4))

    print(f"path_B argmax = {balanced_accuracy_score(y, pB_o.argmax(1)):.5f}")
    prior = np.bincount(y, minlength=3) / len(y)
    bias_b, bal_b = tune_log_bias(pB_o, y, prior)
    print(f"path_B tuned (own bias) = {bal_b:.5f}  bias={bias_b.round(4).tolist()}")
    print(f"path_B tuned (recipe bias) = {bal_at_bias(pB_o, y):.5f}\n")

    # Try iso-cal (matches LB-best calibration pipeline).
    pB_o_iso, pB_t_iso = iso_cal(pB_o, pB_t, y)
    print(f"path_B iso (recipe bias) = {bal_at_bias(pB_o_iso, y):.5f}\n")

    # Anchors.
    bal_lb3 = bal_at_bias(lb3_o, y)
    bal_lb4 = bal_at_bias(lb4_o, y)
    print(f"LB-best 3-stack OOF @recipe-bias = {bal_lb3:.5f}")
    print(f"LB-best 4-stack OOF @recipe-bias = {bal_lb4:.5f}\n")

    # Error geometry.
    pB_err = (np.log(np.clip(pB_o_iso, 1e-12, 1)) + BIAS).argmax(1) != y
    lb3_err = (np.log(np.clip(lb3_o, 1e-12, 1)) + BIAS).argmax(1) != y
    lb4_err = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1) != y
    print(f"errs:    path_B={pB_err.sum()}  LB3={lb3_err.sum()}  LB4={lb4_err.sum()}")
    print(f"Jaccard(path_B, LB3)={jaccard(pB_err, lb3_err):.4f}  "
          f"Jaccard(path_B, LB4)={jaccard(pB_err, lb4_err):.4f}\n")

    # Blend sweep vs LB-best 3-stack and 4-stack at fixed recipe bias.
    print(f"{'α':>6} | {'vs LB3 OOF':>12} {'Δ_LB3':>10} | {'vs LB4 OOF':>12} {'Δ_LB4':>10}")
    print("-" * 70)
    for a in (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        b3 = normed(log_blend([lb3_o, pB_o_iso], np.array([1 - a, a])))
        b4 = normed(log_blend([lb4_o, pB_o_iso], np.array([1 - a, a])))
        s3 = bal_at_bias(b3, y); s4 = bal_at_bias(b4, y)
        print(f"{a:>6.3f} | {s3:>12.5f} {s3 - bal_lb3:>+10.5f} | "
              f"{s4:>12.5f} {s4 - bal_lb4:>+10.5f}")
    print()

    # Per-class recall at α=0 vs α=0.10 vs LB-best 4-stack (guardrail).
    base_pcr = per_class_recall(lb4_o, y)
    for a in (0.05, 0.10, 0.20):
        b4 = normed(log_blend([lb4_o, pB_o_iso], np.array([1 - a, a])))
        pcr = per_class_recall(b4, y)
        d = [pcr[i] - base_pcr[i] for i in range(3)]
        print(f"α={a}: PCR={[f'{v:.4f}' for v in pcr]} "
              f"(Δ={[f'{v:+.4f}' for v in d]})")


if __name__ == "__main__":
    main()
