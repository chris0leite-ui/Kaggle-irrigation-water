"""Blend-gate for recipe_focal vs LB-best 3-way teacher.

Diagnostics:
  - Standalone focal OOF tuned bal_acc
  - Error count vs teacher
  - Jaccard(err) vs recipe, vs 2-way LB-best, vs 3-way LB-best
  - Per-class recall (Low / Medium / High) at focal's own tuned bias
  - Fixed-bias log-blend sweep at teacher's bias (no retune)
  - 3-way blend grid (teacher x focal) and 4-way (adding focal to 3-way)

Gate decisions per LEARNINGS.md rules:
  - Jaccard < 0.80 (orthogonality) AND errs <= anchor (magnitude)
  - OOF delta >= +0.0002 for LB-transfer probability
  - If passes, emit submission; otherwise report diagnostic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    fast_bal_acc, log_blend, tune_log_bias, CLS2IDX,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True, parents=True)

FOCAL_SUFFIX = "_g2h3"  # matches scripts/recipe_focal.py run
W_RECIPE, W_S1, W_S7 = 0.25, 0.35, 0.40  # LB-best 3-way weights


def err_set(oof: np.ndarray, bias: np.ndarray, y: np.ndarray) -> set[int]:
    eps = 1e-9
    pred = (np.log(np.clip(oof, eps, 1.0)) + bias).argmax(1)
    return set(np.where(pred != y)[0].tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def per_class_recall(y: np.ndarray, pred: np.ndarray, K: int = 3) -> np.ndarray:
    cc = np.bincount(y, minlength=K)
    matches = (pred == y)
    hit = np.array([matches[y == k].sum() for k in range(K)], dtype=np.int64)
    return hit / np.maximum(cc, 1)


def main() -> None:
    # ---------------------------------------------------- load
    oof_focal = np.load(ART / f"oof_recipe_focal{FOCAL_SUFFIX}.npy")
    test_focal = np.load(ART / f"test_recipe_focal{FOCAL_SUFFIX}.npy")

    oof_r = np.load(ART / "oof_recipe_full_te.npy")
    test_r = np.load(ART / "test_recipe_full_te.npy")
    oof_s1 = np.load(ART / "oof_recipe_pseudolabel.npy")
    test_s1 = np.load(ART / "test_recipe_pseudolabel.npy")
    oof_s7 = np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")
    test_s7 = np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)

    # ---------------------------------------------------- anchors
    w3 = np.array([W_RECIPE, W_S1, W_S7])
    oof_t3 = log_blend([oof_r, oof_s1, oof_s7], w3)
    test_t3 = log_blend([test_r, test_s1, test_s7], w3)
    oof_t2 = log_blend([oof_r, oof_s1], np.array([0.5, 0.5]))
    test_t2 = log_blend([test_r, test_s1], np.array([0.5, 0.5]))

    bias_t3, bal_t3 = tune_log_bias(oof_t3, y, prior)
    bias_t2, bal_t2 = tune_log_bias(oof_t2, y, prior)
    bias_r, bal_r = tune_log_bias(oof_r, y, prior)
    bias_f, bal_f = tune_log_bias(oof_focal, y, prior)

    print(f"{'anchor':<20s} {'tuned_bal':>10s} {'bias':>28s}")
    print(f"{'recipe':<20s} {bal_r:>10.5f}  {str(bias_r.round(3).tolist()):>26s}")
    print(f"{'LB-best 2-way':<20s} {bal_t2:>10.5f}  {str(bias_t2.round(3).tolist()):>26s}")
    print(f"{'LB-best 3-way':<20s} {bal_t3:>10.5f}  {str(bias_t3.round(3).tolist()):>26s}")
    print(f"{'focal (standalone)':<20s} {bal_f:>10.5f}  {str(bias_f.round(3).tolist()):>26s}")

    # ---------------------------------------------------- error geometry
    errs_r = err_set(oof_r, bias_r, y)
    errs_t3 = err_set(oof_t3, bias_t3, y)
    errs_f = err_set(oof_focal, bias_f, y)

    # Recall deltas at focal's tuned bias
    eps = 1e-9
    pred_r = (np.log(np.clip(oof_r, eps, 1.0)) + bias_r).argmax(1)
    pred_t3 = (np.log(np.clip(oof_t3, eps, 1.0)) + bias_t3).argmax(1)
    pred_f = (np.log(np.clip(oof_focal, eps, 1.0)) + bias_f).argmax(1)

    rec_r = per_class_recall(y, pred_r)
    rec_t3 = per_class_recall(y, pred_t3)
    rec_f = per_class_recall(y, pred_f)

    print(f"\n{'model':<20s} {'errs':>7s} {'recL':>8s} {'recM':>8s} {'recH':>8s}")
    print(f"{'recipe':<20s} {len(errs_r):>7d} {rec_r[0]:>8.4f} {rec_r[1]:>8.4f} {rec_r[2]:>8.4f}")
    print(f"{'LB-best 3-way':<20s} {len(errs_t3):>7d} {rec_t3[0]:>8.4f} {rec_t3[1]:>8.4f} {rec_t3[2]:>8.4f}")
    print(f"{'focal standalone':<20s} {len(errs_f):>7d} {rec_f[0]:>8.4f} {rec_f[1]:>8.4f} {rec_f[2]:>8.4f}")

    j_f_r = jaccard(errs_f, errs_r)
    j_f_t3 = jaccard(errs_f, errs_t3)
    print(f"\nJaccard(err) focal vs recipe    = {j_f_r:.4f}")
    print(f"Jaccard(err) focal vs 3-way     = {j_f_t3:.4f}")

    # ---------------------------------------------------- blend sweeps
    print("\n=== fixed-bias log-blend sweep: alpha_focal * focal + (1-a) * anchor ===")
    print(f"{'alpha':>6} {'vs recipe':>10} {'vs 3-way':>10} {'vs 2-way':>10}")
    grid = [0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
             0.30, 0.40, 0.50]
    best = {"recipe": (-999, 0), "t3": (-999, 0), "t2": (-999, 0)}
    for a in grid:
        logf = np.log(np.clip(oof_focal, eps, 1.0))
        def blend(log_anchor, bias):
            z = a * logf + (1 - a) * log_anchor + bias
            p = z.argmax(1)
            return fast_bal_acc(y, p)
        v_r = blend(np.log(np.clip(oof_r, eps, 1.0)), bias_r)
        v_t3 = blend(np.log(np.clip(oof_t3, eps, 1.0)), bias_t3)
        v_t2 = blend(np.log(np.clip(oof_t2, eps, 1.0)), bias_t2)
        if v_r > best["recipe"][0]: best["recipe"] = (v_r, a)
        if v_t3 > best["t3"][0]: best["t3"] = (v_t3, a)
        if v_t2 > best["t2"][0]: best["t2"] = (v_t2, a)
        print(f"{a:>6.3f} {v_r:>10.5f} {v_t3:>10.5f} {v_t2:>10.5f}")

    d_recipe = best["recipe"][0] - bal_r
    d_t3 = best["t3"][0] - bal_t3
    d_t2 = best["t2"][0] - bal_t2
    print(f"\npeaks:")
    print(f"  vs recipe      : a={best['recipe'][1]:.3f}  bal={best['recipe'][0]:.5f}  delta={d_recipe:+.5f}")
    print(f"  vs LB-best 2-way: a={best['t2'][1]:.3f}  bal={best['t2'][0]:.5f}  delta={d_t2:+.5f}")
    print(f"  vs LB-best 3-way: a={best['t3'][1]:.3f}  bal={best['t3'][0]:.5f}  delta={d_t3:+.5f}")

    # ---------------------------------------------------- gate
    GATE_JACC = 0.80
    GATE_ERRS = len(errs_t3)
    GATE_DELTA = 0.0002
    print(f"\n=== GATE (vs LB-best 3-way) ===")
    print(f"  Jaccard < {GATE_JACC}: {j_f_t3 < GATE_JACC}  ({j_f_t3:.3f})")
    print(f"  focal errs <= 3-way ({GATE_ERRS}): {len(errs_f) <= GATE_ERRS}  ({len(errs_f)})")
    print(f"  peak delta vs 3-way >= {GATE_DELTA}: {d_t3 >= GATE_DELTA}  ({d_t3:+.5f})")
    passed = (j_f_t3 < GATE_JACC) and (len(errs_f) <= GATE_ERRS) and (d_t3 >= GATE_DELTA)
    print(f"  PASSED: {passed}")

    # Emit a blend submission at the best alpha vs 3-way, regardless of gate
    # (for manual inspection). Gate is advisory on whether to LB-probe.
    best_a = best["t3"][1]
    log_fusion_test = (best_a * np.log(np.clip(test_focal, eps, 1.0))
                       + (1 - best_a) * np.log(np.clip(test_t3, eps, 1.0)))
    test_pred_idx = (log_fusion_test + bias_t3).argmax(1)
    cls_idx = {v: k for k, v in CLS2IDX.items()}
    sub = pd.DataFrame({
        "id": te["id"].to_numpy(),
        "Irrigation_Need": [cls_idx[i] for i in test_pred_idx],
    })
    tag = f"a{int(best_a*100):03d}"
    sub_path = SUB / f"submission_focal_blend_{tag}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nwrote {sub_path}  (diagnostic — LB-probe ONLY if gate passed)")
    print(f"  test dist: {dict(sub['Irrigation_Need'].value_counts())}")

    summary = dict(
        focal_suffix=FOCAL_SUFFIX,
        standalone=dict(oof_argmax=float(fast_bal_acc(y, oof_focal.argmax(1))),
                        oof_tuned=float(bal_f),
                        bias=bias_f.tolist(),
                        errors=len(errs_f),
                        recall_L=float(rec_f[0]),
                        recall_M=float(rec_f[1]),
                        recall_H=float(rec_f[2])),
        anchors=dict(recipe=float(bal_r), lb2way=float(bal_t2), lb3way=float(bal_t3)),
        jaccard=dict(vs_recipe=float(j_f_r), vs_3way=float(j_f_t3)),
        blend_sweep_peaks=dict(
            vs_recipe=dict(alpha=best["recipe"][1], bal=best["recipe"][0], delta=d_recipe),
            vs_lb2way=dict(alpha=best["t2"][1], bal=best["t2"][0], delta=d_t2),
            vs_lb3way=dict(alpha=best["t3"][1], bal=best["t3"][0], delta=d_t3),
        ),
        gate_passed=bool(passed),
    )
    out = ART / f"blend_focal{FOCAL_SUFFIX}_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
