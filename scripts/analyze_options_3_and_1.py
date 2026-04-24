"""Cross-experiment analysis for Option 3 (meta-stack) and Option 1 (router).

Reports standalone OOF, Jaccard vs teacher, error magnitude, blend-gate
results, and test-set disagreement counts. No LB spend; every metric is
OOF-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import fast_bal_acc, tune_log_bias
from meta_common import (
    ART, CANDIDATES, EPS, build_teacher, load_y_and_features, log_blend, recipe_bias,
)

TEACHER_OOF_BA = 0.98029   # LB-best 3-way at recipe bias
TEACHER_LB = 0.98005        # verified LB
LB_TRANSFER_THRESHOLD = 0.00020


def row_jaccard(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> float:
    a = pred_a != y; b = pred_b != y
    inter = int((a & b).sum()); union = int((a | b).sum())
    return inter / max(union, 1)


def forecast_lb(oof_ba: float, ref_oof: float = TEACHER_OOF_BA,
                ref_lb: float = TEACHER_LB) -> float:
    """Conservative: ref gap plus half the OOF delta (half-credit for inflation)."""
    delta_oof = oof_ba - ref_oof
    gap = ref_oof - ref_lb
    credit = max(delta_oof, 0) * 0.5
    return ref_lb + credit - max(-delta_oof, 0)


def main() -> None:
    y, tr_score, _, _, _ = load_y_and_features()
    oof_t, test_t = build_teacher()
    bias = recipe_bias()
    t_log = np.log(np.clip(oof_t, EPS, 1.0))
    t_pred = (t_log + bias).argmax(1)
    t_ba = fast_bal_acc(y, t_pred)
    t_errs = int((t_pred != y).sum())
    print(f"Teacher (LB-best 3-way) OOF @ recipe bias = {t_ba:.5f}   errs={t_errs}")
    print(f"Teacher LB = {TEACHER_LB}   gap = +{t_ba - TEACHER_LB:.5f}")
    print()

    prior = np.bincount(y, minlength=3) / len(y)

    # --- Option 3 meta
    meta_oof_p = ART / "oof_disagree_meta.npy"
    meta_res_p = ART / "disagree_meta_results.json"
    if meta_oof_p.exists() and meta_res_p.exists():
        meta_oof = np.load(meta_oof_p)
        meta_res = json.loads(meta_res_p.read_text())
        print("=" * 60)
        print("Option 3: disagreement meta-stack")
        print("=" * 60)
        print(f"n_features = {meta_res['n_features']}")
        print(f"meta tuned OOF = {meta_res['meta_tuned_ba']:.5f}")
        print(f"meta @ recipe bias = {meta_res['meta_fixed_bias_ba']:.5f}  "
              f"(Δ vs teacher {meta_res['meta_fixed_bias_ba'] - t_ba:+.5f})")
        # Fresh argmax Jaccard at each's own best operating point
        tuned_bias, tuned_ba = tune_log_bias(meta_oof, y, prior, high_grid_wide=True)
        meta_pred = (np.log(np.clip(meta_oof, EPS, 1.0)) + tuned_bias).argmax(1)
        meta_errs = int((meta_pred != y).sum())
        j_m_t = row_jaccard(meta_pred, t_pred, y)
        print(f"meta errors = {meta_errs}  (teacher {t_errs})  Jaccard vs teacher = {j_m_t:.4f}")
        peak_a = meta_res["peak_alpha"]; peak_v = meta_res["peak_blend_ba"]
        print(f"blend sweep peak α={peak_a} → {peak_v:.5f}  "
              f"Δ vs teacher = {peak_v - t_ba:+.5f}")
        passes_gate_blend = (peak_v - t_ba) >= LB_TRANSFER_THRESHOLD
        passes_gate_standalone = (meta_res['meta_tuned_ba'] - t_ba) >= LB_TRANSFER_THRESHOLD
        forecast = forecast_lb(max(peak_v, meta_res['meta_tuned_ba']))
        print(f"passes LB-transfer threshold (+{LB_TRANSFER_THRESHOLD}): "
              f"blend={passes_gate_blend}  standalone={passes_gate_standalone}")
        print(f"LB forecast (half-credit) = {forecast:.5f}")
        print()
    else:
        print("Option 3 results NOT yet present.\n")

    # --- Option 1 router
    rt_oof_p = ART / "oof_selective_router.npy"
    rt_res_p = ART / "selective_router_results.json"
    if rt_oof_p.exists() and rt_res_p.exists():
        rt_res = json.loads(rt_res_p.read_text())
        print("=" * 60)
        print("Option 1: per-row selective router")
        print("=" * 60)
        print(f"router tuned standalone = {rt_res['router_standalone_tuned_ba']:.5f}")
        sweep = rt_res["tau_sweep"]
        print(f"{'τ':>6s} {'n_routed':>10s} {'frac':>7s} {'bal_acc':>9s} "
              f"{'Δ vs T':>10s} {'net_wins':>10s}")
        for tau_key, d in sweep.items():
            print(f"{tau_key:>10s}  {d['n_routed']:>7d}  {d['frac_routed']*100:>5.2f}%  "
                  f"{d['bal_acc']:.5f}  {d['bal_acc'] - t_ba:+.5f}  {d['net_wins']:>+6d}")
        peak_tau = rt_res["peak_tau"]; peak_val = rt_res["peak_bal_acc"]
        delta = peak_val - t_ba
        print(f"peak: {peak_tau} → {peak_val:.5f}  Δ vs teacher = {delta:+.5f}")
        print(f"high-class-only gate: {rt_res.get('high_only_gate', {})}")
        passes_gate = delta >= LB_TRANSFER_THRESHOLD
        forecast = forecast_lb(peak_val)
        print(f"passes LB-transfer threshold (+{LB_TRANSFER_THRESHOLD}): {passes_gate}")
        print(f"LB forecast (half-credit) = {forecast:.5f}")
        print()
    else:
        print("Option 1 results NOT yet present.\n")

    print("=" * 60)
    print("Verdict summary")
    print("=" * 60)
    print("gate: blend Δ ≥ +0.00020 (LB-transfer threshold)")
    print("action: only submit if OOF delta clears gate AND")
    print("        forecast LB > current best 0.98005")


if __name__ == "__main__":
    main()
