"""Override-eval helpers: compute OOF macro-recall delta, per-class direction
breakdown, and per-direction OOF precision; emit CSV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import CLASSES, IDX2CLS, SUB, load_test_ids


def macro_recall(y: np.ndarray, pred: np.ndarray) -> float:
    cc = np.bincount(y, minlength=3)
    return float(np.mean([
        ((pred == y) & (y == k)).sum() / max(cc[k], 1) for k in range(3)
    ]))


def direction_breakdown(anchor_argmax: np.ndarray, out_argmax: np.ndarray) -> dict:
    """Per-direction count + net-per-class shift."""
    diffs = {}
    net = {0: 0, 1: 0, 2: 0}
    mask = anchor_argmax != out_argmax
    for i in np.where(mask)[0]:
        a = int(anchor_argmax[i])
        b = int(out_argmax[i])
        key = f"{IDX2CLS[a]}->{IDX2CLS[b]}"
        diffs[key] = diffs.get(key, 0) + 1
        net[b] += 1
        net[a] -= 1
    return {"directions": diffs, "net_per_class": {IDX2CLS[k]: v for k, v in net.items()}}


def per_direction_oof_precision(anchor_oof_argmax: np.ndarray,
                                 out_oof_argmax: np.ndarray,
                                 y: np.ndarray) -> dict:
    """For each (a→b) direction, count overrides and how often the override
    landed on the true class. Macro-recall break-even precision per
    direction = N_b / (N_a + N_b) under macro-recall."""
    cc = np.bincount(y, minlength=3)
    res = {}
    mask = anchor_oof_argmax != out_oof_argmax
    for i in np.where(mask)[0]:
        a, b = int(anchor_oof_argmax[i]), int(out_oof_argmax[i])
        key = f"{IDX2CLS[a]}->{IDX2CLS[b]}"
        d = res.setdefault(key, {"n": 0, "correct_to_b": 0, "wrong_back_to_a": 0,
                                   "true_class_breakdown": {0: 0, 1: 0, 2: 0}})
        d["n"] += 1
        d["true_class_breakdown"][int(y[i])] += 1
        if y[i] == b:
            d["correct_to_b"] += 1
        elif y[i] == a:
            d["wrong_back_to_a"] += 1
    # Compute precision + break-even
    for key, d in res.items():
        a, b = key.split("->")
        a_idx, b_idx = CLASSES.index(a), CLASSES.index(b)
        d["precision"] = d["correct_to_b"] / max(d["n"], 1)
        d["break_even"] = cc[b_idx] / max(cc[a_idx] + cc[b_idx], 1)
        d["expected_macro_delta"] = (
            d["correct_to_b"] / max(cc[b_idx], 1)
            - d["wrong_back_to_a"] / max(cc[a_idx], 1)
        ) / 3.0
    return res


def evaluate(out_argmax: np.ndarray,
             out_oof_argmax: np.ndarray,
             anchor_argmax: np.ndarray,
             anchor_oof_argmax: np.ndarray,
             current_lb_best_argmax: np.ndarray,
             y: np.ndarray) -> dict:
    """Compute the full diagnostic for an override candidate.

    out_*_argmax: candidate predictions (test + OOF analog)
    anchor_*_argmax: the anchor we override on top of (test + OOF analog)
    current_lb_best_argmax: 0.98140 winner test argmax (for row-diff reporting)
    """
    base_macro = macro_recall(y, anchor_oof_argmax)
    cand_macro = macro_recall(y, out_oof_argmax)
    return {
        "anchor_oof_macro": base_macro,
        "candidate_oof_macro": cand_macro,
        "oof_delta_vs_anchor": cand_macro - base_macro,
        "row_diff_vs_anchor_test": int((out_argmax != anchor_argmax).sum()),
        "row_diff_vs_lb_best_0p98140": int((out_argmax != current_lb_best_argmax).sum()),
        "test_overrides_count": int((out_argmax != anchor_argmax).sum()),
        "test_direction_breakdown": direction_breakdown(anchor_argmax, out_argmax),
        "oof_direction_precision": per_direction_oof_precision(anchor_oof_argmax, out_oof_argmax, y),
    }


def emit_csv(out_argmax: np.ndarray, name: str, ids: np.ndarray | None = None) -> Path:
    """Write submission CSV from argmax labels."""
    if ids is None:
        ids = load_test_ids()
    path = SUB / name
    pd.DataFrame({"id": ids,
                  "Irrigation_Need": [IDX2CLS[i] for i in out_argmax]}).to_csv(path, index=False)
    return path


def fmt_summary(label: str, diag: dict) -> str:
    """One-line + breakdown summary."""
    d = diag
    lines = [
        f"=== {label} ===",
        f"  OOF Δ vs anchor:        {d['oof_delta_vs_anchor']:+.6f}  ({d['anchor_oof_macro']:.5f} -> {d['candidate_oof_macro']:.5f})",
        f"  test overrides:         {d['test_overrides_count']}",
        f"  test row-diff vs 98140: {d['row_diff_vs_lb_best_0p98140']}",
        f"  test net-per-class:     {d['test_direction_breakdown']['net_per_class']}",
        f"  test directions:        {d['test_direction_breakdown']['directions']}",
        "  OOF direction precision (per a->b):",
    ]
    for k, dp in sorted(d["oof_direction_precision"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(
            f"    {k:10s}  n={dp['n']:5d}  prec={dp['precision']:.3f}  "
            f"break-even={dp['break_even']:.3f}  Δmacro={dp['expected_macro_delta']:+.6f}"
        )
    return "\n".join(lines)
