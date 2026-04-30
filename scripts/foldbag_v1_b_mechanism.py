"""Idea 4 — Fold-seed-bagged v1' + re-run B's mechanism.

Main has 4 RF natural variants on disk:
  - v1_orig (oof_sklearn_rf_meta_natural.npy)         # NOTE: this is v2 prob array, see below
  - n1000_fs42 (oof_rf_natural_v1_n1000_fs42.npy)
  - n500_fs7   (oof_rf_natural_v1_n500_fs7.npy)
  - n500_fs123 (oof_rf_natural_v1_n500_fs123.npy)

CAVEAT: oof_sklearn_rf_meta_natural.npy was overwritten by v2 (a1lgbm,
LB 0.98098). The v2 prob array applied with v1's bias [0.43, 0.87, 3.20]
reproduces v1 CSV at 99.95%. So v2-prob × v1-bias is a near-equivalent.

Mechanism: prob-level geomean of {v2_orig, n1k, fs7, fs123} → bagged_v1'_probs.
Apply v1's bias [0.43, 0.87, 3.20] → bagged_v1'_argmax. Then run B's
mechanism: where {raw_argmax, tier1b_argmax} unanimously differ from
bagged_v1'_argmax, override to consensus → produce B'.

Compare B' to B (LB 0.98140) on test predictions. If B' differs in
direction-favorable rows (ADD-H, no REMOVE-H regressions), submit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import IDX2CLS  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")

# v1 RF natural variants — fold-seed bagging
RF_BAG_NAMES = [
    "sklearn_rf_meta_natural",       # original (v2 prob array, near-v1)
    "rf_natural_v1_n1000_fs42",
    "rf_natural_v1_n500_fs7",
    "rf_natural_v1_n500_fs123",
]

V1_BIAS = np.array([0.4324, 0.8689, 3.2008])


def normed(a: np.ndarray) -> np.ndarray:
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def log_blend(probs_list: list[np.ndarray], weights: np.ndarray,
              eps: float = 1e-9) -> np.ndarray:
    w = weights / weights.sum()
    logits = np.zeros_like(probs_list[0])
    for wi, p in zip(w, probs_list):
        logits += wi * np.log(np.clip(p, eps, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def csv_to_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def main():
    print(f"=== Idea 4: fold-seed-bagged v1' + B's mechanism ===\n")

    # Load all 4 RF variants test probs
    test_probs = []
    for name in RF_BAG_NAMES:
        p = ART / f"test_{name}.npy"
        if not p.exists():
            print(f"  MISSING: {p}")
            return
        arr = normed(np.load(p).astype(np.float32))
        test_probs.append(arr)
        print(f"  loaded test_{name}.npy: {arr.shape}")

    # Geomean (equal weights since each is structurally similar v1 sibling)
    bagged_test = log_blend(test_probs, np.ones(len(test_probs)))
    print(f"\nbagged v1' test probs: {bagged_test.shape}")

    # Apply v1's bias to get bagged argmax
    bagged_argmax = (np.log(np.clip(bagged_test, 1e-9, 1.0)) + V1_BIAS).argmax(1)
    print(f"bagged v1' argmax class counts: {np.bincount(bagged_argmax, minlength=3).tolist()}")

    # Compare to single-seed v1
    v1_argmax = csv_to_argmax("submission_sklearn_rf_meta_natural_standalone_v1_lb98129")
    diff_v1 = int((bagged_argmax != v1_argmax).sum())
    print(f"bagged v1' vs single-seed v1: {diff_v1} rows differ")

    # Now run B's mechanism: where {raw_argmax, tier1b_argmax} unanimously
    # differ from bagged_v1'_argmax, override to consensus
    raw = csv_to_argmax("submission_rawashishsin_2600_standalone")
    tier1b = csv_to_argmax("submission_tier1b_greedy_meta")

    # Override mask
    raw_eq_t1b = (raw == tier1b)
    diff_anchor = (raw != bagged_argmax)
    override_mask = raw_eq_t1b & diff_anchor

    bp_argmax = bagged_argmax.copy()
    bp_argmax[override_mask] = raw[override_mask]
    n_override = int(override_mask.sum())
    print(f"\nB's mechanism applied: {n_override} overrides")
    print(f"B' class counts: {np.bincount(bp_argmax, minlength=3).tolist()}")

    # Compare to current LB-best B
    b_argmax = csv_to_argmax("submission_2other_raw_tier1b_k2")
    diff_b = int((bp_argmax != b_argmax).sum())
    print(f"\nB' vs B (LB 0.98140): {diff_b} rows differ")

    # Direction breakdown
    directions = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((b_argmax == fr) & (bp_argmax == to)).sum())
            if n > 0:
                directions[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n
    print(f"directions B->B': {directions}")

    # Net High change
    h_added = int(((b_argmax != 2) & (bp_argmax == 2)).sum())
    h_removed = int(((b_argmax == 2) & (bp_argmax != 2)).sum())
    net_h = h_added - h_removed
    print(f"net_H = +{h_added} -{h_removed} = {net_h:+d}")

    # Emit candidate submission
    test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()
    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": [IDX2CLS[i] for i in bp_argmax],
    })
    out_csv = SUB / "submission_idea4_foldbag_v1_b_mech.csv"
    sub.to_csv(out_csv, index=False)
    print(f"\nemitted: {out_csv}")

    # Save diagnostics
    out_json = ART / "idea4_foldbag_v1_b_mech_results.json"
    out_json.write_text(json.dumps({
        "rf_bag_components": RF_BAG_NAMES,
        "diff_bagged_vs_single_seed": diff_v1,
        "n_override": n_override,
        "diff_b_prime_vs_b": diff_b,
        "directions_b_to_b_prime": directions,
        "net_h": net_h,
        "h_added": h_added,
        "h_removed": h_removed,
        "candidate_csv": str(out_csv),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
