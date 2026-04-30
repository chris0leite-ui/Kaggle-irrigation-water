"""Idea 4b iteration 2 — apply triple-consensus override on top of LB 0.98150.

After Idea 4b lifted LB from 0.98140 → 0.98150 (+0.00010), iterate the
same mechanism with the new anchor. The original 4b had 105 H→M flips;
the new anchor (4b at 0.98150) has those flips applied. New iteration
finds rows where:
  (a) bagged_v1' argmax differs from 4b's argmax
  (b) {raw, tier1b} k=2 unanimously say bagged_v1's class
  (c) 14-component bank majority confirms

Most rows that 4b flipped are now resolved (new anchor matches consensus).
This iteration searches for any RESIDUAL rows where bagged_v1' + consensus
disagree with 4b but didn't trigger in the previous round.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def csv_to_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


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


def main():
    print("=== Idea 4b iter2: triple-consensus override on top of LB 0.98150 ===\n")

    # New anchor: 4b at LB 0.98150
    anchor = csv_to_argmax("submission_idea4b_selective_override")
    print(f"Anchor (4b LB 0.98150) class counts: {np.bincount(anchor, minlength=3).tolist()}")

    # Rebuild bagged_v1'
    rf_names = [
        "sklearn_rf_meta_natural",
        "rf_natural_v1_n1000_fs42",
        "rf_natural_v1_n500_fs7",
        "rf_natural_v1_n500_fs123",
    ]
    test_probs = [normed(np.load(ART / f"test_{n}.npy").astype(np.float32))
                  for n in rf_names]
    bagged_test = log_blend(test_probs, np.ones(len(test_probs)))
    V1_BIAS = np.array([0.4324, 0.8689, 3.2008])
    bagged_argmax = (np.log(np.clip(bagged_test, 1e-9, 1.0)) + V1_BIAS).argmax(1)

    # Load consensus axes
    raw = csv_to_argmax("submission_rawashishsin_2600_standalone")
    tier1b = csv_to_argmax("submission_tier1b_greedy_meta")
    maj = np.load(ART / "stability_test_majority.npy")

    # Triple-consensus filter on NEW anchor (4b)
    # (a) bagged_v1' differs from anchor
    diff_a = bagged_argmax != anchor
    # (b) {raw, tier1b} k=2 unanimous, agree on bagged_v1's class
    unan_rt = (raw == tier1b) & (raw == bagged_argmax)
    # (c) 14-bank majority agrees with bagged_v1's class
    bank_agree = maj == bagged_argmax

    flip_mask = diff_a & unan_rt & bank_agree
    n_flips = int(flip_mask.sum())
    print(f"Triple-consensus flip candidates: {n_flips}")

    # Apply
    new_pred = anchor.copy()
    new_pred[flip_mask] = bagged_argmax[flip_mask]

    # Direction breakdown
    directions = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((anchor == fr) & (new_pred == to)).sum())
            if n > 0:
                directions[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n
    print(f"directions vs 4b anchor: {directions}")

    h_added = int(((anchor != 2) & (new_pred == 2)).sum())
    h_removed = int(((anchor == 2) & (new_pred != 2)).sum())
    net_h = h_added - h_removed
    print(f"net_H = +{h_added} -{h_removed} = {net_h:+d}")

    print(f"\nnew_pred class counts: {np.bincount(new_pred, minlength=3).tolist()}")

    # Compare also to B (LB 0.98140) for reference
    b = csv_to_argmax("submission_2other_raw_tier1b_k2")
    diff_vs_b = int((new_pred != b).sum())
    diff_vs_4b = int((new_pred != anchor).sum())
    print(f"\ndiff vs 4b (LB 0.98150): {diff_vs_4b}")
    print(f"diff vs B  (LB 0.98140): {diff_vs_b}")

    if n_flips == 0:
        print("\n*** NO NEW FLIPS — mechanism saturated on 4b anchor ***")
    else:
        # Emit candidate
        test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()
        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
        })
        out_csv = SUB / "submission_idea4b_iter2.csv"
        sub.to_csv(out_csv, index=False)
        print(f"\nemitted: {out_csv}")

    # Stability check on flipped rows (if any)
    if n_flips > 0:
        agr = np.load(ART / "stability_test_agreement.npy")
        flip_agr = agr[flip_mask]
        print(f"\nstability agreement on flip rows:")
        print(f"  p25={np.percentile(flip_agr, 25):.3f}")
        print(f"  p50={np.percentile(flip_agr, 50):.3f}")
        print(f"  p75={np.percentile(flip_agr, 75):.3f}")

    out_json = ART / "idea4b_iter2_results.json"
    out_json.write_text(json.dumps({
        "n_flips": n_flips,
        "directions": directions,
        "net_h": net_h,
        "diff_vs_4b": diff_vs_4b,
        "diff_vs_b": diff_vs_b,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
