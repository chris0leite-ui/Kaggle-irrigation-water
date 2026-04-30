"""Idea 4c — 5-component bagged base + B's mechanism + 14-bank filter.

Extension of Idea 4b: add a 5th sibling to the fold-seed bag that
introduces MODEL CLASS diversity (ExtraTrees) on top of the existing
RF natural fold-seed diversity. ET shares v1's 7-component bank but
uses random-threshold splits instead of best-split (RF default).

5-component bag:
  - sklearn_rf_meta_natural (RF, fs42, v2-bank prob array)
  - rf_natural_v1_n1000_fs42 (RF, fs42, 7-comp bank, n=1000)
  - rf_natural_v1_n500_fs7   (RF, fs7,  7-comp bank)
  - rf_natural_v1_n500_fs123 (RF, fs123, 7-comp bank)
  - et_natural_v1_n500_fs42  (ET, fs42, 7-comp bank) ← NEW

Then run the same triple-consensus filter as Idea 4b:
  (a) 5bagged' argmax differs from 4b anchor (LB 0.98150)
  (b) {raw, tier1b} k=2 unanimously say 5bagged's class
  (c) 14-bank majority confirms

Hypothesis: ET's random-threshold splits will shift different boundary
rows than RF's best-splits. The 5-component bag covers a wider
probability surface, exposing more triple-consensus disagreement points
that pure RF bagging missed.

If 5bagged disagrees with 4b on rows where the strict triple-consensus
fires, we have new candidate flips not in 4b's 105 set.
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
    print("=== Idea 4c: 5-component bag (RF×4 + ET×1) + triple-consensus ===\n")

    # 5-component bag
    rf_names = [
        "sklearn_rf_meta_natural",
        "rf_natural_v1_n1000_fs42",
        "rf_natural_v1_n500_fs7",
        "rf_natural_v1_n500_fs123",
        "et_natural_v1_n500_fs42",   # NEW: ExtraTrees, model-class diversity
    ]
    test_probs = []
    for n in rf_names:
        p = ART / f"test_{n}.npy"
        if not p.exists():
            print(f"MISSING: {p}")
            return
        arr = normed(np.load(p).astype(np.float32))
        test_probs.append(arr)
        print(f"  loaded test_{n}: {arr.shape}")

    bagged = log_blend(test_probs, np.ones(len(test_probs)))
    V1_BIAS = np.array([0.4324, 0.8689, 3.2008])
    bagged_arg = (np.log(np.clip(bagged, 1e-9, 1.0)) + V1_BIAS).argmax(1)
    print(f"\n5-bag argmax class counts: {np.bincount(bagged_arg, minlength=3).tolist()}")

    # Anchor = 4b at LB 0.98150
    anchor = csv_to_argmax("submission_idea4b_selective_override")
    diff_a = bagged_arg != anchor
    print(f"5-bag vs 4b: {diff_a.sum()} rows differ")

    LMH = ["L", "M", "H"]
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            n = int(((anchor == fr) & (bagged_arg == to)).sum())
            if n > 0:
                print(f"  4b={LMH[fr]} -> 5bag={LMH[to]}: {n}")

    # Apply triple-consensus filter
    raw = csv_to_argmax("submission_rawashishsin_2600_standalone")
    tier1b = csv_to_argmax("submission_tier1b_greedy_meta")
    maj = np.load(ART / "stability_test_majority.npy")

    unan_rt = (raw == tier1b) & (raw == bagged_arg)
    bank_agree = maj == bagged_arg

    flip_mask = diff_a & unan_rt & bank_agree
    n_flips = int(flip_mask.sum())
    print(f"\nTriple-consensus flips (5-bag + {{raw,tier1b}} unan + 14-bank maj): {n_flips}")

    if n_flips == 0:
        print("\n*** SATURATED with strict filter ***")
        # Try variants
        v_a = (diff_a & unan_rt).sum()
        v_b = (diff_a & bank_agree).sum()
        print(f"  5-bag + {{raw,tier1b}} unan only: {v_a}")
        print(f"  5-bag + 14-bank maj only:        {v_b}")
        # Variant C: drop {raw, tier1b} unanimous, keep bagged + bank
        v_c_mask = diff_a & bank_agree
        if v_c_mask.sum() > 0:
            print(f"\n=== Variant C (drop OTHERS, keep bagged + bank): {v_c_mask.sum()} flips ===")
            new_pred_c = anchor.copy()
            new_pred_c[v_c_mask] = bagged_arg[v_c_mask]
            for fr in range(3):
                for to in range(3):
                    if fr == to: continue
                    n = int(((anchor == fr) & (new_pred_c == to)).sum())
                    if n > 0:
                        print(f"    4b={LMH[fr]} -> {LMH[to]}: {n}")
            test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()
            sub_c = pd.DataFrame({
                "id": test_ids,
                "Irrigation_Need": pd.Series(new_pred_c).map({0: "Low", 1: "Medium", 2: "High"}),
            })
            out_c = SUB / "submission_idea4c_5bag_variantC.csv"
            sub_c.to_csv(out_c, index=False)
            print(f"  emitted: {out_c}")
        return

    # Strict triple-consensus produced flips — emit
    new_pred = anchor.copy()
    new_pred[flip_mask] = bagged_arg[flip_mask]

    print(f"\nclass counts: {np.bincount(new_pred, minlength=3).tolist()}")
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            n = int(((anchor == fr) & (new_pred == to)).sum())
            if n > 0:
                print(f"  4b={LMH[fr]} -> 4c={LMH[to]}: {n}")

    h_added = int(((anchor != 2) & (new_pred == 2)).sum())
    h_removed = int(((anchor == 2) & (new_pred != 2)).sum())
    print(f"net_H = +{h_added} -{h_removed} = {h_added - h_removed:+d}")

    test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()
    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
    })
    out_csv = SUB / "submission_idea4c_5bag_strict.csv"
    sub.to_csv(out_csv, index=False)
    print(f"\nemitted: {out_csv}")

    out_json = ART / "idea4c_5bag_results.json"
    out_json.write_text(json.dumps({
        "n_flips": n_flips,
        "h_added": h_added,
        "h_removed": h_removed,
    }, indent=2))


if __name__ == "__main__":
    main()
