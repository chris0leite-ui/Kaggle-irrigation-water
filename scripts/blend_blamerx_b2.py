"""Blend-gate + diagnostic analysis for blamerx τ=0.92 and B2 GroupKFold.

Runs after both production jobs complete. Reports for each:
- Standalone tuned OOF vs baselines
- Error count and Jaccard vs recipe + vs LB-best 2-way + vs LB-best 3-way
- Fixed-bias log-blend α sweep vs each anchor
- Emit-gate decision: Δ ≥ +0.0002 means worth an LB probe

B2 GroupKFold is a DIAGNOSTIC (not a blend candidate) — interpret
its Δ vs StratifiedKFold baseline to judge OOF-honesty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import fast_bal_acc, tune_log_bias


ART = Path("scripts/artifacts")


def bal_at_bias(probs: np.ndarray, y: np.ndarray, bias: np.ndarray) -> float:
    return fast_bal_acc(y.astype(np.int32),
                        (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1))


def log_blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """(1-alpha) * log(a) + alpha * log(b), renormalised in prob space."""
    eps = 1e-9
    z = (1 - alpha) * np.log(np.clip(a, eps, 1.0)) + alpha * np.log(np.clip(b, eps, 1.0))
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def sweep(anchor_oof: np.ndarray, cand_oof: np.ndarray, y: np.ndarray,
          bias: np.ndarray, alphas=None) -> dict:
    if alphas is None:
        alphas = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25,
                  0.30, 0.40, 0.50]
    rows = []
    for a in alphas:
        p = log_blend(anchor_oof, cand_oof, a)
        rows.append(dict(alpha=a, bal=bal_at_bias(p, y, bias)))
    return rows


def jaccard(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    ea = a != y
    eb = b != y
    inter = (ea & eb).sum()
    union = (ea | eb).sum()
    return float(inter) / float(max(union, 1))


def main():
    y = pd.read_csv("data/train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}).to_numpy(np.int32)

    # Anchors on disk
    anchors = {}
    for name, path in [
        ("recipe_full_te", ART / "oof_recipe_full_te.npy"),
        ("LB_best_2way_pseudo_s1", ART / "oof_recipe_pseudolabel.npy"),
    ]:
        if path.exists():
            anchors[name] = np.load(path)

    # Labeler biases
    with open(ART / "recipe_full_te_results.json") as f:
        recipe_bias = np.array(json.loads(f.read())["log_bias"])

    # ----- blamerx τ=0.92 -----
    p_path = ART / "oof_recipe_pseudolabel_tau092.npy"
    if p_path.exists():
        tau092 = np.load(p_path)
        bias, tuned = tune_log_bias(
            tau092, y, np.bincount(y, minlength=3) / len(y))
        print(f"\n=== blamerx τ=0.92 ===")
        print(f"  standalone tuned  = {tuned:.5f}  "
              f"bias={bias.round(4).tolist()}")
        print(f"  Δ vs recipe       = {tuned - 0.97967:+.5f}")
        print(f"  Δ vs LB-best 2way = {tuned - 0.98012:+.5f}")

        stage1_pseudo = anchors.get("LB_best_2way_pseudo_s1")
        if stage1_pseudo is not None:
            errs_tau092 = (tau092.argmax(1) != y).sum()
            errs_s1 = (stage1_pseudo.argmax(1) != y).sum()
            errs_recipe = (anchors["recipe_full_te"].argmax(1) != y).sum()
            print(f"  errors at argmax:")
            print(f"    τ=0.92  : {errs_tau092:,}")
            print(f"    τ=0.98 s1: {errs_s1:,}")
            print(f"    recipe  : {errs_recipe:,}")
            print(f"  Jaccard vs recipe   = "
                  f"{jaccard(tau092.argmax(1), anchors['recipe_full_te'].argmax(1), y):.4f}")
            print(f"  Jaccard vs pseudo_s1 = "
                  f"{jaccard(tau092.argmax(1), stage1_pseudo.argmax(1), y):.4f}")

        # Blend vs recipe (fixed recipe bias)
        print(f"\n  Blend vs recipe (fixed recipe bias):")
        for row in sweep(anchors["recipe_full_te"], tau092, y, recipe_bias):
            mark = " ←" if row["bal"] > 0.97967 + 1e-5 else ""
            print(f"    α={row['alpha']:5.3f}  {row['bal']:.5f}"
                  f"  Δ={row['bal']-0.97967:+.5f}{mark}")

        # Blend vs LB-best 2-way (if present)
        if stage1_pseudo is not None:
            # LB-best 2-way anchor = 0.5 log(recipe) + 0.5 log(pseudo_s1)
            lb_best_2way = log_blend(anchors["recipe_full_te"], stage1_pseudo, 0.5)
            base_2way = bal_at_bias(lb_best_2way, y, recipe_bias)
            print(f"\n  Blend vs LB-best 2-way (base {base_2way:.5f}):")
            for row in sweep(lb_best_2way, tau092, y, recipe_bias):
                mark = " ←" if row["bal"] > base_2way + 1e-5 else ""
                print(f"    α={row['alpha']:5.3f}  {row['bal']:.5f}"
                      f"  Δ={row['bal']-base_2way:+.5f}{mark}")
    else:
        print(f"skipping blamerx τ=0.92 — {p_path} not found yet")

    # ----- B2 GroupKFold diagnostic -----
    b2_path = ART / "oof_b2_groupkfold_region.npy"
    if b2_path.exists():
        b2 = np.load(b2_path)
        bias_b2, tuned_b2 = tune_log_bias(
            b2, y, np.bincount(y, minlength=3) / len(y))
        print(f"\n=== B2 GroupKFold (by Region) ===")
        print(f"  tuned OOF (GroupKFold)    = {tuned_b2:.5f}  "
              f"bias={bias_b2.round(4).tolist()}")
        print(f"  tuned OOF (StratifiedKF)  = 0.97967 (recipe baseline)")
        print(f"  Δ                         = {tuned_b2 - 0.97967:+.5f}")
        print()
        if tuned_b2 < 0.97967 - 0.005:
            print("  DIAGNOSIS: MATERIAL DROP — region leakage exists in "
                  "StratifiedKFold. True frontier is lower than we think.")
        elif tuned_b2 < 0.97967 - 0.002:
            print("  DIAGNOSIS: Moderate drop — some region-specific signal "
                  "the StratifiedKFold split exploits. OOF slightly optimistic.")
        else:
            print("  DIAGNOSIS: OOF holds — StratifiedKFold is honest. "
                  "Ceiling is real structural saturation, not a CV artifact.")
    else:
        print(f"skipping B2 — {b2_path} not found yet")


if __name__ == "__main__":
    main()
