"""Analyze the 2-seed bag of the greedy winner.

Runs AFTER xgb_dist_routed_v3_seed7.py AND xgb_specialist_678_seed7.py
have produced their OOFs/test arrays. This script does no training —
it just averages saved OOFs and rebuilds the hybrid + greedy stacks.

Inputs (all in scripts/artifacts/):
  oof_xgb_dist_routed_v3.npy         (seed=42, existing)
  oof_xgb_dist_routed_v3_seed7.npy   (new, from Task 1)
  oof_xgb_spec_678.npy               (seed=42, existing)
  oof_xgb_spec_678_seed7.npy         (new, from Task 1)
  test_xgb_dist_routed_v3.npy / test_xgb_dist_routed_v3_seed7.npy
  test_xgb_spec_678.npy              / test_xgb_spec_678_seed7.npy

Pipeline:
  1. Average routed_v3 (seed=42) + routed_v3 (seed=7)  -> routed_bag.
  2. Average spec_678  (seed=42) + spec_678  (seed=7)  -> spec_bag.
  3. Rebuild hybrid_bag: routed_bag, then override rows where
     dgp_score in {6,7,8} with spec_bag (exact recipe from
     hybrid_routed_spec.py).
  4. Rebuild greedy_bag: 0.45*hybrid_bag + 0.40*routed_bag
     + 0.15*spec_bag in LOG space (same formula as the LB-0.97296
     greedy_w045_040_015 submission).
  5. Tune log-bias on greedy_bag via coord-ascent.
  6. Report standalone bagged OOFs AND per-component delta vs seed=42.

Outputs:
  scripts/artifacts/oof_greedy_blend_bag.npy
  scripts/artifacts/test_greedy_blend_bag.npy
  scripts/artifacts/oof_xgb_dist_routed_v3_bag.npy
  scripts/artifacts/oof_xgb_spec_678_bag.npy
  scripts/artifacts/test_xgb_dist_routed_v3_bag.npy
  scripts/artifacts/test_xgb_spec_678_bag.npy
  scripts/artifacts/seed_bag_greedy_results.json
  submissions/submission_blend_greedy_bag.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")
SPEC_SCORES = (6, 7, 8)
ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(ACTIVE_STAGES).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).values


def main() -> None:
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    tr_scores = compute_dgp_score(tr)
    te_scores = compute_dgp_score(te)
    tr_spec_mask = np.isin(tr_scores, SPEC_SCORES)
    te_spec_mask = np.isin(te_scores, SPEC_SCORES)

    # --- load seed=42 baselines ---
    oof_routed_42 = np.load(ART / "oof_xgb_dist_routed_v3.npy")
    test_routed_42 = np.load(ART / "test_xgb_dist_routed_v3.npy")
    oof_spec_42 = np.load(ART / "oof_xgb_spec_678.npy")
    test_spec_42 = np.load(ART / "test_xgb_spec_678.npy")

    # --- load seed=7 siblings ---
    oof_routed_7 = np.load(ART / "oof_xgb_dist_routed_v3_seed7.npy")
    test_routed_7 = np.load(ART / "test_xgb_dist_routed_v3_seed7.npy")
    oof_spec_7 = np.load(ART / "oof_xgb_spec_678_seed7.npy")
    test_spec_7 = np.load(ART / "test_xgb_spec_678_seed7.npy")

    # --- 2-seed bags (simple arithmetic mean in prob space) ---
    oof_routed_bag = 0.5 * (oof_routed_42 + oof_routed_7)
    test_routed_bag = 0.5 * (test_routed_42 + test_routed_7)
    oof_spec_bag = 0.5 * (oof_spec_42 + oof_spec_7)
    test_spec_bag = 0.5 * (test_spec_42 + test_spec_7)

    np.save(ART / "oof_xgb_dist_routed_v3_bag.npy", oof_routed_bag)
    np.save(ART / "test_xgb_dist_routed_v3_bag.npy", test_routed_bag)
    np.save(ART / "oof_xgb_spec_678_bag.npy", oof_spec_bag)
    np.save(ART / "test_xgb_spec_678_bag.npy", test_spec_bag)

    # --- standalone per-component diagnostics ---
    print("\n=== Per-component OOF tuned bal_acc (seed=42 / seed=7 / bag) ===")
    results = {}
    for name, oof42, oof7, oofbag in [
        ("routed_v3", oof_routed_42, oof_routed_7, oof_routed_bag),
        ("spec_678",  oof_spec_42,   oof_spec_7,   oof_spec_bag),
    ]:
        # For spec_678, restrict eval to the spec domain (matches
        # xgb_specialist_678.py's reporting convention); for routed_v3
        # eval on full 630k.
        if name == "spec_678":
            y_eval = y[tr_spec_mask]
            a42 = balanced_accuracy_score(y_eval, oof42[tr_spec_mask].argmax(1))
            a7  = balanced_accuracy_score(y_eval, oof7[tr_spec_mask].argmax(1))
            ab  = balanced_accuracy_score(y_eval, oofbag[tr_spec_mask].argmax(1))
            raw42 = (oof42[tr_spec_mask].argmax(1) == y_eval).mean()
            raw7  = (oof7[tr_spec_mask].argmax(1)  == y_eval).mean()
            rawb  = (oofbag[tr_spec_mask].argmax(1) == y_eval).mean()
            print(f"  {name}   argmax_bal  seed42={a42:.5f}  seed7={a7:.5f}  "
                  f"bag={ab:.5f}  Δ_bag-vs-seed42={ab-a42:+.5f}")
            results[name] = {
                "seed42_argmax_bal": float(a42),
                "seed7_argmax_bal": float(a7),
                "bag_argmax_bal": float(ab),
                "delta_bag_vs_seed42": float(ab - a42),
                "seed42_raw_acc": float(raw42),
                "seed7_raw_acc": float(raw7),
                "bag_raw_acc": float(rawb),
            }
        else:
            _, t42 = tune_log_bias(oof42, y, prior)
            _, t7  = tune_log_bias(oof7,  y, prior)
            _, tb  = tune_log_bias(oofbag, y, prior)
            print(f"  {name}   tuned_bal   seed42={t42:.5f}  seed7={t7:.5f}  "
                  f"bag={tb:.5f}  Δ_bag-vs-seed42={tb-t42:+.5f}")
            results[name] = {
                "seed42_tuned": float(t42),
                "seed7_tuned": float(t7),
                "bag_tuned": float(tb),
                "delta_bag_vs_seed42": float(tb - t42),
            }

    # --- rebuild hybrid_bag: routed_bag overridden by spec_bag on {6,7,8} ---
    oof_hybrid_bag = oof_routed_bag.copy()
    oof_hybrid_bag[tr_spec_mask] = oof_spec_bag[tr_spec_mask]
    test_hybrid_bag = test_routed_bag.copy()
    test_hybrid_bag[te_spec_mask] = test_spec_bag[te_spec_mask]

    _, hyb_bag_bal = tune_log_bias(oof_hybrid_bag, y, prior)
    print(f"\nhybrid_bag (routed_bag + spec_bag on {{6,7,8}}) tuned: {hyb_bag_bal:.5f}")

    # also compute the analogous non-bagged hybrid for reference
    oof_hybrid_42 = oof_routed_42.copy()
    oof_hybrid_42[tr_spec_mask] = oof_spec_42[tr_spec_mask]
    _, hyb_42_bal = tune_log_bias(oof_hybrid_42, y, prior)
    print(f"hybrid seed=42 (reference)                             : {hyb_42_bal:.5f}")
    print(f"Δ hybrid_bag - hybrid_seed42                            : {hyb_bag_bal - hyb_42_bal:+.5f}")

    # --- rebuild greedy_bag: 0.45*hybrid_bag + 0.40*routed_bag + 0.15*spec_bag in log space ---
    log_hyb = np.log(np.clip(oof_hybrid_bag, 1e-9, 1.0))
    log_rou = np.log(np.clip(oof_routed_bag, 1e-9, 1.0))
    log_spc = np.log(np.clip(oof_spec_bag,   1e-9, 1.0))
    log_greedy = 0.45 * log_hyb + 0.40 * log_rou + 0.15 * log_spc
    oof_greedy_bag = np.exp(log_greedy - log_greedy.max(axis=1, keepdims=True))
    oof_greedy_bag /= oof_greedy_bag.sum(axis=1, keepdims=True)

    log_hyb_t = np.log(np.clip(test_hybrid_bag, 1e-9, 1.0))
    log_rou_t = np.log(np.clip(test_routed_bag, 1e-9, 1.0))
    log_spc_t = np.log(np.clip(test_spec_bag,   1e-9, 1.0))
    log_greedy_t = 0.45 * log_hyb_t + 0.40 * log_rou_t + 0.15 * log_spc_t
    test_greedy_bag = np.exp(log_greedy_t - log_greedy_t.max(axis=1, keepdims=True))
    test_greedy_bag /= test_greedy_bag.sum(axis=1, keepdims=True)

    bias, greedy_bag_bal = tune_log_bias(oof_greedy_bag, y, prior)
    print(f"\n=== greedy_bag (log-blend 0.45/0.40/0.15) ===")
    print(f"  tuned bal_acc      : {greedy_bag_bal:.5f}")
    print(f"  tuned bias         : {dict(zip(CLASSES, bias.round(4)))}")
    print(f"  greedy seed=42 (ref): 0.97375  LB 0.97296")
    print(f"  Δ vs seed=42 greedy: {greedy_bag_bal - 0.97375:+.5f}")

    cm = confusion_matrix(
        y, (np.log(np.clip(oof_greedy_bag, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    print(f"  OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # --- save artefacts + submission ---
    np.save(ART / "oof_greedy_blend_bag.npy", oof_greedy_bag)
    np.save(ART / "test_greedy_blend_bag.npy", test_greedy_bag)

    tuned_test_idx = (np.log(np.clip(test_greedy_bag, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        SUB / "submission_blend_greedy_bag.csv", index=False
    )

    out = {
        "components": results,
        "hybrid_seed42_tuned": float(hyb_42_bal),
        "hybrid_bag_tuned": float(hyb_bag_bal),
        "delta_hybrid_bag_vs_seed42": float(hyb_bag_bal - hyb_42_bal),
        "greedy_bag_tuned": float(greedy_bag_bal),
        "greedy_seed42_reference": 0.97375,
        "greedy_seed42_lb": 0.97296,
        "delta_greedy_bag_vs_seed42": float(greedy_bag_bal - 0.97375),
        "tuned_bias": bias.tolist(),
        "blend_weights_log_space": {"hybrid": 0.45, "routed": 0.40, "spec": 0.15},
    }
    with open(ART / "seed_bag_greedy_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nwrote {ART / 'seed_bag_greedy_results.json'}")
    print(f"wrote {SUB / 'submission_blend_greedy_bag.csv'}")


if __name__ == "__main__":
    main()
