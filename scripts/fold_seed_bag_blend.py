"""Aggregate per-fold-seed OOF arrays into a bagged greedy+nonrule blend.

Reads `oof_xgb_dist_routed_v3_fs{seed}.npy`, `oof_xgb_spec_678_fs{seed}.npy`,
`oof_xgb_nonrule_fs{seed}.npy` (and test counterparts) for every fold seed
passed via env var `FOLD_SEEDS` (comma-separated) or default
[42, 7, 123, 2024, 9999].

Pipeline (matches the single-seed greedy+nonrule LB-best recipe):
  1. Per seed: hybrid_v3 = routed_v3 with spec_678 override on
     rows where dgp_score in {6, 7, 8}.
  2. Per seed: greedy = exp( 0.45*log(hybrid) + 0.40*log(routed) + 0.15*log(spec) )
     (spec is outside its domain = zero prob; for rows outside {6,7,8}
     we fall back to hybrid-only -> equals routed_v3 by construction).
  3. Per seed: greedy_plus_nonrule = exp( 0.85*log(greedy) + 0.15*log(nonrule) )
     (matches single-seed LB-best alpha=0.15 on nonrule, fixed).
  4. Bag: mean across seeds in probability space.
  5. Tune log-bias on bagged OOF (coord ascent). Apply same bias to
     bagged test probs, argmax, write submission.

Compare:
  - single-seed greedy+nonrule: OOF 0.97421 / LB 0.97352 (LB-best)
  - bagged greedy+nonrule:      OOF ??????? / LB ???????

Artefacts:
  scripts/artifacts/oof_bagged_greedy_nonrule.npy
  scripts/artifacts/test_bagged_greedy_nonrule.npy
  scripts/artifacts/fold_seed_bag_blend_results.json
  submissions/submission_bagged_greedy_nonrule.csv
"""
from __future__ import annotations

import json
import os
import time
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

# Greedy log-blend weights from 2026-04-21 greedy_binhigh_minimal.py
W_HYB = 0.45
W_ROUTED = 0.40
W_SPEC = 0.15
# Fixed alpha for nonrule into greedy (single-seed LB-best)
ALPHA_NONRULE = 0.15

ART = Path("scripts/artifacts")
OUT = Path("submissions")
OUT.mkdir(exist_ok=True)

FOLD_SEEDS_ENV = os.environ.get("FOLD_SEEDS", "42,7,123,2024,9999")
FOLD_SEEDS = [int(s.strip()) for s in FOLD_SEEDS_ENV.split(",") if s.strip()]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
    sm = df["Soil_Moisture"].astype(float).values
    rf = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = df["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)


def log_blend_weighted(parts: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Weighted geometric mean in log-space, normalized row-wise."""
    logs = None
    for p, w in parts:
        lp = np.log(np.clip(p, 1e-9, 1.0)) * w
        logs = lp if logs is None else logs + lp
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
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
    return bias, float(best)


def build_hybrid(routed: np.ndarray, spec: np.ndarray, spec_mask: np.ndarray) -> np.ndarray:
    out = routed.copy()
    out[spec_mask] = spec[spec_mask]
    return out


def build_greedy(hybrid: np.ndarray, routed: np.ndarray, spec: np.ndarray,
                 spec_mask: np.ndarray) -> np.ndarray:
    """Log-blend of the 3 greedy components. For rows outside the spec
    domain, spec is all-zero (not a valid prob), so we clip to a uniform
    prior on those rows so the log-blend reduces to a weighted combo of
    hybrid+routed with the spec weight evenly distributed (neutral)."""
    spec_safe = spec.copy()
    neutral = np.ones(3) / 3.0
    spec_safe[~spec_mask] = neutral
    return log_blend_weighted([
        (hybrid, W_HYB),
        (routed, W_ROUTED),
        (spec_safe, W_SPEC),
    ])


def load_seed_arrays(seed: int, n_train: int, n_test: int):
    paths = {
        "oof_routed":  ART / f"oof_xgb_dist_routed_v3_fs{seed}.npy",
        "test_routed": ART / f"test_xgb_dist_routed_v3_fs{seed}.npy",
        "oof_spec":    ART / f"oof_xgb_spec_678_fs{seed}.npy",
        "test_spec":   ART / f"test_xgb_spec_678_fs{seed}.npy",
        "oof_nonrule": ART / f"oof_xgb_nonrule_fs{seed}.npy",
        "test_nonrule":ART / f"test_xgb_nonrule_fs{seed}.npy",
    }
    for k, p in paths.items():
        if not p.exists():
            return None
    arrays = {k: np.load(p) for k, p in paths.items()}
    # sanity
    assert arrays["oof_routed"].shape == (n_train, 3)
    assert arrays["test_routed"].shape == (n_test, 3)
    return arrays


def main() -> None:
    log(f"fold seeds: {FOLD_SEEDS}")
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    tr_scores = compute_dgp_score(tr)
    te_scores = compute_dgp_score(te)
    tr_spec_mask = np.isin(tr_scores, SPEC_SCORES)
    te_spec_mask = np.isin(te_scores, SPEC_SCORES)

    n_train = len(tr)
    n_test = len(te)
    log(f"train rows: {n_train}, test rows: {n_test}")
    log(f"spec-domain rows (train/test): {tr_spec_mask.sum()}/{te_spec_mask.sum()}")

    per_seed_oof_greedy_plus_nonrule = []
    per_seed_test_greedy_plus_nonrule = []
    per_seed_oof_greedy = []
    per_seed_test_greedy = []
    per_seed_oof_nonrule = []
    per_seed_test_nonrule = []
    per_seed_diag = {}

    seeds_loaded = []
    for seed in FOLD_SEEDS:
        a = load_seed_arrays(seed, n_train, n_test)
        if a is None:
            log(f"  seed {seed}: artefacts incomplete - SKIPPING")
            continue
        seeds_loaded.append(seed)

        hybrid_oof = build_hybrid(a["oof_routed"], a["oof_spec"], tr_spec_mask)
        hybrid_test = build_hybrid(a["test_routed"], a["test_spec"], te_spec_mask)

        greedy_oof = build_greedy(hybrid_oof, a["oof_routed"], a["oof_spec"], tr_spec_mask)
        greedy_test = build_greedy(hybrid_test, a["test_routed"], a["test_spec"], te_spec_mask)

        gn_oof = log_blend_weighted([
            (greedy_oof, 1 - ALPHA_NONRULE),
            (a["oof_nonrule"], ALPHA_NONRULE),
        ])
        gn_test = log_blend_weighted([
            (greedy_test, 1 - ALPHA_NONRULE),
            (a["test_nonrule"], ALPHA_NONRULE),
        ])

        # per-seed stats: tune bias on this seed alone and record
        _, seed_tuned = tune_log_bias(gn_oof, y, prior)
        _, seed_tuned_greedy = tune_log_bias(greedy_oof, y, prior)
        _, seed_tuned_nonrule = tune_log_bias(a["oof_nonrule"], y, prior)
        per_seed_diag[seed] = {
            "greedy_tuned_oof": float(seed_tuned_greedy),
            "nonrule_tuned_oof": float(seed_tuned_nonrule),
            "greedy_plus_nonrule_tuned_oof": float(seed_tuned),
        }
        log(f"  seed {seed}: greedy={seed_tuned_greedy:.5f}  "
            f"nonrule={seed_tuned_nonrule:.5f}  "
            f"greedy+nonrule={seed_tuned:.5f}")

        per_seed_oof_greedy_plus_nonrule.append(gn_oof)
        per_seed_test_greedy_plus_nonrule.append(gn_test)
        per_seed_oof_greedy.append(greedy_oof)
        per_seed_test_greedy.append(greedy_test)
        per_seed_oof_nonrule.append(a["oof_nonrule"])
        per_seed_test_nonrule.append(a["test_nonrule"])

    if not seeds_loaded:
        log("no seed artefacts found - aborting")
        return

    log(f"bagging {len(seeds_loaded)} seeds: {seeds_loaded}")

    # Bagged averages: probability-space mean across seeds
    bag_gn_oof = np.mean(per_seed_oof_greedy_plus_nonrule, axis=0)
    bag_gn_test = np.mean(per_seed_test_greedy_plus_nonrule, axis=0)
    bag_greedy_oof = np.mean(per_seed_oof_greedy, axis=0)
    bag_greedy_test = np.mean(per_seed_test_greedy, axis=0)
    bag_nonrule_oof = np.mean(per_seed_oof_nonrule, axis=0)
    bag_nonrule_test = np.mean(per_seed_test_nonrule, axis=0)

    # Follow the single-seed methodology to stay apples-to-apples with LB:
    #   1) Tune log-bias on bagged GREEDY only
    #   2) Apply that bias FIXED to bagged greedy+nonrule
    # This matches nonrule_features_only.py which held greedy bias fixed
    # when sweeping alpha. The single-seed LB-best 0.97352 came from that
    # recipe, so the bagged version must use the same to be comparable.
    bias_g, tuned_g = tune_log_bias(bag_greedy_oof, y, prior)
    log(f"BAGGED greedy tuned OOF           = {tuned_g:.5f}  bias={bias_g.round(4).tolist()}")

    log_bag_gn_oof = np.log(np.clip(bag_gn_oof, 1e-9, 1.0))
    tuned_gn_fixed = balanced_accuracy_score(y, (log_bag_gn_oof + bias_g).argmax(1))
    log(f"BAGGED greedy+nonrule (fixed bias) = {tuned_gn_fixed:.5f}")

    # Also report the retuned-bias variant for diagnostics (not for LB).
    bias_gn, tuned_gn_retune = tune_log_bias(bag_gn_oof, y, prior)
    log(f"BAGGED greedy+nonrule (retune)    = {tuned_gn_retune:.5f}  "
        f"bias={bias_gn.round(4).tolist()}")

    # Baseline for comparison: the LB-best single-seed greedy+nonrule
    # computed under the same fixed-bias recipe (nonrule_features_only.py).
    baseline_single_seed = 0.97421
    delta_vs_single = tuned_gn_fixed - baseline_single_seed
    log(f"Δ bagged(fixed bias) vs single-seed greedy+nonrule: {delta_vs_single:+.5f}")

    # Save bagged arrays + submission (FIXED bias from greedy only)
    np.save(ART / "oof_bagged_greedy_nonrule.npy", bag_gn_oof)
    np.save(ART / "test_bagged_greedy_nonrule.npy", bag_gn_test)

    # Submission with FIXED greedy-only bias on bagged test probs (LB-comparable)
    test_logp = np.log(np.clip(bag_gn_test, 1e-9, 1.0)) + bias_g
    preds = test_logp.argmax(1)
    sub_path = OUT / "submission_bagged_greedy_nonrule.csv"
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
        sub_path, index=False
    )
    log(f"wrote {sub_path} (fixed bias)")

    # Also write retuned-bias submission for diagnostic comparison
    test_logp_rt = np.log(np.clip(bag_gn_test, 1e-9, 1.0)) + bias_gn
    preds_rt = test_logp_rt.argmax(1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds_rt]}).to_csv(
        OUT / "submission_bagged_greedy_nonrule_retune.csv", index=False
    )
    log(f"wrote submissions/submission_bagged_greedy_nonrule_retune.csv (retune bias)")

    # Also write a 'greedy only' bagged submission for safe-fallback
    test_logp_g = np.log(np.clip(bag_greedy_test, 1e-9, 1.0)) + bias_g
    preds_g = test_logp_g.argmax(1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds_g]}).to_csv(
        OUT / "submission_bagged_greedy.csv", index=False
    )
    log(f"wrote submissions/submission_bagged_greedy.csv (tuned={tuned_g:.5f})")

    # Confusion matrix on bagged greedy+nonrule (fixed bias)
    cm = confusion_matrix(y, (log_bag_gn_oof + bias_g).argmax(1),
                          labels=[0, 1, 2])
    log(f"bagged greedy+nonrule OOF confusion (fixed bias):\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    results = {
        "fold_seeds_requested": FOLD_SEEDS,
        "fold_seeds_loaded": seeds_loaded,
        "n_seeds_bagged": len(seeds_loaded),
        "greedy_weights": {"hybrid": W_HYB, "routed": W_ROUTED, "spec": W_SPEC},
        "alpha_nonrule": ALPHA_NONRULE,
        "per_seed": per_seed_diag,
        "bagged_greedy_tuned_oof": float(tuned_g),
        "bagged_greedy_bias": bias_g.tolist(),
        "bagged_greedy_plus_nonrule_fixed_bias_oof": float(tuned_gn_fixed),
        "bagged_greedy_plus_nonrule_retune_oof": float(tuned_gn_retune),
        "bagged_greedy_plus_nonrule_retune_bias": bias_gn.tolist(),
        "single_seed_baseline_oof": baseline_single_seed,
        "delta_vs_single_seed": float(delta_vs_single),
        "submission_path": str(sub_path),
    }
    with open(ART / "fold_seed_bag_blend_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/fold_seed_bag_blend_results.json")


if __name__ == "__main__":
    main()
