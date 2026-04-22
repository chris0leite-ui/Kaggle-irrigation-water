"""Session B analysis: load per-seed OOFs, report cross-seed variance,
build multi-seed test-prob bag, emit LB-candidate submission if stable.

Reads:
  per-seed artefacts saved by session_b_pipeline.py for fold_seeds in
  {42, 7, 123}. For seed=42 we use the existing oof_greedy_blend.npy /
  test_greedy_blend.npy (saved before the per-seed pipeline was built)
  as the reference, and reconstruct greedy+nonrule LB-best from
  oof_xgb_nonrule.npy / test_xgb_nonrule.npy.

Outputs:
  scripts/artifacts/session_b_multi_seed_summary.json
  submissions/submission_lb_best_multi_seed_bag.csv  (if stable)

Stability gate (from the session plan):
  If per-seed OOF spread σ < 0.0005 across {42, 7, 123}: STABLE ->
    emit multi-seed bag as a submission candidate (avg test probs,
    tune log-bias on avg OOF). Expected: tighter OOF→LB gap.
  If σ >= 0.0005: UNSTABLE -> the 0.97421 number was partly lucky-
    split luck; log this and rethink.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


ART = Path("scripts/artifacts")
OUT = Path("submissions")
ROOT = Path(".")
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ID = "id"
TARGET = "Irrigation_Need"
SEEDS = [42, 7, 123]


def log(msg):
    print(msg, flush=True)


def log_blend(probs_list, weights, eps=1e-9):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    logits = np.zeros_like(probs_list[0])
    for wi, p in zip(w, probs_list):
        logits += wi * np.log(np.clip(p, eps, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p


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
    return bias, float(best)


def load_components_for_seed(seed, y):
    """Return (oof_lb, test_lb, tuned_oof_on_lb, per-component summaries).

    For seed=42 we prefer the historical artefacts (oof_greedy_blend /
    oof_xgb_nonrule) — that IS the LB-validated stack. For seeds 7,123
    we use the newly-saved per-seed LB-best (oof_lb_best_fs{seed}.npy).
    """
    if seed == 42:
        # historical / saved
        oof_greedy = np.load(ART / "oof_greedy_blend.npy")
        test_greedy = np.load(ART / "test_greedy_blend.npy")
        oof_nr = np.load(ART / "oof_xgb_nonrule.npy")
        test_nr = np.load(ART / "test_xgb_nonrule.npy")
        oof_lb = log_blend([oof_greedy, oof_nr], [0.85, 0.15])
        test_lb = log_blend([test_greedy, test_nr], [0.85, 0.15])
        # per-component OOFs for variance table
        components = {
            "routed_v3": np.load(ART / "oof_xgb_dist_routed_v3.npy"),
            "spec_678": np.load(ART / "oof_xgb_spec_678.npy"),
            "nonrule": oof_nr,
            "greedy": oof_greedy,
            "lb_best": oof_lb,
        }
        tests = {"lb_best": test_lb}
    else:
        oof_greedy = np.load(ART / f"oof_greedy_fs{seed}.npy")
        test_greedy = np.load(ART / f"test_greedy_fs{seed}.npy")
        oof_lb = np.load(ART / f"oof_lb_best_fs{seed}.npy")
        test_lb = np.load(ART / f"test_lb_best_fs{seed}.npy")
        oof_nr = np.load(ART / f"oof_nonrule_fs{seed}.npy")
        components = {
            "routed_v3": np.load(ART / f"oof_routed_v3_fs{seed}.npy"),
            "spec_678": np.load(ART / f"oof_spec_678_fs{seed}.npy"),
            "nonrule": oof_nr,
            "greedy": oof_greedy,
            "lb_best": oof_lb,
        }
        tests = {"lb_best": test_lb}
    return components, tests, oof_lb, test_lb


def main():
    log("=== Session B multi-seed analysis ===")
    tr = pd.read_csv(ROOT / "data/train.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    results = {"seeds": SEEDS, "per_seed": {}}

    # Per-seed OOF bal_acc (tuned) for each component
    per_seed_lb_oof = {}
    per_seed_lb_test = {}
    for seed in SEEDS:
        try:
            comps, tests, oof_lb, test_lb = load_components_for_seed(seed, y)
        except FileNotFoundError as e:
            log(f"skip seed={seed}: missing artefact ({e})")
            continue
        per_seed = {}
        for name, o in comps.items():
            bias, tuned = tune_log_bias(o, y, prior)
            per_seed[name] = float(tuned)
        per_seed_lb_oof[seed] = oof_lb
        per_seed_lb_test[seed] = test_lb
        log(f"seed {seed}:  " + "  ".join(f"{k}={v:.5f}" for k, v in per_seed.items()))
        results["per_seed"][seed] = per_seed

    if len(per_seed_lb_oof) < 2:
        log("Need at least 2 seeds to compute spread; exiting.")
        with open(ART / "session_b_multi_seed_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # Cross-seed spread on each component
    log("\n=== Cross-seed spread (tuned OOF bal_acc) ===")
    spread = {}
    for name in ["routed_v3", "spec_678", "nonrule", "greedy", "lb_best"]:
        vals = [results["per_seed"][s][name] for s in SEEDS
                if s in results["per_seed"]]
        if not vals:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        minv, maxv = float(min(vals)), float(max(vals))
        log(f"  {name:12s}  mean={mean:.5f}  std={std:.5f}  "
            f"min={minv:.5f}  max={maxv:.5f}  spread={maxv-minv:.5f}")
        spread[name] = {"mean": mean, "std": std, "min": minv, "max": maxv,
                        "spread": maxv - minv, "values": vals}
    results["spread"] = spread

    lb_std = spread["lb_best"]["std"]
    lb_mean = spread["lb_best"]["mean"]
    log(f"\nLB-best cross-seed: mean {lb_mean:.5f}  std {lb_std:.5f}")

    # --- Multi-seed bag: average OOF & test probs in LOG space ---
    oofs = [per_seed_lb_oof[s] for s in SEEDS if s in per_seed_lb_oof]
    tests = [per_seed_lb_test[s] for s in SEEDS if s in per_seed_lb_test]
    oof_bag = log_blend(oofs, np.ones(len(oofs)))
    test_bag = log_blend(tests, np.ones(len(tests)))
    bias_bag, tuned_bag = tune_log_bias(oof_bag, y, prior)
    log(f"\nMulti-seed bag (log-avg of {len(oofs)} LB-best OOFs): tuned OOF = {tuned_bag:.5f}")
    results["bag"] = {
        "n_seeds": len(oofs),
        "tuned_oof": float(tuned_bag),
        "delta_vs_seed42": float(tuned_bag - results["per_seed"][42]["lb_best"])
            if 42 in results["per_seed"] else None,
        "log_bias": bias_bag.tolist(),
    }

    # Also try prob-avg bag as a sanity
    prob_bag_oof = np.mean(oofs, axis=0)
    prob_bag_oof = prob_bag_oof / prob_bag_oof.sum(axis=1, keepdims=True)
    bias_p, tuned_p = tune_log_bias(prob_bag_oof, y, prior)
    log(f"Multi-seed prob-avg bag:                                tuned OOF = {tuned_p:.5f}")
    results["bag_prob_avg"] = {"tuned_oof": float(tuned_p)}

    # Confusion matrix for the bag
    pred_bag = (np.log(np.clip(oof_bag, 1e-9, 1.0)) + bias_bag).argmax(axis=1)
    cm = confusion_matrix(y, pred_bag)
    log(f"Bag confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")
    results["bag_confusion_matrix"] = cm.tolist()

    # Gate + emit submission if stable
    stable = lb_std < 0.0005
    log(f"\n=== Stability gate: std {lb_std:.5f} {'<' if stable else '>='} 0.0005 => "
        f"{'STABLE' if stable else 'UNSTABLE'} ===")
    results["stability"] = {"std": lb_std, "stable": bool(stable),
                            "threshold": 0.0005}

    if stable and tuned_bag > lb_mean - 0.0001:
        pred_test = (np.log(np.clip(test_bag, 1e-9, 1.0)) + bias_bag).argmax(axis=1)
        te = pd.read_csv(ROOT / "data/test.csv", usecols=[ID])
        sub = pd.DataFrame({ID: te[ID],
                           TARGET: [CLASSES[i] for i in pred_test]})
        fn = "submission_lb_best_multi_seed_bag.csv"
        sub.to_csv(OUT / fn, index=False)
        log(f"LB candidate emitted: {fn}")
        results["submission_emitted"] = fn

    with open(ART / "session_b_multi_seed_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved -> {ART}/session_b_multi_seed_summary.json")


if __name__ == "__main__":
    main()
