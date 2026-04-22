"""Rank-sum / Borda blend over saved OOFs.

Tests the "sum" lever from the 2026-04-21 brainstorm. All prior blends
were prob-space or log-space. Rank-averaging is calibration-invariant
and robust to confidence-scale differences across model families.

Run:
    python scripts/rank_blend.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score


ART = Path(__file__).parent / "artifacts"
DATA = Path(__file__).parent.parent / "data"


def tune_bias(p: np.ndarray, y: np.ndarray, prior: np.ndarray):
    """Coord-ascent on per-class log-bias to maximise bal_acc."""
    lp = np.log(np.clip(p, 1e-9, 1.0))
    b = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + b).argmax(1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = b.copy()
            sc = []
            for g in grid:
                base[k] = b[k] + g
                sc.append(balanced_accuracy_score(y, (lp + base).argmax(1)))
            j = int(np.argmax(sc))
            if sc[j] > best + 1e-6:
                b[k] = b[k] + grid[j]
                best = sc[j]
                improved = True
        if not improved:
            break
    return b, float(best)


def col_ranks(p: np.ndarray) -> np.ndarray:
    """Per-column rank normalised to [0, 1]."""
    out = np.empty_like(p, dtype=np.float64)
    n = p.shape[0]
    for c in range(p.shape[1]):
        out[:, c] = rankdata(p[:, c], method="average") / n
    return out


def softmax_row(a: np.ndarray) -> np.ndarray:
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)


def main():
    y = (
        pd.read_csv(DATA / "train.csv")["Irrigation_Need"]
        .map({"Low": 0, "Medium": 1, "High": 2})
        .values.astype(np.int64)
    )
    prior = np.bincount(y) / len(y)
    print(f"y: n={len(y)} prior={prior.round(4).tolist()}")

    components = {
        "hybrid_lgbmxgb_blend": ART / "oof_hybrid_lgbmxgb_blend.npy",
        "xgb_dist_routed_v3": ART / "oof_xgb_dist_routed_v3.npy",
        "xgb_vanilla_dist":   ART / "oof_xgb_vanilla_dist.npy",
        "lgbm_te_orig":       ART / "oof_lgbm_te_orig.npy",
    }
    oofs = {k: np.load(v) for k, v in components.items()}
    for k, a in oofs.items():
        print(f"{k}: shape={a.shape} range=[{a.min():.3g}, {a.max():.3g}]")

    baselines = {}
    for k, p in oofs.items():
        _, tuned = tune_bias(p, y, prior)
        baselines[k] = tuned
        print(f"baseline tuned bal_acc {k:<26s} = {tuned:.5f}")

    ranks = {k: col_ranks(p) for k, p in oofs.items()}

    subsets = {
        "all4":           list(components.keys()),
        "no_hybrid":      ["xgb_dist_routed_v3", "xgb_vanilla_dist", "lgbm_te_orig"],
        "hybrid+xgb_v3":  ["hybrid_lgbmxgb_blend", "xgb_dist_routed_v3"],
        "hybrid+all_base":["hybrid_lgbmxgb_blend", "xgb_dist_routed_v3",
                          "xgb_vanilla_dist", "lgbm_te_orig"],
    }

    results = {}

    # Equal-weight rank-avg -> row-softmax -> log-bias.
    for name, keys in subsets.items():
        r_avg = np.mean([ranks[k] for k in keys], axis=0)
        r_norm = r_avg / r_avg.sum(1, keepdims=True)
        bias, tuned = tune_bias(r_norm, y, prior)
        key = f"rank_avg_{name}"
        results[key] = {"members": keys, "tuned_bal_acc": tuned, "bias": bias.tolist()}
        print(f"{key:<40s} tuned={tuned:.5f}")

    # Weighted rank-avg (weight by standalone tuned bal_acc, softmax on scores).
    for name, keys in subsets.items():
        scores = np.array([baselines[k] for k in keys])
        w = np.exp((scores - scores.max()) * 200.0)
        w /= w.sum()
        r_avg = np.zeros_like(ranks[keys[0]])
        for k, wi in zip(keys, w):
            r_avg += wi * ranks[k]
        r_norm = r_avg / r_avg.sum(1, keepdims=True)
        bias, tuned = tune_bias(r_norm, y, prior)
        key = f"rank_wavg_{name}"
        results[key] = {
            "members": keys,
            "weights": w.tolist(),
            "tuned_bal_acc": tuned,
            "bias": bias.tolist(),
        }
        print(f"{key:<40s} tuned={tuned:.5f} weights={w.round(3).tolist()}")

    # Borda row-wise: sum class-ranks across models, argmax + log-bias on softmax.
    for name, keys in subsets.items():
        borda = np.mean([ranks[k] for k in keys], axis=0)
        logits = np.log(np.clip(borda, 1e-9, 1.0))
        pseudo = softmax_row(logits)
        bias, tuned = tune_bias(pseudo, y, prior)
        key = f"borda_softmax_{name}"
        results[key] = {"members": keys, "tuned_bal_acc": tuned, "bias": bias.tolist()}
        print(f"{key:<40s} tuned={tuned:.5f}")

    # Per-model-α sweep between rank-avg and prob-avg (just the hybrid + xgb_v3).
    keys = ["hybrid_lgbmxgb_blend", "xgb_dist_routed_v3"]
    r_avg = np.mean([ranks[k] for k in keys], axis=0)
    r_norm = r_avg / r_avg.sum(1, keepdims=True)
    p_avg = np.mean([oofs[k] for k in keys], axis=0)
    sweep = []
    for alpha in np.linspace(0.0, 1.0, 11):
        mix = alpha * r_norm + (1 - alpha) * p_avg
        mix /= mix.sum(1, keepdims=True)
        _, tuned = tune_bias(mix, y, prior)
        sweep.append((float(alpha), tuned))
        print(f"mix rank↔prob alpha={alpha:.2f} tuned={tuned:.5f}")
    results["mix_rank_prob_hybrid_xgbv3"] = {"sweep": sweep}

    # Current-best reference.
    results["baselines"] = baselines
    best = max(
        (v for k, v in results.items() if isinstance(v, dict) and "tuned_bal_acc" in v),
        key=lambda d: d["tuned_bal_acc"],
    )
    print(f"\nbest rank/Borda variant tuned bal_acc = {best['tuned_bal_acc']:.5f}")
    print(f"current-best hybrid_lgbmxgb_blend      = {baselines['hybrid_lgbmxgb_blend']:.5f}")

    out = ART / "rank_blend_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
