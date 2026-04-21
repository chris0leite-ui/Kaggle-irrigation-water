"""CatBoost Jaccard-overlap check + 3-way blend.

Question: does CatBoost bring orthogonal error signal to (LGBM, XGB)?
Even though CatBoost standalone OOF (0.97128) is below both
LGBM-dist (0.97266) and XGB-dist (0.97304), its errors may not
overlap with theirs — in which case a 3-way blend gains from diversity.

Pre-check gate: Jaccard on OOF error sets.
  < 0.80 -> commit to seed-bag CatBoost + 3-way blend
  >= 0.80 -> CatBoost is a parameter-shifted version of LGBM/XGB,
             skip.

Then: try geometric-mean 3-way blend over (LGBM-TE, XGB-vanilla, CatBoost)
and log-blend alpha sweep.

Reference OOFs available:
  oof_lgbm_te_orig.npy    = LGBM-dist + redundant TE cols (0.97270 ≈ LGBM-dist)
  oof_xgb_vanilla_dist.npy = XGB-dist trained on all rows (0.97304)
  oof_catboost_dist.npy   = CatBoost-dist (0.97128)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ART = Path("scripts/artifacts")


def tune_log_bias(p, y, prior):
    lp = np.log(np.clip(p, 1e-9, 1.0))
    b = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + b).argmax(axis=1))
    grid = np.linspace(-3, 3, 61)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = b.copy()
            sc = []
            for g in grid:
                base[k] = b[k] + g
                sc.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(sc))
            if sc[j] > best + 1e-6:
                b[k] = b[k] + grid[j]
                best = sc[j]
                imp = True
        if not imp:
            break
    return b, best


def main():
    tr = pd.read_csv("data/train.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    oof_lgbm = np.load(ART / "oof_lgbm_te_orig.npy")  # ≈ LGBM-dist
    oof_xgb = np.load(ART / "oof_xgb_vanilla_dist.npy")  # XGB-dist no route
    oof_cb = np.load(ART / "oof_catboost_dist.npy")

    # Tune bias for each and compute error sets
    results = {}
    err_sets = {}
    for name, oof in [("LGBM", oof_lgbm), ("XGB", oof_xgb), ("CatBoost", oof_cb)]:
        bias, tuned = tune_log_bias(oof, y, prior)
        pred = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
        errs = np.where(pred != y)[0]
        err_sets[name] = set(errs.tolist())
        results[name] = {"tuned_bal": float(tuned), "n_errors": len(errs)}
        print(f"{name:10s}  tuned={tuned:.5f}  n_errors={len(errs)}")

    # Jaccard overlap
    print("\n=== Jaccard overlap of error sets ===")
    jaccards = {}
    for a, b in [("LGBM", "XGB"), ("LGBM", "CatBoost"), ("XGB", "CatBoost")]:
        inter = len(err_sets[a] & err_sets[b])
        union = len(err_sets[a] | err_sets[b])
        j = inter / union if union else 0
        jaccards[f"{a}_vs_{b}"] = j
        print(f"  {a} vs {b}: inter={inter}  union={union}  Jaccard={j:.4f}")
    inter_all = len(err_sets["LGBM"] & err_sets["XGB"] & err_sets["CatBoost"])
    union_all = len(err_sets["LGBM"] | err_sets["XGB"] | err_sets["CatBoost"])
    jaccards["all_three"] = inter_all / union_all if union_all else 0
    print(f"  all three (inter/union): {inter_all}/{union_all} = "
          f"{jaccards['all_three']:.4f}")

    print("\n=== Blend sweeps ===")

    # 2-way LGBM + XGB reference
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = np.exp(w * np.log(np.clip(oof_lgbm, 1e-9, 1.0)) +
                       (1 - w) * np.log(np.clip(oof_xgb, 1e-9, 1.0)))
        blend /= blend.sum(axis=1, keepdims=True)
        _, t = tune_log_bias(blend, y, prior)
        print(f"  LGBM*{w:.1f} + XGB*{1-w:.1f}:            tuned={t:.5f}")

    # 3-way geometric blends
    print("  3-way blends (log-space, coarse grid):")
    best_3way = -1
    best_3way_w = None
    for w_l in np.arange(0.1, 0.6, 0.1):
        for w_x in np.arange(0.1, 0.6, 0.1):
            w_c = 1 - w_l - w_x
            if w_c <= 0.0 or w_c >= 0.9:
                continue
            blend = np.exp(w_l * np.log(np.clip(oof_lgbm, 1e-9, 1.0)) +
                           w_x * np.log(np.clip(oof_xgb, 1e-9, 1.0)) +
                           w_c * np.log(np.clip(oof_cb, 1e-9, 1.0)))
            blend /= blend.sum(axis=1, keepdims=True)
            _, t = tune_log_bias(blend, y, prior)
            if t > best_3way:
                best_3way = t
                best_3way_w = (w_l, w_x, w_c)
            print(f"    w=(L={w_l:.1f}, X={w_x:.1f}, C={w_c:.1f}):  tuned={t:.5f}")

    print(f"\nbest 3-way blend: weights={best_3way_w} tuned={best_3way:.5f}")
    print(f"  vs LGBM*0.45+XGB*0.55 reference: 0.97327 (current best blend)")
    print(f"  Δ: {best_3way - 0.97327:+.5f}")

    with open(ART / "catboost_jaccard_blend_results.json", "w") as f:
        json.dump({
            "standalone": results,
            "jaccards": jaccards,
            "best_3way": {"weights": list(best_3way_w), "tuned": float(best_3way)},
        }, f, indent=2)


if __name__ == "__main__":
    main()
