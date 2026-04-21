"""Class-asymmetric blend: keep hybrid_v3 for Low/Medium probs, but
upweight High prob contributions from other models.

Motivation: under macro-recall with 3.3% High prior, every correct
promotion to High is worth ~18× a correct Low. Plain prob-averaging
treats all three classes symmetrically and tends to pull High down
when one or two models are less High-confident on boundary rows.
We want the OPPOSITE: on rows where multiple models' P_high is
elevated, let that consensus override hybrid_v3's calmer estimate.

Strategy implemented:
    P_out[Low]    = hybrid_v3[Low]                     (unchanged)
    P_out[Medium] = hybrid_v3[Medium]                  (unchanged)
    P_out[High]   = hybrid_v3[High] + gamma * (max(other_models[High])
                                               - hybrid_v3[High])
                    where gamma > 0 strengthens consensus-on-High.
Then re-normalize rowwise and tune log-bias as usual.

Gamma is swept over [0, 1] and the OOF-best value is picked. Only
writes a submission if tuned OOF beats hybrid_v3 standalone
(0.97352) by >= 3e-4.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def fast_bal_acc(y, pred, cc):
    m = pred == y
    hit = np.array([m[y == k].sum() for k in range(3)])
    return float((hit / np.maximum(cc, 1)).mean())

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def tune_log_bias(oof, y, prior, cc):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = fast_bal_acc(y, (log_oof + bias).argmax(axis=1), cc)
    gd = np.linspace(-3.0, 3.0, 61)
    gh = np.linspace(-3.0, 6.0, 91)
    for _ in range(20):
        improved = False
        for k in range(3):
            grid = gh if k == 2 else gd
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(fast_bal_acc(y, (log_oof + base).argmax(axis=1), cc))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def per_class_recall(y, pred):
    return {c: float(((pred == i) & (y == i)).sum() / max((y == i).sum(), 1))
            for i, c in enumerate(CLASSES)}


def main():
    tr = pd.read_csv("data/train.csv", usecols=[ID, TARGET])
    te_ids = pd.read_csv("data/test.csv", usecols=[ID])[ID].values
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    cc = np.bincount(y, minlength=3)

    # anchor: hybrid_v3 (current best).
    oof_h = np.load(ART / "oof_xgb_hybrid_v3.npy")
    test_h = np.load(ART / "test_xgb_hybrid_v3.npy")

    # other models whose P_high we trust. Don't include spec_678 — its
    # OOF is only meaningful on 56k rows.
    others = []
    for name in ["oof_lgbm_dgp.npy", "oof_xgb_dist.npy", "oof_xgb_dist_routed_v3.npy"]:
        p = ART / name
        if p.exists():
            others.append((name.replace("oof_", "").replace(".npy", ""), np.load(p)))
    others_test = []
    for name in ["test_lgbm_dgp.npy", "test_xgb_dist.npy", "test_xgb_dist_routed_v3.npy"]:
        p = ART / name
        if p.exists():
            others_test.append(np.load(p))
    print(f"hybrid_v3 OOF shape {oof_h.shape}; {len(others)} other models for consensus")

    bias_h, bal_h = tune_log_bias(oof_h, y, prior, cc)
    pred_h = (np.log(np.clip(oof_h, 1e-9, 1.0)) + bias_h).argmax(axis=1)
    pcr_h = per_class_recall(y, pred_h)
    print(f"hybrid_v3 standalone: bal={bal_h:.5f}  "
          f"rec_L={pcr_h['Low']:.4f} rec_M={pcr_h['Medium']:.4f} rec_H={pcr_h['High']:.4f}")

    # consensus High prob: take MAX across others (willing to promote
    # based on single strong voter), and MEAN (require agreement).
    oof_high_max = np.maximum.reduce([o[:, 2] for _, o in others])
    oof_high_mean = np.mean(np.stack([o[:, 2] for _, o in others], axis=0), axis=0)
    test_high_max = np.maximum.reduce([t[:, 2] for t in others_test])
    test_high_mean = np.mean(np.stack([t[:, 2] for t in others_test], axis=0), axis=0)

    results = {"hybrid_v3_standalone": float(bal_h)}

    for variant_name, oof_hi, test_hi in [
            ("max_other_high", oof_high_max, test_high_max),
            ("mean_other_high", oof_high_mean, test_high_mean)]:
        rows = []
        best = (-1.0, None, None, None, None)
        for gamma in np.linspace(-0.5, 1.5, 21):
            oof_adj = oof_h.copy()
            # only adjust High column; gamma>0 lifts toward others
            oof_adj[:, 2] = oof_h[:, 2] + gamma * (oof_hi - oof_h[:, 2])
            # re-normalize row-sums
            oof_adj = np.clip(oof_adj, 1e-9, None)
            oof_adj /= oof_adj.sum(axis=1, keepdims=True)
            test_adj = test_h.copy()
            test_adj[:, 2] = test_h[:, 2] + gamma * (test_hi - test_h[:, 2])
            test_adj = np.clip(test_adj, 1e-9, None)
            test_adj /= test_adj.sum(axis=1, keepdims=True)

            bias, tuned = tune_log_bias(oof_adj, y, prior, cc)
            pred = (np.log(np.clip(oof_adj, 1e-9, 1.0)) + bias).argmax(axis=1)
            pcr = per_class_recall(y, pred)
            rows.append({"gamma": float(gamma), "tuned": float(tuned),
                         **{f"rec_{k}": v for k, v in pcr.items()}})
            if tuned > best[0]:
                best = (tuned, gamma, bias, oof_adj, test_adj)
            print(f"  {variant_name:18s}  gamma={gamma:+.2f}  tuned={tuned:.5f}  "
                  f"rec_L={pcr['Low']:.4f} rec_M={pcr['Medium']:.4f} rec_H={pcr['High']:.4f}")

        results[variant_name] = {
            "best_gamma": best[1], "best_tuned": best[0],
            "sweep": rows,
        }
        print(f"  {variant_name} best: gamma={best[1]:+.2f}  tuned={best[0]:.5f}  "
              f"Δ={best[0] - bal_h:+.5f}")

        # write submission if it beats hybrid_v3 standalone
        if best[0] > bal_h + 3e-4:
            bias = best[2]
            test_adj = best[4]
            pred_idx = (np.log(np.clip(test_adj, 1e-9, 1.0)) + bias).argmax(axis=1)
            fname = f"submission_blend_high_weighted_{variant_name}.csv"
            pd.DataFrame({ID: te_ids, TARGET: [IDX2CLS[i] for i in pred_idx]}).to_csv(
                SUB / fname, index=False
            )
            print(f"  wrote {fname}")

    with open(ART / "blend_high_weighted_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
