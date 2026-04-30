"""Validate the n11 layered candidate (BMA + raw + t1b 3-way consensus on winner) on OOF.

Build OOF analog of:
  oof_winner   = v1_oof argmax + k=2 unanimous override (raw, t1b)
  oof_bma      = per-class BMA over 5 LB-validated bases at OOF
  oof_layered  = oof_winner with second-pass override:
                   if BMA == raw == t1b agree on class C != oof_winner[i] → flip

Compare macro-recall: oof_winner vs oof_layered.
Per-direction precision of the layered override.

Decision: emit submission for LB probe only if OOF Δ ≥ +0.0002 vs winner.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def _normed(a, eps=1e-9):
    a = np.clip(a, eps, 1.0)
    return a / a.sum(axis=1, keepdims=True)


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def break_even(prior, a, c):
    return prior[c] / (prior[a] + prior[c])


def biased_arg(probs, bias, eps=1e-9):
    return (np.log(np.clip(probs, eps, 1.0)) + bias).argmax(1)


def main():
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)

    bases = [
        ("v1",     "oof_sklearn_rf_meta_natural_v1_lb98129.npy", 0.98129),
        ("raw",    "oof_rawashishsin_2600.npy", 0.98109),
        ("t1b",    "oof_tier1b_greedy_meta.npy", 0.98094),
        ("recipe", "oof_recipe_full_te.npy", 0.97939),
        ("cb",     "oof_recipe_full_te_catboost.npy", 0.97935),
    ]
    pool = []
    for label, path, lb in bases:
        oof = _normed(np.load(ART / path).astype(np.float32))
        bias, tuned = tune_log_bias(oof, y, prior)
        argm = biased_arg(oof, bias)
        pcr = per_class_recall(y, argm)
        # Biased softmax
        log_p = np.log(np.clip(oof, 1e-9, 1.0)) + bias
        log_p -= log_p.max(axis=1, keepdims=True)
        p_b = np.exp(log_p)
        p_b /= p_b.sum(axis=1, keepdims=True)
        pool.append(dict(label=label, lb=lb, tuned=tuned, bias=bias,
                         pcr=pcr, argm=argm, p_biased=p_b))
        print(f"  {label} (LB {lb:.5f}): tuned {tuned:.5f}")

    v1_arg = pool[0]["argm"]
    raw_arg = pool[1]["argm"]
    t1b_arg = pool[2]["argm"]

    # OOF winner = v1 + k=2 unanimous (raw, t1b)
    cand = (raw_arg == t1b_arg) & (raw_arg != v1_arg)
    oof_winner = v1_arg.copy()
    oof_winner[cand] = raw_arg[cand]
    bal_winner = balanced_accuracy_score(y, oof_winner)
    pcr_winner = per_class_recall(y, oof_winner)
    print(f"\nOOF winner (v1 + k=2 unanimous): {bal_winner:.5f}")
    print(f"  PCR=[L={pcr_winner[0]:.4f} M={pcr_winner[1]:.4f} H={pcr_winner[2]:.4f}]")

    # Build BMA: per-class weights = LB * recall, normalized
    n = len(y)
    w = np.zeros((len(pool), 3))
    for i, p in enumerate(pool):
        for k in range(3):
            w[i, k] = p["lb"] * p["pcr"][k]
    w /= w.sum(axis=0, keepdims=True)

    log_bma = np.zeros((n, 3))
    for i, p in enumerate(pool):
        log_p = np.log(np.clip(p["p_biased"], 1e-9, 1.0))
        for k in range(3):
            log_bma[:, k] += w[i, k] * log_p[:, k]
    log_bma -= log_bma.max(axis=1, keepdims=True)
    p_bma = np.exp(log_bma)
    p_bma /= p_bma.sum(axis=1, keepdims=True)
    bias_bma, tuned_bma = tune_log_bias(p_bma, y, prior)
    bma_arg = biased_arg(p_bma, bias_bma)
    bal_bma = balanced_accuracy_score(y, bma_arg)
    print(f"\nBMA standalone: tuned bias {bias_bma.round(3).tolist()}  bal_acc {bal_bma:.5f}")

    # Layered: where BMA agrees with raw AND t1b on a class != oof_winner
    # I.e., 3-way consensus says oof_winner is wrong on this row
    consensus = (raw_arg == t1b_arg) & (raw_arg == bma_arg) & (raw_arg != oof_winner)
    n_layered = consensus.sum()
    oof_layered = oof_winner.copy()
    oof_layered[consensus] = raw_arg[consensus]
    bal_layered = balanced_accuracy_score(y, oof_layered)
    pcr_layered = per_class_recall(y, oof_layered)
    print(f"\nOOF layered (winner + 3-way consensus): {bal_layered:.5f}")
    print(f"  Δ vs winner = {bal_layered - bal_winner:+.5f}")
    print(f"  Layered overrides: {n_layered}")
    print(f"  PCR=[L={pcr_layered[0]:.4f} M={pcr_layered[1]:.4f} H={pcr_layered[2]:.4f}]")

    # Per-direction
    print(f"\n=== Layered override direction breakdown ===")
    print(f"{'A':<8}{'C':<8}{'n':>5}{'prec':>8}{'BE':>8}{'margin':>9}")
    direction_stats = {}
    for a in range(3):
        for c in range(3):
            if a == c: continue
            mask = consensus & (oof_winner == a) & (raw_arg == c)
            n_d = int(mask.sum())
            if n_d == 0: continue
            n_correct = int((y[mask] == c).sum())
            prec = n_correct / n_d
            be = break_even(prior, a, c)
            direction_stats[(a, c)] = dict(n=n_d, n_correct=n_correct, prec=float(prec),
                                           be=float(be), margin=float(prec - be))
            print(f"{IDX2CLS[a]:<8}{IDX2CLS[c]:<8}{n_d:>5}{prec:>8.4f}{be:>8.4f}{prec-be:>+9.4f}")

    # Save summary
    summary = {
        "oof_winner_bal_acc": float(bal_winner),
        "oof_layered_bal_acc": float(bal_layered),
        "oof_delta_layered_vs_winner": float(bal_layered - bal_winner),
        "n_layered_overrides": int(n_layered),
        "direction_stats": {f"{IDX2CLS[a]}->{IDX2CLS[c]}": s
                            for (a, c), s in direction_stats.items()},
    }
    json_path = ART / "n11_layered_oof_validation.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
