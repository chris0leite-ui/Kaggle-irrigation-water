"""#14 2-of-3 plurality override using n2 disagree-meta as 3rd OTHER.

OOF result: 0.98093 (vs LB-best k=2 unanimous 0.98088, +0.00005).

Mechanism:
  Anchor = v1 RF natural (LB 0.98129)
  OTHERS = {raw, t1b, n2_disagree}
  Override if 2-of-3 OTHERS agree on a class C != v1's argmax (plurality)

OOF found 946 candidate overrides (vs k=2 unanimous's 356). Per-direction
analysis to predict LB transferability.

Build candidate submission for LB probe consideration.
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
    return a / np.clip(a.sum(1, keepdims=True), eps, None)


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def break_even(prior, a, c):
    return prior[c] / (prior[a] + prior[c])


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    # Load 4 components
    v1 = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_t = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    raw = _normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    raw_t = _normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))
    t1b = _normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))
    t1b_t = _normed(np.load(ART / "test_tier1b_greedy_meta.npy").astype(np.float32))
    n2 = _normed(np.load(ART / "oof_n2_disagree.npy").astype(np.float32))
    n2_t = _normed(np.load(ART / "test_n2_disagree.npy").astype(np.float32))

    bv1, _ = tune_log_bias(v1, y, prior)
    braw, _ = tune_log_bias(raw, y, prior)
    bt1b, _ = tune_log_bias(t1b, y, prior)
    bn2, _ = tune_log_bias(n2, y, prior)

    def biased_arg(p, b):
        return (np.log(np.clip(p, 1e-9, 1.0)) + b).argmax(1)

    v1_oa = biased_arg(v1, bv1)
    raw_oa = biased_arg(raw, braw)
    t1b_oa = biased_arg(t1b, bt1b)
    n2_oa = biased_arg(n2, bn2)
    v1_ta = biased_arg(v1_t, bv1)
    raw_ta = biased_arg(raw_t, braw)
    t1b_ta = biased_arg(t1b_t, bt1b)
    n2_ta = biased_arg(n2_t, bn2)

    # Use v1 SUBMISSION CSV for test anchor (per CLAUDE.md alignment)
    v1_csv = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")
    v1_anchor_test = v1_csv[TARGET].map(CLS2IDX).to_numpy()

    n_oof = len(y)
    n_test = len(test_ids)

    # 2-of-3 plurality on OOF (consensus class != v1, voted by 2+ of {raw, t1b, n2})
    oargs_oof = np.stack([raw_oa, t1b_oa, n2_oa], axis=1)  # (n, 3)
    votes_oof = np.zeros((n_oof, 3), dtype=np.int32)
    for c in range(3):
        votes_oof[:, c] = (oargs_oof == c).sum(axis=1)
    # Override class with most votes (≥2) and ≠ v1
    cls_idx = np.arange(3)[None, :]
    not_v1 = (cls_idx != v1_oa[:, None])
    eligible = (votes_oof >= 2) & not_v1
    votes_eligible = np.where(eligible, votes_oof, -1)
    any_elig = eligible.any(axis=1)
    chosen = votes_eligible.argmax(axis=1)
    pred_oof = v1_oa.copy()
    pred_oof[any_elig] = chosen[any_elig]

    base_v1 = balanced_accuracy_score(y, v1_oa)
    bal_2of3 = balanced_accuracy_score(y, pred_oof)
    pcr_2of3 = per_class_recall(y, pred_oof)
    n_overrides_oof = int(any_elig.sum())
    print(f"v1 baseline: {base_v1:.5f}")
    print(f"2-of-3 plurality (raw, t1b, n2): {bal_2of3:.5f}  Δ={bal_2of3-base_v1:+.5f}")
    print(f"  Overrides: {n_overrides_oof}")
    print(f"  PCR=[L={pcr_2of3[0]:.4f} M={pcr_2of3[1]:.4f} H={pcr_2of3[2]:.4f}]")

    # Compare to k=2 unanimous
    k2_mask = (raw_oa == t1b_oa) & (raw_oa != v1_oa)
    k2_pred = v1_oa.copy()
    k2_pred[k2_mask] = raw_oa[k2_mask]
    k2_bal = balanced_accuracy_score(y, k2_pred)
    print(f"k=2 unanimous (LB winner mechanism): {k2_bal:.5f}  Δ={k2_bal-base_v1:+.5f}")
    print(f"\n2-of-3 vs k=2 OOF: {bal_2of3-k2_bal:+.5f}")

    # Per-direction analysis on 2-of-3 overrides
    print(f"\n=== Per-direction 2-of-3 plurality ===")
    direction_stats = {}
    for a in range(3):
        for c in range(3):
            if a == c: continue
            m = any_elig & (v1_oa == a) & (chosen == c)
            n_d = int(m.sum())
            if n_d == 0: continue
            n_correct = int((y[m] == c).sum())
            prec = n_correct / n_d
            be = break_even(prior, a, c)
            direction_stats[(a, c)] = dict(n=n_d, prec=float(prec), be=float(be))
            print(f"  {IDX2CLS[a]:<7}->{IDX2CLS[c]:<7}: n={n_d:4d}  prec={prec:.4f}  BE={be:.4f}  margin={prec-be:+.4f}")

    # ===== Test side =====
    oargs_test = np.stack([raw_ta, t1b_ta, n2_ta], axis=1)
    votes_test = np.zeros((n_test, 3), dtype=np.int32)
    for c in range(3):
        votes_test[:, c] = (oargs_test == c).sum(axis=1)
    not_anchor = (np.arange(3)[None, :] != v1_anchor_test[:, None])
    eligible_test = (votes_test >= 2) & not_anchor
    votes_elig_test = np.where(eligible_test, votes_test, -1)
    any_elig_test = eligible_test.any(axis=1)
    chosen_test = votes_elig_test.argmax(axis=1)
    new_test = v1_anchor_test.copy()
    new_test[any_elig_test] = chosen_test[any_elig_test]
    n_test_over = int(any_elig_test.sum())
    print(f"\nTest 2-of-3 overrides: {n_test_over}")
    print(f"Test class distribution after override:")
    cnt = np.bincount(new_test, minlength=3)
    print(f"  L={cnt[0]} M={cnt[1]} H={cnt[2]}")

    winner = pd.read_csv(SUB / "submission_2other_raw_tier1b_k2.csv")
    winner_pred = winner[TARGET].map(CLS2IDX).to_numpy()
    diff_winner = int((new_test != winner_pred).sum())
    diff_v1 = int((new_test != v1_anchor_test).sum())
    print(f"\nTest diff vs LB-best winner (0.98140): {diff_winner}")
    print(f"Test diff vs v1: {diff_v1}")

    # Per-direction breakdown of test overrides
    print(f"\nTest direction breakdown:")
    for a in range(3):
        for c in range(3):
            if a == c: continue
            cnt_d = int((any_elig_test & (v1_anchor_test == a) & (chosen_test == c)).sum())
            if cnt_d > 0:
                print(f"  {IDX2CLS[a]:<7}->{IDX2CLS[c]:<7}: {cnt_d}")

    # Save
    path = SUB / "submission_n14_2of3_plurality.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in new_test]}).to_csv(path, index=False)
    print(f"\nSaved: {path}")

    summary = {
        "v1_baseline_OOF": float(base_v1),
        "k2_unanimous_OOF": float(k2_bal),
        "2of3_plurality_OOF": float(bal_2of3),
        "delta_2of3_vs_k2": float(bal_2of3 - k2_bal),
        "delta_2of3_vs_v1": float(bal_2of3 - base_v1),
        "n_oof_overrides": n_overrides_oof,
        "n_test_overrides": n_test_over,
        "diff_vs_winner": diff_winner,
        "diff_vs_v1_anchor": diff_v1,
        "direction_stats": {f"{IDX2CLS[a]}->{IDX2CLS[c]}": s
                            for (a, c), s in direction_stats.items()},
        "submission": str(path),
    }
    with open(ART / "n14_2of3_plurality_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary")


if __name__ == "__main__":
    main()
