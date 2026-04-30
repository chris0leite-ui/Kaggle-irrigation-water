"""Option A: Model-class-diverse OTHERS pool for k=2 unanimous override.

Anchor: 4b (LB 0.98150)
OTHERS pool (4 distinct model classes):
  - raw (XGB + sklearn TargetEncoder, LB 0.98109)
  - tier1b (XGB-meta on 63-component bank, LB 0.98094)
  - recipe_full_te_catboost (CatBoost, LB 0.97935)
  - extratrees_dist_digits (ExtraTrees, OOF only)

k=2 unanimous: 2 of 4 OTHERS must agree on a class != anchor's argmax.

Why this might work despite Idea 4c saturation:
  - 4c added 5th components to bagged_v1 (kept OTHERS as {raw, tier1b})
  - Same OTHERS pool → "same signal counted twice" via 14-bank overlap
  - This experiment changes the OTHERS pool itself: 4 distinct
    model-class gradients. CB and ET are tree-based but with
    fundamentally different training (ordered boosting vs
    randomized splits) — independent signal axes.

OOF analysis: per-direction precision on each variant
  (k=2 of 4, k=3 of 4, k=4 of 4 unanimous).
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


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)
    n_tr, n_te = len(train), len(test)

    bases = [
        ("v1",     "oof_sklearn_rf_meta_natural_v1_lb98129.npy",
                   "test_sklearn_rf_meta_natural_v1_lb98129.npy"),
        ("raw",    "oof_rawashishsin_2600.npy",
                   "test_rawashishsin_2600.npy"),
        ("t1b",    "oof_tier1b_greedy_meta.npy",
                   "test_tier1b_greedy_meta.npy"),
        ("cb",     "oof_recipe_full_te_catboost.npy",
                   "test_recipe_full_te_catboost.npy"),
        ("et",     "oof_extratrees_dist_digits.npy",
                   "test_extratrees_dist_digits.npy"),
    ]
    pool = {}
    for label, oof_p, test_p in bases:
        oof_path = ART / oof_p
        test_path = ART / test_p
        if not oof_path.exists() or not test_path.exists():
            print(f"  SKIP {label}: missing")
            continue
        oof = _normed(np.load(oof_path).astype(np.float32))
        tst = _normed(np.load(test_path).astype(np.float32))
        bias, tuned = tune_log_bias(oof, y, prior)
        oof_arg = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(1)
        test_arg = (np.log(np.clip(tst, 1e-9, 1.0)) + bias).argmax(1)
        pool[label] = dict(oof=oof, test=tst, bias=bias, tuned=tuned,
                           oof_arg=oof_arg, test_arg=test_arg)
        print(f"  {label}: tuned {tuned:.5f}")

    # Anchor v1 (for OOF analog) and 4b (for test side)
    v1_oof_arg = pool["v1"]["oof_arg"]
    base_v1 = balanced_accuracy_score(y, v1_oof_arg)
    print(f"\nv1 baseline OOF: {base_v1:.5f}")

    # Test anchor = 4b (LB 0.98150)
    anchor_4b = pd.read_csv(SUB / "submission_idea4b_selective_override.csv")[TARGET].map(CLS2IDX).to_numpy()
    # OOF anchor = B (= v1 + k=2 unanimous (raw, t1b)) which corresponds to LB 0.98140
    raw_oa = pool["raw"]["oof_arg"]
    t1b_oa = pool["t1b"]["oof_arg"]
    cb_oa = pool["cb"]["oof_arg"]
    et_oa = pool["et"]["oof_arg"]
    raw_ta = pool["raw"]["test_arg"]
    t1b_ta = pool["t1b"]["test_arg"]
    cb_ta = pool["cb"]["test_arg"]
    et_ta = pool["et"]["test_arg"]

    # B_oof = v1 + k=2 unanimous(raw, t1b)
    k2_mask = (raw_oa == t1b_oa) & (raw_oa != v1_oof_arg)
    B_oof = v1_oof_arg.copy()
    B_oof[k2_mask] = raw_oa[k2_mask]
    B_bal = balanced_accuracy_score(y, B_oof)
    print(f"B (k=2 unanimous on raw,t1b, OOF analog of LB 0.98140): {B_bal:.5f}")

    # ===== Apply k_thresh-of-4 unanimous on OOF =====
    others_oof = np.stack([raw_oa, t1b_oa, cb_oa, et_oa], axis=1)  # (n_tr, 4)
    others_test = np.stack([raw_ta, t1b_ta, cb_ta, et_ta], axis=1)  # (n_te, 4)

    # Per row, count votes per class
    votes_oof = np.zeros((n_tr, 3), dtype=np.int32)
    votes_test = np.zeros((n_te, 3), dtype=np.int32)
    for c in range(3):
        votes_oof[:, c] = (others_oof == c).sum(axis=1)
        votes_test[:, c] = (others_test == c).sum(axis=1)

    print(f"\n=== Sweep k_thresh on OOF (anchor = B = LB-best k=2 OOF) ===")
    print(f"  vs B_OOF = {B_bal:.5f}")
    not_B = (np.arange(3)[None, :] != B_oof[:, None])
    not_4b = (np.arange(3)[None, :] != anchor_4b[:, None])

    summary_per_k = {}
    for k_thresh in [2, 3, 4]:
        elig_oof = (votes_oof >= k_thresh) & not_B
        votes_elig = np.where(elig_oof, votes_oof, -1)
        any_elig = elig_oof.any(axis=1)
        chosen_oof = votes_elig.argmax(axis=1)
        n_oof_over = int(any_elig.sum())

        # Apply override on B_oof (find rows where anchor B != consensus)
        new_oof = B_oof.copy()
        m = any_elig & (chosen_oof != B_oof)
        new_oof[m] = chosen_oof[m]
        # Apply override only on rows where consensus differs from B
        new_oof_apply = B_oof.copy()
        new_oof_apply[m] = chosen_oof[m]
        n_applied = int(m.sum())
        bal = balanced_accuracy_score(y, new_oof_apply)
        pcr = per_class_recall(y, new_oof_apply)

        # Per-direction precision
        direction_stats = {}
        for a in range(3):
            for c in range(3):
                if a == c: continue
                dm = m & (B_oof == a) & (chosen_oof == c)
                n_d = int(dm.sum())
                if n_d == 0: continue
                n_correct = int((y[dm] == c).sum())
                direction_stats[(a, c)] = dict(
                    n=n_d, prec=float(n_correct / n_d),
                    be=float(prior[c] / (prior[a] + prior[c])))

        # Also test side: anchor = 4b
        elig_test = (votes_test >= k_thresh) & not_4b
        votes_elig_t = np.where(elig_test, votes_test, -1)
        any_elig_t = elig_test.any(axis=1)
        chosen_test = votes_elig_t.argmax(axis=1)
        m_test = any_elig_t & (chosen_test != anchor_4b)
        n_test_over = int(m_test.sum())

        print(f"\n  k>={k_thresh}: applied {n_applied} OOF overrides ({n_oof_over} candidates)")
        print(f"    OOF bal_acc: {bal:.5f}  Δ vs B={bal-B_bal:+.5f}")
        print(f"    Test overrides on 4b: {n_test_over}")
        print(f"    Per-direction precision (OOF):")
        for (a, c), s in direction_stats.items():
            print(f"      {IDX2CLS[a]}->{IDX2CLS[c]}: n={s['n']}  prec={s['prec']:.4f}  "
                  f"BE={s['be']:.4f}  margin={s['prec']-s['be']:+.4f}")

        # Build test submission
        test_pred = anchor_4b.copy()
        test_pred[m_test] = chosen_test[m_test]
        path = SUB / f"submission_n16_4way_k{k_thresh}_on4b.csv"
        pd.DataFrame({"id": test_ids,
                      TARGET: [IDX2CLS[i] for i in test_pred]}).to_csv(path, index=False)
        print(f"    Saved: {path}")

        summary_per_k[f"k>={k_thresh}"] = {
            "n_oof_overrides": n_applied,
            "n_test_overrides": n_test_over,
            "OOF_bal_acc": float(bal),
            "delta_vs_B": float(bal - B_bal),
            "directions": {f"{IDX2CLS[a]}->{IDX2CLS[c]}": s
                           for (a, c), s in direction_stats.items()},
        }

    with open(ART / "n16_4way_diverse_results.json", "w") as f:
        json.dump(summary_per_k, f, indent=2)
    print(f"\nSaved summary")


if __name__ == "__main__":
    main()
