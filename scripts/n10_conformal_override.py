"""#10 Conformal-set override on the LB-best 0.98140 winner.

Mechanism:
  Build Mondrian (per-class) split-conformal prediction sets on v1 anchor's
  OOF probabilities. For each test row, override v1's argmax to consensus
  class C ONLY IF:
    (a) raw and t1b unanimously agree on class C (k=2 unanimous), AND
    (b) C is in v1's conformal prediction set at coverage 1-α

  Different from current LB-best 0.98140:
    - 0.98140 winner overrides whenever raw == t1b ≠ v1, regardless of
      v1's confidence on the consensus class
    - Conformal-set filter requires the consensus class to be PLAUSIBLE
      under v1's calibrated distribution
    - Filters out weak-direction overrides (M→L 57%, L→M 25% precision)
      because consensus class often outside the conformal set on those

Three coverage levels: α ∈ {0.05, 0.10, 0.15} (90%, 95%, 85%)

Per CLAUDE.md: emits candidate CSVs only; no auto-submit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS = ("Low", "Medium", "High")
CLS2IDX = {c: i for i, c in enumerate(CLS)}
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


def mondrian_conformal_thresholds(probs_calib, y_calib, alpha):
    """Per-class split-conformal nonconformity threshold.

    Score s_i = 1 - p_i(y_i)  (LAC nonconformity)
    For each class k separately, compute q_hat_k from {s_i : y_i = k}
    so that 90% (when alpha=0.1) of class-k calibration rows have s_i <= q_hat_k.

    Returns: array (n_class,) of q_hat per class.
    """
    scores = 1.0 - probs_calib[np.arange(len(y_calib)), y_calib]
    q_hat = np.zeros(3)
    for k in range(3):
        m = y_calib == k
        n = m.sum()
        if n == 0:
            q_hat[k] = 1.0
            continue
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        q_hat[k] = float(np.quantile(scores[m], q_level))
    return q_hat


def conformal_set_membership(probs_query, q_hat):
    """Return (n, 3) bool array: cls k in set iff (1 - p_query[k]) <= q_hat[k]."""
    score_query = 1.0 - probs_query  # shape (n, 3)
    return score_query <= q_hat[None, :]


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    # Load v1 anchor + 2 OTHERS (k=2 unanimous override)
    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy"))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy"))
    raw_oof = _normed(np.load(ART / "oof_rawashishsin_2600.npy"))
    raw_test = _normed(np.load(ART / "test_rawashishsin_2600.npy"))
    t1b_oof = _normed(np.load(ART / "oof_tier1b_greedy_meta.npy"))
    t1b_test = _normed(np.load(ART / "test_tier1b_greedy_meta.npy"))

    # Tune log-bias for argmax computation
    v1_bias, _ = tune_log_bias(v1_oof, y, prior)
    raw_bias, _ = tune_log_bias(raw_oof, y, prior)
    t1b_bias, _ = tune_log_bias(t1b_oof, y, prior)

    # Compute argmax + biased probabilities for each model
    def biased_probs(p, b):
        log_p = np.log(np.clip(p, 1e-9, 1.0)) + b
        log_p = log_p - log_p.max(axis=1, keepdims=True)
        ep = np.exp(log_p)
        return ep / ep.sum(axis=1, keepdims=True)

    v1_oof_b = biased_probs(v1_oof, v1_bias)
    v1_test_b = biased_probs(v1_test, v1_bias)
    v1_oof_arg = v1_oof_b.argmax(1)
    v1_test_arg = v1_test_b.argmax(1)
    raw_oof_arg = (np.log(np.clip(raw_oof, 1e-9, 1.0)) + raw_bias).argmax(1)
    raw_test_arg = (np.log(np.clip(raw_test, 1e-9, 1.0)) + raw_bias).argmax(1)
    t1b_oof_arg = (np.log(np.clip(t1b_oof, 1e-9, 1.0)) + t1b_bias).argmax(1)
    t1b_test_arg = (np.log(np.clip(t1b_test, 1e-9, 1.0)) + t1b_bias).argmax(1)

    # Sanity: v1's test argmax should match v1 CSV
    v1_csv = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")
    v1_csv_pred = v1_csv[TARGET].map(CLS2IDX).to_numpy()
    print(f"v1 OOF-argmax vs v1 CSV diff: {(v1_test_arg != v1_csv_pred).sum()}")
    # Use OOF-argmax internally for OOF analysis; use CSV for test anchor

    # Anchor for test side = LB-best winner (CSV)
    winner_csv = pd.read_csv(SUB / "submission_2other_raw_tier1b_k2.csv")
    winner_pred = winner_csv[TARGET].map(CLS2IDX).to_numpy()

    # ===== OOF: split-conformal calibration via 5-fold (already aligned) =====
    # We use LEAVE-ONE-FOLD-OUT calibration: for each fold f, compute conformal
    # thresholds on OOF rows from folds != f, apply to fold-f rows
    # This is a bit awkward; simpler: use a single calib/query split via random shuffle
    rng = np.random.RandomState(42)
    n = len(y)
    perm = rng.permutation(n)
    half = n // 2
    cal_idx, query_idx = perm[:half], perm[half:]

    def evaluate(alpha, label):
        q_hat = mondrian_conformal_thresholds(v1_oof_b[cal_idx], y[cal_idx], alpha)
        # Compute set membership on the QUERY half (held-out OOF for unbiased eval)
        sets_query = conformal_set_membership(v1_oof_b[query_idx], q_hat)
        avg_size = sets_query.sum(axis=1).mean()
        coverage = sets_query[np.arange(len(query_idx)), y[query_idx]].mean()
        # Mark rows where consensus class C is in v1's conformal set
        sets_test = conformal_set_membership(v1_test_b, q_hat)
        sets_oof = conformal_set_membership(v1_oof_b, q_hat)
        return q_hat, sets_oof, sets_test, avg_size, coverage

    print(f"\n=== Conformal calibration ===")
    print(f"v1 anchor OOF tuned bias: {v1_bias.round(3).tolist()}")

    # ===== Apply to test winner with override gate =====
    # WINNER's overrides: where winner_pred ≠ v1_csv_pred (286 rows on test)
    # For each, the consensus class is winner_pred[i] (= raw_test_arg[i] = t1b_test_arg[i])
    # Conformal filter: only KEEP override if winner_pred[i] in v1's conformal set
    # Rows where winner_pred not in conformal set: REVERT to v1

    diff_mask_test = winner_pred != v1_csv_pred
    print(f"\nWinner test diff vs v1: {diff_mask_test.sum()} rows")

    # OOF-side analog of winner's mechanism: apply k=2 unanimous override on OOF
    raw_t1b_unanimous_oof = (raw_oof_arg == t1b_oof_arg) & (raw_oof_arg != v1_oof_arg)
    oof_consensus = raw_oof_arg
    oof_winner = v1_oof_arg.copy()
    oof_winner[raw_t1b_unanimous_oof] = oof_consensus[raw_t1b_unanimous_oof]

    base_v1_bal = balanced_accuracy_score(y, v1_oof_arg)
    base_winner_bal = balanced_accuracy_score(y, oof_winner)
    print(f"\nOOF baselines:")
    print(f"  v1                 = {base_v1_bal:.5f}")
    print(f"  k=2 unanimous winner = {base_winner_bal:.5f}  Δ={base_winner_bal-base_v1_bal:+.5f}")

    # Sweep alpha
    results = []
    print(f"\n=== Conformal-filtered override sweep ===")
    print(f"{'α':>5}{'avg_set':>9}{'cov_q':>8}{'overrides_OOF':>16}{'OOF_bal':>10}"
          f"{'Δ_v1':>9}{'Δ_winner':>10}{'overrides_test':>16}")
    for alpha in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        q_hat, sets_oof, sets_test, avg_size, cov_q = evaluate(alpha, f"α={alpha}")

        # OOF: filter raw_t1b_unanimous to rows where consensus is in v1's conf set
        oof_filter = raw_t1b_unanimous_oof.copy()
        for i in np.where(raw_t1b_unanimous_oof)[0]:
            cls_c = oof_consensus[i]
            if not sets_oof[i, cls_c]:
                oof_filter[i] = False  # consensus class not in v1's set → no override
        n_oof_over = oof_filter.sum()
        oof_filtered = v1_oof_arg.copy()
        oof_filtered[oof_filter] = oof_consensus[oof_filter]
        bal = balanced_accuracy_score(y, oof_filtered)
        d_v1 = bal - base_v1_bal
        d_w = bal - base_winner_bal

        # Test: filter winner's overrides similarly
        test_keep = diff_mask_test.copy()
        for i in np.where(diff_mask_test)[0]:
            cls_c = winner_pred[i]
            if not sets_test[i, cls_c]:
                test_keep[i] = False
        n_test_over = test_keep.sum()

        print(f"{alpha:>5.2f}{avg_size:>9.3f}{cov_q:>8.4f}{n_oof_over:>16}"
              f"{bal:>10.5f}{d_v1:>+9.5f}{d_w:>+10.5f}{n_test_over:>16}")

        # Build test-side candidate
        new_pred = v1_csv_pred.copy()
        new_pred[test_keep] = winner_pred[test_keep]
        diff_winner = int((new_pred != winner_pred).sum())
        path = SUB / f"submission_n10_conformal_a{int(alpha*100):03d}.csv"
        pd.DataFrame({"id": test_ids,
                      TARGET: [IDX2CLS[i] for i in new_pred]}).to_csv(path, index=False)
        results.append({
            "alpha": alpha, "q_hat": q_hat.tolist(),
            "avg_set_size": float(avg_size), "coverage_query": float(cov_q),
            "n_oof_override": int(n_oof_over), "OOF_bal_acc": float(bal),
            "delta_vs_v1": float(d_v1), "delta_vs_winner": float(d_w),
            "n_test_override": int(n_test_over),
            "test_diff_vs_winner": diff_winner, "submission": path.name,
        })

    # ===== Save summary =====
    with open(ART / "n10_conformal_override_results.json", "w") as f:
        json.dump({"v1_bias": v1_bias.tolist(),
                   "OOF_baseline_v1": float(base_v1_bal),
                   "OOF_baseline_winner": float(base_winner_bal),
                   "sweep": results}, f, indent=2)
    print(f"\nSaved: scripts/artifacts/n10_conformal_override_results.json")


if __name__ == "__main__":
    main()
