"""P2: hard-quota decision rule on LB-best 4-stack OOF + test.

Mechanism: instead of argmax(log p + bias), use rank-based assignment.
For target counts N_L, N_M, N_H summing to N:
  1. Assign top-N_H rows by P(High) → High
  2. Among remaining rows, assign top-N_M by P(Medium) → Medium
  3. Rest → Low
This is a constrained assignment. Order of class processing matters when
quotas don't add up cleanly to argmax. We try (H, M, L), (M, H, L), and
a Hungarian-style greedy assignment (each row goes to its best-prob class
that still has quota).

Quota source candidates:
  Q1. Train y prior counts: (N_te * 0.5872, 0.3795, 0.0333)
  Q2. Test rule_pred prior counts (slightly different)
  Q3. Predicted test y prior counts (P1 calibrated)
  Q4. OOF-optimal: sweep N_H around train prior, find OOF-best
       (sanity check: shouldn't differ much from Q1)

Compared to log-bias decision rule [1.43, 1.47, 3.40], quota rules
have ZERO learnable parameters (only the assignment ORDER) — robust
to bias-overfit / private-LB drift.

This is a diagnostic. If any quota rule lifts OOF over the
fixed-bias 0.98084 with no fitted parameters, that's a robustness gain.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, SUB, TARGET, build_lbbest_stack,
    iso_cal, log, normed,
)


def assign_sequential(p: np.ndarray, quotas: tuple[int, int, int],
                      order: tuple[int, int, int]) -> np.ndarray:
    """Sequential rank assignment in the given class order.
    For c1, c2, c3 in order: assign top-quotas[c] of remaining rows by P(c)."""
    n = len(p)
    pred = -np.ones(n, dtype=np.int8)
    remaining = np.ones(n, dtype=bool)
    for c in order[:-1]:
        idx_remaining = np.where(remaining)[0]
        scores = p[idx_remaining, c]
        if quotas[c] >= len(idx_remaining):
            chosen = idx_remaining
        else:
            top = np.argpartition(-scores, quotas[c] - 1)[:quotas[c]]
            chosen = idx_remaining[top]
        pred[chosen] = c
        remaining[chosen] = False
    pred[remaining] = order[-1]
    return pred


def assign_greedy_hungarian(p: np.ndarray, quotas: tuple[int, int, int]) -> np.ndarray:
    """Greedy: sort rows by max-prob descending; each row goes to its
    argmax-class if quota remains, else next-best with quota.
    """
    n = len(p)
    pred = -np.ones(n, dtype=np.int8)
    quota_left = list(quotas)
    # Process rows by decreasing top1-prob (most-confident first)
    top1 = p.max(axis=1)
    order_rows = np.argsort(-top1)
    rank = np.argsort(-p, axis=1)
    for i in order_rows:
        for c in rank[i]:
            if quota_left[c] > 0:
                pred[i] = c
                quota_left[c] -= 1
                break
    return pred


def bal(y, pred):
    return balanced_accuracy_score(y, pred)


def main():
    t0 = time.time()
    log("loading train/test/y")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    n_tr, n_te = len(train), len(test)

    # Build LB-best 4-stack OOF + test
    log("building LB-best 4-stack")
    s2_o, s2_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    p_oof = log_blend([s2_o, meta_o_iso], np.array([0.7, 0.3]))
    p_test = log_blend([s2_t, meta_t_iso], np.array([0.7, 0.3]))

    # Baseline: log-bias argmax
    pred_bias_oof = (np.log(np.clip(p_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_bias_test = (np.log(np.clip(p_test, 1e-12, 1)) + BIAS).argmax(1)
    base_oof = bal(y, pred_bias_oof)
    log(f"\nBASELINE (log-bias [1.43,1.47,3.40] argmax):")
    log(f"  OOF bal_acc                = {base_oof:.5f}")
    log(f"  OOF predicted counts        = {np.bincount(pred_bias_oof, minlength=3).tolist()}")
    log(f"  test predicted counts       = {np.bincount(pred_bias_test, minlength=3).tolist()}")

    # Quota sources
    y_tr_counts = np.bincount(y, minlength=3)
    y_tr_prior = y_tr_counts / n_tr
    rule_test = add_distance_features(test)["rule_pred"].to_numpy().astype(int)
    rule_test_dist = np.bincount(rule_test, minlength=3) / n_te
    # Predicted test y prior via train confusion P(y | rule)
    rule_train = add_distance_features(train)["rule_pred"].to_numpy().astype(int)
    cm = np.zeros((3, 3))
    for r, yr in zip(rule_train, y):
        cm[r, yr] += 1
    cm_rownorm = cm / cm.sum(1, keepdims=True)
    pred_test_prior = (rule_test_dist[:, None] * cm_rownorm).sum(axis=0)

    quota_sources = {
        "Q1_train_y": y_tr_prior,
        "Q2_test_rule": rule_test_dist,
        "Q3_pred_test_y": pred_test_prior,
    }

    # Use n_tr for OOF, n_te for test
    log("\n=== OOF QUOTA SWEEP (varying quota source × assignment order) ===")
    results = {}
    for q_name, q_prior in quota_sources.items():
        # Counts must integerize summing to n_tr
        q_counts_oof = np.round(q_prior * n_tr).astype(int)
        # Adjust to exact n_tr: add/sub from the largest class
        diff = n_tr - q_counts_oof.sum()
        if diff:
            q_counts_oof[np.argmax(q_counts_oof)] += diff
        q_counts_oof = tuple(int(x) for x in q_counts_oof)

        # Try different orderings + greedy
        for order_name, order in [
            ("HML", (2, 1, 0)),
            ("MHL", (1, 2, 0)),
            ("HLM", (2, 0, 1)),
            ("greedy", None),
        ]:
            if order is None:
                pred = assign_greedy_hungarian(p_oof, q_counts_oof)
            else:
                pred = assign_sequential(p_oof, q_counts_oof, order)
            sc = bal(y, pred)
            cc = np.bincount(pred, minlength=3).tolist()
            key = f"{q_name}/{order_name}"
            results[key] = dict(quota=list(q_counts_oof), order=order_name, oof=float(sc), counts=cc)
            d = sc - base_oof
            marker = " ✓" if sc > base_oof else ""
            log(f"  {key:<28} quota={q_counts_oof}  order={order_name:<7}  OOF={sc:.5f}  Δ={d:+.5f}{marker}")

    # Best non-fitted rule
    best_key = max(results, key=lambda k: results[k]["oof"])
    best = results[best_key]
    log(f"\nbest non-fitted: {best_key}  OOF={best['oof']:.5f}  Δ={best['oof']-base_oof:+.5f}")

    # Apply best to test
    q_name = best_key.split("/")[0]
    order_name = best_key.split("/")[1]
    q_prior_test = quota_sources[q_name]
    q_counts_test = np.round(q_prior_test * n_te).astype(int)
    diff = n_te - q_counts_test.sum()
    if diff:
        q_counts_test[np.argmax(q_counts_test)] += diff
    q_counts_test = tuple(int(x) for x in q_counts_test)
    if order_name == "greedy":
        pred_test_quota = assign_greedy_hungarian(p_test, q_counts_test)
    else:
        order_map = {"HML": (2, 1, 0), "MHL": (1, 2, 0), "HLM": (2, 0, 1)}
        pred_test_quota = assign_sequential(p_test, q_counts_test, order_map[order_name])

    diff_vs_bias = (pred_test_quota != pred_bias_test).sum()
    log(f"\nbest quota rule on TEST:")
    log(f"  quota counts        = {q_counts_test}")
    log(f"  predicted counts    = {np.bincount(pred_test_quota, minlength=3).tolist()}")
    log(f"  rows differing from log-bias prediction: {diff_vs_bias} / {n_te}")

    out = dict(
        base_oof=float(base_oof),
        results=results,
        best_key=best_key,
        best_oof=float(best["oof"]),
        delta_vs_bias=float(best["oof"] - base_oof),
        n_test_pred_diff_vs_bias=int(diff_vs_bias),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "p2_quota_decision_results.json").write_text(json.dumps(out, indent=2))

    # Emit candidate submission only if we beat baseline OOF
    if best["oof"] > base_oof:
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_test_quota]
        path = SUB / f"submission_p2_quota_{q_name}_{order_name}.csv"
        sub.to_csv(path, index=False)
        log(f"  wrote {path}")

    # Confusion-matrix diff vs baseline
    log(f"\nbaseline log-bias confusion (rows=true y, cols=pred):")
    cm_base = confusion_matrix(y, pred_bias_oof)
    log(f"  {cm_base.tolist()}")
    log(f"\nbest quota confusion:")
    if order_name == "greedy":
        pred_oof_quota = assign_greedy_hungarian(p_oof, tuple(np.round(quota_sources[q_name] * n_tr).astype(int)))
    else:
        order_map = {"HML": (2, 1, 0), "MHL": (1, 2, 0), "HLM": (2, 0, 1)}
        # Recompute (cheap, just to be safe)
        q_counts_oof_recompute = np.round(quota_sources[q_name] * n_tr).astype(int)
        diff = n_tr - q_counts_oof_recompute.sum()
        if diff:
            q_counts_oof_recompute[np.argmax(q_counts_oof_recompute)] += diff
        pred_oof_quota = assign_sequential(p_oof, tuple(int(x) for x in q_counts_oof_recompute), order_map[order_name])
    cm_q = confusion_matrix(y, pred_oof_quota)
    log(f"  {cm_q.tolist()}")
    log(f"\n  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
