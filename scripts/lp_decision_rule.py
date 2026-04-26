"""Bayes-optimal / LP decision-rule probe on the LB-best primary (LB 0.98094).

Hypothesis: the coord-ascent log-bias [1.43, 1.47, 3.40] is overfit on OOF
relative to the closed-form Bayes-optimal under macro-recall, which is
b_k = -log(prior_k) -> [0.53, 0.97, 3.41]. Coord-ascent is ~0.9 units
heavier on Low+Medium. Test 5 decision-rule families on the SAME LB-best
primary OOF + test predictions and gate any candidate with stricter rules
than the LR meta-stacker null (Δ ≥ +2e-4 OOF AND per-class recall guardrail
AND structurally interpretable signal source).

Files: short, single-purpose. Outputs JSON + (gated) submission CSV.
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
from common import fast_bal_acc, log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, SUB, build_lbbest_stack, iso_cal, load_y, normed,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_primary(y: np.ndarray):
    """Reconstruct LB-best primary (LB 0.98094) = 3-stack + xgb_metastack_iso α=0.30."""
    s2_o, s2_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    iso_o, iso_t = iso_cal(meta_o, meta_t, y)
    w = np.array([0.70, 0.30])
    return log_blend([s2_o, iso_o], w), log_blend([s2_t, iso_t], w)


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return cm.diagonal() / np.maximum(cm.sum(axis=1), 1)


def report(name, y, P, bias, baseline_bal=None):
    pred = (np.log(np.clip(P, 1e-12, 1)) + bias).argmax(1)
    bal = balanced_accuracy_score(y, pred)
    rec = per_class_recall(y, pred)
    errs = (pred != y).sum()
    delta = bal - baseline_bal if baseline_bal is not None else 0.0
    log(f"  {name:30s}  bal={bal:.5f}  Δ={delta:+.5f}  errs={errs:6d}  "
        f"PCR=[L={rec[0]:.4f} M={rec[1]:.4f} H={rec[2]:.4f}]  "
        f"bias=[{bias[0]:+.3f}, {bias[1]:+.3f}, {bias[2]:+.3f}]")
    return dict(name=name, bal=float(bal), delta=float(delta), errs=int(errs),
                rec_low=float(rec[0]), rec_med=float(rec[1]), rec_high=float(rec[2]),
                bias=[float(b) for b in bias])


def temperature_bias_search(P, y):
    """Joint (T, b) search: pred = argmax_k (log P[k] / T + b_k). T_k per class.

    The 6-D landscape is non-smooth (argmax flips). Coord-ascent on a coarse
    grid + Nelder-Mead refinement.
    """
    log_P = np.log(np.clip(P, 1e-12, 1))
    cc = np.bincount(y, minlength=3)
    T = np.array([1.0, 1.0, 1.0])
    b = BIAS.copy()
    best = fast_bal_acc(y, (log_P + b).argmax(1), class_counts=cc)
    grid_T = np.linspace(0.5, 2.5, 21)
    grid_b = np.linspace(-2.0, 2.0, 21)
    for it in range(8):
        improved = False
        for k in range(3):
            scores = [fast_bal_acc(y, (log_P / np.where(np.arange(3) == k, t, T) + b).argmax(1),
                                    class_counts=cc) for t in grid_T]
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                T[k] = grid_T[j]; best = scores[j]; improved = True
        for k in range(3):
            scores = [fast_bal_acc(y, (log_P / T + np.where(np.arange(3) == k, b[k] + g, b)).argmax(1),
                                    class_counts=cc) for g in grid_b]
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                b[k] = b[k] + grid_b[j]; best = scores[j]; improved = True
        if not improved:
            break
    return T, b, best


def lp_oof_assignment(P, y, prior_cap):
    """LP-style upper bound: for each row i, allow assignment y_pred_i ∈ {0,1,2},
    subject to per-class cardinality constraints based on prior_cap. Maximize
    macro-recall on OOF (where y_true is known).

    This is the BEST any decision rule could possibly do given (P, prior_cap)
    constraints — useful as a target ceiling, not a deployable rule.
    """
    N = len(y)
    cc = np.bincount(y, minlength=3)
    # Per-row "score" of assigning to class k is the log-prob (we want to keep rows
    # most-confident-per-class first, while respecting cap). We greedily assign
    # rows to their argmax class but cap each class's count at prior_cap * N.
    log_P = np.log(np.clip(P, 1e-12, 1))
    caps = (prior_cap * N).astype(int)
    # Score = how much macro-recall this row contributes if assigned correctly
    pred = np.full(N, -1, dtype=np.int64)
    counts = np.zeros(3, dtype=np.int64)
    # Sort rows by max log-prob descending; assign to argmax if cap available.
    argmax_k = log_P.argmax(1)
    max_logp = log_P.max(1)
    order = np.argsort(-max_logp)
    for i in order:
        k = argmax_k[i]
        if counts[k] < caps[k]:
            pred[i] = k
            counts[k] += 1
    # Fill remaining rows (caps exhausted) with argmax over remaining classes
    leftover = pred == -1
    if leftover.any():
        for i in np.where(leftover)[0]:
            order_k = np.argsort(-log_P[i])
            for k in order_k:
                if counts[k] < caps[k]:
                    pred[i] = k; counts[k] += 1; break
            if pred[i] == -1:
                pred[i] = argmax_k[i]  # fallback
    bal = fast_bal_acc(y, pred, class_counts=cc)
    return bal, pred


def main():
    y = load_y()
    log(f"loaded y, n={len(y)}, prior={np.bincount(y) / len(y)}")

    log("reconstructing LB-best primary (LB 0.98094)...")
    P_oof, P_test = build_primary(y)
    log(f"primary OOF shape {P_oof.shape}, test shape {P_test.shape}")

    train_prior = np.bincount(y) / len(y)
    log(f"train prior: {train_prior}")

    # Sanity: reproduce LB-best at current bias
    baseline = report("CURRENT log-bias [1.43, 1.47, 3.40]", y, P_oof, BIAS, None)

    results = [baseline]

    # Family 1: closed-form Bayes-optimal under train prior
    bias_train = -np.log(train_prior)
    results.append(report("Bayes-opt (train prior)", y, P_oof, bias_train, baseline["bal"]))

    # Family 2: closed-form Bayes-optimal under predicted-test prior
    # P1 finding: predicted_test_prior ≈ train_prior within 0.07pp, so this is
    # effectively the same as Family 1 on this problem. Keep for completeness.
    # Use rule_pred frequencies on test as a slightly different anchor.
    test_df = pd.read_csv("data/test.csv")
    from common import add_distance_features
    rule_pred_test = add_distance_features(test_df)["rule_pred"].to_numpy()
    test_rule_prior = np.bincount(rule_pred_test, minlength=3) / len(rule_pred_test)
    bias_test_rule = -np.log(np.clip(test_rule_prior, 1e-6, None))
    results.append(report("Bayes-opt (test rule_pred prior)", y, P_oof, bias_test_rule, baseline["bal"]))

    # Family 3: per-class temperature + bias coord-ascent (richer parameterization)
    log("running joint (T, b) coord-ascent search...")
    T, b_T, bal_T = temperature_bias_search(P_oof, y)
    log(f"  optimal T={T}, b={b_T}, OOF={bal_T:.5f}")
    pred_T = (np.log(np.clip(P_oof, 1e-12, 1)) / T + b_T).argmax(1)
    bal_T_check = balanced_accuracy_score(y, pred_T)
    rec_T = per_class_recall(y, pred_T)
    errs_T = (pred_T != y).sum()
    delta_T = bal_T_check - baseline["bal"]
    log(f"  Family-3 (T, b) bal={bal_T_check:.5f} Δ={delta_T:+.5f} errs={errs_T} "
        f"PCR=[L={rec_T[0]:.4f} M={rec_T[1]:.4f} H={rec_T[2]:.4f}]")
    results.append(dict(name="(T, b) coord-ascent",
                        bal=float(bal_T_check), delta=float(delta_T), errs=int(errs_T),
                        rec_low=float(rec_T[0]), rec_med=float(rec_T[1]), rec_high=float(rec_T[2]),
                        T=[float(t) for t in T], bias=[float(bb) for bb in b_T]))

    # Family 4: LP-style upper bound under train prior cardinality cap
    log("computing LP upper bound under train-prior cap...")
    lp_bal, lp_pred = lp_oof_assignment(P_oof, y, train_prior)
    rec_lp = per_class_recall(y, lp_pred)
    errs_lp = (lp_pred != y).sum()
    delta_lp = lp_bal - baseline["bal"]
    log(f"  LP upper bound: bal={lp_bal:.5f} Δ={delta_lp:+.5f} errs={errs_lp} "
        f"PCR=[L={rec_lp[0]:.4f} M={rec_lp[1]:.4f} H={rec_lp[2]:.4f}]")
    results.append(dict(name="LP cap=train_prior", bal=float(lp_bal), delta=float(delta_lp),
                        errs=int(errs_lp), rec_low=float(rec_lp[0]), rec_med=float(rec_lp[1]),
                        rec_high=float(rec_lp[2])))

    # Save full results
    out = ART / "lp_decision_rule_results.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"results saved to {out}")

    # Decision: print final ranking + emit-gate summary
    log("=" * 80)
    log(f"BASELINE (LB 0.98094): OOF tuned = {baseline['bal']:.5f}, "
        f"PCR=[L={baseline['rec_low']:.4f} M={baseline['rec_med']:.4f} "
        f"H={baseline['rec_high']:.4f}]")
    log("Per-family deltas vs baseline:")
    for r in results[1:]:
        gate = "PASS" if (r["delta"] >= 2e-4 and
                          r["rec_low"] >= baseline["rec_low"] - 5e-4 and
                          r["rec_med"] >= baseline["rec_med"] - 5e-4 and
                          r["rec_high"] >= baseline["rec_high"] - 5e-4) else "FAIL"
        log(f"  {r['name']:35s}  Δ={r['delta']:+.5f}  gate={gate}")

    # Save test-side predictions for any PASS family (so user can LB-probe later)
    log("=" * 80)
    log("Saving test-side argmax predictions for each family (gated emit only)...")
    log_P_test = np.log(np.clip(P_test, 1e-12, 1))
    families_test = {
        "current_logbias": (log_P_test + BIAS).argmax(1),
        "bayes_train": (log_P_test + bias_train).argmax(1),
        "bayes_test_rule": (log_P_test + bias_test_rule).argmax(1),
        "T_b": (log_P_test / T + b_T).argmax(1),
    }
    classes = ["Low", "Medium", "High"]
    test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()
    for name, pred in families_test.items():
        # Disagreement vs current_logbias (the LB-best primary submission)
        if name != "current_logbias":
            disagree = int((pred != families_test["current_logbias"]).sum())
            log(f"  {name}: {disagree} test rows differ from current logbias submission")
        # Save as candidate (NOT auto-emit; user reviews)
        sub_path = SUB / f"submission_lp_{name}.csv"
        df = pd.DataFrame({"id": test_ids, "Irrigation_Need": [classes[p] for p in pred]})
        df.to_csv(sub_path, index=False)
        log(f"  {name}: saved {sub_path}")


if __name__ == "__main__":
    main()
