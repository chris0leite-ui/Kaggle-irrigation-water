"""P1: characterize test set and recalibrate log-bias to predicted-test-prior.

Mechanism:
  1. Compute rule_pred() distribution on test vs train. If test rule_pred
     proportions differ from train, our log-bias [1.43, 1.47, 3.40]
     (tuned on train OOF) is calibrated to the wrong prior.
  2. Use rule_pred-on-test as a proxy for the unknown true test class
     distribution (rule has 96.1% bal_acc on synthetic, so it captures
     class-prior structure even where individual labels flip).
  3. Re-tune log-bias on train OOF using a recalibrated objective:
     macro-recall weighted by predicted-test-prior, OR raw accuracy
     under a class-prior reweighting.
  4. Apply new bias to LB-best 4-stack TEST probs, emit candidate
     submission.

If test prior differs by ≥0.5pp on any class, this could be a free
recalibration with zero compute cost. If priors match within noise,
verdict is "no shift" and primary is already optimal.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, SUB, TARGET, build_lbbest_stack,
    iso_cal, load_y, log, normed,
)
from common import log_blend  # noqa: E402


def compute_rule_pred(df: pd.DataFrame) -> np.ndarray:
    """rule_pred via add_distance_features — returns int array shape (N,)."""
    fe = add_distance_features(df)
    return fe["rule_pred"].to_numpy().astype(int)


def coord_ascent_bias(p_oof: np.ndarray, y: np.ndarray, bias0: np.ndarray,
                      target_prior: np.ndarray | None = None,
                      grid_step: float = 0.01,
                      grid_half_width: float = 0.30,
                      n_passes: int = 6) -> tuple[np.ndarray, float]:
    """Coord ascent on bias around bias0; objective = balanced_accuracy_score
    on y, optionally with class-weight = target_prior (re-weights recall to
    match expected test prior, not train).
    """
    bias = bias0.copy()
    logp = np.log(np.clip(p_oof, 1e-12, 1))

    def score(b):
        pred = (logp + b).argmax(1)
        if target_prior is None:
            return balanced_accuracy_score(y, pred)
        # Per-class recall reweighted by target_prior
        rec = np.zeros(3, dtype=np.float64)
        for c in range(3):
            mask = y == c
            n = mask.sum()
            if n == 0:
                continue
            rec[c] = (pred[mask] == c).sum() / n
        return float((rec * target_prior).sum() / target_prior.sum())

    best = score(bias)
    grid = np.arange(-grid_half_width, grid_half_width + 1e-9, grid_step)
    for _ in range(n_passes):
        improved = False
        for k in range(3):
            base = bias[k]
            for d in grid:
                b2 = bias.copy()
                b2[k] = base + d
                s = score(b2)
                if s > best + 1e-9:
                    best = s
                    bias = b2
                    improved = True
        if not improved:
            break
    return bias, best


def main():
    t0 = time.time()
    log("loading train/test")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    rule_train = compute_rule_pred(train)
    rule_test = compute_rule_pred(test)
    n_tr, n_te = len(train), len(test)

    # True train label distribution
    y_dist = np.bincount(y, minlength=3) / n_tr
    rule_train_dist = np.bincount(rule_train, minlength=3) / n_tr
    rule_test_dist = np.bincount(rule_test, minlength=3) / n_te

    log(f"\n  Train true y prior:        Low {y_dist[0]:.4f}  Med {y_dist[1]:.4f}  High {y_dist[2]:.4f}")
    log(f"  Train rule_pred prior:      Low {rule_train_dist[0]:.4f}  Med {rule_train_dist[1]:.4f}  High {rule_train_dist[2]:.4f}")
    log(f"  Test  rule_pred prior:      Low {rule_test_dist[0]:.4f}  Med {rule_test_dist[1]:.4f}  High {rule_test_dist[2]:.4f}")

    delta_test_vs_train = rule_test_dist - rule_train_dist
    log(f"\n  Test - Train rule diff:    Low {delta_test_vs_train[0]:+.5f}  Med {delta_test_vs_train[1]:+.5f}  High {delta_test_vs_train[2]:+.5f}")

    # Predict test true-y prior. Rule on synthetic train hits ~96.1% bal_acc,
    # so rule_test is a noisy estimate of test y_dist. Best estimate:
    # `train_y | train_rule` joint probabilities applied to test rule_pred.
    # P(y | rule on train) ≈ confusion-matrix row-normalized.
    confusion = np.zeros((3, 3), dtype=np.float64)
    for tr_rule, tr_y in zip(rule_train, y):
        confusion[tr_rule, tr_y] += 1
    # Row-normalize: confusion[r, c] = P(y=c | rule_pred=r)
    cm_rownorm = confusion / confusion.sum(axis=1, keepdims=True)
    log("\n  P(y | rule_pred) on train:")
    for r in range(3):
        log(f"    rule={CLASSES[r]:<6}: " + " ".join(f"{CLASSES[c]} {cm_rownorm[r, c]:.4f}" for c in range(3)))

    pred_test_prior = (rule_test_dist[:, None] * cm_rownorm).sum(axis=0)
    log(f"\n  Predicted test y prior:    Low {pred_test_prior[0]:.4f}  Med {pred_test_prior[1]:.4f}  High {pred_test_prior[2]:.4f}")
    log(f"  Train y prior:              Low {y_dist[0]:.4f}  Med {y_dist[1]:.4f}  High {y_dist[2]:.4f}")
    delta_y = pred_test_prior - y_dist
    log(f"  Predicted test - train y:   Low {delta_y[0]:+.5f}  Med {delta_y[1]:+.5f}  High {delta_y[2]:+.5f}")

    # Build LB-best 4-stack OOF + test (anchor for primary submission)
    log("\nbuilding LB-best stack as anchor")
    s2_o, s2_t = build_lbbest_stack(y)
    # Add the meta-stacker iso component matching the LB 0.98094 primary
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    p4_oof = log_blend([s2_o, meta_o_iso], np.array([0.7, 0.3]))
    p4_test = log_blend([s2_t, meta_t_iso], np.array([0.7, 0.3]))
    base_score = balanced_accuracy_score(y, (np.log(np.clip(p4_oof, 1e-12, 1)) + BIAS).argmax(1))
    log(f"  LB-best 4-stack OOF (BIAS={BIAS.tolist()}): {base_score:.5f}")

    # Re-tune bias on train OOF with target_prior=predicted test prior
    log("\ncoord-ascent bias retune (target_prior = predicted_test_prior)")
    new_bias_test, _ = coord_ascent_bias(p4_oof, y, BIAS, target_prior=pred_test_prior)
    score_new_oof = balanced_accuracy_score(y, (np.log(np.clip(p4_oof, 1e-12, 1)) + new_bias_test).argmax(1))
    log(f"  new bias (test-prior weighted): {new_bias_test.tolist()}")
    log(f"  OOF bal_acc at new bias (UNWEIGHTED for comparability): {score_new_oof:.5f}")

    # Sanity: re-tune on train OOF with no target prior (should reproduce ~BIAS)
    new_bias_unw, _ = coord_ascent_bias(p4_oof, y, BIAS)
    log(f"  sanity: unweighted retune        : bias={new_bias_unw.tolist()}  bal_acc={balanced_accuracy_score(y, (np.log(np.clip(p4_oof, 1e-12, 1)) + new_bias_unw).argmax(1)):.5f}")

    # Compare test predictions: BIAS vs new_bias_test
    pred_orig = (np.log(np.clip(p4_test, 1e-12, 1)) + BIAS).argmax(1)
    pred_new = (np.log(np.clip(p4_test, 1e-12, 1)) + new_bias_test).argmax(1)
    diff = (pred_orig != pred_new).sum()
    log(f"\n  Test predictions differing (orig BIAS vs recalibrated): {diff} / {n_te}")
    if diff > 0:
        log("  Per-class delta (new - orig) on test:")
        for c in range(3):
            log(f"    {CLASSES[c]:<6}: orig {(pred_orig == c).sum():>6}  new {(pred_new == c).sum():>6}  Δ {(pred_new == c).sum() - (pred_orig == c).sum():+5d}")

    out = dict(
        train_y_prior=y_dist.tolist(),
        train_rule_prior=rule_train_dist.tolist(),
        test_rule_prior=rule_test_dist.tolist(),
        rule_diff=delta_test_vs_train.tolist(),
        pred_test_y_prior=pred_test_prior.tolist(),
        train_y_prior_diff=delta_y.tolist(),
        cm_rownorm=cm_rownorm.tolist(),
        oof_4stack_at_orig_bias=float(base_score),
        oof_4stack_at_test_prior_bias=float(score_new_oof),
        orig_bias=BIAS.tolist(),
        new_bias_test_prior=new_bias_test.tolist(),
        sanity_unweighted_retune_bias=new_bias_unw.tolist(),
        n_test_pred_diff=int(diff),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "p1_test_prior_results.json").write_text(json.dumps(out, indent=2))

    # Emit candidate submission only if a meaningful number of test predictions
    # change. This is NOT auto-emitted to LB — user must approve.
    if diff > 0:
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_new]
        path = SUB / "submission_p1_test_prior_recal.csv"
        sub.to_csv(path, index=False)
        log(f"  wrote candidate {path}")

    log(f"\n  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
