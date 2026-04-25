"""J7: Conformal-gated overrides on score=6 missed-High boundary.

Mechanism: split-conformal calibration of the score=6 missed-High detector
(spec6_mh_v2, AUC 0.938, see CLAUDE.md 2026-04-25 entry). Instead of an ad-
hoc theta sweep, pick threshold τ such that calibrated precision >= 8.1%
break-even (under macro-recall) with confidence >= 90%.

Per CLAUDE.md, prior theta=0.15 sweep gave 28% OOF precision but only 2
test overrides. Conformal provides PRINCIPLED threshold selection that
guarantees coverage; the goal here is to determine if the lever is fully
closed (no τ produces both precision >= break-even AND a useful number of
test overrides) or has small residual headroom.
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
from tier1b_helpers import (BIAS, ART, SUB, DATA, SEED, N_FOLDS, CLS2IDX,
                            CLASSES, TARGET, normed, iso_cal,
                            build_lbbest_stack, load_y, bal_at_bias)  # noqa: E402

# Score=6 ∩ teacher_argmax=Medium is the override domain; ~330 truly-H per
# CLAUDE.md error analysis. Break-even precision under macro-recall ~8.1%.
BREAKEVEN_PREC = 0.081
CONFORMAL_CONF = 0.90  # 90% lower bound on precision


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def conformal_threshold(scores, labels, target_prec, conf):
    """Smallest τ such that lower CI on precision >= target_prec at conf level.

    For binary detector with score in [0,1] and label in {0,1}: try thresholds
    in descending order. For each, count overrides (n) and correct (k). The
    Wilson lower bound on prec p̂=k/n at confidence conf must be >= target_prec.
    Returns (τ, n, k, lower_ci) for the smallest τ that satisfies.
    """
    z = 1.6449  # 90% one-sided
    order = np.argsort(-scores)  # descending
    s_sorted = scores[order]
    l_sorted = labels[order]
    cum_correct = np.cumsum(l_sorted)
    n = np.arange(1, len(scores) + 1)
    p_hat = cum_correct / n
    # Wilson lower bound:
    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))) / denom
    lower = center - margin
    valid = lower >= target_prec
    if not valid.any():
        return None
    # Take the LARGEST n with lower >= target_prec (loosest threshold).
    idxs = np.where(valid)[0]
    pick = idxs[-1]
    tau = s_sorted[pick]
    return float(tau), int(n[pick]), int(cum_correct[pick]), float(lower[pick])


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    # --- load detector ---
    p_oof = np.load(ART / "oof_spec6_mh_v2.npy").astype(np.float32)
    p_test = np.load(ART / "test_spec6_mh_v2.npy").astype(np.float32)
    log(f"loaded spec6_mh_v2: oof shape={p_oof.shape} test shape={p_test.shape}")

    # --- domain: score=6 ∩ teacher_argmax=Medium ---
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    score_tr = tr_d["dgp_score"].to_numpy()
    score_te = te_d["dgp_score"].to_numpy()

    log("building LB-best 4-stack teacher")
    lb3_o, lb3_t = build_lbbest_stack(y)
    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    v1_iso_o, v1_iso_t = iso_cal(v1_o, v1_t, y)
    from common import log_blend
    teacher_o = log_blend([lb3_o, v1_iso_o], np.array([0.7, 0.3]))
    teacher_t = log_blend([lb3_t, v1_iso_t], np.array([0.7, 0.3]))
    teacher_argmax_tr = (np.log(np.clip(teacher_o, 1e-12, 1)) + BIAS).argmax(1)
    teacher_argmax_te = (np.log(np.clip(teacher_t, 1e-12, 1)) + BIAS).argmax(1)

    # train override domain: score=6 ∩ teacher pred Medium
    in_dom_tr = (score_tr == 6) & (teacher_argmax_tr == 1)
    in_dom_te = (score_te == 6) & (teacher_argmax_te == 1)
    n_dom_tr = in_dom_tr.sum()
    n_dom_te = in_dom_te.sum()
    truly_H_tr = ((y == 2) & in_dom_tr).sum()
    log(f"override domain (train): {n_dom_tr} rows; truly-High = {truly_H_tr}")
    log(f"override domain (test):  {n_dom_te} rows")
    log(f"break-even precision: {BREAKEVEN_PREC:.4f}")

    # --- split-conformal calibration ---
    # Hold out 30% of in-domain train rows as calibration set.
    rng = np.random.default_rng(SEED)
    dom_idx = np.where(in_dom_tr)[0]
    rng.shuffle(dom_idx)
    n_cal = int(0.3 * len(dom_idx))
    cal_idx = dom_idx[:n_cal]
    rest_idx = dom_idx[n_cal:]

    cal_scores = p_oof[cal_idx]
    cal_labels = (y[cal_idx] == 2).astype(np.int32)
    log(f"calibration set: {len(cal_idx)} rows ({cal_labels.sum()} truly-H)")

    res = conformal_threshold(cal_scores, cal_labels, BREAKEVEN_PREC, CONFORMAL_CONF)
    if res is None:
        log(f"\nNO conformal threshold satisfies precision >= {BREAKEVEN_PREC} at "
            f"{CONFORMAL_CONF*100:.0f}% confidence. Lever fully closed.")
        out = dict(n_dom_tr=int(n_dom_tr), n_dom_te=int(n_dom_te),
                   truly_H_tr=int(truly_H_tr), tau=None, deployable=False,
                   note="no τ satisfies break-even precision at calibration confidence")
        (ART / "j7_conformal_spec6_results.json").write_text(json.dumps(out, indent=2))
        return

    tau, n_picked, k_picked, lower_ci = res
    log(f"\nConformal threshold: τ={tau:.5f}")
    log(f"  calibration set: {n_picked} overrides, {k_picked} correct "
        f"(precision {k_picked/n_picked:.4f}, Wilson lower {lower_ci:.4f})")

    # --- evaluate on rest of train (out-of-calibration) ---
    rest_overrides = (p_oof[rest_idx] >= tau) & in_dom_tr[rest_idx]
    n_rest_over = rest_overrides.sum()
    correct_rest = (rest_overrides & (y[rest_idx] == 2)).sum()
    rest_prec = correct_rest / max(n_rest_over, 1)
    log(f"  out-of-cal train: {n_rest_over} overrides, {correct_rest} correct "
        f"({rest_prec:.4f})")

    # --- evaluate macro-recall lift on full train OOF ---
    pred_post = teacher_argmax_tr.copy()
    flip_mask = in_dom_tr & (p_oof >= tau)
    pred_post[flip_mask] = 2  # override to High
    n_flips = flip_mask.sum()
    correct_flips = (flip_mask & (y == 2)).sum()
    log(f"\n  full-train OOF: {n_flips} overrides, {correct_flips} correct "
        f"(precision {correct_flips/max(n_flips, 1):.4f})")

    bal_pre = balanced_accuracy_score(y, teacher_argmax_tr)
    bal_post = balanced_accuracy_score(y, pred_post)
    log(f"  macro-recall pre={bal_pre:.5f} post={bal_post:.5f} Δ={bal_post-bal_pre:+.5f}")

    # --- test-side override count ---
    test_flip_mask = in_dom_te & (p_test >= tau)
    n_test_flips = test_flip_mask.sum()
    log(f"  test-side override count: {n_test_flips}")

    # --- gate ---
    delta_oof = bal_post - bal_pre
    gate_pass = (delta_oof >= 2e-4) and (n_test_flips >= 10)
    log(f"\nGATE: Δ_OOF >= +2e-4 AND test_flips >= 10 → {gate_pass}")
    if gate_pass:
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        post_te = teacher_argmax_te.copy()
        post_te[test_flip_mask] = 2
        sub[TARGET] = [CLASSES[i] for i in post_te]
        path = SUB / "submission_j7_conformal_spec6.csv"
        sub.to_csv(path, index=False)
        log(f"emitted {path} (AWAITS USER LB APPROVAL)")
    else:
        log(f"gate fails — lever closed.")

    out = dict(
        tau=float(tau), n_picked_cal=int(n_picked), k_picked_cal=int(k_picked),
        wilson_lower=float(lower_ci),
        n_train_overrides=int(n_flips), correct_train_overrides=int(correct_flips),
        train_precision=float(correct_flips/max(n_flips, 1)),
        bal_pre=float(bal_pre), bal_post=float(bal_post), delta_oof=float(delta_oof),
        n_test_overrides=int(n_test_flips), gate_pass=bool(gate_pass),
        elapsed_sec=float(time.time()-t0),
    )
    (ART / "j7_conformal_spec6_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote j7_conformal_spec6_results.json")


if __name__ == "__main__":
    main()
