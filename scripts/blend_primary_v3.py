#!/usr/bin/env python3
"""Compound the two LB-validated submissions:
  - rawashishsin v3 standalone (NEW LB-best 0.98109, n_est=2600 faithful)
  - tier1b_greedy_meta (prior LB-best 0.98094, our 4-stack)

Both LB-validated. Blend at multiple α to find sweet spot.
Different from earlier rawashishsin direct-blend tests (those used iso-cal'd v3,
this uses v3 RAW since iso flips its calibration negative against our primary's
log-bias).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal, load_y, normed)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
INT2LABEL = {0: "Low", 1: "Medium", 2: "High"}


def per_class_recall(y, pred):
    return np.array([(pred[y == c] == c).mean() for c in range(3)])


def main():
    print("[load] y + LB-best primary + rawashishsin v3")
    y = load_y()

    # rawashishsin v3 (LB 0.98109) — RAW (no iso)
    raw_o = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    raw_t = normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))
    # iso-cal version too
    raw_o_iso, raw_t_iso = iso_cal(raw_o, raw_t, y)

    # LB-best primary (LB 0.98094): LB-3-stack + xgb_metastack_iso × 0.30
    lb3_o, lb3_t = build_lbbest_stack(y)
    mv_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    mv_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    mv_o_iso, mv_t_iso = iso_cal(mv_o, mv_t, y)
    primary_o = log_blend([lb3_o, mv_o_iso], np.array([0.7, 0.3]))
    primary_t = log_blend([lb3_t, mv_t_iso], np.array([0.7, 0.3]))

    # Standalone evaluations
    BIAS_RAW = np.array([-1.357, -1.193, 0.0])  # rawashishsin's own tuned bias
    p_raw_own = (np.log(np.clip(raw_o, 1e-12, 1)) + BIAS_RAW).argmax(1)
    bal_raw_own = balanced_accuracy_score(y, p_raw_own)

    p_primary = (np.log(np.clip(primary_o, 1e-12, 1)) + BIAS).argmax(1)
    bal_primary = balanced_accuracy_score(y, p_primary)
    pcr_primary = per_class_recall(y, p_primary)

    print(f"\nrawashishsin v3 standalone (own bias) OOF: {bal_raw_own:.6f}  → LB 0.98109")
    print(f"LB-best primary (recipe bias) OOF:        {bal_primary:.6f}  → LB 0.98094")

    # Test-side comparison
    p_raw_test = (np.log(np.clip(raw_t, 1e-12, 1)) + BIAS_RAW).argmax(1)
    p_primary_test = (np.log(np.clip(primary_t, 1e-12, 1)) + BIAS).argmax(1)
    diff_test = (p_raw_test != p_primary_test).sum()
    print(f"Test-side disagreement (LB-validated): {diff_test} / 270000 ({100*diff_test/270000:.2f}%)")

    print(f"\n{'-'*70}")
    print("Blending strategy: each pipeline at its OWN tuned bias, log-blend test softmax")
    print(f"{'-'*70}\n")
    # Build raw test softprobs at each pipeline's own bias
    # Apply bias before exp/normalize so each pipeline's bias-tuned posterior is preserved
    raw_logp = np.log(np.clip(raw_t, 1e-12, 1)) + BIAS_RAW  # rawashishsin's own bias
    raw_logp -= raw_logp.max(1, keepdims=True)
    raw_post = np.exp(raw_logp); raw_post /= raw_post.sum(1, keepdims=True)
    pri_logp = np.log(np.clip(primary_t, 1e-12, 1)) + BIAS
    pri_logp -= pri_logp.max(1, keepdims=True)
    pri_post = np.exp(pri_logp); pri_post /= pri_post.sum(1, keepdims=True)

    # OOF for evaluation (apply same bias-then-softmax)
    raw_oof_logp = np.log(np.clip(raw_o, 1e-12, 1)) + BIAS_RAW
    raw_oof_logp -= raw_oof_logp.max(1, keepdims=True)
    raw_oof_post = np.exp(raw_oof_logp); raw_oof_post /= raw_oof_post.sum(1, keepdims=True)
    pri_oof_logp = np.log(np.clip(primary_o, 1e-12, 1)) + BIAS
    pri_oof_logp -= pri_oof_logp.max(1, keepdims=True)
    pri_oof_post = np.exp(pri_oof_logp); pri_oof_post /= pri_oof_post.sum(1, keepdims=True)

    print(f"  {'α_raw':>6}  {'OOF':>8}  {'Δ vs prim':>10}  {'errs':>6}  {'recL':>7} {'recM':>7} {'recH':>7}  {'test diff'}")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]:
        # Log-blend the BIAS-CORRECTED posteriors (each pipeline's argmax-equivalent)
        w = np.array([1.0 - alpha, alpha])
        blend_o = log_blend([pri_oof_post, raw_oof_post], w)
        blend_t = log_blend([pri_post, raw_post], w)
        # Argmax — no extra bias since each input is already bias-corrected
        p_oof = blend_o.argmax(1)
        bal = balanced_accuracy_score(y, p_oof)
        pcr = per_class_recall(y, p_oof)
        errs = (p_oof != y).sum()
        # Test-side prediction
        p_test = blend_t.argmax(1)
        diff_vs_primary = (p_test != p_primary_test).sum()

        print(f"  {alpha:>6.3f}  {bal:.6f}  {bal - bal_primary:+10.5f}  {errs:>6}  {pcr[0]:.5f} {pcr[1]:.5f} {pcr[2]:.5f}  {diff_vs_primary}")

        # Save as candidate at α=0.50, 0.30, 0.40
        if alpha in [0.30, 0.40, 0.50]:
            sub = sample.copy()
            sub["Irrigation_Need"] = [INT2LABEL[p] for p in p_test]
            sub_path = SUB / f"submission_blend_primary_v3_a{int(alpha*100):03d}.csv"
            sub.to_csv(sub_path, index=False)
            print(f"    [save] {sub_path}  diff vs primary: {diff_vs_primary}")


if __name__ == "__main__":
    main()
