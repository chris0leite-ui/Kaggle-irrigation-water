"""Verify the new candidate: LB-best 3-stack × xgb_metastack__iso α=0.300.

Reports:
  - OOF tuned bal_acc, error count, per-class recall
  - Jaccard vs LB-best 3-stack
  - Jaccard vs other recent submissions
  - Test class distribution
  - Expected-LB scenario range
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
BIAS = np.array([1.4324, 1.4689, 3.4008])


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def build_lbbest_stack(y):
    r = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    rt = _normed(np.load(ART / "test_recipe_full_te.npy"))
    s1 = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1t = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7 = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7t = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm = _normed(np.load(ART / "oof_realmlp.npy"))
    rmt = _normed(np.load(ART / "test_realmlp.npy"))
    nr = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nrt = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    nr_iso_o, nr_iso_t = iso_cal(nr, nrt, y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    st1_o = log_blend([lb3_o, rm], np.array([0.8, 0.2]))
    st1_t = log_blend([lb3_t, rmt], np.array([0.8, 0.2]))
    st2_o = log_blend([st1_o, nr_iso_o], np.array([0.925, 0.075]))
    st2_t = log_blend([st1_t, nr_iso_t], np.array([0.925, 0.075]))
    return st2_o, st2_t


def main():
    train = pd.read_csv(DATA / "train.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    lb_o, lb_t = build_lbbest_stack(y)
    meta_o = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = _normed(np.load(ART / "test_xgb_metastack.npy"))
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)

    alpha = 0.300
    new_o = log_blend([lb_o, meta_iso_o], np.array([1 - alpha, alpha]))
    new_t = log_blend([lb_t, meta_iso_t], np.array([1 - alpha, alpha]))

    pred_lb_o = (np.log(np.clip(lb_o, 1e-12, 1)) + BIAS).argmax(1)
    pred_lb_t = (np.log(np.clip(lb_t, 1e-12, 1)) + BIAS).argmax(1)
    pred_new_o = (np.log(np.clip(new_o, 1e-12, 1)) + BIAS).argmax(1)
    pred_new_t = (np.log(np.clip(new_t, 1e-12, 1)) + BIAS).argmax(1)

    bal_lb = balanced_accuracy_score(y, pred_lb_o)
    bal_new = balanced_accuracy_score(y, pred_new_o)
    errs_lb = int((pred_lb_o != y).sum())
    errs_new = int((pred_new_o != y).sum())
    delta = bal_new - bal_lb

    print("=" * 70)
    print(f"LB-best 3-stack (anchor):  OOF={bal_lb:.5f}  errs={errs_lb}")
    print(f"+ meta_iso α=0.300:        OOF={bal_new:.5f}  errs={errs_new}")
    print(f"Δ OOF = {delta:+.5f}")
    print()

    # Per-class
    for tag, pred in [("LB-best", pred_lb_o), ("new blend", pred_new_o)]:
        recs = []
        for k in range(3):
            mk = y == k
            recs.append(((pred == k) & mk).sum() / max(mk.sum(), 1))
        print(f"{tag:12s}  L={recs[0]:.4f}  M={recs[1]:.4f}  H={recs[2]:.4f}  "
              f"mean={np.mean(recs):.5f}")

    # Jaccard
    errs_lb_mask = pred_lb_o != y
    errs_new_mask = pred_new_o != y
    inter = int((errs_lb_mask & errs_new_mask).sum())
    union = int((errs_lb_mask | errs_new_mask).sum())
    print(f"\nerror-Jaccard(new, LB-best) = {inter/max(union,1):.4f}")
    print(f"  LB-best-errs only new got right: {int((errs_lb_mask & ~errs_new_mask).sum())}")
    print(f"  new introduced new errors       : {int((~errs_lb_mask & errs_new_mask).sum())}")

    # Test class distribution
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub["Irrigation_Need"] = [CLASSES[i] for i in pred_new_t]

    # Compare test dist with LB-best
    sub_lb = sample.copy()
    sub_lb["Irrigation_Need"] = [CLASSES[i] for i in pred_lb_t]
    print(f"\nclass distribution (test predictions):")
    print(f"{'class':>8} {'LB-best':>8} {'new':>8} {'Δ':>6}")
    for c in CLASSES:
        n_lb = int((sub_lb["Irrigation_Need"] == c).sum())
        n_new = int((sub["Irrigation_Need"] == c).sum())
        print(f"{c:>8} {n_lb:>8} {n_new:>8} {n_new-n_lb:>+6}")

    # Difference count
    diff = int((sub["Irrigation_Need"] != sub_lb["Irrigation_Need"]).sum())
    print(f"\nrows differing from LB-best: {diff:,} ({diff/len(sub)*100:.2f}%)")

    # Save submission
    path = SUB / "submission_tier1b_greedy_meta.csv"
    sub.to_csv(path, index=False)
    print(f"\nwrote {path}")

    # Expected LB scenarios
    lb_lb_best = 0.98008
    lb_hist_gap_median = 0.00053  # from LB-best's gap
    print("\n=== expected LB scenarios ===")
    print(f"if OOF→LB gap matches LB-best's +{lb_hist_gap_median:.5f}:  "
          f"LB ≈ {bal_new - lb_hist_gap_median:.5f}")
    print(f"if gap stays at +0.00053 but OOF Δ fully transfers: "
          f"LB ≈ {lb_lb_best + delta:.5f}")
    print(f"if gap inflates to +0.00080 (typical for meta-stacks): "
          f"LB ≈ {bal_new - 0.00080:.5f}")

    summary = dict(
        lb_best_oof=float(bal_lb), lb_best_errs=errs_lb,
        new_oof=float(bal_new), new_errs=errs_new,
        delta=float(delta), alpha=alpha,
        jaccard=float(inter / max(union, 1)),
        rows_diff_from_lb=diff,
    )
    (ART / "tier1b_verify_meta_iso_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
