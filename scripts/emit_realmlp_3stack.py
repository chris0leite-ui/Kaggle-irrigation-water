"""Emit the 3-component candidate found by greedy_realmlp_refit.py:
  anchor = LB-best 3-way (recipe 0.25 + pseudo_s1 0.35 + pseudo_s7 0.40)
  step 1: + realmlp α=0.200           (OOF 0.98029 -> 0.98047)
  step 2: + xgb_nonrule__iso α=0.075  (OOF 0.98047 -> 0.98061)

Reports:
  - final OOF at recipe's fixed bias [1.4324, 1.4689, 3.4008]
  - error count
  - Jaccard vs LB-best 3-way and vs recipe
  - per-class recall breakdown

Writes submission to submissions/submission_lb3_realmlp_nonruleiso.csv
for human review. NO automated LB probe.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"
BIAS = np.array([1.4324, 1.4689, 3.4008])


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    oo = oo / np.clip(oo.sum(1, keepdims=True), 1e-9, None)
    tt = tt / np.clip(tt.sum(1, keepdims=True), 1e-9, None)
    return oo, tt


def _normed(arr):
    return arr / np.clip(arr.sum(1, keepdims=True), 1e-9, None)


def _pred(probs, bias=BIAS):
    return (np.log(np.clip(probs, 1e-12, 1.0)) + bias).argmax(1)


def _errmask(probs, y, bias=BIAS):
    return _pred(probs, bias) != y


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    # Load components (normalized for log_blend stability).
    recipe_oof = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    recipe_test = _normed(np.load(ART / "test_recipe_full_te.npy"))
    pseudo_s1_oof = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    pseudo_s1_test = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    pseudo_s7_oof = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    pseudo_s7_test = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    realmlp_oof = _normed(np.load(ART / "oof_realmlp.npy"))
    realmlp_test = _normed(np.load(ART / "test_realmlp.npy"))
    nonrule_oof = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nonrule_test = _normed(np.load(ART / "test_xgb_nonrule.npy"))

    # Isotonic-calibrate xgb_nonrule per class using y labels as target.
    nonrule_iso_oof, nonrule_iso_test = iso_cal(nonrule_oof, nonrule_test, y)

    # Step 0: LB-best 3-way (anchor).
    lb3_oof = log_blend(
        [recipe_oof, pseudo_s1_oof, pseudo_s7_oof],
        np.array([0.25, 0.35, 0.40]),
    )
    lb3_test = log_blend(
        [recipe_test, pseudo_s1_test, pseudo_s7_test],
        np.array([0.25, 0.35, 0.40]),
    )
    bal_lb3 = balanced_accuracy_score(y, _pred(lb3_oof))
    errs_lb3 = int(_errmask(lb3_oof, y).sum())
    print(f"LB-best 3-way             OOF={bal_lb3:.5f}  errs={errs_lb3}")

    # Step 1: + realmlp α=0.200 (log-space).
    stack1_oof = log_blend([lb3_oof, realmlp_oof], np.array([0.8, 0.2]))
    stack1_test = log_blend([lb3_test, realmlp_test], np.array([0.8, 0.2]))
    bal_s1 = balanced_accuracy_score(y, _pred(stack1_oof))
    errs_s1 = int(_errmask(stack1_oof, y).sum())
    print(f"  + realmlp α=0.200        OOF={bal_s1:.5f}  errs={errs_s1}  "
          f"Δ={bal_s1 - bal_lb3:+.5f}")

    # Step 2: + nonrule__iso α=0.075.
    stack2_oof = log_blend([stack1_oof, nonrule_iso_oof],
                           np.array([0.925, 0.075]))
    stack2_test = log_blend([stack1_test, nonrule_iso_test],
                            np.array([0.925, 0.075]))
    bal_s2 = balanced_accuracy_score(y, _pred(stack2_oof))
    errs_s2 = int(_errmask(stack2_oof, y).sum())
    print(f"  + nonrule_iso α=0.075    OOF={bal_s2:.5f}  errs={errs_s2}  "
          f"Δ={bal_s2 - bal_s1:+.5f}")
    print(f"  TOTAL Δ vs LB3          = {bal_s2 - bal_lb3:+.5f}")

    # Diagnostics.
    print()
    errs_final = _errmask(stack2_oof, y)
    errs_lb3_mask = _errmask(lb3_oof, y)
    inter = int((errs_final & errs_lb3_mask).sum())
    union = int((errs_final | errs_lb3_mask).sum())
    jacc_lb3 = inter / max(union, 1)
    print(f"Jaccard(stack, LB3)        = {jacc_lb3:.4f}")

    recipe_errs_mask = _errmask(recipe_oof, y)
    inter_r = int((errs_final & recipe_errs_mask).sum())
    union_r = int((errs_final | recipe_errs_mask).sum())
    jacc_recipe = inter_r / max(union_r, 1)
    print(f"Jaccard(stack, recipe)     = {jacc_recipe:.4f}")

    # Per-class recall.
    pred = _pred(stack2_oof)
    print(f"\nper-class recall at recipe bias:")
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = y == cls_idx
        recall = ((pred == cls_idx) & mask).sum() / max(mask.sum(), 1)
        print(f"  {cls_name:7s} recall = {recall:.4f}  "
              f"(n={int(mask.sum())}, correct={int(((pred == cls_idx) & mask).sum())})")

    # Emit submission.
    sample = pd.read_csv(DATA / "sample_submission.csv")
    pred_test = _pred(stack2_test)
    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred_test]
    path = SUB / "submission_lb3_realmlp_nonruleiso.csv"
    sub.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print(f"class counts: {sub[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
