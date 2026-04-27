"""Build and verify the B candidate submission CSV.

Recipe:
  L3 = 0.5 × XGB_meta_iso + 0.5 × MLP_meta_iso  (in prob space, renormed)
  blend = log_blend(LB_best_3stack, L3, [0.5, 0.5])
  pred = argmax(log(blend) + bias[1.4324, 1.4689, 3.4008])

OOF gate-pass diagnostic:
  blended OOF bal_acc = 0.98118
  Δ vs LB-best 4-stack 0.98084 = +0.00033
  PCR delta: L=-7e-5 / M=+3.6e-4 / H=+7.1e-4 (positive on rare class)
  errs 9341 vs anchor 9415

Outputs:
  submissions/submission_meta_l3_xgb_mlp_blend_a050.csv

DOES NOT run kaggle submit — that requires a separate explicit invocation
per the CLAUDE.md "ALWAYS ASK FIRST" rule.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_xgb_metastack import (  # noqa: E402
    build_lbbest_stack, iso_cal, _normed,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
BIAS = np.array([1.4324, 1.4689, 3.4008])
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log("loading data + LB-best 3-stack anchor")
    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train[TARGET].map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int32)

    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best 3-stack test rows: {lb_test.shape}")

    # XGB meta on disk; iso-cal vs y.
    log("loading + iso-cal'ing XGB meta-stacker (committed at tier1b)")
    xgb_meta_oof = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    xgb_meta_test = _normed(np.load(ART / "test_xgb_metastack.npy"))
    xgb_iso_oof, xgb_iso_test = iso_cal(xgb_meta_oof, xgb_meta_test, y)

    # MLP meta on disk; iso-cal vs y.
    log("loading + iso-cal'ing MLP meta-stacker (this session)")
    mlp_meta_oof = _normed(np.load(ART / "oof_mlp_metastack.npy"))
    mlp_meta_test = _normed(np.load(ART / "test_mlp_metastack.npy"))
    mlp_iso_oof, mlp_iso_test = iso_cal(mlp_meta_oof, mlp_meta_test, y)

    # L3 weighted average (W_MLP=0.5, arithmetic per cdeotte) + renorm.
    log("L3 weighted average (W_MLP=0.5)")
    l3_test = 0.5 * xgb_iso_test + 0.5 * mlp_iso_test
    l3_test = l3_test / l3_test.sum(axis=1, keepdims=True)

    # Blend into LB-best 3-stack at α=0.50 (log-blend).
    log("blending into LB-best 3-stack at α=0.50 (log-blend)")
    test_blend = log_blend([lb_test, l3_test], np.array([0.5, 0.5]))

    # Apply fixed bias + argmax.
    eps = 1e-12
    pred = (np.log(np.clip(test_blend, eps, 1.0)) + BIAS).argmax(1)

    # Sanity-check OOF blend matches the gate result.
    l3_oof = 0.5 * xgb_iso_oof + 0.5 * mlp_iso_oof
    l3_oof = l3_oof / l3_oof.sum(axis=1, keepdims=True)
    oof_blend = log_blend([lb_oof, l3_oof], np.array([0.5, 0.5]))
    oof_pred = (np.log(np.clip(oof_blend, eps, 1.0)) + BIAS).argmax(1)
    from sklearn.metrics import balanced_accuracy_score
    oof_bal = balanced_accuracy_score(y, oof_pred)
    log(f"  OOF blend bal@bias = {oof_bal:.5f}  (expected ~0.98118)")
    pcr = np.array([(oof_pred[y == k] == k).mean() for k in range(3)])
    log(f"  PCR: L={pcr[0]:.5f} M={pcr[1]:.5f} H={pcr[2]:.5f}")
    log(f"  errs vs y: {int((oof_pred != y).sum())}")

    # Build submission CSV.
    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred]
    out_path = SUB / "submission_meta_l3_xgb_mlp_blend_a050.csv"
    sub.to_csv(out_path, index=False)
    log(f"wrote {out_path}")

    # Class distribution sanity-check.
    counts = pd.Series(sub[TARGET]).value_counts(normalize=True)
    log(f"  test class dist: Low={counts.get('Low', 0):.4f}  "
        f"Medium={counts.get('Medium', 0):.4f}  High={counts.get('High', 0):.4f}")
    log(f"  total rows: {len(sub):,}")
    log(f"  matches sample_submission rows: {len(sub) == len(sample)}")
    log(f"  id column equal: {bool((sub['id'].values == sample['id'].values).all())}")


if __name__ == "__main__":
    main()
