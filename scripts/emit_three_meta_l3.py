"""Emit the 3-meta L3 winning blend submission CSV.

Best from three_meta_l3_results.json:
  L3 = 0.00 × xgb_iso + 0.90 × mlp_iso + 0.10 × lr_iso  (renormed)
  blend = log_blend(LB-best 3-stack, L3, [0.40, 0.60])
  pred = argmax(log(blend) + bias[1.4324, 1.4689, 3.4008])

OOF gate result:
  blended OOF bal_acc = 0.98152
  Δ vs LB-best 4-stack 0.98084 = +0.00068
  PCR delta: L=-0.0004 / M=+0.0007 / H=+0.0018
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_xgb_metastack import (  # noqa: E402
    BIAS, build_lbbest_stack, iso_cal, _normed,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log("loading data + LB-best 3-stack anchor")
    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train[TARGET].map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int32)

    lb_oof, lb_test = build_lbbest_stack(y)

    log("loading + iso-cal'ing 3 meta-stackers")
    xgb_meta_oof = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    xgb_meta_test = _normed(np.load(ART / "test_xgb_metastack.npy"))
    xgb_iso_oof, xgb_iso_test = iso_cal(xgb_meta_oof, xgb_meta_test, y)

    mlp_meta_oof = _normed(np.load(ART / "oof_mlp_metastack.npy"))
    mlp_meta_test = _normed(np.load(ART / "test_mlp_metastack.npy"))
    mlp_iso_oof, mlp_iso_test = iso_cal(mlp_meta_oof, mlp_meta_test, y)

    lr_meta_oof = _normed(np.load(ART / "oof_lr_metastack_v2.npy"))
    lr_meta_test = _normed(np.load(ART / "test_lr_metastack_v2.npy"))
    lr_iso_oof, lr_iso_test = iso_cal(lr_meta_oof, lr_meta_test, y)

    # Winning weights from the sweep.
    w_xgb, w_mlp, w_lr = 0.00, 0.90, 0.10
    alpha = 0.60
    log(f"L3 weights: xgb={w_xgb} mlp={w_mlp} lr={w_lr}; α={alpha}")

    l3_oof = w_xgb * xgb_iso_oof + w_mlp * mlp_iso_oof + w_lr * lr_iso_oof
    l3_oof = l3_oof / l3_oof.sum(1, keepdims=True)
    l3_test = w_xgb * xgb_iso_test + w_mlp * mlp_iso_test + w_lr * lr_iso_test
    l3_test = l3_test / l3_test.sum(1, keepdims=True)

    blend_oof = log_blend([lb_oof, l3_oof], np.array([1 - alpha, alpha]))
    blend_test = log_blend([lb_test, l3_test], np.array([1 - alpha, alpha]))

    eps = 1e-12
    pred_test = (np.log(np.clip(blend_test, eps, 1.0)) + BIAS).argmax(1)

    # Sanity-check OOF reproduction.
    pred_oof = (np.log(np.clip(blend_oof, eps, 1.0)) + BIAS).argmax(1)
    from sklearn.metrics import balanced_accuracy_score
    oof_bal = balanced_accuracy_score(y, pred_oof)
    log(f"  OOF blend bal@bias = {oof_bal:.5f}  (expected 0.98152)")
    pcr = np.array([(pred_oof[y == k] == k).mean() for k in range(3)])
    log(f"  PCR: L={pcr[0]:.5f} M={pcr[1]:.5f} H={pcr[2]:.5f}")
    log(f"  errs vs y: {int((pred_oof != y).sum())}")

    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred_test]
    out_path = SUB / "submission_three_meta_l3_mlp090_lr010_a060.csv"
    sub.to_csv(out_path, index=False)
    log(f"wrote {out_path}")

    counts = pd.Series(sub[TARGET]).value_counts(normalize=True)
    log(f"  test class dist: Low={counts.get('Low', 0):.4f}  "
        f"Medium={counts.get('Medium', 0):.4f}  High={counts.get('High', 0):.4f}")
    log(f"  total rows: {len(sub):,}")
    log(f"  matches sample_submission rows: {len(sub) == len(sample)}")
    log(f"  id column equal: {bool((sub['id'].values == sample['id'].values).all())}")


if __name__ == "__main__":
    main()
