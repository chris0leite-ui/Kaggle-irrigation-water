"""Emit submission CSV for v3 iso-blend at α=0.30 (peak Δ=+0.00015 OOF).

Below the +2e-4 internal gate the script enforces, but submitted
explicitly per user direction to validate whether OOF→LB transfer
follows the v1 meta-stacker pattern (+0.00023 OOF → +0.00086 LB).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
ALPHA = 0.30


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def main():
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    train_y = pd.read_csv(DATA / "train.csv")[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    # Reconstruct LB-best 4-stack test side
    r = _normed(np.load(ART / "test_recipe_full_te.npy"))
    s1 = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7 = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm = _normed(np.load(ART / "test_realmlp.npy"))

    # nonrule iso-cal needs to be built from oof + test against y
    nr_oof = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nr_test = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    from sklearn.isotonic import IsotonicRegression
    nr_iso_t = np.zeros_like(nr_test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(nr_oof[:, c], (train_y == c).astype(np.float32))
        nr_iso_t[:, c] = ir.predict(nr_test[:, c])
    nr_iso_t = _normed(nr_iso_t)

    w3 = np.array([0.25, 0.35, 0.40])
    lb3_t = log_blend([r, s1, s7], w3)
    s1_t = log_blend([lb3_t, rm], np.array([0.8, 0.2]))
    lb_test = log_blend([s1_t, nr_iso_t], np.array([0.925, 0.075]))

    # Load v3 iso meta-stacker test
    meta_iso_t = _normed(np.load(ART / "test_xgb_metastack_v3_iso.npy"))

    # Final blend at α=0.30
    blend_t = log_blend([lb_test, meta_iso_t], np.array([1 - ALPHA, ALPHA]))
    pred_idx = (np.log(np.clip(blend_t, 1e-12, 1)) + BIAS).argmax(1)

    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred_idx]
    out = SUB / f"submission_metastack_v3_iso_a{int(ALPHA * 1000):03d}.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out}")
    print(f"  shape: {sub.shape}")
    print(f"  dist: {dict(sub[TARGET].value_counts())}")


if __name__ == "__main__":
    main()
