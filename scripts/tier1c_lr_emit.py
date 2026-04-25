"""Build LB-best-4-stack-anchored submissions for the LR meta-stacker.

The original tier1c_lr_metastack.py emitted vs the LB-best 3-stack
(OOF 0.98061). The current LB best is the 4-stack (OOF 0.98084 / LB 0.98094);
we emit anchored on that instead. Three candidates for review:
  - α=0.30 (conservative, all PCR within -0.0005 guardrail)
  - α=0.50 (moderate, Low -0.0003 still in guardrail)
  - α=0.65 (aggressive, Low -0.0005 borderline; OOF Δ +0.00098 best inside guardrail)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, SUB, TARGET,
    build_lbbest_stack, iso_cal, log,
)


def main():
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building LB-best 4-stack (primary, OOF 0.98084 / LB 0.98094)")
    lb3_o, lb3_t = build_lbbest_stack(y)
    xgb_oof = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    xgb_test = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    xgb_iso_o, xgb_iso_t = iso_cal(xgb_oof, xgb_test, y)
    lb4_o = log_blend([lb3_o, xgb_iso_o], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, xgb_iso_t], np.array([0.7, 0.3]))

    log("loading LR meta-stacker, iso-cal")
    lr_oof = np.load(ART / "oof_lr_metastack.npy").astype(np.float32)
    lr_test = np.load(ART / "test_lr_metastack.npy").astype(np.float32)
    lr_iso_o, lr_iso_t = iso_cal(lr_oof, lr_test, y)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    for a in (0.30, 0.50, 0.65):
        test_blend = log_blend([lb4_t, lr_iso_t], np.array([1 - a, a]))
        pred_t = (np.log(np.clip(test_blend, 1e-12, 1)) + BIAS).argmax(1)
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_t]
        # Class-distribution diagnostic
        cnt = np.bincount(pred_t, minlength=3)
        log(f"  α={a:.2f}  test classes [L {cnt[0]} M {cnt[1]} H {cnt[2]}]")
        path = SUB / f"submission_tier1c_lr_iso_4stack_a{int(a*100):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"    wrote {path}")


if __name__ == "__main__":
    main()
