"""Apply per-class isotonic calibration to v8b's OOF + test, then run
the 4-gate analyzer.

Hypothesis: iso-cal smooths per-class probability scales. v8b's
G4 RESHUFFLE failure (ratio 0.09 = high churn relative to net change)
may be caused by per-row prob-scale noise that iso-cal smooths out,
potentially preserving net_H direction while reducing churn → higher
G4 ratio.

Mechanism distinct from prior nulls: iso-cal on the v8b output is
NOT bank-extension or model retrain. Just per-class monotonic
remapping. Risk: flips direction back to REMOVE.

Outputs: oof_xgb_metastack_v8b_iso.npy + test + 4-gate JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tier1b_helpers import ART, BIAS, bal_at_bias, load_y, normed  # noqa: E402

EPS = 1e-12


def main():
    y = load_y()
    oof = np.load(ART / "oof_xgb_metastack_v8b.npy").astype(np.float32)
    test = np.load(ART / "test_xgb_metastack_v8b.npy").astype(np.float32)
    print(f"v8b OOF shape: {oof.shape}, test shape: {test.shape}")
    print(f"v8b @ recipe-bias = {bal_at_bias(normed(oof), y):.5f}")

    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    oo = normed(oo)
    tt = normed(tt)
    print(f"v8b_iso @ recipe-bias = {bal_at_bias(oo, y):.5f}")
    print(f"  Δ vs v8b raw = {bal_at_bias(oo, y) - bal_at_bias(normed(oof), y):+.5f}")

    np.save(ART / "oof_xgb_metastack_v8b_iso.npy", oo)
    np.save(ART / "test_xgb_metastack_v8b_iso.npy", tt)
    print(f"\nWrote oof_xgb_metastack_v8b_iso.npy + test")


if __name__ == "__main__":
    main()
