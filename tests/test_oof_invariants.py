"""Smoke tests for committed OOF/test artefacts.

Guards against silent regression if a producing script drifts (wrong
seed, reordered classes, changed fold split). Run: `pytest tests/`.

Expected committed artefacts and their conventions are documented in
OOFS.md and scripts/common.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ART = Path(__file__).parent.parent / "scripts" / "artifacts"

# name -> (is_sparse_carrier, expected_zero_row_frac_range)
# sparse carriers zero-fill rows outside their domain; dense OOFs must
# have all rows summing to ~1.
EXPECTED = {
    "xgb_vanilla_dist":      (False, None),
    "xgb_dist_routed_v3":    (False, None),
    "xgb_nonrule":           (False, None),
    "xgb_bin_high":          (False, None),
    "lgbm_te_orig":          (False, None),
    "hybrid_lgbmxgb_blend":  (False, None),
    "hybrid_binhigh":        (False, None),
    "greedy_blend":          (False, None),
    "xgb_spec_678":          (True, (0.85, 0.95)),   # ~91% non-spec rows zero
}


@pytest.mark.parametrize("name,is_sparse,zero_frac_range", [
    (n, s, r) for n, (s, r) in EXPECTED.items()
])
def test_oof_artifact(name, is_sparse, zero_frac_range):
    oof_path = ART / f"oof_{name}.npy"
    test_path = ART / f"test_{name}.npy"
    if not oof_path.exists():
        pytest.skip(f"{oof_path.name} not present (run regeneration chain)")

    oof = np.load(oof_path)
    test = np.load(test_path)

    assert oof.shape == (630_000, 3), f"{name} oof shape {oof.shape}"
    assert test.shape == (270_000, 3), f"{name} test shape {test.shape}"
    assert np.isfinite(oof).all(), f"{name} oof has non-finite entries"
    assert np.isfinite(test).all(), f"{name} test has non-finite entries"

    row_sums = oof.sum(1)
    zero_frac = float((row_sums == 0).mean())

    if is_sparse:
        lo, hi = zero_frac_range
        assert lo <= zero_frac <= hi, (
            f"{name} is a sparse carrier: zero-row frac {zero_frac:.4f} "
            f"outside expected [{lo}, {hi}]"
        )
        # non-zero rows must still be valid probability vectors
        nz = row_sums > 0
        assert np.allclose(row_sums[nz], 1.0, atol=1e-5), (
            f"{name} non-zero rows don't sum to 1"
        )
    else:
        assert zero_frac == 0.0, f"{name} dense OOF has {zero_frac:.4%} zero rows"
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"{name} rows don't sum to 1 (min={row_sums.min()}, max={row_sums.max()})"
        )

    # argmax class distribution sanity: no class can collapse to 0 or 100%
    # (even the rule predicts all three classes).
    if not is_sparse:
        counts = np.bincount(oof.argmax(1), minlength=3) / len(oof)
        assert (counts > 0.01).all() and (counts < 0.99).all(), (
            f"{name} degenerate argmax distribution: {counts}"
        )


def test_fold_convention_import():
    """common.py exports the pinned fold convention."""
    from scripts.common import SEED, N_FOLDS, CLASSES
    assert SEED == 42
    assert N_FOLDS == 5
    assert CLASSES == ("Low", "Medium", "High")
