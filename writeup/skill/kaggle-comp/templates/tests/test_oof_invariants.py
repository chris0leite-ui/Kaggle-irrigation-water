"""Smoke tests for committed OOF / test artifacts.

Guards against silent regression if a producing script drifts (wrong
seed, reordered classes, changed fold split). Run: `pytest tests/`.

Usage: drop a new artifact into scripts/artifacts/ and add an entry to
EXPECTED below. The test then validates shape, finiteness, and that
rows are valid probability vectors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ART = Path(__file__).parent.parent / "scripts" / "artifacts"

# name -> dict(n_train, n_test, n_class, sparse_carrier=False)
# sparse carriers zero-fill rows outside their domain; dense OOFs must
# have all rows summing to ~1.
# Populate after the first baseline run (kickoff Day 1).
EXPECTED: dict[str, dict] = {
    # "baseline_lgbm": {"n_train": ?, "n_test": ?, "n_class": ?},
}


@pytest.mark.parametrize("name,spec", list(EXPECTED.items()))
def test_oof_artifact(name, spec):
    oof_path = ART / f"oof_{name}.npy"
    test_path = ART / f"test_{name}.npy"
    if not oof_path.exists():
        pytest.skip(f"{oof_path.name} not present")

    oof = np.load(oof_path)
    test = np.load(test_path)

    assert oof.shape == (spec["n_train"], spec["n_class"]), \
        f"{name} oof shape {oof.shape}"
    assert test.shape == (spec["n_test"], spec["n_class"]), \
        f"{name} test shape {test.shape}"
    assert np.isfinite(oof).all(), f"{name} oof has non-finite entries"
    assert np.isfinite(test).all(), f"{name} test has non-finite entries"

    row_sums = oof.sum(1)
    if spec.get("sparse_carrier", False):
        nz = row_sums > 0
        assert np.allclose(row_sums[nz], 1.0, atol=1e-5), \
            f"{name} non-zero rows don't sum to 1"
    else:
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f"{name} rows don't sum to 1"
        # argmax sanity: no class collapses to 0% or 100%
        counts = np.bincount(oof.argmax(1), minlength=spec["n_class"]) / len(oof)
        assert (counts > 0.001).all() and (counts < 0.999).all(), \
            f"{name} degenerate argmax distribution: {counts}"


def test_fold_convention_import():
    """common.py exports the pinned fold convention."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from common import SEED, N_FOLDS  # noqa: E402
    assert SEED == 42
    assert N_FOLDS == 5
