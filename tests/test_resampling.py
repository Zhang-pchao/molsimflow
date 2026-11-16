"""Seed, block, tie and missing-observation contracts."""

import numpy as np
import pytest
from molsimflow.postprocess.resampling import (
    residualize,
    spearman,
    bh_qvalues,
    block_bootstrap_partial_spearman,
)


def test_rank_ties_missingness_and_affine_residuals():
    assert spearman([1, 1, 2, np.nan], [3, 3, 4, 100]) == pytest.approx(1)
    np.testing.assert_allclose(residualize(2 * np.arange(8) + 3, np.arange(8)), 0, atol=1e-13)
    assert np.isnan(residualize([1, 2], [0, 1])).all()
    with pytest.raises(ValueError):
        residualize([1, 2], [1])


def test_fdr_known_values_and_permutations():
    p = np.array([0.04, 0.001, 0.03, 0.02])
    expected = [0.04, 0.004, 0.04, 0.04]
    np.testing.assert_allclose(bh_qvalues(p), expected)
    permutation = [2, 0, 3, 1]
    np.testing.assert_allclose(bh_qvalues(p[permutation]), np.asarray(expected)[permutation])
    assert bh_qvalues([]).size == 0
    with pytest.raises(ValueError):
        bh_qvalues([0.2, np.nan])


def test_block_bootstrap_is_reproducible_with_unequal_blocks():
    rng = np.random.default_rng(1)
    control = rng.normal(size=24)
    x = control + rng.normal(size=24)
    y = 2 * control + x + rng.normal(size=24)
    labels = np.repeat([10, 20, 30], [3, 7, 14])
    a = block_bootstrap_partial_spearman(x, y, control, labels, samples=80, seed=7)
    b = block_bootstrap_partial_spearman(x, y, control, labels, samples=80, seed=7)
    assert a == b and a["blocks"] == 3 and a["bootstrap_valid"] == 80
    with pytest.raises(ValueError):
        block_bootstrap_partial_spearman(x, y, control, labels, samples=0)
