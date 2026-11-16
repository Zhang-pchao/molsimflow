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


def test_bootstrap_mean_preserves_facade_draws_and_generator_state():
    from molsimflow.postprocess.resampling import bootstrap_mean
    from molsimflow.postprocess.tpcl_renewal_rewiring import bootstrap_mean as legacy

    assert legacy is bootstrap_mean
    values = np.array([1.0, 1.0, 3.0, 9.0])
    expected_rng = np.random.default_rng(12)
    actual_rng = np.random.default_rng(12)
    draws = expected_rng.choice(values, size=(101, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    assert bootstrap_mean(values, 101, actual_rng) == (float(values.mean()), float(low), float(high))
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


@pytest.mark.parametrize("values", [np.array([4.0]), np.array([1.0, np.nan])])
def test_bootstrap_mean_keeps_single_unit_and_nonfinite_behavior(values):
    from molsimflow.postprocess.resampling import bootstrap_mean

    result = bootstrap_mean(values, 100, np.random.default_rng(2))
    if np.isfinite(values).all():
        assert result == (4.0, 4.0, 4.0)
    else:
        assert np.isnan(result).all()


def test_bootstrap_mean_keeps_empty_input_error():
    from molsimflow.postprocess.resampling import bootstrap_mean

    with pytest.raises(ValueError, match="empty block series"):
        bootstrap_mean(np.array([]), 100, np.random.default_rng(2))
