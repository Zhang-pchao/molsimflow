"""Sampling-contract tests for reusable time-series diagnostics."""

import numpy as np
import pytest
from molsimflow.postprocess.timeseries import (
    autocorrelation_fft,
    normalized_autocorrelation,
    positive_acf_time_ps,
    statistical_inefficiency,
)


def test_fft_matches_overlap_normalized_direct_reference():
    values = np.random.default_rng(8).normal(size=251)
    np.testing.assert_allclose(
        autocorrelation_fft(values), normalized_autocorrelation(values, 250), atol=1e-13
    )


def test_truncation_definitions_are_explicit():
    acf = np.array([1, 0.5, -0.1, 0.2, -0.1])
    assert statistical_inefficiency(acf) == 2
    assert statistical_inefficiency(acf, method="initial_positive_pair") == 2
    assert positive_acf_time_ps(acf, 0.1) == 0.1
    assert statistical_inefficiency([1, 0.5, -0.3], method="initial_positive_pair") == 1.4


def test_legacy_aliases_and_constant_missing_policies():
    from molsimflow.postprocess.constant_force_kinematics import normalized_autocorrelation as old

    assert old is normalized_autocorrelation
    np.testing.assert_array_equal(autocorrelation_fft([3, 3, 3]), [1, 0, 0])
    assert np.isnan(old([3, 3, 3], 2)[1:]).all()
    np.testing.assert_array_equal(old([1, np.nan, 2], 1), old([1, 2], 1))
    with pytest.raises(ValueError):
        autocorrelation_fft([1, np.nan, 2])


@pytest.mark.parametrize("values", [[], [1], [[1, 2]], [float("inf"), 1]])
def test_invalid_fft_inputs(values):
    with pytest.raises(ValueError):
        autocorrelation_fft(values)


def test_empirical_overlap_normalizes_unequal_counts_and_keeps_common_support():
    from molsimflow.postprocess.umbrella_wham import histogram_overlap

    assert histogram_overlap([0, 0, 1], [1, 1], [0, 0.5, 1]) == pytest.approx(1 / 3)
    assert histogram_overlap([0], [1], [0, 0.5, 1]) == 0
    with pytest.raises(ValueError):
        histogram_overlap([0], [2], [0, 1])
