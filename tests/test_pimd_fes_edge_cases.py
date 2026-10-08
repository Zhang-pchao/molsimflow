"""Numerical and support regressions for complete-frame PIMD estimators."""

import numpy as np
import pytest

from molsimflow.postprocess.pimd_fes import (
    normalized_log_weights,
    quantum_fes_1d,
    quantum_histogram_masses,
)
from molsimflow.postprocess.pimd_reweight import (
    normalized_log_weights as workflow_log_weights,
)


@pytest.mark.parametrize("normalize", [normalized_log_weights, workflow_log_weights])
@pytest.mark.parametrize("offset", [-1e16, 1e16])
def test_log_weights_preserve_normalization_under_large_constant_offset(normalize, offset):
    # Equal finite energies must give a uniform distribution, regardless of zero.
    result = normalize(np.full(3, offset))
    np.testing.assert_allclose(result, np.full(3, -np.log(3)), rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.exp(result).sum(), 1, rtol=0, atol=1e-15)


def test_weighted_conditional_handles_a_zero_mass_conditioning_bin():
    result = quantum_histogram_masses(
        [[0, 0], [1, 1]],
        [-1000, 0],
        [-0.5, 0.5, 1.5],
        conditioning=[0, 1],
        conditioning_edges=[-0.5, 0.5, 1.5],
    )
    # The first finite log weight underflows to zero after normalization.
    np.testing.assert_array_equal(result["direct"], [0, 1])
    np.testing.assert_array_equal(result["conditional"], result["direct"])


def test_probability_fes_survives_disjoint_bead_support():
    result = quantum_fes_1d(
        [[0, 1], [0, 1]], [0, 0], [-0.5, 0.5, 1.5, 2.5], kbt=1,
    )
    np.testing.assert_array_equal(result["probability_mean"], [0, 0, np.inf])
    assert np.isposinf(result["logmean_diagnostic"]).all()
    np.testing.assert_array_equal(result["probability_support"], [True, True, False])
    assert not result["support"].any()  # Retain the legacy common-support key.


def test_probability_fes_zero_uses_its_own_support():
    result = quantum_fes_1d(
        [[0, 0, 0], [1, 1, 0]], np.log([1, 10]),
        [-0.5, 0.5, 1.5], kbt=1,
    )
    # Probability masses are [13, 20]/33. The diagnostic is missing bin 1.
    np.testing.assert_allclose(
        result["probability_mean"], np.log(20 / np.array([13, 20])), atol=1e-15,
    )


def test_probability_fes_rejects_no_in_range_observations():
    with pytest.raises(ValueError):
        quantum_fes_1d([[2, 3]], [0], [-0.5, 0.5], kbt=1)


def test_workflow_normalization_preserves_the_raw_keyword():
    np.testing.assert_allclose(
        workflow_log_weights(raw=[0, 0]), [-np.log(2), -np.log(2)],
    )
