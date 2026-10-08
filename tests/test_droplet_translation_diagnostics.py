import numpy as np

from molsimflow.postprocess.droplet_translation_diagnostics import (
    periodic_arithmetic_center,
    periodic_connectivity_winding,
    periodic_density_profile,
    register_periodic_density,
    smooth_contact_weights,
    weighted_quantile,
)


def test_periodic_arithmetic_center_reconstructs_across_boundary():
    result = periodic_arithmetic_center(np.array([9.7, 9.9, 0.1, 0.3]), 0.0, 10.0)

    assert np.isclose(result.value, 0.0, atol=1.0e-12)
    assert np.isclose(result.occupied_arc, 0.6)
    assert np.max(np.abs(result.offsets)) <= 0.3 + 1.0e-12


def test_smooth_contact_weights_and_weighted_quantiles():
    weights = smooth_contact_weights(
        np.array([4.0, 4.5, 5.0]), 0.0, midpoint_offset=4.5, width=0.5
    )

    assert weights[0] > weights[1] > weights[2]
    assert np.isclose(weights[1], 0.5)
    quantiles = weighted_quantile(np.array([0.0, 1.0, 2.0]), np.ones(3), [0.5])
    assert np.isclose(quantiles[0], 1.0)


def test_periodic_density_registration_recovers_subbin_shift():
    reference_values = np.linspace(1.0, 3.0, 500)
    shifted_values = (reference_values + 1.37) % 10.0
    reference = periodic_density_profile(reference_values, 0.0, 10.0, bins=512, sigma=0.2)
    shifted = periodic_density_profile(shifted_values, 0.0, 10.0, bins=512, sigma=0.2)
    result = register_periodic_density(reference, shifted, 10.0)

    assert np.isclose(result.displacement, 1.37, atol=0.03)
    assert result.residual < 0.03


def test_periodic_connectivity_detects_x_winding_ring():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    ring = np.column_stack(
        (np.arange(0.0, 10.0, 1.0), np.full(10, 5.0), np.full(10, 5.0))
    )
    finite = ring[:5]

    assert periodic_connectivity_winding(ring, bounds, 1.1) == (True, False, False)
    assert periodic_connectivity_winding(finite, bounds, 1.1) == (False, False, False)
