"""Independent identities and finite differences for the contraction contract."""

import numpy as np
import pytest

from molsimflow.postprocess.path_contraction import (
    contract_coordinates, pullback_bias_forces, validate_contraction,
    validate_contraction_metadata,
)


@pytest.mark.parametrize("beads", [1, 2, 3, 8, 32])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_centroid_covariance_and_no_mutation(beads, fraction):
    real = np.random.default_rng(71).normal(size=(beads, 2, 3))
    saved = real.copy()
    virtual = contract_coordinates(real, fraction)
    mean = real.mean(axis=0)
    np.testing.assert_allclose(virtual.mean(axis=0), mean, rtol=1e-12, atol=1e-12)
    u = (real - mean).reshape(beads, -1)
    v = (virtual - mean).reshape(beads, -1)
    np.testing.assert_allclose(v.T @ v, fraction**2 * (u.T @ u), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(real, saved)
    assert not np.shares_memory(real, virtual)


def test_affine_and_quadratic_cv_identities():
    x = np.array([[-0.4], [0.2], [0.9]])
    c = x.mean()
    for fraction in [0.0, 0.1, 0.5, 1.0]:
        z = contract_coordinates(x, fraction)
        assert np.mean(2.3*z - 0.7) == pytest.approx(2.3*c - 0.7, abs=1e-12)
        assert np.mean(z*z) == pytest.approx((1-fraction**2)*c*c + fraction**2*np.mean(x*x))
        virtual_force = np.full_like(x, -2.3 / len(x))
        np.testing.assert_allclose(pullback_bias_forces(virtual_force, fraction), virtual_force)


@pytest.mark.parametrize("fraction", [0.0, 0.125, 0.5, 1.0])
def test_cubic_cv_exact_moments_and_internal_force(fraction):
    x = np.array([[-0.2], [0.3], [0.8]])
    c = x.mean()
    u = x - c
    m2, m3 = np.mean(u**2), np.mean(u**3)
    z = contract_coordinates(x, fraction)
    s = np.mean(z + z**3)
    assert s == pytest.approx(c+c**3+3*c*fraction**2*m2+fraction**3*m3, abs=1e-12)
    slope = 1.7*(s-0.1)
    force = pullback_bias_forces(-slope*(1+3*z*z)/len(x), fraction)
    expected_internal = -3*slope/len(x)*(2*c*fraction**2*u+fraction**3*(u*u-m2))
    np.testing.assert_allclose(force-force.mean(axis=0), expected_internal, rtol=1e-12, atol=1e-12)


def _vector_bias(x, fraction):
    z = contract_coordinates(x, fraction)
    s0, s1 = np.mean(z**3), np.mean(np.sin(z))
    energy = 0.7*s0*s0 + 0.4*s0*s1 + 0.9*s1*s1
    gradient = ((1.4*s0+0.4*s1)*3*z*z + (0.4*s0+1.8*s1)*np.cos(z)) / z.size
    return energy, pullback_bias_forces(-gradient, fraction)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_joint_vector_bias_finite_difference(fraction):
    x = np.random.default_rng(87).uniform(-0.8, 0.9, size=(3, 2, 3))
    energy, force = _vector_bias(x, fraction)
    for step in [1e-4, 1e-5, 1e-6]:
        finite_difference = np.empty_like(x)
        for index in np.ndindex(x.shape):
            plus, minus = x.copy(), x.copy()
            plus[index] += step
            minus[index] -= step
            finite_difference[index] = -(_vector_bias(plus, fraction)[0]-_vector_bias(minus, fraction)[0])/(2*step)
        np.testing.assert_allclose(force, finite_difference, atol=2e-7, rtol=2e-6)
    virtual = contract_coordinates(x, fraction)
    permuted = np.roll(x, 1, axis=0)
    np.testing.assert_allclose(_vector_bias(permuted, fraction)[1], np.roll(force, 1, axis=0))
    assert _vector_bias(permuted, fraction)[0] == pytest.approx(energy)
    np.testing.assert_allclose(contract_coordinates(x+1.3, fraction), virtual+1.3)


def test_small_lambda_internal_force_order():
    x = np.array([[-0.2], [0.3], [0.8]])
    norms = []
    for fraction in [0.01, 0.005, 0.0025]:
        z = contract_coordinates(x, fraction)
        force = pullback_bias_forces(-(1+3*z*z)/len(x), fraction)
        norms.append(np.linalg.norm(force-force.mean(axis=0)))
    np.testing.assert_allclose(np.array(norms[:-1])/norms[1:], 4.0, rtol=0.01)


def test_force_sum_and_work_conjugacy():
    rng = np.random.default_rng(65)
    f = rng.normal(size=(3, 2, 3))
    dx = rng.normal(size=f.shape)
    pulled = pullback_bias_forces(f, 0.37)
    np.testing.assert_allclose(pulled.sum(axis=0), f.sum(axis=0), atol=1e-12)
    assert np.sum(pulled*dx) == pytest.approx(np.sum(f*contract_coordinates(dx, 0.37)), abs=1e-12)


@pytest.mark.parametrize("bad", [True, -0.1, 1.1, np.nan, np.inf, [0.5], "invalid"])
def test_invalid_parameter(bad):
    with pytest.raises(ValueError, match="finite scalar"):
        validate_contraction(bad)


@pytest.mark.parametrize("bad", [[], [1.0, 2.0], [[np.nan]], np.empty((2, 0, 3))])
def test_invalid_arrays(bad):
    for operation in [contract_coordinates, pullback_bias_forces]:
        with pytest.raises(ValueError):
            operation(bad, 0.5)


def test_real_observable_metadata_required():
    record = {"lambda": 0.5, "coordinate_lift": "pimd_unwrapped", "observable_coordinates": "real_beads"}
    assert validate_contraction_metadata(record) == 0.5
    for key, bad in [("lambda", -1), ("coordinate_lift", "wrapped"), ("observable_coordinates", "virtual_beads")]:
        with pytest.raises(ValueError):
            validate_contraction_metadata({**record, key: bad})
    with pytest.raises(ValueError):
        validate_contraction_metadata({"lambda": 0.5})
