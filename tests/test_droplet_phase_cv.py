import numpy as np

from molsimflow.postprocess.droplet_phase_cv import (
    harmonic_periodic_restraint,
    periodic_phase,
    render_plumed_phase_restraint,
)


def test_periodic_phase_handles_boundary_and_derivative_sum():
    phase = periodic_phase(np.array([9.8, 0.2]), 0.0, 10.0)

    assert np.isclose(phase.value, 0.0, atol=1.0e-12)
    assert np.isclose(np.sum(phase.derivatives), 1.0)


def test_periodic_phase_derivative_matches_finite_difference():
    values = np.array([9.7, 0.1, 0.6, 1.2])
    analytic = periodic_phase(values, 0.0, 10.0)
    step = 1.0e-6

    numerical = []
    for index in range(len(values)):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        q_plus = periodic_phase(plus, 0.0, 10.0).value
        q_minus = periodic_phase(minus, 0.0, 10.0).value
        delta = (q_plus - q_minus + 5.0) % 10.0 - 5.0
        numerical.append(delta / (2.0 * step))

    np.testing.assert_allclose(analytic.derivatives, numerical, atol=2.0e-9, rtol=0.0)


def test_constant_weight_reduces_outlier_phase_leverage():
    values = np.array([1.0, 1.1, 1.2, 6.0])
    equal = periodic_phase(values, 0.0, 10.0)
    weighted = periodic_phase(values, 0.0, 10.0, weights=np.array([1.0, 1.0, 1.0, 0.1]))

    assert abs(weighted.value - 1.1) < abs(equal.value - 1.1)
    assert abs(weighted.derivatives[-1]) < abs(equal.derivatives[-1])


def test_harmonic_restraint_uses_periodic_shortest_path():
    displacement, energy, force = harmonic_periodic_restraint(9.9, 0.4, 0.02, 10.0)

    assert np.isclose(displacement, 0.5)
    assert np.isclose(energy, 0.0025)
    assert np.isclose(force, 0.01)


def test_render_plumed_phase_restraint_is_fixed_center_and_periodic():
    text = render_plumed_phase_restraint(
        [10, 20, 30], lower=0.0, upper=74.0, target=22.5, kappa=0.02
    )

    assert "waterO: GROUP ATOMS=10,20,30" in text
    assert "center: CENTER ATOMS=waterO PHASES" in text
    assert "PERIODIC=0,74" in text
    assert "rest: RESTRAINT ARG=qx AT=22.5 KAPPA=0.02" in text
    assert "MOVINGRESTRAINT" not in text
