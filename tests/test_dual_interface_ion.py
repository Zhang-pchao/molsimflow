"""Focused tests for the dual-interface formal-charge proxy."""

import math

import numpy as np

from molsimflow.postprocess.dual_interface_ion import (
    field_proxy,
    local_basis,
    minimum_image,
    validate_species_charge,
)


def test_minimum_image_and_basis_are_consistent():
    lengths = np.array([100.0, 80.0, 120.0])
    delta = minimum_image(np.array([-94.0, 0.0, 0.0]), lengths)
    assert np.allclose(delta, [6.0, 0.0, 0.0])
    _, e_s, e_u, e_z = local_basis(np.array([97.0, 40.0, 50.0]), np.array([3.0, 40.0, 50.0]), lengths)
    basis = np.vstack([e_s, e_u, e_z])
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    assert e_z[2] > 0


def test_softened_field_has_expected_symmetry():
    charges = np.array([1.0, -1.0])
    positions = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    field = field_proxy(points, positions, charges, epsilon_r=1.0, softening_A=1.0)
    assert field[0, 0] > 0
    assert math.isclose(field[0, 1], 0.0, abs_tol=1e-12)
    assert math.isclose(field[1, 1], 0.0, abs_tol=1e-12)


def test_species_charge_contract_is_fail_closed():
    assert validate_species_charge("Na_plus", 1.0) == 1.0
    assert validate_species_charge("Cl_minus", -1.0) == -1.0
    try:
        validate_species_charge("water_O", -0.82)
    except ValueError as exc:
        assert "unsupported species" in str(exc)
    else:
        raise AssertionError("Unapproved fixed partial charge was accepted")


if __name__ == "__main__":
    test_minimum_image_and_basis_are_consistent()
    test_softened_field_has_expected_symmetry()
    test_species_charge_contract_is_fail_closed()
