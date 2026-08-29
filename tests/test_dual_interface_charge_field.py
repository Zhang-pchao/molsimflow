"""Focused invariants for the full-cell fixed-charge field proxy."""

import math

import numpy as np

from molsimflow.io.lammps_dump import LammpsDumpFrame
from molsimflow.postprocess.dual_interface_charge_field import (
    SPCE_Q_H,
    SPCE_Q_O,
    TIP3P_Q_H,
    TIP3P_Q_O,
    ChargeFieldConfig,
    _sample_geometry,
    classify_charge_sources,
    deposit_cic,
    solve_periodic_field,
)


def _frame(rows):
    return LammpsDumpFrame(
        frame_index=0,
        timestep=0,
        bounds=np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]]),
        box_header="ITEM: BOX BOUNDS pp pp pp",
        atom_fields=("id", "type", "x", "y", "z"),
        atom_rows=tuple(tuple(map(str, row)) for row in rows),
    )


def test_water_models_are_neutral():
    assert math.isclose(SPCE_Q_O + 2 * SPCE_Q_H, 0.0, abs_tol=1e-15)
    assert math.isclose(TIP3P_Q_O + 2 * TIP3P_Q_H, 0.0, abs_tol=1e-15)


def test_cic_conserves_charge_and_is_pbc_equivalent():
    bounds = np.array([[0.0, 10.0], [0.0, 12.0], [0.0, 14.0]])
    positions = np.array([[0.2, 11.8, 7.0], [9.7, 0.4, 13.9]])
    charges = np.array([1.0, -1.0])
    first = deposit_cic(positions, charges, bounds, (8, 9, 10))
    second = deposit_cic(positions + np.array([10.0, -12.0, 14.0]), charges, bounds, (8, 9, 10))
    assert math.isclose(first.sum(), charges.sum(), abs_tol=1e-12)
    assert np.allclose(first, second, atol=1e-12)


def test_periodic_poisson_solver_matches_one_mode():
    shape = (32, 8, 8)
    lengths = np.array([32.0, 8.0, 8.0])
    x = np.arange(shape[0], dtype=float)
    wave_number = 2.0 * np.pi / lengths[0]
    rho = np.sin(wave_number * x)[:, None, None] * np.ones((1, shape[1], shape[2]))
    cell_volume = float(np.prod(lengths / np.asarray(shape)))
    _, ex, ey, ez = solve_periodic_field(rho * cell_volume, lengths, 1e-9)
    expected = -(4.0 * np.pi * 14.399645478 / wave_number) * np.cos(wave_number * x)
    assert np.allclose(ex[:, 0, 0], expected, rtol=1e-10, atol=1e-10)
    assert np.allclose(ey, 0.0, atol=1e-12)
    assert np.allclose(ez, 0.0, atol=1e-12)


def test_oh_coordination_reconstructs_neutral_sources():
    rows = [
        (1, 2, 2.0, 2.0, 2.0),
        (2, 1, 2.9, 2.0, 2.0),
        (3, 1, 1.1, 2.0, 2.0),
        (4, 2, 8.0, 8.0, 8.0),
        (5, 1, 8.9, 8.0, 8.0),
        (6, 1, 7.1, 8.0, 8.0),
        (7, 1, 8.0, 8.9, 8.0),
        (8, 2, 14.0, 14.0, 14.0),
        (9, 1, 14.9, 14.0, 14.0),
    ]
    sources, counts = classify_charge_sources(_frame(rows), False, ChargeFieldConfig())
    assert counts["n_intact_water"] == 1
    assert counts["n_H3O_plus"] == 1
    assert counts["n_OH_minus_total"] == 1
    assert math.isclose(counts["formal_net_charge_e"], 0.0, abs_tol=1e-12)
    assert all(len(positions) == len(charges) for positions, charges in sources.values())


def test_true_unassigned_hydrogen_is_a_charge_conserving_free_proton():
    rows = [
        (1, 2, 2.0, 2.0, 2.0),
        (2, 1, 2.9, 2.0, 2.0),
        (3, 1, 3.5, 2.0, 2.0),
    ]
    _, counts = classify_charge_sources(_frame(rows), False, ChargeFieldConfig())
    assert counts["n_OH_minus_total"] == 1
    assert counts["n_H_plus_free"] == 1
    assert math.isclose(counts["formal_net_charge_e"], 0.0, abs_tol=1e-12)


def test_inner_and_outer_caps_are_three_dimensional_and_disjoint():
    trace = {
        "bubble_A_center_x_A": 4.0,
        "bubble_A_center_y_A": 10.0,
        "bubble_A_center_z_A": 10.0,
        "bubble_B_center_x_A": 16.0,
        "bubble_B_center_y_A": 10.0,
        "bubble_B_center_z_A": 10.0,
    }
    case = {"has_tio2": "0", "nominal_radius_a_A": "3.0", "nominal_radius_b_A": "3.0"}
    geometry = _sample_geometry(trace, _frame([]), case, ChargeFieldConfig())
    inner = geometry["inner_mask"]
    outer = geometry["outer_mask"]
    assert inner.any() and outer.any()
    assert not np.any(inner & outer)
    assert geometry["film_probe_count"] > 1


if __name__ == "__main__":
    test_water_models_are_neutral()
    test_cic_conserves_charge_and_is_pbc_equivalent()
    test_periodic_poisson_solver_matches_one_mode()
    test_oh_coordination_reconstructs_neutral_sources()
    test_true_unassigned_hydrogen_is_a_charge_conserving_free_proton()
    test_inner_and_outer_caps_are_three_dimensional_and_disjoint()
