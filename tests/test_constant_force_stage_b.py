from __future__ import annotations

import numpy as np
import pytest

from molsimflow.postprocess.constant_force_stage_b_flux import (
    decompose_track_center,
    plane_crossing_counts,
)
from molsimflow.postprocess.constant_force_stage_b_layers import (
    layer_flux_closure,
    occupancy_weighted_velocity,
)


def test_phase_averaged_plane_crossings_preserve_direction() -> None:
    previous = np.asarray([9.4, 4.0])
    current = np.asarray([10.6, 3.0])
    positive, negative, net = plane_crossing_counts(
        previous,
        current,
        lower_bound=0.0,
        box_length=10.0,
        plane_count=10,
    )
    assert positive == pytest.approx(0.2)
    assert negative == pytest.approx(0.1)
    assert net == pytest.approx(0.1)


def test_track_center_decomposition_is_exact_with_membership_change() -> None:
    atom_ids = np.asarray([1, 2, 3, 4])
    previous = np.asarray([0.0, 2.0, 8.0, 10.0])
    current = np.asarray([1.0, 3.0, 7.0, 11.0])
    result = decompose_track_center(
        {1, 2, 3},
        {1, 2, 4},
        atom_ids,
        previous,
        current,
    )
    assert result["advective_displacement_A"] == pytest.approx(1.0 / 3.0)
    assert result["membership_displacement_A"] == pytest.approx(4.0 / 3.0)
    assert result["total_center_displacement_A"] == pytest.approx(5.0 / 3.0)
    assert result["closure_residual_A"] == pytest.approx(0.0)


def test_occupancy_weighting_suppresses_sparse_layer_outlier() -> None:
    value = occupancy_weighted_velocity([100.0, 1.0], [1.0, 100.0])
    assert value == pytest.approx(200.0 / 101.0)


def test_layer_flux_closes_against_global_com_velocity() -> None:
    result = layer_flux_closure(
        [2.0, 1.0],
        [100.0, -100.0],
        global_velocity_Aps=1.0 / 3.0,
    )
    assert result["layer_velocity_sum_molecule_A_per_ps"] == pytest.approx(1.0)
    assert result["global_velocity_sum_molecule_A_per_ps"] == pytest.approx(1.0)
    assert result["closure_residual_molecule_A_per_ps"] == pytest.approx(0.0)
