"""Tests for the periodicity-correct front-pattern observables.

Each test pins a defect that the open-boundary module exhibited, so a
regression back to the earlier behaviour fails loudly.
"""

import numpy as np
import pytest

from molsimflow.postprocess.periodic_front_pattern import (
    complete_window_advance_fraction,
    connected_components,
    footprint_connectivity,
    minimum_image_delta,
    periodic_neighbour_matrix,
    selectivity_with_support,
    sublattice_connectivity,
    synchronous_tip_bulk_gap,
    tip_isolation_full_surface,
)


def test_minimum_image_wraps_only_periodic_axes():
    coords = np.array([[0.5, 0.5, 0.0], [9.5, 0.5, 0.0]])
    box = [10.0, 0.0, 0.0]  # x periodic, y and z not
    delta = minimum_image_delta(coords, box)
    # the convention is pts[0] - pts[1]; 0.5 - 9.5 wraps to +1.0
    assert delta[0, 1, 0] == pytest.approx(1.0)
    assert delta[0, 1, 1] == pytest.approx(0.0)


def test_periodic_neighbour_matrix_closes_the_box():
    # Two points 9 A apart in a 10 A box are 1 A apart under minimum image.
    coords = np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]])
    matrix = periodic_neighbour_matrix(coords, [10.0, 20.0, 20.0], cutoff=2.0)
    assert matrix[0, 1]
    assert not matrix[0, 0]


def test_sublattice_connectivity_recovers_a_periodic_loop():
    # A ring of four sites around a 10 A periodic box in y. Open
    # boundaries would cut the ring into pieces; the periodic image
    # closes it.
    box = [100.0, 10.0, 100.0]
    coords = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 6.5, 0.0],
        [0.0, 9.0, 0.0],
    ])
    selection = np.ones(4, dtype=bool)
    result = sublattice_connectivity(coords, box, selection, cutoff=3.0)
    assert result["n_fragments"] == 1
    assert result["connected_fraction"] == pytest.approx(1.0)


def test_sublattice_connectivity_reports_x_span_of_largest_fragment():
    # Fragment A: two sites 1 A apart. Fragment B: four sites 2 A apart
    # spanning x. The gap between A and B is 9 A, above the cutoff, so
    # the two fragments stay separate.
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],          # fragment A
        [10.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],
        [14.0, 0.0, 0.0],
        [16.0, 0.0, 0.0],         # fragment B spans x
    ])
    selection = np.ones(6, dtype=bool)
    result = sublattice_connectivity(coords, [200.0, 50.0, 50.0], selection,
                                     cutoff=6.0)
    assert result["n_fragments"] == 2
    assert result["largest_fragment"] == 4
    assert result["largest_x_span_A"] == pytest.approx(6.0)
    assert result["connected_fraction"] == pytest.approx(4.0 / 6.0)


def test_tip_isolation_counts_seam_neighbours():
    # The tip sits on the destination face but is connected to a parent
    # on the departure face across the seam. It must not be isolated.
    parent_xyz = np.array([
        [0.0, 0.0, 0.0],    # departure face
        [2.0, 0.0, 0.0],    # destination face, the tip
    ])
    contacted = [np.array([0, 1])]
    half_mask = np.array([False, True])
    box = [100.0, 100.0, 100.0]
    result = tip_isolation_full_surface(contacted, parent_xyz, box, half_mask,
                                        cutoff=5.3)
    assert result["isolated_fraction"] == pytest.approx(0.0)


def test_tip_isolation_detects_a_genuinely_isolated_tip():
    parent_xyz = np.array([
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
    ])
    contacted = [np.array([0, 1])]
    half_mask = np.array([False, True])
    box = [200.0, 100.0, 100.0]
    result = tip_isolation_full_surface(contacted, parent_xyz, box, half_mask,
                                        cutoff=5.3)
    assert result["isolated_fraction"] == pytest.approx(1.0)


def test_footprint_connectivity_separates_full_and_half_face():
    # Three parents in a row on the departure face, one tip across the
    # seam. The full footprint is connected; the destination half alone
    # is not.
    parent_xyz = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
    ])
    contacted = [np.array([0, 1, 2, 3])]
    half_mask = np.array([False, False, False, True])
    box = [100.0, 100.0, 100.0]
    result = footprint_connectivity(contacted, parent_xyz, box, half_mask,
                                    cutoff=5.3)
    assert result["median_full_largest"] == pytest.approx(4.0)
    assert result["median_half_largest"] == pytest.approx(1.0)


def test_synchronous_gap_uses_one_frame_at_a_time():
    tip = np.array([0.0, 5.0, 2.0])
    bulk = np.array([0.0, 1.0, 1.0])
    result = synchronous_tip_bulk_gap(tip, bulk)
    # gaps are 4, 1, 1 A -> median 1, max 4, final 1
    assert result["median_gap_A"] == pytest.approx(1.0)
    assert result["max_gap_A"] == pytest.approx(4.0)
    assert result["final_gap_A"] == pytest.approx(1.0)


def test_complete_window_fraction_keeps_the_failed_column():
    # The tip reaches column 2 and never leaves it. Those windows are
    # the evidence of arrest and must not be discarded.
    columns = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    result = complete_window_advance_fraction(columns, lookahead=2)
    # column 2 is occupied in the last 4 frames; only 2 of those leave a
    # complete 2-frame window
    assert result[2]["n_complete_windows"] == 2
    assert result[2]["n_advanced"] == 0
    assert result[2]["forward_fraction"] == pytest.approx(0.0)
    assert result[0]["n_advanced"] == 2


def test_complete_window_fraction_ignores_the_incomplete_tail():
    columns = np.array([0, 0, 0, 1, 1])  # lookahead 2
    result = complete_window_advance_fraction(columns, lookahead=2)
    # column 0 occupies frames 0-2, so all three leave a complete window;
    # column 1 occupies the last two frames and leaves none
    assert result[0]["n_complete_windows"] == 3
    assert result[1]["n_complete_windows"] == 0
    assert result[1]["forward_fraction"] is None


def test_selectivity_with_support_flags_a_sparse_tail():
    # Two well-supported columns and one tail column with a huge ratio.
    rows = [
        {"selectivity": 1.1, "contacted_observations": 1000.0},
        {"selectivity": 1.2, "contacted_observations": 900.0},
        {"selectivity": 90.0, "contacted_observations": 2.0},
    ]
    summary = selectivity_with_support(rows)
    assert summary["median"] == pytest.approx(1.2)
    assert summary["n_columns_carrying_80pct_support"] == 2
    assert summary["median_over_dominant_columns"] == pytest.approx(1.15)
    assert summary["total_support"] == pytest.approx(1902.0)


def test_connected_components_labels_correctly():
    adjacency = np.array([
        [False, True, False, False],
        [True, False, True, False],
        [False, True, False, True],
        [False, False, True, False],
    ])
    n, labels, sizes = connected_components(adjacency)
    assert n == 1
    assert sizes.max() == 4
