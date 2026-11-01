"""Reject ambiguous geometry and temporal inputs before numerical reduction."""
import numpy as np
import pytest

from molsimflow.postprocess.periodic_front_pattern import (
    complete_window_advance_fraction, connected_components, minimum_image_delta,
    periodic_neighbour_matrix, selectivity_with_support, synchronous_tip_bulk_gap,
)
from molsimflow.postprocess.umbrella_wham import histogram_pmf


@pytest.mark.parametrize("coords,box,periodic", [
    ([[np.nan, 0]], [1, 1], None), ([[0, 0]], [-1, 1], None),
    ([[0, 0]], [np.inf, 1], None), ([[0, 0]], [[1, 1]], None),
    ([[0, 0]], [0, 1], [True, False]),
])
def test_invalid_geometry_is_not_silently_nonperiodic(coords, box, periodic):
    with pytest.raises(ValueError):
        minimum_image_delta(coords, box, periodic)


@pytest.mark.parametrize("cutoff", [0, -1, np.nan, np.inf])
def test_invalid_cutoff_does_not_become_empty_connectivity(cutoff):
    with pytest.raises(ValueError):
        periodic_neighbour_matrix([[0, 0]], [1, 1], cutoff)


@pytest.mark.parametrize("adjacency", [
    np.ones((2, 3), bool), np.array([[False, True], [False, False]]),
    np.array([[0, 1], [1, 0]]),
])
def test_graph_contract_rejects_ambiguous_edges(adjacency):
    with pytest.raises(ValueError):
        connected_components(adjacency)


def test_empty_graph_and_explicit_isolates_keep_identity():
    count, labels, sizes = connected_components(np.zeros((0, 0), bool))
    assert count == 0 and labels.size == sizes.size == 0
    count, labels, sizes = connected_components(np.zeros((3, 3), bool))
    assert count == 3
    np.testing.assert_array_equal(sizes, [1, 1, 1])


@pytest.mark.parametrize("columns,lookahead", [
    ([0.5, 1.5], 1), ([[0, 1]], 1), ([0, -2], 1),
    ([0, 1], 1.5), ([0, 1], True),
])
def test_advance_windows_do_not_truncate_invalid_inputs(columns, lookahead):
    with pytest.raises(ValueError):
        complete_window_advance_fraction(columns, lookahead)


def test_synchronous_comparison_requires_one_frame_axis():
    with pytest.raises(ValueError):
        synchronous_tip_bulk_gap([[0, 1]], [[0, 1]])


@pytest.mark.parametrize("support", [-1, np.nan, np.inf])
def test_invalid_support_is_not_a_weight(support):
    with pytest.raises(ValueError):
        selectivity_with_support([{"selectivity": 1., "contacted_observations": support}])


@pytest.mark.parametrize("reference", [np.nan, np.inf])
def test_invalid_pmf_reference_does_not_select_an_arbitrary_bin(reference):
    with pytest.raises(ValueError):
        histogram_pmf(np.array([0.]), np.array([0.]), np.array([-1., 1.]),
                      1., reference_coordinate=reference)
