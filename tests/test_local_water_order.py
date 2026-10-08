import math

import numpy as np

from molsimflow.postprocess.local_water_order import (
    hbond_network_metrics,
    local_structure_index,
    tetrahedral_order,
    water_hbond_edges,
)


def test_tetrahedral_order_is_one_for_regular_tetrahedron():
    vectors = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    assert math.isclose(tetrahedral_order(vectors), 1.0, abs_tol=1.0e-12)


def test_local_structure_index_uses_first_neighbor_beyond_cutoff():
    distances = np.asarray([2.5, 2.7, 3.4, 3.5, 4.2])
    value, count = local_structure_index(distances, 3.7)
    gaps = np.asarray([0.2, 0.7, 0.1, 0.7])
    assert count == 4
    assert math.isclose(value, float(np.mean((gaps - np.mean(gaps)) ** 2)))


def test_hbond_network_reports_induced_connected_component():
    oxygen = np.asarray([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [8.0, 0.0, 0.0]])
    oh_vectors = [
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ]
    bounds = np.asarray([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])
    rows, summary = hbond_network_metrics(
        oxygen,
        oh_vectors,
        np.asarray([0, 1]),
        bounds,
        oo_cutoff_A=3.5,
        angle_cutoff_deg=30.0,
    )
    assert rows["degree"].tolist() == [1, 1]
    assert rows["internal_degree"].tolist() == [1, 1]
    assert summary["incident_edges"] == 1
    assert summary["induced_edges"] == 1
    assert summary["component_count"] == 1
    assert summary["largest_component_fraction"] == 1.0


def test_water_hbond_edges_preserve_both_directions_without_double_counting():
    oxygen = np.asarray([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
    oh_vectors = [
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([[-1.0, 0.0, 0.0]]),
    ]
    bounds = np.asarray([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])
    edges = water_hbond_edges(
        oxygen,
        oh_vectors,
        np.asarray([0, 1]),
        bounds,
        oo_cutoff_A=3.5,
        angle_cutoff_deg=30.0,
    )
    assert edges == [(0, 1), (1, 0)]

    rows, summary = hbond_network_metrics(
        oxygen,
        oh_vectors,
        np.asarray([0, 1]),
        bounds,
        oo_cutoff_A=3.5,
        angle_cutoff_deg=30.0,
    )
    assert rows["donor"].tolist() == [1, 1]
    assert rows["acceptor"].tolist() == [1, 1]
    assert rows["degree"].tolist() == [1, 1]
    assert summary["incident_edges"] == 1
