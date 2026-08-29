"""Focused tests for matched-gap H-bond connectivity."""

import math

from molsimflow.postprocess.dual_interface_hbond import (
    HbondCandidate,
    classify_layer,
    graph_metrics,
    select_edges,
)


def test_layer_partition_remains_nonoverlapping_at_narrow_gap():
    assert classify_layer(-1.9, 6.0, 3.0) == "A"
    assert classify_layer(0.0, 6.0, 3.0) == "central"
    assert classify_layer(1.9, 6.0, 3.0) == "B"
    assert classify_layer(-1.2, 4.0, 3.0) == "A"
    assert classify_layer(0.0, 4.0, 3.0) == "central"
    assert classify_layer(1.2, 4.0, 3.0) == "B"


def test_graph_metrics_detect_spanning_and_shortest_path():
    nodes = {1, 2, 3, 4}
    edges = {(1, 2), (2, 3), (3, 4)}
    layers = {1: "A", 2: "central", 3: "central", 4: "B"}
    result = graph_metrics(nodes, edges, layers)
    assert result["spanning_indicator"] == 1
    assert result["shortest_A_B_path_edges"] == 3.0
    assert result["largest_component_fraction"] == 1.0


def test_default_hbond_geometry_is_fail_closed():
    candidates = [
        HbondCandidate(1, 10, 2, 3.49, 2.44, 150.1),
        HbondCandidate(2, 11, 3, 3.51, 2.20, 170.0),
        HbondCandidate(3, 12, 4, 3.20, 2.46, 170.0),
        HbondCandidate(4, 13, 5, 3.20, 2.20, 149.9),
    ]
    selected = select_edges(candidates, 3.5, 2.45, 150.0)
    assert set(selected) == {(1, 2)}


def test_no_spanning_path_is_nan():
    result = graph_metrics({1, 2}, set(), {1: "A", 2: "B"})
    assert result["spanning_indicator"] == 0
    assert math.isnan(result["shortest_A_B_path_edges"])


if __name__ == "__main__":
    test_layer_partition_remains_nonoverlapping_at_narrow_gap()
    test_graph_metrics_detect_spanning_and_shortest_path()
    test_default_hbond_geometry_is_fail_closed()
    test_no_spanning_path_is_nan()
