import numpy as np

from molsimflow.postprocess.precontact_n2_enrichment import frame_metrics, projection_counts


def test_frame_metrics_separates_main_and_disconnected_n2():
    row, values = frame_metrics(
        np.array([2.0, 3.0, 8.0, 20.0]),
        np.array([0, 1]),
        near_z_min=0.0,
        near_z_max=10.0,
    )
    assert row["near_surface_main_bubble_count"] == 2
    assert row["near_surface_disconnected_count"] == 1
    np.testing.assert_array_equal(values["disconnected"], [8.0, 20.0])


def test_projection_counts_separates_bubble_satellite_and_remote_n2():
    centers = np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [7.0, 5.0, 5.0], [1.0, 1.0, 5.0]])
    bounds = np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])
    result = projection_counts(
        centers,
        np.full(4, 5.0),
        np.array([0, 1]),
        bounds,
        near_z_min=0.0,
        near_z_max=10.0,
        margin=1.0,
    )
    assert result["near_surface_disconnected_inside_projection_count"] == 1
    assert result["near_surface_disconnected_outside_projection_count"] == 1
