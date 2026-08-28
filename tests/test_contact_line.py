import math

import numpy as np

from molsimflow.postprocess.contact_line import block_rows, polygon_metrics


def test_polygon_geometry_and_operational_jump_detection():
    geometry, boundary = polygon_metrics(
        np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    )
    assert len(boundary) == 4
    assert math.isclose(geometry["contact_line_area_A2"], 4.0)
    assert math.isclose(geometry["contact_line_perimeter_A"], 8.0)
    assert np.allclose([geometry["centroid_dx_A"], geometry["centroid_dy_A"]], 0.0)

    rows = []
    radii = [10.0, 10.2, 10.4, 10.6, 14.6]
    for step, radius in enumerate(np.repeat(radii, 4)):
        rows.append({"step": step, "contact_line_equivalent_radius_A": radius})
    blocks, threshold = block_rows(
        rows, block_frames=4, timestep_fs=0.5, jump_sigma=4.0, minimum_jump=2.0
    )
    assert threshold == 2.0
    assert [row["radius_jump_candidate"] for row in blocks] == [False] * 4 + [True]
