import math

import numpy as np

from molsimflow.postprocess.surface_site_enrichment import read_contact_lines, region_metrics


def test_surface_site_enrichment_regions_and_boundary_proxy():
    sites = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [4.0, 4.0]])
    types = np.array(["CH3", "SiOH", "CH3", "SiOH"])
    boundary = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    row = region_metrics(
        sites,
        types,
        boundary,
        np.zeros(2),
        np.array([20.0, 20.0]),
        tpcl_half_width=1.5,
        boundary_proximity=1.0,
    )
    assert row["surface_ch3_fraction"] == 0.5
    assert row["footprint_site_count"] == 3
    assert math.isclose(row["footprint_ch3_fraction"], 2.0 / 3.0)
    assert row["tpcl_site_count"] == 3
    assert math.isfinite(row["mean_contact_line_boundary_proxy_A"])


def test_read_contact_lines_omits_frames_without_boundary_points(tmp_path):
    lines = tmp_path / "contact_line.csv"
    points = tmp_path / "contact_line_points.csv"
    lines.write_text("step,contact_line_center_x_A\n10,1.0\n20,nan\n")
    points.write_text("step,x_A,y_A\n10,0.0,0.0\n10,1.0,0.0\n")

    rows, boundaries = read_contact_lines(lines, points)

    assert [int(row["step"]) for row in rows] == [10]
    assert set(boundaries) == {10}
