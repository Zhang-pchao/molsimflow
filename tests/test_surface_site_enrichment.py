import math

import numpy as np

from molsimflow.postprocess.surface_site_enrichment import region_metrics


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
