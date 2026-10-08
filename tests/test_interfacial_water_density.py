import numpy as np

from molsimflow.postprocess.interfacial_water_density import analyze_frame, classify_regions


def test_regions_partition_periodic_points():
    boundary = np.array([[8.0, 8.0], [2.0, 8.0], [2.0, 2.0], [8.0, 2.0]])
    points = np.array([[0.0, 0.0], [3.0, 0.0], [5.0, 5.0]])
    masks = classify_regions(
        points, boundary, np.zeros(2), np.array([10.0, 10.0]), tpcl_half_width=1.1
    )
    assert masks["footprint"].tolist() == [True, False, False]
    assert masks["tpcl"].tolist() == [False, True, False]
    assert masks["far_field"].tolist() == [False, False, True]
    assert np.all(sum(masks.values()) == 1)


def test_early_small_contact_line_allows_empty_footprint_core():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 20.0]])
    coordinates = np.array([[5.0, 5.0, 3.0], [8.0, 8.0, 3.0]])
    boundary = np.array([[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]])
    _, areas, row = analyze_frame(
        coordinates,
        bounds,
        boundary,
        np.array([5.0, 5.0]),
        surface_z=0.0,
        tpcl_half_width=2.0,
        grid_spacing=1.0,
        z_edges=np.arange(0.0, 6.5, 0.5),
        hydration_z_max=6.0,
    )
    assert areas["footprint"] == 0.0
    assert np.isnan(row["footprint_hydration_areal_density_A-2"])
