from pathlib import Path

import numpy as np

from molsimflow.postprocess.nanodroplet_spreading import (
    DropletFrame,
    add_unwrapped_centers,
    analyze_frame,
    periodic_weighted_center,
)


def test_periodic_weighted_center_crosses_boundary():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    center = periodic_weighted_center(np.array([[9.8, 5, 5], [0.2, 5, 5]]), np.ones(2), bounds)
    assert np.allclose(center, [0.0, 5.0, 5.0])


def test_add_unwrapped_centers_handles_cluster_and_mass_centers():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    rows = []
    for x in (9.8, 0.2):
        row = {}
        for prefix in ("droplet_oxygen_center", "inserted_water_mass_com"):
            row.update({f"{prefix}_x_A": x, f"{prefix}_y_A": 5.0, f"{prefix}_z_A": 5.0})
        rows.append(row)
    add_unwrapped_centers(rows, bounds)
    assert np.isclose(rows[1]["droplet_oxygen_center_x_unwrapped_A"], 10.2)
    assert np.isclose(rows[1]["inserted_water_mass_com_x_unwrapped_A"], 10.2)


def test_spreading_metrics_use_oxygen_cluster_and_contacts():
    bounds = np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])
    oxygen = np.array([[5, 5, 4], [6, 5, 4], [5, 6, 4], [6, 6, 5]], dtype=float)
    water = np.repeat(oxygen, 3, axis=0)
    types = np.tile([2, 1, 1], 4)
    frame = DropletFrame(Path("drop.dump"), 0, 10, bounds, np.array([[5, 5, 2]]), types, water)
    row = analyze_frame(frame, oxygen_type=2, surface_z=2, cluster_cutoff=2, contact_cutoff=3)
    assert row["largest_cluster_water_count"] == 4
    assert row["contact_water_count"] == 3
    assert row["footprint_convex_hull_area_A2"] > 0
