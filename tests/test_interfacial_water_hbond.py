import numpy as np

from molsimflow.postprocess.interfacial_water_hbond import (
    sampling_summary,
    water_water_hbond_count,
)


def test_water_water_hbond_count_detects_directed_pair():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    oxygen = np.array([[2.0, 2.0, 2.0], [4.8, 2.0, 2.0]])
    oh_vectors = [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    assert water_water_hbond_count(
        oxygen, oh_vectors, bounds, oo_cutoff=3.5, angle_cutoff_deg=30.0
    ) == 1


def test_sampling_summary_exposes_sparse_regions():
    rows = []
    for footprint_count in (0, 2):
        row = {f"{region}_h2o_count": 1 for region in ("footprint", "tpcl", "far_field")}
        row["footprint_h2o_count"] = footprint_count
        rows.append(row)
    summary = sampling_summary(rows)
    assert summary["footprint"] == {
        "mean_h2o_count_per_frame": 1.0,
        "nonempty_frame_count": 1,
        "total_h2o_frame_samples": 2,
    }
