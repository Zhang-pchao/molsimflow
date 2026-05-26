import csv
import math

import numpy as np

from molsimflow.postprocess.water_orientation import (
    WaterOrientationSummaryConfig,
    analyze_water_orientation,
    compute_water_orientation_sample,
)


def _write_orientation_samples(path):
    rows = [
        {
            "frame": 0,
            "time": 0.0,
            "s": -1.0,
            "rho": 0.5,
            "theta_AB_deg": 30.0,
            "cos_theta_AB": math.cos(math.radians(30.0)),
            "S_AB": 0.625,
            "theta_A_deg": 40.0,
            "cos_theta_A": math.cos(math.radians(40.0)),
            "S_A": 0.380,
            "theta_B_deg": 50.0,
            "cos_theta_B": math.cos(math.radians(50.0)),
            "S_B": 0.120,
            "d3d_all": 1.0,
        },
        {
            "frame": 0,
            "time": 0.0,
            "s": 0.0,
            "rho": 1.5,
            "theta_AB_deg": 60.0,
            "cos_theta_AB": 0.5,
            "S_AB": -0.125,
            "theta_A_deg": 70.0,
            "cos_theta_A": math.cos(math.radians(70.0)),
            "S_A": -0.325,
            "theta_B_deg": 80.0,
            "cos_theta_B": math.cos(math.radians(80.0)),
            "S_B": -0.455,
            "d3d_all": 2.0,
        },
        {
            "frame": 1,
            "time": 0.1,
            "s": 1.0,
            "rho": 0.5,
            "theta_AB_deg": 90.0,
            "cos_theta_AB": 0.0,
            "S_AB": -0.5,
            "theta_A_deg": 100.0,
            "cos_theta_A": math.cos(math.radians(100.0)),
            "S_A": -0.455,
            "theta_B_deg": 110.0,
            "cos_theta_B": math.cos(math.radians(110.0)),
            "S_B": -0.325,
            "d3d_all": 3.0,
        },
        {
            "frame": 1,
            "time": 0.1,
            "s": 2.0,
            "rho": 2.5,
            "theta_AB_deg": 120.0,
            "cos_theta_AB": -0.5,
            "S_AB": -0.125,
            "theta_A_deg": 130.0,
            "cos_theta_A": math.cos(math.radians(130.0)),
            "S_A": 0.120,
            "theta_B_deg": 140.0,
            "cos_theta_B": math.cos(math.radians(140.0)),
            "S_B": 0.380,
            "d3d_all": 4.0,
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_compute_water_orientation_sample_geometry():
    sample = compute_water_orientation_sample(
        oxygen_position=np.array([0.0, 0.0, 0.0]),
        hydrogen_positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        center_a=np.array([-1.0, 0.0, 0.0]),
        center_b=np.array([1.0, 0.0, 0.0]),
        bounds_or_lengths=np.array([10.0, 10.0, 10.0]),
        frame=7,
        time=0.2,
        oxygen_id=10,
        hydrogen_ids=[11, 12],
    )

    assert sample["frame"] == 7
    assert sample["oxygen_id"] == 10
    assert math.isclose(sample["cos_theta_AB"], 1.0 / math.sqrt(2.0), rel_tol=1e-12)
    assert math.isclose(sample["theta_AB_deg"], 45.0, rel_tol=1e-12)
    assert math.isclose(sample["S_AB"], 0.25, rel_tol=1e-12)
    assert math.isclose(sample["s"], 0.0, abs_tol=1e-12)
    assert math.isclose(sample["rho"], 0.0, abs_tol=1e-12)


def test_analyze_water_orientation_writes_summaries(tmp_path):
    input_path = tmp_path / "water_orientation_samples.csv"
    _write_orientation_samples(input_path)

    outputs = analyze_water_orientation(
        input_csv=input_path,
        output_dir=tmp_path / "water_orientation",
        config=WaterOrientationSummaryConfig(
            rho_bins=2,
            rho_max=2.0,
            s_bins=2,
            s_min=-1.0,
            s_max=1.0,
            cv_bins=2,
            angle_bins=6,
        ),
    )

    assert outputs["frame_summary"].exists()
    assert outputs["radial_profile"].exists()
    assert outputs["sr_map"].exists()
    assert outputs["cv_summary"].exists()
    assert outputs["angle_distribution"].exists()

    with outputs["radial_profile"].open() as handle:
        radial = list(csv.DictReader(handle))
    assert [int(row["count"]) for row in radial] == [2, 1]

    with outputs["angle_distribution"].open() as handle:
        angle = list(csv.DictReader(handle))
    assert sum(int(row["count"]) for row in angle) == 4

    with outputs["state_statistics"].open() as handle:
        stats = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert stats["n_orientation_samples"] == "4"
    assert stats["n_frames"] == "2"
