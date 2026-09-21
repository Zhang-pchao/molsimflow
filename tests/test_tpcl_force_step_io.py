from pathlib import Path

import pytest

from molsimflow.postprocess.tpcl_force_step_io import (
    expected_multirate_steps,
    expected_regular_steps,
    project_production_size,
)


def test_expected_multirate_steps_has_one_boundary_frame():
    assert expected_multirate_steps(1000, 100, 40, 10, 20) == (
        1010,
        1020,
        1030,
        1040,
        1060,
        1080,
        1100,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 100, 0, 10, 20),
        (0, 100, 110, 10, 20),
        (0, 100, 45, 10, 20),
        (0, 105, 40, 10, 20),
    ],
)
def test_expected_multirate_steps_rejects_incomplete_cadence(arguments):
    with pytest.raises(ValueError):
        expected_multirate_steps(*arguments)


def test_expected_regular_steps_excludes_parent_frame():
    assert expected_regular_steps(1000, 40, 10) == (1010, 1020, 1030, 1040)


def test_size_projection_scales_each_output_by_its_own_cadence(tmp_path: Path):
    sizes = {
        "tpcl_coordinates.lammpstrj.zst": 200,
        "tpcl_dynamics.lammpstrj.zst": 40,
        "full_reference.lammpstrj.zst": 4,
        "motion_energy_stress_0p01ps.dat": 200,
        "force_sums_0p01ps.dat": 200,
        "final.restart": 10,
        "final.data": 20,
    }
    for name, size in sizes.items():
        (tmp_path / name).write_bytes(b"x" * size)
    result = project_production_size(
        output_dir=tmp_path,
        observed_counts={
            "coordinates": 200,
            "dynamics": 40,
            "full_reference": 4,
            "motion": 200,
            "force": 200,
        },
        start_step=36_200_000,
        projection_total_steps=200_000,
        projection_fast_steps=40_000,
        fast_coordinate_stride=20,
        slow_coordinate_stride=100,
        dynamics_stride=100,
        full_stride=1000,
        table_stride=20,
        restart_stride=10_000,
    )
    assert result["projected_coordinate_frames"] == 3600
    assert result["projected_dynamics_frames"] == 2000
    assert result["projected_full_reference_frames"] == 200
    assert result["projected_table_rows"] == 10000
    assert result["projected_restart_checkpoints"] == 20
    components = result["components_bytes"]
    assert components["tpcl_coordinates.lammpstrj.zst"] == 3600
    assert components["tpcl_dynamics.lammpstrj.zst"] == 2000
    assert components["full_reference.lammpstrj.zst"] == 200
    assert components["motion_energy_stress_0p01ps.dat"] == 10000
    assert components["restart_checkpoints_and_final"] == 210
