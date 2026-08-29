import math
from pathlib import Path

import numpy as np

from molsimflow.postprocess.dual_interface_water import (
    DualInterfaceConfig,
    assign_intact_waters,
    build_axial_profile,
    build_sz_map,
    local_basis,
    parse_segment_specs,
)


def test_local_basis_is_orthonormal_and_z_positive():
    midpoint, e_s, e_u, e_z = local_basis(
        np.array([9.0, 2.0, 4.0]),
        np.array([1.0, 2.0, 4.0]),
        np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
    )

    assert np.allclose(midpoint, [0.0, 2.0, 4.0])
    assert np.allclose([np.linalg.norm(e_s), np.linalg.norm(e_u), np.linalg.norm(e_z)], 1.0)
    assert math.isclose(float(np.dot(e_s, e_u)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(np.dot(e_s, e_z)), 0.0, abs_tol=1e-12)
    assert float(np.dot(e_z, [0.0, 0.0, 1.0])) > 0.0


def test_assign_intact_waters_handles_periodic_bonds_and_rejects_non_water():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    oxygen = np.array([[0.1, 5.0, 5.0], [5.0, 5.0, 5.0]])
    hydrogen = np.array([[9.9, 5.0, 5.0], [0.1, 5.9, 5.0], [5.8, 5.0, 5.0]])

    assignments, status = assign_intact_waters(oxygen, hydrogen, bounds, cutoff_A=1.1)

    assert assignments[0] == (0, 1)
    assert assignments[1] == (-1, -1)
    assert status.tolist() == [2, 1]


def test_map_and_block_profile_keep_gap_windows_separate():
    config = DualInterfaceConfig(
        gap_windows_A=((4.0, 6.0),),
        s_min_A=-1.0,
        s_max_A=1.0,
        z_min_A=-1.0,
        z_max_A=1.0,
        transverse_half_width_A=1.0,
        s_bins=2,
        z_bins=2,
        rho_bins=2,
        bootstrap_samples=50,
        block_ns=0.02,
    )
    samples = []
    frames = []
    for block, time in enumerate((0.0, 0.02, 0.04)):
        frames.append({"gap_window": "4-6A", "surface_z_mid_A": math.nan})
        samples.append(
            {
                "gap_window": "4-6A",
                "s_A": -0.5,
                "z_mid_A": -0.5,
                "rho_A": 0.5,
                "mu_s": 0.2 + 0.1 * block,
                "mu_z": 0.4,
                "mu_u": 0.0,
                "time_ns": time,
            }
        )

    sz = build_sz_map(samples, frames, config)
    occupied = [row for row in sz if row["count"]]
    profile = build_axial_profile(samples, config)
    axial = [row for row in profile if row["count"]]

    assert len(occupied) == 1
    assert occupied[0]["count"] == 3
    assert math.isclose(occupied[0]["mu_s_mean"], 0.3)
    assert axial[0]["effective_block_count"] == 3
    assert math.isfinite(axial[0]["mu_s_ci95_low"])


def test_parse_segment_specs_preserves_explicit_labels():
    assert parse_segment_specs("0-0.5=/a/dump;0.5-1=/b/dump") == (
        ("0-0.5", Path("/a/dump")),
        ("0.5-1", Path("/b/dump")),
    )
