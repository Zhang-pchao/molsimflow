import math

import numpy as np

from molsimflow.cli import build_parser
from molsimflow.postprocess.surface_functional_group_orientation import (
    analyze_geometry,
    build_blocks,
    identify_functional_sites,
)


def _synthetic_surface():
    elements = np.asarray(
        ["Si", "Si", "Si", "Si", "O", "C", "O", "C"]
        + ["H"] * 8
    )
    coordinates = np.asarray(
        [
            [4.0, 4.0, 5.0],
            [12.0, 4.0, 5.0],
            [4.0, 12.0, 5.0],
            [12.0, 12.0, 5.0],
            [4.0, 4.0, 6.6],
            [12.0, 4.0, 6.8],
            [4.0, 12.0, 6.6],
            [12.0, 12.0, 6.8],
            [4.0, 4.0, 7.55],
            [12.9, 4.0, 7.1],
            [11.55, 4.78, 7.1],
            [11.55, 3.22, 7.1],
            [4.0, 12.0, 7.55],
            [12.9, 12.0, 7.1],
            [11.55, 12.78, 7.1],
            [11.55, 11.22, 7.1],
        ]
    )
    return elements, coordinates, np.asarray([20.0, 20.0, 20.0])


def test_identify_and_measure_global_and_local_axes():
    elements, coordinates, lengths = _synthetic_surface()
    sites = identify_functional_sites(
        elements,
        coordinates,
        lengths,
        surface_range=(1, len(elements)),
        surface_z_A=6.6,
        surface_depth_A=3.0,
        oh_cutoff_A=1.25,
        ch_cutoff_A=1.30,
        si_terminal_cutoff_A=2.20,
        local_normal_neighbors=4,
    )
    assert sum(site.site_type == "CH3" for site in sites) == 2
    assert sum(site.site_type == "SiOH" for site in sites) == 2
    assert {site.anchor_si_id for site in sites} == {1, 2, 3, 4}

    atom_ids = np.arange(1, len(elements) + 1)
    rows = analyze_geometry(
        step=20_000,
        segment_index=0,
        bounds=np.column_stack((np.zeros(3), lengths)),
        surface=coordinates,
        candidate_oxygen_ids=atom_ids[elements == "O"],
        candidate_oxygen=coordinates[elements == "O"],
        hydrogen_ids=atom_ids[elements == "H"],
        hydrogen=coordinates[elements == "H"],
        sites=sites,
        surface_range=(1, len(elements)),
        plane_z_A=6.6,
        timestep_fs=0.5,
        oh_cutoff_A=1.25,
        ch_cutoff_A=1.30,
        si_terminal_cutoff_A=2.20,
    )
    assert len(rows) == 4
    assert all(row["group_integrity"] for row in rows)
    assert all(row["site_axis_valid"] for row in rows)
    assert all(math.isclose(row["site_axis_tilt_global_deg"], 0.0) for row in rows)
    assert all(math.isclose(row["site_axis_tilt_local_deg"], 0.0) for row in rows)
    sioh = [row for row in rows if row["site_type"] == "SiOH"]
    assert all(math.isclose(row["oh_axis_tilt_global_deg"], 0.0) for row in sioh)
    assert all(math.isclose(row["si_o_h_angle_deg"], 180.0) for row in sioh)


def test_blocks_never_cross_segment_boundaries():
    rows = []
    for segment, steps in ((0, (20_000, 40_000)), (1, (60_000, 80_000))):
        for step in steps:
            row = {
                "step": step,
                "time_ns": step * 0.5 / 1.0e6,
                "segment_index": segment,
                "site_type": "CH3",
            }
            row.update({metric: 1.0 for metric in (
                "mean_site_axis_cos_global",
                "mean_site_axis_cos_local",
                "mean_site_axis_tilt_global_deg",
                "mean_site_axis_tilt_local_deg",
                "mean_terminal_height_A",
                "mean_local_normal_tilt_deg",
                "mean_local_plane_rms_A",
                "group_integrity_fraction",
                "mean_oh_axis_cos_global",
                "mean_oh_axis_cos_local",
                "mean_oh_axis_tilt_global_deg",
                "mean_oh_axis_tilt_local_deg",
                "mean_si_o_h_angle_deg",
            )})
            rows.append(row)
    blocks, interval = build_blocks(rows, block_frames=4)
    assert interval == 20_000
    assert len(blocks) == 2
    assert {block["segment_index"] for block in blocks} == {0, 1}
    assert not any(block["complete_block"] for block in blocks)


def test_cli_exposes_surface_functional_group_orientation(tmp_path):
    args = build_parser().parse_args(
        [
            "postprocess",
            "surface-functional-group-orientation",
            "--trajectory",
            str(tmp_path / "bubble.dump"),
            "--output-dir",
            str(tmp_path / "out"),
            "--initial-xyz",
            str(tmp_path / "model.xyz"),
            "--surface-range",
            "1:10",
            "--water-range",
            "11:30",
            "--surface-z-A",
            "5",
            "--no-plots",
        ]
    )
    assert args.postprocess_command == "surface-functional-group-orientation"
    assert args.block_frames == 25
