import csv
import math

import numpy as np

from molsimflow.postprocess.bridge_descriptors import (
    BridgeCylinder,
    analyze_bridge_ion_occupancy,
    analyze_bridge_water_density,
    build_bridge_ion_occupancy_table,
    build_bridge_water_frame_table,
    canonical_ion_species,
    cylinder_volume_nm3,
    summarize_by_gap_bins,
)


def test_bridge_cylinder_contains_points_with_periodic_wrapping():
    cylinder = BridgeCylinder(
        center=np.array([9.5, 0.0, 0.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        radius_A=1.0,
        length_A=4.0,
    )

    mask = cylinder.contains(
        np.array(
            [
                [0.5, 0.0, 0.0],
                [0.5, 2.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ),
        box_dims=np.array([10.0, 10.0, 10.0]),
    )

    assert mask.tolist() == [True, False, False]


def test_bridge_water_density_and_gap_binning():
    rows = [
        {"time_ns": "0.0", "surface_gap_estimate_A": "1.0", "bridge_cyl_env.sum": "2.0", "bridge_cyl_env.mean": "0.2"},
        {"time_ns": "0.1", "surface_gap_estimate_A": "3.0", "bridge_cyl_env.sum": "4.0", "bridge_cyl_env.mean": "0.4"},
    ]

    frame_rows = build_bridge_water_frame_table(
        rows,
        bridge_radius_A=1.0,
        bridge_length_A=10.0,
        case_label="caseA",
    )
    volume = cylinder_volume_nm3(1.0, 10.0)

    assert len(frame_rows) == 2
    assert math.isclose(frame_rows[0]["bridge_water_density_proxy_per_nm3"], 2.0 / volume)

    binned = summarize_by_gap_bins(
        frame_rows,
        ["bridge_water_count_proxy", "bridge_water_density_proxy_per_nm3"],
        gap_bin_width_A=5.0,
        min_bin_count=1,
    )

    assert len(binned) == 1
    assert math.isclose(binned[0]["bridge_water_count_proxy_mean"], 3.0)


def test_bridge_ion_occupancy_counts_charge_and_density():
    positions = [
        {"time_ns": "0.0", "current_trace_species": "H3O+", "in_bridge_region": "1"},
        {"time_ns": "0.0", "current_trace_species": "surface OH-", "in_bridge_region": "1"},
        {"time_ns": "0.0", "current_trace_species": "Cl-", "in_bridge_region": "1"},
        {"time_ns": "0.0", "current_trace_species": "Na+", "in_bridge_region": "0"},
        {"time_ns": "0.1", "current_trace_species": "Na+", "in_bridge_region": "1"},
    ]
    gaps = [
        {"time_ns": 0.0, "surface_gap_A": 1.0, "state": "near"},
        {"time_ns": 0.1, "surface_gap_A": 6.0, "state": "open"},
    ]

    frame_rows = build_bridge_ion_occupancy_table(
        positions,
        bridge_radius_A=1.0,
        bridge_length_A=10.0,
        case_label="caseB",
        gap_rows=gaps,
    )

    first = frame_rows[0]
    assert first["n_bridge_total_ions"] == 3.0
    assert first["n_bridge_h3o"] == 1.0
    assert first["n_bridge_oh_surface"] == 1.0
    assert first["n_bridge_cl"] == 1.0
    assert first["bridge_net_charge_e"] == -1.0
    assert canonical_ion_species("Bulk OH-") == "oh_bulk"


def test_bridge_descriptor_file_outputs(tmp_path):
    water_csv = tmp_path / "state.csv"
    with water_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time_ns", "surface_gap_estimate_A", "bridge_cyl_env.sum", "bridge_cyl_env.mean"],
        )
        writer.writeheader()
        writer.writerow({"time_ns": "0.0", "surface_gap_estimate_A": "1.0", "bridge_cyl_env.sum": "2", "bridge_cyl_env.mean": "0.2"})
        writer.writerow({"time_ns": "0.1", "surface_gap_estimate_A": "3.0", "bridge_cyl_env.sum": "4", "bridge_cyl_env.mean": "0.4"})

    water_outputs = analyze_bridge_water_density(
        input_csv=water_csv,
        output_dir=tmp_path / "water",
        bridge_radius_A=1.0,
        bridge_length_A=10.0,
        gap_bin_width_A=5.0,
    )
    assert water_outputs["frame_table"].exists()
    assert water_outputs["binned"].exists()
    assert water_outputs["window_summary"].exists()

    ion_csv = tmp_path / "ions.csv"
    with ion_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_ns", "current_trace_species", "in_bridge_region"])
        writer.writeheader()
        writer.writerow({"time_ns": "0.0", "current_trace_species": "H3O+", "in_bridge_region": "1"})
        writer.writerow({"time_ns": "0.0", "current_trace_species": "Cl-", "in_bridge_region": "1"})
        writer.writerow({"time_ns": "0.1", "current_trace_species": "Na+", "in_bridge_region": "1"})

    ion_outputs = analyze_bridge_ion_occupancy(
        positions_csv=ion_csv,
        gap_table=water_csv,
        output_dir=tmp_path / "ion",
        bridge_radius_A=1.0,
        bridge_length_A=10.0,
        gap_bin_width_A=5.0,
    )
    assert ion_outputs["frame_table"].exists()
    assert ion_outputs["binned"].exists()
    assert ion_outputs["window_summary"].exists()
