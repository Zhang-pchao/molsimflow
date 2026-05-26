import csv

from molsimflow.workflows.double_bubble_merge.microstate import analyze_bridge_microstate


def _write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_bridge_microstate_writes_tables(tmp_path):
    frame_index = tmp_path / "frame_index.csv"
    water_trace = tmp_path / "water_trace.csv"
    ion_trace = tmp_path / "ion_trace.csv"
    ion_positions = tmp_path / "ion_positions.csv"
    water_positions = tmp_path / "water_positions.csv"

    _write_rows(
        frame_index,
        [
            {
                "global_frame": 10,
                "segment_label": "seg0",
                "segment_frame": 0,
                "time_ns": 0.0,
                "d3d_all": 5.0,
                "water_segment": "seg0",
                "water_local_frame": 0,
                "ion_segment": "seg0",
                "ion_local_frame": 0,
            },
            {
                "global_frame": 11,
                "segment_label": "seg0",
                "segment_frame": 1,
                "time_ns": 0.1,
                "d3d_all": 4.0,
                "water_segment": "seg0",
                "water_local_frame": 1,
                "ion_segment": "seg0",
                "ion_local_frame": 1,
            },
        ],
    )
    _write_rows(
        water_trace,
        [
            {
                "segment": "seg0",
                "local_frame": 0,
                "bridge_center_x_A": 0.0,
                "bridge_center_y_A": 0.0,
                "bridge_center_z_A": 0.0,
                "n_current_bridge_waters": 2,
                "n_seed_retained_in_bridge": 1,
                "n_new_bridge_waters": 1,
                "n_untracked_current_bridge_waters": 0,
                "seed_retention_fraction": 0.5,
            },
            {
                "segment": "seg0",
                "local_frame": 1,
                "bridge_center_x_A": 1.0,
                "bridge_center_y_A": 0.0,
                "bridge_center_z_A": 0.0,
                "n_current_bridge_waters": 3,
                "n_seed_retained_in_bridge": 2,
                "n_new_bridge_waters": 1,
                "n_untracked_current_bridge_waters": 0,
                "seed_retention_fraction": 0.67,
            },
        ],
    )
    _write_rows(
        ion_trace,
        [
            {
                "segment": "seg0",
                "local_frame": 0,
                "n_current_bridge_ions": 2,
                "n_current_na": 1,
                "n_current_cl": 1,
                "n_current_oh_bulk": 0,
                "n_current_oh_surface": 0,
                "n_current_h3o": 0,
                "new_bridge_ion_fraction": 0.0,
            },
            {
                "segment": "seg0",
                "local_frame": 1,
                "n_current_bridge_ions": 1,
                "n_current_na": 1,
                "n_current_cl": 0,
                "n_current_oh_bulk": 0,
                "n_current_oh_surface": 0,
                "n_current_h3o": 0,
                "new_bridge_ion_fraction": 0.2,
            },
        ],
    )
    _write_rows(
        ion_positions,
        [
            {
                "time_ns": 0.0,
                "segment": "seg0",
                "local_frame": 0,
                "atom_id": 100,
                "current_trace_species": "Na",
                "current_trace_region": "bridge",
                "in_bridge": 1,
                "in_trace_region": 1,
                "in_bridge_region": 1,
                "x_A": 1.0,
                "y_A": 0.0,
                "z_A": 0.0,
            }
        ],
    )
    _write_rows(
        water_positions,
        [
            {"global_frame": 10, "atom_id": 1, "in_bridge": 1, "in_shell_only": 0},
            {"global_frame": 10, "atom_id": 2, "in_bridge": 1, "in_shell_only": 0},
            {"global_frame": 11, "atom_id": 1, "in_bridge": 1, "in_shell_only": 0},
        ],
    )

    outputs = analyze_bridge_microstate(
        frame_index=frame_index,
        water_trace=water_trace,
        ion_trace=ion_trace,
        output_dir=tmp_path / "microstate",
        bridge_rho_max_A=2.0,
        bridge_s_min_A=-1.0,
        bridge_s_max_A=1.0,
        ion_positions=ion_positions,
        water_positions=water_positions,
    )

    assert outputs["frame_table"].exists()
    assert outputs["ion_positions"].exists()
    assert outputs["species_region_summary"].exists()
    assert outputs["qc"].exists()

    with outputs["frame_table"].open() as handle:
        frames = list(csv.DictReader(handle))
    assert frames[0]["Nw_bridge_core"] == "2"
    assert frames[0]["Nion_bridge_current"] == "2"
    assert frames[0]["bridge_net_charge_proxy_e"] == "0"
    assert frames[1]["water_geometry_minus_trace_count"] == "-2"

    with outputs["species_region_summary"].open() as handle:
        summary = list(csv.DictReader(handle))
    assert summary[0]["species_canonical"] == "Na+"
    assert summary[0]["n_atoms"] == "1"
