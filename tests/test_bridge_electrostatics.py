import csv

from molsimflow.postprocess.bridge_electrostatics import (
    BridgeElectrostaticsConfig,
    add_bridge_coordinates,
    analyze_bridge_electrostatics,
    charge_profile,
    electrostatic_coupling_metrics,
    frame_electrostatics,
    poisson_proxy,
)


def _config():
    return BridgeElectrostaticsConfig(
        profile_s_min_A=-5.0,
        profile_s_max_A=5.0,
        profile_bin_width_A=5.0,
        profile_rho_max_A=5.0,
        core_s_half_width_A=1.5,
        core_rho_max_A=3.0,
        surface_shell_width_A=0.75,
        shell_s_inner_A=2.0,
        shell_s_outer_A=5.0,
        shell_rho_max_A=5.0,
        gap_bin_width_A=2.0,
    )


def _rows():
    frames = [
        {
            "global_frame": 0,
            "time_ns": 0.0,
            "analysis_surface_gap_A": 6.0,
            "dynamic_surface_gap_est_A": 6.0,
            "d3d_all": 44.0,
            "bridge_core_volume_A3": 30.0,
            "Nw_bridge_core": 5,
            "fes_free_energy_relative_raw_interp": 2.0,
            "bubble_A_center_x_A": -5.0,
            "bubble_A_center_y_A": 0.0,
            "bubble_A_center_z_A": 0.0,
            "bubble_B_center_x_A": 5.0,
            "bubble_B_center_y_A": 0.0,
            "bubble_B_center_z_A": 0.0,
            "bridge_center_x_A": 0.0,
            "bridge_center_y_A": 0.0,
            "bridge_center_z_A": 0.0,
        },
        {
            "global_frame": 1,
            "time_ns": 0.1,
            "analysis_surface_gap_A": 8.0,
            "dynamic_surface_gap_est_A": 8.0,
            "d3d_all": 46.0,
            "bridge_core_volume_A3": 30.0,
            "Nw_bridge_core": 4,
            "fes_free_energy_relative_raw_interp": 3.0,
            "bubble_A_center_x_A": -5.0,
            "bubble_A_center_y_A": 0.0,
            "bubble_A_center_z_A": 0.0,
            "bubble_B_center_x_A": 5.0,
            "bubble_B_center_y_A": 0.0,
            "bubble_B_center_z_A": 0.0,
            "bridge_center_x_A": 0.0,
            "bridge_center_y_A": 0.0,
            "bridge_center_z_A": 0.0,
        },
        {
            "global_frame": 2,
            "time_ns": 0.2,
            "analysis_surface_gap_A": 10.0,
            "dynamic_surface_gap_est_A": 10.0,
            "d3d_all": 48.0,
            "bridge_core_volume_A3": 30.0,
            "Nw_bridge_core": 3,
            "fes_free_energy_relative_raw_interp": 5.0,
            "bubble_A_center_x_A": -5.0,
            "bubble_A_center_y_A": 0.0,
            "bubble_A_center_z_A": 0.0,
            "bubble_B_center_x_A": 5.0,
            "bubble_B_center_y_A": 0.0,
            "bubble_B_center_z_A": 0.0,
            "bridge_center_x_A": 0.0,
            "bridge_center_y_A": 0.0,
            "bridge_center_z_A": 0.0,
        },
    ]
    ions = [
        {
            "global_frame": 0,
            "time_ns": 0.0,
            "species_canonical": "H3O_plus",
            "species_charge_e": 1.0,
            "bridge_axis_s_A": 0.0,
            "bridge_axis_rho_A": 1.0,
        },
        {
            "global_frame": 0,
            "time_ns": 0.0,
            "species_canonical": "Cl_minus",
            "species_charge_e": -1.0,
            "bridge_axis_s_A": -3.0,
            "bridge_axis_rho_A": 1.0,
        },
        {
            "global_frame": 1,
            "time_ns": 0.1,
            "species_canonical": "Na_plus",
            "species_charge_e": 1.0,
            "bridge_axis_s_A": 4.0,
            "bridge_axis_rho_A": 1.0,
        },
        {
            "global_frame": 1,
            "time_ns": 0.1,
            "species_canonical": "OH_minus_bulk",
            "species_charge_e": -1.0,
            "bridge_axis_s_A": 0.5,
            "bridge_axis_rho_A": 2.0,
        },
    ]
    return frames, ions


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_charge_profile_and_poisson_proxy():
    frames, ions = _rows()
    profile = charge_profile(ions, frames, config=_config())

    assert len(profile) == 2
    assert profile[0]["n_frames"] == 3
    assert profile[0]["net_charge_e_per_frame"] == -1.0 / 3.0
    assert profile[1]["net_charge_e_per_frame"] == 1.0 / 3.0

    potential = poisson_proxy(profile)
    assert "poisson_potential_proxy_mV_zero_edge" in potential[0]
    assert potential[-1]["poisson_potential_proxy_mV_zero_edge"] == 0.0


def test_frame_electrostatics_and_coupling_metrics():
    frames, ions = _rows()
    config = _config()

    edl = frame_electrostatics(ions, frames, config=config)
    by_frame = {row["global_frame"]: row for row in edl}
    assert by_frame[0]["core_net_charge_e"] == 1.0
    assert by_frame[0]["surface_shell_abs_charge_e"] == 1.0
    assert by_frame[1]["core_net_charge_e"] == -1.0

    coupling, species_budget = electrostatic_coupling_metrics(ions, frames, config=config)
    coupling_by_frame = {row["global_frame"]: row for row in coupling}
    assert coupling_by_frame[0]["left_shell_net_charge_e"] == -1.0
    assert coupling_by_frame[1]["right_shell_net_charge_e"] == 1.0
    assert any(row["region"] == "core" and row["species"] == "H3O_plus" for row in species_budget)


def test_add_bridge_coordinates_from_frame_centers():
    frames, _ions = _rows()
    ion = {
        "global_frame": 0,
        "x_A": 2.0,
        "y_A": 3.0,
        "z_A": 0.0,
        "species_canonical": "Na_plus",
    }

    out = add_bridge_coordinates([ion], frames, config=_config(), force=True)[0]

    assert out["bridge_axis_s_A"] == 2.0
    assert out["bridge_axis_rho_A"] == 3.0


def test_analyze_bridge_electrostatics_writes_tables(tmp_path):
    frames, ions = _rows()
    frame_table = tmp_path / "frames.csv"
    ion_table = tmp_path / "ions.csv"
    _write_csv(frame_table, frames)
    _write_csv(ion_table, ions)

    outputs = analyze_bridge_electrostatics(
        ion_table=ion_table,
        frame_table=frame_table,
        output_dir=tmp_path / "out",
        config=_config(),
        disjoining_area_A2=100.0,
    )

    assert outputs["charge_profile"].exists()
    assert outputs["poisson_proxy"].exists()
    assert outputs["frame_electrostatics"].exists()
    assert outputs["coupling_frame_table"].exists()
    assert outputs["manifest"].exists()

    with outputs["gap_summary"].open() as handle:
        gap_rows = list(csv.DictReader(handle))
    assert gap_rows
