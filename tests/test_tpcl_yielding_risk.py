import csv
import json

from molsimflow.postprocess.tpcl_yielding_risk import run_analysis


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_yielding_risk_balances_event_and_shifted_controls(tmp_path):
    cluster = {
        "cluster_id": 1,
        "primary_event_id": 7,
        "primary_arc_index": 2,
        "transition_time_ns": 0.020,
    }
    environment = []
    for kind, shift, anchor in (("event", 0, 200), ("circular_shift_control", 20, 400)):
        for relative in (-4, -3, -2, -1, 0):
            environment.append(
                {
                    "event_id": 7,
                    "sample_kind": kind,
                    "circular_shift_frames": shift,
                    "relative_frame": relative,
                    "step": anchor + 10 * relative,
                    "nearest_site_distance_A": 1,
                    "local_site_count": 3,
                    "local_ch3_fraction": 0.5,
                    "nearest_ch3_distance_A": 2,
                    "nearest_sioh_distance_A": 3,
                    "chemical_boundary_distance_proxy_A": 4,
                    "local_hydration_areal_density_A-2": 5 if relative < 0 else 999,
                    "local_water_dipole_cos_z": 0.1,
                    "local_water_water_hbond_degree": 1.5,
                    "local_surface_water_hbond_per_h2o": 0.2,
                    "local_n2_min_distance_A": 6,
                }
            )
    pre_steps = {190, 390}
    modes = [
        {
            "step": step,
            "mean_radius_A": 10,
            "mode_2_amplitude_A": 1,
            "mode_3_amplitude_A": 2,
            "mode_4_amplitude_A": 3,
            "unresolved_mode_rms_A": 0.2,
        }
        for step in pre_steps
    ]
    geometry = [
        {
            "step": step,
            "contact_contour_area_A2": 100,
            "contact_contour_circularity": 0.8,
            "molecular_center_cap_angle_candidate_deg": 90,
        }
        for step in pre_steps
    ]
    stress = [
        {
            "step": step,
            "global_pressure_trace_bar": 1,
            "global_normal_minus_tangential_bar": 2,
            "global_shear_norm_bar": 3,
        }
        for step in pre_steps
    ]
    paths = {}
    for name, rows in (
        ("event_size_metrics", [cluster]),
        ("frame_modes", modes),
        ("geometry", geometry),
        ("global_stress", stress),
        ("event_environment", environment),
    ):
        paths[name] = tmp_path / f"{name}.csv"
        _write_csv(paths[name], rows)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "PASS",
                "case_id": "case",
                "first_time_ns": 0.0,
                "last_time_ns": 0.1,
                "event_cluster_count": 1,
            }
        ),
        encoding="utf-8",
    )
    sources = tmp_path / "sources.tsv"
    sources.write_text(
        "case_id\tpropagation_summary\tevent_size_metrics\tframe_modes\tgeometry\t"
        "global_stress\tevent_environment\n"
        f"case\t{summary}\t{paths['event_size_metrics']}\t{paths['frame_modes']}\t"
        f"{paths['geometry']}\t{paths['global_stress']}\t{paths['event_environment']}\n",
        encoding="utf-8",
    )

    result = run_analysis(sources, tmp_path / "out", time_ps_per_step=0.1)

    assert result["row_count"] == 2
    with (tmp_path / "out/yielding_risk_sets.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["sample_kind"] for row in rows} == {"event", "circular_shift_control"}
    assert sum(float(row["risk_set_weight"]) for row in rows) == 1.0
    assert all(float(row["pre_local_local_hydration_areal_density_A-2"]) == 5 for row in rows)
    assert all(float(row["sample_time_ns"]) in {0.02, 0.04} for row in rows)
    assert result["time_ps_per_step"] == 0.1
