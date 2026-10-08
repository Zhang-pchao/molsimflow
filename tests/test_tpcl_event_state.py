import csv
import json

from molsimflow.postprocess.tpcl_event_state import EventStateConfig, run_analysis


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_event_state_uses_only_pre_event_descriptors(tmp_path):
    times_ps = [20.0, 23.0, 31.0, 60.0]
    arcs = [0, 1, 4, 0]
    clusters = []
    modes = []
    geometry = []
    stress = []
    environment = []
    for index, (time_ps, arc) in enumerate(zip(times_ps, arcs), start=1):
        step = int(time_ps * 10)
        pre_step = step - 5
        clusters.append(
            {
                "cluster_id": index,
                "transition_time_ns": time_ps / 1000.0,
                "transition_step": step,
                "primary_event_id": index,
                "primary_arc_index": arc,
                "pre_step": pre_step,
                **{
                    field: 0.1 * index
                    for field in (
                        "affected_arc_fraction",
                        "event_size_residual_A2",
                        "event_size_radius_A2",
                        "mean_radius_change_A",
                        "primary_residual_change_A",
                        "delta_mode_2_amplitude_A",
                        "delta_mode_3_amplitude_A",
                        "delta_mode_4_amplitude_A",
                        "delta_mode_5_amplitude_A",
                        "delta_mode_6_amplitude_A",
                        "delta_contact_contour_area_A2",
                        "delta_contact_contour_perimeter_A",
                        "delta_contact_contour_circularity",
                        "delta_molecular_center_cap_angle_candidate_deg",
                    )
                },
            }
        )
        modes.append(
            {
                "step": pre_step,
                "mean_radius_A": 10 + index,
                "mode_2_amplitude_A": 1,
                "mode_3_amplitude_A": 2,
                "mode_4_amplitude_A": 3,
                "unresolved_mode_rms_A": 0.2,
            }
        )
        geometry.append(
            {
                "step": pre_step,
                "contact_contour_area_A2": 100,
                "contact_contour_circularity": 0.8,
                "molecular_center_cap_angle_candidate_deg": 90,
            }
        )
        stress.append(
            {
                "step": pre_step,
                "global_pressure_trace_bar": 1,
                "global_normal_minus_tangential_bar": 2,
                "global_shear_norm_bar": 3,
            }
        )
        for relative in (-4, -3, -2, -1, 1):
            environment.append(
                {
                    "sample_kind": "event",
                    "event_id": index,
                    "relative_frame": relative,
                    "nearest_site_distance_A": 1,
                    "local_ch3_fraction": 0.5,
                    "chemical_boundary_distance_proxy_A": "nan",
                    "local_hydration_areal_density_A-2": index if relative < 0 else 999,
                    "local_water_dipole_cos_z": 0.1,
                    "local_water_water_hbond_degree": 1.5,
                    "local_surface_water_hbond_per_h2o": 0.2,
                    "local_n2_min_distance_A": 5,
                }
            )
    paths = {}
    for name, rows in (
        ("event_size_metrics", clusters),
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
                "arc_count": 8,
                "first_time_ns": 0.0,
                "last_time_ns": 0.1,
                "event_cluster_count": 4,
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

    result = run_analysis(
        sources,
        tmp_path / "out",
        config=EventStateConfig(arc_bins=(("near", 1, 2), ("far", 2, 5))),
    )

    assert result["row_count"] == 4
    with (tmp_path / "out/event_state_table.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["future_cross_arc_fast_count"] == "1"
    assert rows[1]["past_history_fast_near_count"] == "1"
    assert rows[2]["past_history_slow_far_count"] == "2"
    assert float(rows[0]["pre_local_local_hydration_areal_density_A-2"]) == 1.0
    assert rows[0]["pre_local_sample_count"] == "4"
    assert rows[0]["stress_alignment_lag_steps"] == "0"
