import csv
import math

from molsimflow.postprocess.tpcl_history_prediction import PredictionConfig, run_analysis


def test_blocked_history_prediction_writes_complete_tables(tmp_path):
    rows = []
    for block in range(6):
        for offset in range(10):
            index = block * 10 + offset
            history = int(offset % 3 == 0)
            row = {
                "case_id": "case",
                "cluster_id": index + 1,
                "primary_arc_index": index % 8,
                "arc_count": 8,
                "transition_time_ns": (block * 200.0 + offset * 10.0) / 1000.0,
                "time_block_200ps": block,
                "future_cross_arc_fast_count": history,
                "future_cross_arc_slow_count": history + (offset % 2),
                "past_cross_arc_fast_count": history,
                "past_cross_arc_slow_count": history + (offset % 2),
            }
            for direction in ("past", "future"):
                for window in ("fast", "slow"):
                    for bin_name in ("near", "mid", "far"):
                        row[f"{direction}_history_{window}_{bin_name}_count"] = (
                            history if bin_name == "near" else 0
                        )
            for feature_index, feature in enumerate(
                (
                    "global_mean_radius_A",
                    "global_mode_2_amplitude_A",
                    "global_mode_3_amplitude_A",
                    "global_mode_4_amplitude_A",
                    "global_unresolved_mode_rms_A",
                    "global_footprint_area_A2",
                    "global_footprint_circularity",
                    "global_cap_angle_candidate_deg",
                    "global_pressure_trace_bar",
                    "global_normal_minus_tangential_bar",
                    "global_shear_norm_bar",
                    "pre_local_nearest_site_distance_A",
                    "pre_local_local_ch3_fraction",
                    "pre_local_chemical_boundary_distance_proxy_A",
                    "pre_local_local_hydration_areal_density_A-2",
                    "pre_local_local_water_dipole_cos_z",
                    "pre_local_local_water_water_hbond_degree",
                    "pre_local_local_surface_water_hbond_per_h2o",
                    "pre_local_local_n2_min_distance_A",
                )
            ):
                row[feature] = math.sin(index + feature_index)
            rows.append(row)
    table = tmp_path / "event_state.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = run_analysis(
        table,
        tmp_path / "out",
        config=PredictionConfig(
            fold_count=3,
            embargo_blocks=0,
            bootstrap_samples=20,
            null_samples=2,
            max_iterations=100,
        ),
    )

    assert result["status"] == "PASS"
    with (tmp_path / "out/model_scores.csv").open() as handle:
        scores = list(csv.DictReader(handle))
    with (tmp_path / "out/history_evidence.csv").open() as handle:
        evidence = list(csv.DictReader(handle))
    assert len(scores) == 8
    assert len(evidence) == 2
    assert all(math.isfinite(float(row["mean_poisson_deviance"])) for row in scores)
    assert all(0.0 <= float(row["block_bootstrap_bh_q_primary_family"]) <= 1.0 for row in evidence)
    assert (tmp_path / "out/history_null_controls.csv").is_file()
