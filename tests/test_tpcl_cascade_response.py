import csv
import json
import math
from argparse import Namespace

import pytest

from molsimflow.postprocess.tpcl_cascade_response import run_analysis


def _write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_run_analysis_builds_response_scales_and_water_mode_links(tmp_path):
    hazard = [
        (0, 1, 0.0, 1.0, 10, 8, 3, 13, True),
        (1, 2, 0.0, 1.0, 10, 5, 1, 8, True),
        (2, 4, 1.0, 2.0, 6, 5, 1, 9, True),
        (4, 5, 1.0, 2.0, 3, 1, 0, 4, False),
    ]
    hazard_rows = [
        {
            "arc_distance_start_bins": start,
            "arc_distance_end_bins_exclusive": end,
            "lag_start_ps": lag_start,
            "lag_end_ps": lag_end,
            "observed_pairs": observed,
            "null_mean_pairs": null_mean,
            "null_q025_pairs": low,
            "null_q975_pairs": high,
            "observed_to_null_ratio": observed / null_mean,
            "empirical_upper_p": 0.01,
            "informative_null_count": informative,
            "same_arc_null_preserves_intervals": start == 0,
        }
        for start, end, lag_start, lag_end, observed, null_mean, low, high, informative in hazard
    ]
    frame_rows = [
        {"step": step, "time_ns": step / 1000, "mean_radius_A": 10.0}
        for step in range(5)
    ]
    event_rows = [
        {
            "primary_event_id": event_id,
            "event_size_residual_A2": value,
            "affected_arc_fraction": value / 10.0,
            "delta_mode_1_amplitude_A": value,
            "delta_mode_2_amplitude_A": 4.0 - value,
        }
        for event_id, value in enumerate((1.0, 2.0, 3.0), start=1)
    ]
    water_rows = [
        {
            "event_id": event_id,
            "arc_index": 0,
            "transition_step": event_id,
            "metric": "mean_q_tet",
            "pre_mean": 0.0,
            "post_mean": value,
            "post_minus_pre": value,
        }
        for event_id, value in enumerate((1.0, 2.0, 3.0), start=1)
    ]
    null_rows = [
        {
            "control_type": control,
            "metric": "mean_q_tet",
            "observed_mean_post_minus_pre": 2.0,
            "null_sample_count": 20,
            "null_mean": 0.0,
            "null_sd": 0.1,
            "null_q025": -0.5,
            "null_q975": 0.5,
            "empirical_two_sided_p": 0.01,
            "inference_status": "within_trajectory_randomization_diagnostic",
        }
        for control in ("per_arc_block_circular_shift", "event_time_permutation")
    ]
    paths = {
        "pair_hazard": tmp_path / "hazard.csv",
        "frame_modes": tmp_path / "modes.csv",
        "event_sizes": tmp_path / "sizes.csv",
        "water_effects": tmp_path / "water.csv",
        "water_null_statistics": tmp_path / "nulls.csv",
    }
    for name, rows in (
        ("pair_hazard", hazard_rows),
        ("frame_modes", frame_rows),
        ("event_sizes", event_rows),
        ("water_effects", water_rows),
        ("water_null_statistics", null_rows),
    ):
        _write(paths[name], rows)
    propagation_summary = tmp_path / "propagation-summary.json"
    propagation_summary.write_text(
        json.dumps({"status": "PASS", "case_id": "other", "arc_count": 8})
    )
    output = tmp_path / "out"
    args = Namespace(
        case_id="case",
        propagation_summary=propagation_summary,
        output_dir=output,
        max_mode=2,
        **paths,
    )
    with pytest.raises(ValueError, match="case does not match"):
        run_analysis(args)
    propagation_summary.write_text(
        json.dumps({"status": "PASS", "case_id": "case", "arc_count": 8})
    )
    summary = run_analysis(args)

    arc_length = 2.0 * math.pi * 10.0 / 8.0
    assert summary["cross_arc_bins_above_null_q975"] == 1
    assert math.isclose(summary["response_weighted_arc_distance_bins"], 1.75)
    assert math.isclose(summary["response_weighted_arc_distance_A"], 1.75 * arc_length)
    assert math.isclose(summary["response_weighted_lag_ps"], 2.0 / 3.0)
    assert summary["water_metrics_outside_both_null_q95"] == ["mean_q_tet"]
    with (output / "water_cascade_associations.csv").open() as handle:
        associations = list(csv.DictReader(handle))
    mode_1 = next(row for row in associations if row["cascade_metric"] == "delta_mode_1_amplitude_A")
    mode_2 = next(row for row in associations if row["cascade_metric"] == "delta_mode_2_amplitude_A")
    assert math.isclose(float(mode_1["spearman_rho"]), 1.0)
    assert math.isclose(float(mode_2["spearman_rho"]), -1.0)
    assert json.loads((output / "manifest.json").read_text())["max_mode"] == 2
