import csv
import json

import numpy as np

from molsimflow.postprocess.event_aligned_circular_field import (
    AnalysisConfig,
    ColumnSpec,
    load_event_data,
    load_field_data,
    read_sources,
    run_analysis,
    signed_arc_offsets,
)


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _toy_inputs(tmp_path):
    field_rows = []
    for frame in range(41):
        time_ps = 0.5 * frame
        for arc in range(8):
            local = 0.0
            low = 0.0
            mean = 0.0
            if 2.0 <= time_ps <= 8.0:
                local = 2.0 if arc == 0 else 1.0 if arc in {1, 7} else 0.0
                low = 0.2
                mean = 0.1
            if 12.0 <= time_ps <= 18.0:
                local = -2.0 if arc == 4 else -1.0 if arc in {3, 5} else 0.0
                low = -0.2
                mean = -0.1
            field_rows.append(
                {
                    "step": frame * 1000,
                    "time_ns": time_ps / 1000.0,
                    "arc_index": arc,
                    "local_radius_A": local + low + mean,
                    "local_residual_A": local,
                    "low_order_shape_component_A": low,
                    "mean_radius_component_A": mean,
                }
            )
    event_rows = [
        {
            "primary_event_id": 1,
            "transition_step": 4000,
            "primary_arc_index": 0,
            "primary_residual_change_A": 2.0,
        },
        {
            "primary_event_id": 2,
            "transition_step": 24000,
            "primary_arc_index": 4,
            "primary_residual_change_A": -2.0,
        },
        {
            "primary_event_id": 3,
            "transition_step": 18000,
            "primary_arc_index": 2,
            "primary_residual_change_A": 1.0,
        },
    ]
    field_path = tmp_path / "field.csv"
    event_path = tmp_path / "events.csv"
    summary_path = tmp_path / "summary.json"
    _write_csv(field_path, field_rows)
    _write_csv(event_path, event_rows)
    summary_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "case_id": "case",
                "arc_count": 8,
                "event_cluster_count": 3,
            }
        )
    )
    sources_path = tmp_path / "sources.tsv"
    with sources_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=("case_id", "arc_kinematics", "event_sizes", "propagation_summary"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case",
                "arc_kinematics": field_path,
                "event_sizes": event_path,
                "propagation_summary": summary_path,
            }
        )
    return sources_path, field_path, event_path


def _config(write_event_table=True):
    return AnalysisConfig(
        fields=(
            "local_radius_A",
            "local_residual_A",
            "low_order_shape_component_A",
            "mean_radius_component_A",
        ),
        primary_field="local_radius_A",
        residual_field="local_residual_A",
        low_order_field="low_order_shape_component_A",
        mean_field="mean_radius_component_A",
        lags_ps=(-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 6.0),
        reference_lag_ps=-0.5,
        block_ps=10.0,
        null_samples=20,
        bootstrap_samples=20,
        random_seed=7,
        write_event_table=write_event_table,
    )


def test_even_circular_offsets_use_half_open_convention():
    assert np.array_equal(signed_arc_offsets(8), np.arange(-4, 4))
    assert np.array_equal(signed_arc_offsets(7), np.arange(-3, 4))


def test_support_gate_and_sign_aligned_response(tmp_path):
    sources_path, field_path, event_path = _toy_inputs(tmp_path)
    source = read_sources(sources_path)[0]
    assert source.case_id == "case"
    field = load_field_data(field_path, _config().fields, ColumnSpec())
    events = load_event_data(
        event_path,
        field,
        ColumnSpec(),
        _config(),
    )
    assert events.admitted_count == 2
    assert events.excluded_boundary_count == 1
    assert np.array_equal(events.signs, np.asarray([1.0, -1.0]))


def test_run_analysis_writes_maps_and_preregistered_estimands(tmp_path):
    sources_path, _, _ = _toy_inputs(tmp_path)
    output = tmp_path / "output"
    summary = run_analysis(sources_path, output, _config())
    assert summary["status"] == "PASS"
    assert summary["case_count"] == 1
    assert summary["primary_estimand_count"] == 6
    assert summary["case_summaries"][0]["admitted_event_count"] == 2
    assert summary["case_summaries"][0]["excluded_block_boundary_count"] == 1
    with (output / "primary_estimands.csv").open(newline="") as handle:
        estimands = list(csv.DictReader(handle))
    persistence = next(
        row
        for row in estimands
        if row["estimand"] == "primary_arc_residual_persistence_fast"
    )
    far = next(row for row in estimands if row["estimand"] == "far_total_response_fast")
    assert float(persistence["observed"]) == 2.0
    assert np.isclose(float(far["observed"]), 0.3)
    assert "bh_q_primary_family" in persistence
    with (output / "event_level_response.csv").open(newline="") as handle:
        event_rows = list(csv.DictReader(handle))
    assert len(event_rows) == 2 * 7 * 8 * 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["block_ps"] == 10.0
