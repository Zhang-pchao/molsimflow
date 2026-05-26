import csv
import math

from molsimflow.postprocess.bridge_water_dynamics import (
    BridgeWaterDynamicsConfig,
    TraceInputSpec,
    analyze_bridge_water_flux,
    analyze_seed_water_survival,
    build_bridge_water_flux_frame_table,
    build_seed_water_survival_frame_table,
    load_trace_input_manifest,
)


def _write_trace(path):
    rows = [
        {
            "time_ns": "0.000",
            "dynamic_surface_gap_est_A": "6.0",
            "state": "separated",
            "n_current_bridge_waters": "3",
            "n_newly_tracked_waters_this_frame": "3",
            "n_seed_waters": "3",
            "n_seed_retained_in_bridge": "3",
            "seed_retention_fraction": "1.0",
            "n_new_bridge_waters": "0",
            "new_bridge_water_fraction": "0.0",
            "median_seed_displacement_A": "0.1",
            "p90_seed_displacement_A": "0.2",
        },
        {
            "time_ns": "0.001",
            "dynamic_surface_gap_est_A": "4.0",
            "state": "thin_bridge",
            "n_current_bridge_waters": "4",
            "n_newly_tracked_waters_this_frame": "1",
            "n_seed_waters": "3",
            "n_seed_retained_in_bridge": "2",
            "seed_retention_fraction": "0.6666667",
            "n_new_bridge_waters": "1",
            "new_bridge_water_fraction": "0.25",
            "median_seed_displacement_A": "0.4",
            "p90_seed_displacement_A": "0.8",
        },
        {
            "time_ns": "0.002",
            "dynamic_surface_gap_est_A": "2.0",
            "state": "thin_bridge",
            "n_current_bridge_waters": "2",
            "n_newly_tracked_waters_this_frame": "0",
            "n_seed_waters": "3",
            "n_seed_retained_in_bridge": "1",
            "seed_retention_fraction": "0.3333333",
            "n_new_bridge_waters": "0",
            "new_bridge_water_fraction": "0.0",
            "median_seed_displacement_A": "0.7",
            "p90_seed_displacement_A": "1.1",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_state(path):
    rows = [
        {"time_ns": "0.000", "surface_gap_estimate_A": "5.5", "state": "separated"},
        {"time_ns": "0.001", "surface_gap_estimate_A": "3.5", "state": "thin_bridge"},
        {"time_ns": "0.002", "surface_gap_estimate_A": "1.5", "state": "thin_bridge"},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path):
    trace = tmp_path / "bridge_water_trace_metrics.csv"
    state = tmp_path / "coalescence_state_table.csv"
    _write_trace(trace)
    _write_state(state)
    return [TraceInputSpec(case_label="caseA", trace_metrics=trace, state_table=state)]


def test_manifest_loads_relative_paths(tmp_path):
    trace = tmp_path / "trace.csv"
    _write_trace(trace)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("case_label,trace_metrics\ncaseA,trace.csv\n", encoding="utf-8")

    specs = load_trace_input_manifest(manifest)

    assert specs[0].case_label == "caseA"
    assert specs[0].trace_metrics == trace


def test_bridge_water_flux_frame_table(tmp_path):
    rows = build_bridge_water_flux_frame_table(
        _inputs(tmp_path),
        config=BridgeWaterDynamicsConfig(gap_source="coalescence", gap_bin_width_A=2.0),
    )

    assert [row["entry_count_proxy"] for row in rows] == [0.0, 1.0, 0.0]
    assert [row["exit_count_proxy"] for row in rows] == [0.0, 0.0, 2.0]
    assert [row["surface_gap_A"] for row in rows] == [5.5, 3.5, 1.5]
    assert math.isclose(rows[2]["exit_rate_per_ps_proxy"], 2.0)


def test_seed_water_survival_frame_table(tmp_path):
    rows = build_seed_water_survival_frame_table(
        _inputs(tmp_path),
        config=BridgeWaterDynamicsConfig(gap_source="trace", gap_bin_width_A=2.0),
    )

    assert [row["monotonic_seed_retained_count_proxy"] for row in rows] == [3.0, 2.0, 1.0]
    assert [row["seed_exit_proxy_count_this_frame"] for row in rows] == [0.0, 1.0, 1.0]
    assert math.isclose(rows[-1]["seed_survival_fraction_monotonic_proxy"], 1.0 / 3.0)


def test_bridge_water_dynamics_writes_outputs(tmp_path):
    inputs = _inputs(tmp_path)
    flux_outputs = analyze_bridge_water_flux(
        inputs,
        output_dir=tmp_path / "flux",
        config=BridgeWaterDynamicsConfig(gap_source="coalescence", gap_bin_width_A=2.0),
    )
    seed_outputs = analyze_seed_water_survival(
        inputs,
        output_dir=tmp_path / "seed",
        config=BridgeWaterDynamicsConfig(gap_source="coalescence", gap_bin_width_A=2.0),
    )

    assert flux_outputs["frame_table"].exists()
    assert flux_outputs["binned"].exists()
    assert seed_outputs["exit_events"].exists()

    with seed_outputs["exit_events"].open() as handle:
        events = list(csv.DictReader(handle))

    assert [float(row["seed_exit_proxy_count_this_frame"]) for row in events] == [1.0, 1.0]
