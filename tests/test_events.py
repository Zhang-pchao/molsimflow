import csv

from molsimflow.postprocess.events import (
    TransitionEventConfig,
    add_derivative_features,
    analyze_transition_events,
    detect_transition_events,
    extract_event_aligned_profiles,
    load_feature_table,
    summarize_event_aligned_profiles,
)


def _write_feature_table(path):
    rows = [
        {"frame": 0, "time_ns": 0.000, "Nw_bridge": 5, "dewet_fraction": 0.10, "water_bridge_connected_flag": "True", "d3d_all": 8.0, "feature_x": 1.0},
        {"frame": 1, "time_ns": 0.001, "Nw_bridge": 4, "dewet_fraction": 0.20, "water_bridge_connected_flag": "True", "d3d_all": 7.0, "feature_x": 2.0},
        {"frame": 2, "time_ns": 0.002, "Nw_bridge": 2, "dewet_fraction": 0.60, "water_bridge_connected_flag": "False", "d3d_all": 6.0, "feature_x": 4.0},
        {"frame": 3, "time_ns": 0.003, "Nw_bridge": 2, "dewet_fraction": 0.70, "water_bridge_connected_flag": "False", "d3d_all": 5.0, "feature_x": 5.0},
        {"frame": 4, "time_ns": 0.004, "Nw_bridge": 3, "dewet_fraction": 0.50, "water_bridge_connected_flag": "True", "d3d_all": 4.0, "feature_x": 6.0},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_detect_transition_events_and_profiles(tmp_path):
    feature_path = tmp_path / "features.csv"
    _write_feature_table(feature_path)
    rows = load_feature_table(feature_path)
    rows, basis, created = add_derivative_features(rows)

    assert basis == "time"
    assert "dNw_bridge_dt" in created

    events, thresholds = detect_transition_events(
        rows,
        config=TransitionEventConfig(
            event_method="connectivity_loss",
            min_event_separation=1,
            event_window_before=1,
            event_window_after=1,
        ),
    )

    assert thresholds["events_detected"] == 1
    assert events[0]["event_anchor_index"] == 2
    assert events[0]["event_type"] == "connectivity_loss"

    profiles = extract_event_aligned_profiles(rows, events, event_window_before=1, event_window_after=1)
    summary = summarize_event_aligned_profiles(profiles)

    assert [row["relative_index"] for row in profiles] == [-1, 0, 1]
    assert any(row["feature"] == "Nw_bridge" and row["relative_index"] == 0 for row in summary)


def test_analyze_transition_events_writes_outputs(tmp_path):
    feature_path = tmp_path / "features.csv"
    _write_feature_table(feature_path)

    outputs = analyze_transition_events(
        input_csv=feature_path,
        output_dir=tmp_path / "events",
        config=TransitionEventConfig(
            event_method="hybrid",
            min_event_separation=1,
            event_window_before=1,
            event_window_after=1,
            lag_window=1,
            change_window=1,
        ),
    )

    assert outputs["events"].exists()
    assert outputs["event_aligned_profiles"].exists()
    assert outputs["feature_lag_correlation"].exists()

    with outputs["events"].open() as handle:
        events = list(csv.DictReader(handle))

    assert events[0]["event_type"]
