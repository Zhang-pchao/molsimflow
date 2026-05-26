import csv

from molsimflow.postprocess.coupling import (
    CouplingConfig,
    analyze_coupling,
    compute_pairwise_coupling,
    compute_state_comparison,
    infer_predictor_columns,
)
from molsimflow.postprocess.events import load_feature_table


def _write_feature_table(path):
    rows = [
        {"frame": 0, "time_ns": 0.000, "Nw_bridge": 5, "dewet_fraction": 0.10, "n_bridge_na": 0, "n_bridge_cl": 0},
        {"frame": 1, "time_ns": 0.001, "Nw_bridge": 4, "dewet_fraction": 0.20, "n_bridge_na": 1, "n_bridge_cl": 0},
        {"frame": 2, "time_ns": 0.002, "Nw_bridge": 2, "dewet_fraction": 0.60, "n_bridge_na": 2, "n_bridge_cl": 1},
        {"frame": 3, "time_ns": 0.003, "Nw_bridge": 2, "dewet_fraction": 0.70, "n_bridge_na": 3, "n_bridge_cl": 1},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_events(path):
    rows = [{"event_id": 1, "event_anchor_index": 2, "event_frame": 2, "event_time": 0.002, "event_type": "nw_drop"}]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_coupling_infers_predictors_and_computes_pairs(tmp_path):
    feature_table = tmp_path / "features.csv"
    _write_feature_table(feature_table)
    rows = load_feature_table(feature_table)

    predictors = infer_predictor_columns(rows, target_columns=("Nw_bridge", "dewet_fraction"))
    coupling = compute_pairwise_coupling(rows, predictors=predictors, targets=("dewet_fraction",))

    assert set(predictors) == {"n_bridge_cl", "n_bridge_na"}
    assert any(row["predictor"] == "n_bridge_na" and row["n_pairs"] == 4 for row in coupling)


def test_coupling_state_comparison(tmp_path):
    feature_table = tmp_path / "features.csv"
    _write_feature_table(feature_table)
    rows = load_feature_table(feature_table)

    state = compute_state_comparison(
        rows,
        predictors=("n_bridge_na",),
        state_target="dewet_fraction",
        low_quantile=0.25,
        high_quantile=0.75,
    )

    delta = [row for row in state if row["state_group"] == "high_minus_low"][0]
    assert delta["mean"] > 0


def test_analyze_coupling_writes_outputs(tmp_path):
    feature_table = tmp_path / "features.csv"
    events = tmp_path / "events.csv"
    _write_feature_table(feature_table)
    _write_events(events)

    outputs = analyze_coupling(
        feature_table=feature_table,
        transition_events=events,
        output_dir=tmp_path / "out",
        predictor_columns=("n_bridge_na", "n_bridge_cl"),
        target_columns=("Nw_bridge", "dewet_fraction"),
        config=CouplingConfig(lag_window=1, event_window_before=1, event_window_after=1),
    )

    assert outputs["coupling"].exists()
    assert outputs["lag_correlation"].exists()
    assert outputs["event_aligned_summary"].exists()

    with outputs["event_aligned_profiles"].open() as handle:
        profiles = list(csv.DictReader(handle))

    assert [int(row["relative_index"]) for row in profiles] == [-1, 0, 1]
