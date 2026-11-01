from __future__ import annotations

import pandas as pd

from molsimflow.postprocess.nanobubble_edge_mechanism import (
    _edge_process,
    _filter_patch_edges,
    _m1_features,
    _null_changes,
    _qualified_patch_keys,
    _recent_mean,
    _segments,
    _validation_groups,
)


def test_patch_segments_do_not_bridge_observer_gaps() -> None:
    table = pd.DataFrame({
        "time_ps": [0.0, 1.0, 2.0, 3.0, 4.0],
        "observer_valid": [1, 1, 0, 1, 1],
        "edge_coordinate_A": [0.0, 0.1, float("nan"), 1.0, 1.1],
    })
    segments = _segments(table, 1.0)
    assert [len(segment) for segment in segments] == [2, 2]


def test_null_changes_and_process_direction() -> None:
    table = pd.DataFrame({
        "time_ps": list(range(30)), "observer_valid": [1] * 30,
        "edge_coordinate_A": [0.0] * 15 + [3.0] * 15,
    })
    values = _null_changes(
        table, cadence_ps=1.0, smoothing_ps=3.0,
        pre_window_ps=5.0, post_window_ps=5.0,
    )
    assert values.max() == 3.0
    assert _edge_process("front", 3.0) == "gas_advance_liquid_retreat"
    assert _edge_process("rear", 3.0) == "gas_retreat_liquid_advance"


def test_high_cadence_m1_and_recent_mean() -> None:
    features = _m1_features({"water_feature_mode": "high_cadence"})
    assert "dipole_rotation_history_deg" in features
    assert "raw_force_impulse_abs_history" in features
    table = pd.DataFrame({"impulse": [-2.0, float("nan"), 4.0]})
    assert _recent_mean(table, "impulse", absolute=True) == 3.0


def test_validation_groups_are_case_aware() -> None:
    risk = pd.DataFrame({
        "case_id": ["oh", "mixed"], "branch_id": ["fx", "fx"],
        "block_id": [0, 0], "patch_id": ["P00", "P00"],
    })
    groups = _validation_groups(risk)
    assert groups["held_branch"].tolist() == ["oh:fx", "mixed:fx"]
    assert groups["held_block"].tolist() == ["oh:fx:0", "mixed:fx:0"]
    assert groups["held_patch"].tolist() == ["oh:fx:P00", "mixed:fx:P00"]


def test_observer_qualification_filters_all_downstream_patch_rows() -> None:
    calibrations = [
        {"patch_id": "P00", "edge": "front", "calibration_status": "HOLD_LOW_COVERAGE"},
        {"patch_id": "P01", "edge": "front", "calibration_status": "PASS"},
        {"patch_id": "P01", "edge": "rear", "calibration_status": "PASS"},
    ]
    qualified = _qualified_patch_keys(calibrations)
    assert qualified == {("P01", "front"), ("P01", "rear")}
    table = pd.DataFrame({
        "patch_id": ["P00", "P01", "P01"],
        "edge": ["front", "front", "rear"],
        "value": [0, 1, 2],
    })
    filtered = _filter_patch_edges(table, qualified)
    assert filtered.value.tolist() == [1, 2]
