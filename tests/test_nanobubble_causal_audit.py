from __future__ import annotations

import numpy as np
import pandas as pd

from molsimflow.postprocess.nanobubble_causal_audit import (
    _paired_summary,
    causal_rolling_median,
    detect_past_only_onsets,
    eligible_patch_ids,
    match_events_one_to_one,
    membership_turnover,
    periodic_site_weights,
    window_membership,
)


def test_causal_median_does_not_see_future_jump() -> None:
    values = [0.0, 0.0, 0.0, 10.0, 10.0]
    filtered = causal_rolling_median(values, 3)
    assert filtered[2] == 0.0
    assert filtered[3] == 0.0
    assert filtered[4] == 10.0


def test_past_only_onset_uses_future_only_for_persistence() -> None:
    values = [0.0] * 12 + [4.0] * 18
    table = pd.DataFrame({
        "step": np.arange(30), "time_ps": np.arange(30, dtype=float),
        "edge_coordinate_A": values, "observer_valid": [1] * 30,
    })
    events = detect_past_only_onsets(
        table, threshold_A=3.0, cadence_ps=1.0, smoothing_ps=1.0,
        reference_ps=2.0, persistence_ps=3.0, persistence_fraction=1.0,
        refractory_ps=5.0,
    )
    assert len(events) == 1
    assert events[0]["onset_time_ps"] == 12.0
    changed = table.copy(); changed.loc[16:, "edge_coordinate_A"] = -20.0
    changed_events = detect_past_only_onsets(
        changed, threshold_A=3.0, cadence_ps=1.0, smoothing_ps=1.0,
        reference_ps=2.0, persistence_ps=3.0, persistence_fraction=1.0,
        refractory_ps=5.0,
    )
    assert changed_events[0]["onset_time_ps"] == 12.0


def test_event_matching_is_one_to_one_and_stratified() -> None:
    reference = pd.DataFrame({
        "event_id": ["a", "b"], "branch_id": ["x", "x"],
        "patch_id": ["P1", "P1"], "edge": ["front", "front"],
        "process": ["advance", "advance"], "anchor_time_ps": [10.0, 20.0],
        "displacement_A": [4.0, 5.0],
    })
    candidate = pd.DataFrame({
        "event_id": ["c", "d", "e"], "branch_id": ["x", "x", "x"],
        "patch_id": ["P1", "P1", "P1"], "edge": ["front", "front", "rear"],
        "process": ["advance", "advance", "advance"],
        "anchor_time_ps": [10.1, 20.4, 10.0], "displacement_A": [3.8, 4.9, 9.0],
    })
    matches, metrics = match_events_one_to_one(reference, candidate, tolerance_ps=0.5)
    assert len(matches) == 2
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 2 / 3


def test_stationary_water_separates_fixed_and_swept_membership() -> None:
    ids = np.asarray([1, 2, 3])
    coordinates = np.asarray([[1.0, 2.0, 1.0], [3.0, 2.0, 1.0], [7.0, 2.0, 1.0]])
    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    first = window_membership(
        ids, coordinates, bounds, drive_index=0, transverse_index=1,
        reference_drive_A=2.0, drive_halfwidth_A=2.0,
        transverse_lo_A=0.0, transverse_hi_A=5.0, surface_top_A=0.0, height_A=3.0,
    )
    fixed = window_membership(
        ids, coordinates, bounds, drive_index=0, transverse_index=1,
        reference_drive_A=2.0, drive_halfwidth_A=2.0,
        transverse_lo_A=0.0, transverse_hi_A=5.0, surface_top_A=0.0, height_A=3.0,
    )
    swept = window_membership(
        ids, coordinates, bounds, drive_index=0, transverse_index=1,
        reference_drive_A=6.0, drive_halfwidth_A=2.0,
        transverse_lo_A=0.0, transverse_hi_A=5.0, surface_top_A=0.0, height_A=3.0,
    )
    assert membership_turnover(first, fixed) == 0.0
    assert membership_turnover(first, swept) > 0.0


def test_periodic_site_weights_wrap_across_boundary() -> None:
    sites = np.asarray([[0.2, 5.0], [5.0, 5.0], [9.8, 5.0]])
    point = np.asarray([0.0, 5.0])
    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    weights = periodic_site_weights(sites, point, bounds, sigma_A=1.0, cutoff_A=2.0)
    assert np.isclose(weights[0], weights[2])
    assert weights[1] == 0.0
    assert np.sum(weights) == 1.0


def test_empty_event_summary_is_explicit() -> None:
    rows = _paired_summary(pd.DataFrame())
    assert len(rows) == 5
    assert all(row["paired_events"] == 0 for row in rows)


def test_explicit_patch_exclusion_is_consistent_across_branches() -> None:
    first = eligible_patch_ids(["P00", "P01", "P02"], ["P00", "P01"], ["P00"])
    second = eligible_patch_ids(["P00", "P01", "P03"], ["P00", "P01"], ["P00"])
    assert first == ["P01"]
    assert second == ["P01"]
