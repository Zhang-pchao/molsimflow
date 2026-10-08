import csv
import gzip
import math
from argparse import Namespace

import numpy as np

from molsimflow.postprocess.event_aligned_water_order import (
    Event,
    event_effect,
    load_event_windows,
    match_same_arc_dwells,
    randomization_null_rows,
    run_analysis,
)


def _grid(steps, arcs):
    return {
        (step, arc): {
            "metric": float(step + arc),
            "local_radius_A": 10.0 + 0.01 * step,
            "local_residual_A": 0.05 * math.sin(step),
        }
        for step in steps
        for arc in arcs
    }


def test_event_effect_uses_pre_and_post_without_transition():
    steps = list(range(10))
    grid = _grid(steps, [0])
    pre, post, effect = event_effect(
        5,
        0,
        steps,
        grid,
        "metric",
        [-2, -1],
        [1, 2],
        circular=False,
    )
    assert pre == 3.5
    assert post == 6.5
    assert effect == 3.0


def test_same_arc_matching_avoids_observed_event_neighborhood():
    steps = list(range(80))
    grid = _grid(steps, [0])
    event = Event(1, 0, 40, 10.4, 1.0)
    controls = match_same_arc_dwells(
        [event],
        steps,
        grid,
        [-2, -1, 0, 1, 2],
        controls_per_event=3,
        exclusion_frames=5,
    )
    assert len(controls) == 3
    assert all(abs(control.anchor_step - 40) > 5 for control in controls)


def test_randomization_null_is_deterministic_and_complete():
    steps = list(range(100))
    grid = _grid(steps, [0, 1])
    events = [Event(1, 0, 30, 10.3, 1.0), Event(2, 1, 60, 10.6, 1.0)]
    first = randomization_null_rows(
        events,
        steps,
        grid,
        ["metric"],
        [-2, -1],
        [1, 2],
        null_samples=5,
        block_frames=10,
        random_seed=9,
    )
    second = randomization_null_rows(
        events,
        steps,
        grid,
        ["metric"],
        [-2, -1],
        [1, 2],
        null_samples=5,
        block_frames=10,
        random_seed=9,
    )
    assert first == second
    assert len(first) == 10
    assert {row["control_type"] for row in first} == {
        "per_arc_block_circular_shift",
        "event_time_permutation",
    }
    assert np.all(np.isfinite([row["mean_post_minus_pre"] for row in first]))


def _write_csv(path, rows, *, compressed=False):
    if compressed:
        with gzip.open(path, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_event_windows_exclude_incomplete_boundary_support(tmp_path):
    events = [Event(event_id, 0, 20 * event_id, 10.0, 1.0) for event_id in range(1, 4)]
    rows = [
        {
            "event_id": event.event_id,
            "arc_index": 0,
            "step": event.transition_step + relative,
            "relative_frame": relative,
            "phase": "pre" if relative < 0 else "post" if relative > 0 else "transition",
        }
        for event in events
        for relative in range(-2, 3)
        if not (event.event_id == 3 and relative == 2)
    ]
    path = tmp_path / "windows.csv"
    _write_csv(path, rows)
    windows, eligibility = load_event_windows(path, events)
    assert set(windows) == {1, 2}
    assert eligibility[-1]["support_status"].startswith("excluded_")


def test_run_analysis_writes_all_control_families(tmp_path):
    steps = list(range(80))
    water_rows = [
        {"step": step, "arc_index": 0, "metric": float(step)} for step in steps
    ]
    track_rows = [
        {
            "step": step,
            "arc_index": 0,
            "local_radius_A": 10.0 + 0.01 * step,
            "local_residual_A": 0.05 * math.sin(step),
        }
        for step in steps
    ]
    event_rows = [
        {
            "event_id": 1,
            "arc_index": 0,
            "transition_step": 40,
            "quality_status": "candidate",
            "pre_local_radius_A": 10.4,
            "dwell_tolerance_A": 1.0,
        }
    ]
    window_rows = [
        {
            "event_id": 1,
            "arc_index": 0,
            "step": 40 + relative,
            "relative_frame": relative,
            "phase": "pre" if relative < 0 else "post" if relative > 0 else "transition",
        }
        for relative in range(-2, 3)
    ]
    water_path = tmp_path / "water.csv.gz"
    track_path = tmp_path / "arc.csv"
    event_path = tmp_path / "events.csv"
    windows_path = tmp_path / "windows.csv"
    _write_csv(water_path, water_rows, compressed=True)
    _write_csv(track_path, track_rows)
    _write_csv(event_path, event_rows)
    _write_csv(windows_path, window_rows)
    summary = run_analysis(
        Namespace(
            case_id="case",
            water_order_by_arc=water_path,
            arc_kinematics=track_path,
            events=event_path,
            event_windows=windows_path,
            output_dir=tmp_path / "out",
            event_status="candidate",
            metrics="metric",
            matched_controls_per_event=2,
            event_exclusion_frames=5,
            block_frames=10,
            null_samples=5,
            random_seed=4,
        )
    )
    assert summary["status"] == "PASS"
    assert summary["catalog_event_count"] == 1
    assert summary["event_count"] == 1
    assert summary["support_excluded_event_count"] == 0
    assert summary["matched_dwell_count"] == 2
    assert summary["null_effect_rows"] == 10
    assert (tmp_path / "out/randomization_statistics.csv").is_file()
    assert (tmp_path / "out/event_window_support_eligibility.csv").is_file()
