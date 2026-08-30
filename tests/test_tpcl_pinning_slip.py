import json
import math

import numpy as np

from molsimflow.cli import build_parser
from molsimflow.postprocess.tpcl_pinning_slip import (
    EVENT_METRICS,
    SurfaceSites,
    TpclConfig,
    contour_differential,
    detect_events,
    local_site_metrics,
    ray_polygon_radii,
)


def _config(tmp_path, *, arc_bins=8):
    return TpclConfig(
        kind="nanodroplet",
        trajectories=(tmp_path / "trajectory.lammpstrj",),
        initial_xyz=tmp_path / "model.xyz",
        surface_range=(1, 10),
        phase_range=(11, 20),
        water_range=(11, 20),
        surface_z_A=5.0,
        cluster_cutoff_A=3.5,
        contact_cutoff_A=3.5,
        arc_bins=arc_bins,
        minimum_contact_points=4,
    )


def test_ray_resampling_and_circle_differential():
    square = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    theta = np.arange(8) * math.pi / 4
    radii = ray_polygon_radii(square, theta)
    np.testing.assert_allclose(radii[::2], 2.0)
    np.testing.assert_allclose(radii[1::2], 2.0 * math.sqrt(2.0))

    circle_theta = np.arange(36) * 2 * math.pi / 36
    circle = 10.0 * np.column_stack((np.cos(circle_theta), np.sin(circle_theta)))
    normal, curvature, arc_length = contour_differential(circle, 2 * math.pi / 36)
    np.testing.assert_allclose(normal, circle / 10.0, atol=1e-12)
    np.testing.assert_allclose(curvature, 0.1, rtol=0.01)
    assert np.all(arc_length > 0)


def test_config_and_top_level_cli(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "kind": "nanobubble",
                "trajectories": [str(tmp_path / "a.dump")],
                "initial_xyz": str(tmp_path / "model.xyz"),
                "surface_range": "1:10",
                "phase_range": [11, 20],
                "water_range": [21, 50],
                "surface_z_A": 5.0,
            }
        )
    )
    config = TpclConfig.load(path)
    assert config.surface_range == (1, 10)
    assert config.minimum_dwell_frames == 4
    args = build_parser().parse_args(
        [
            "postprocess",
            "tpcl-pinning-slip",
            "--config",
            str(path),
            "--output-dir",
            str(tmp_path / "out"),
            "--no-plots",
        ]
    )
    assert args.postprocess_command == "tpcl-pinning-slip"
    assert args.no_plots


def test_site_mapping_is_periodic_and_labels_boundary(tmp_path):
    config = _config(tmp_path, arc_bins=8)
    sites = SurfaceSites(
        atom_ids=np.array([1, 2, 3]),
        site_types=np.array(["CH3", "SiOH", "CH3"]),
        sioh_hydrogen_ids={},
        ch3_count=2,
        sioh_count=1,
    )
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 20.0]])
    site_coordinates = np.array([[0.2, 5.0, 5.0], [9.8, 5.0, 5.0], [5.0, 5.0, 5.0]])
    rows = local_site_metrics(
        np.array([[0.05, 5.0], [9.85, 5.0]]), site_coordinates, sites, bounds, config
    )
    assert rows[0]["nearest_site_id"] == 1
    assert rows[1]["nearest_site_id"] == 2
    assert rows[1]["near_chemical_boundary"]


def _local_row(step, arc, position, residual, center):
    row = {
        "step": step,
        "arc_index": arc,
        "theta_deg": arc * 45.0,
        "absolute_normal_position_A": position,
        "local_residual_A": residual,
        "phase_center_normal_position_A": center,
        "nearest_site_id": 7,
        "nearest_site_type": "SiOH",
        "localization_noise_A": 0.5,
        "consecutive_from_previous": step > 0,
    }
    row.update({name: 1.0 for name in EVENT_METRICS})
    return row


def test_repeated_local_dwell_jump_is_candidate(tmp_path):
    config = _config(tmp_path)
    steps = [index * 20_000 for index in range(12)]
    local_by_step = {}
    for index, step in enumerate(steps):
        state = 0 if index < 4 else 1 if index < 10 else 2
        rows = []
        for arc in range(config.arc_bins):
            if arc == 0:
                position = 10.0 + 3.0 * state
                residual = 3.0 * state
            elif arc == 1:
                position = 10.0 + 2.5 * state
                residual = 2.5 * state
            else:
                position = 10.0
                residual = 0.0
            rows.append(_local_row(step, arc, position, residual, 0.0))
        local_by_step[step] = rows
    frame_rows = [{"step": step} for step in steps]
    events = detect_events(
        frame_rows,
        local_by_step,
        config,
        dwell_tolerance=1.0,
        jump_threshold=2.0,
        expected_step=20_000,
    )
    candidates = [event for event in events if event["quality_status"] == "candidate_stick_slip"]
    assert len(candidates) >= 2
    assert all(event["dwell_frames"] >= 4 for event in candidates)
    assert len({event["event_cluster_id"] for event in candidates}) == 2
    assert all(event["event_cluster_arc_count"] == 2 for event in candidates)


def test_whole_object_jump_is_rejected(tmp_path):
    config = _config(tmp_path)
    steps = [index * 20_000 for index in range(6)]
    local_by_step = {}
    for index, step in enumerate(steps):
        state = 0.0 if index < 4 else 3.0
        local_by_step[step] = [
            _local_row(step, arc, 10.0 + state, 0.0, state)
            for arc in range(config.arc_bins)
        ]
    events = detect_events(
        [{"step": step} for step in steps],
        local_by_step,
        config,
        dwell_tolerance=1.0,
        jump_threshold=2.0,
        expected_step=20_000,
    )
    assert events
    assert all(event["quality_status"] == "rejected" for event in events)
    assert all("whole_object_translation_dominated" in event["rejection_reason"] for event in events)
