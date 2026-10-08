import csv
import json

import numpy as np

from molsimflow.postprocess.tpcl_propagation import PropagationConfig, analyze_tpcl_propagation


def _write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_modes_clusters_sizes_and_hazard(tmp_path):
    arcs = []
    for frame in range(12):
        for arc in range(8):
            radius = 10.0 + 2.0 * np.cos(2.0 * np.pi * 2 * arc / 8)
            if frame >= 5 and arc in {0, 1}:
                radius += 2.0
            if frame >= 7 and arc == 2:
                radius += 2.0
            arcs.append(
                {
                    "step": frame,
                    "time_ns": frame / 1000.0,
                    "arc_index": arc,
                    "local_radius_A": radius,
                    "local_residual_A": radius - 10.0,
                    "localization_noise_A": 0.5,
                }
            )
    events = [
        {
            "event_id": 1,
            "arc_index": 0,
            "transition_step": 5,
            "transition_time_ns": 0.005,
            "end_step": 4,
            "post_end_step": 5,
            "quality_status": "candidate",
            "jump_distance_A": 2.0,
            "localization_noise_A": 0.5,
        },
        {
            "event_id": 2,
            "arc_index": 1,
            "transition_step": 5,
            "transition_time_ns": 0.005,
            "end_step": 4,
            "post_end_step": 5,
            "quality_status": "candidate",
            "jump_distance_A": 2.0,
            "localization_noise_A": 0.5,
        },
        {
            "event_id": 3,
            "arc_index": 2,
            "transition_step": 7,
            "transition_time_ns": 0.007,
            "end_step": 6,
            "post_end_step": 7,
            "quality_status": "candidate",
            "jump_distance_A": 2.0,
            "localization_noise_A": 0.5,
        },
        {
            "event_id": 4,
            "arc_index": 7,
            "transition_step": 9,
            "transition_time_ns": 0.009,
            "end_step": 8,
            "post_end_step": 9,
            "quality_status": "rejected",
            "jump_distance_A": 2.0,
            "localization_noise_A": 0.5,
        },
    ]
    geometry = [
        {"step": frame, "contact_contour_area_A2": 100.0 + frame}
        for frame in range(12)
    ]
    arc_path, event_path, geometry_path = tmp_path / "arc.csv", tmp_path / "events.csv", tmp_path / "geometry.csv"
    _write(arc_path, arcs)
    _write(event_path, events)
    _write(geometry_path, geometry)

    outputs = analyze_tpcl_propagation(
        "case",
        arc_path,
        event_path,
        tmp_path / "out",
        geometry_path,
        PropagationConfig(
            max_mode=3,
            cluster_window_ps=0.5,
            null_samples=20,
            random_seed=7,
            time_bin_edges_ps=(0.0, 1.0, 3.0, 10.0),
            event_status="candidate",
        ),
    )
    summary = json.loads(outputs["summary"].read_text())
    assert summary["frame_count"] == 12
    assert summary["admitted_event_rows"] == 3
    assert summary["event_cluster_count"] == 2
    with outputs["frame_modes"].open() as handle:
        modes = list(csv.DictReader(handle))
    assert abs(float(modes[0]["mode_2_amplitude_A"]) - 2.0) < 1.0e-12
    with outputs["event_sizes"].open() as handle:
        sizes = list(csv.DictReader(handle))
    assert len(sizes) == 2
    assert int(sizes[0]["affected_arc_count"]) == 2
    assert float(sizes[0]["event_size_residual_A2"]) > 0.0
    with outputs["pair_hazard"].open() as handle:
        hazard = list(csv.DictReader(handle))
    assert hazard
