from pathlib import Path

import numpy as np

from molsimflow.postprocess.nanobubble_attachment import (
    SelectedFrame,
    analyze_frame,
    first_persistent_contact,
    molecule_centers,
    run_analysis,
)


def test_molecule_centers_respect_periodic_boundary():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    nitrogen = np.array([[9.8, 5.0, 5.0], [0.2, 5.0, 5.0]])
    assert np.allclose(molecule_centers(nitrogen, bounds), [[0.0, 5.0, 5.0]])


def test_attachment_metrics_and_persistence():
    bounds = np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])
    frame = SelectedFrame(
        Path("segment.dump"), 0, 10, bounds,
        np.array([[5.0, 5.0, 2.0], [10.0, 10.0, 2.0]]),
        np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.5], [6.0, 5.0, 5.0], [6.0, 5.0, 5.5]]),
    )
    row = analyze_frame(frame, surface_z=2.0, cluster_cutoff=3.0, contact_cutoff=4.0)
    assert row["largest_cluster_n2_count"] == 2
    assert row["bubble_contact_n2_count"] == 2
    assert row["dissolved_or_disconnected_n2_count"] == 0
    assert row["bubble_lateral_radius_p90_A"] > 0
    rows = [{"bubble_contact_n2_count": value, "step": step} for step, value in enumerate([0, 2, 2, 1])]
    assert first_persistent_contact(rows, minimum=2, persistence=2)["step"] == 1


def test_per_trajectory_step_window_counts_only_selected_frames(monkeypatch, tmp_path: Path):
    bounds = np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])
    frame = SelectedFrame(
        Path("segment.dump"), 0, 10, bounds,
        np.array([[5.0, 5.0, 2.0], [10.0, 10.0, 2.0]]),
        np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.5], [6.0, 5.0, 5.0], [6.0, 5.0, 5.5]]),
    )

    def fake_frames(*_args):
        for step in (0, 10, 15, 20):
            yield SelectedFrame(frame.source, frame.source_frame, step, frame.bounds, frame.surface, frame.nitrogen)

    monkeypatch.setattr("molsimflow.postprocess.nanobubble_attachment.iter_selected_frames", fake_frames)
    monkeypatch.setattr("molsimflow.postprocess.nanobubble_attachment.write_plot", lambda *_args: None)
    args = type("Args", (), {
        "trajectory": [Path("segment.dump")], "min_step": [10], "max_step": [15],
        "surface_range": (1, 2), "nitrogen_range": (3, 6), "surface_z_A": 2.0,
        "reference_structure": None, "cluster_cutoff_A": 3.0, "contact_cutoff_A": 4.0,
        "max_frames": None, "drop_first_frame": False, "output_dir": tmp_path / "out",
        "output_stem": "core", "minimum_contact_n2": 1, "persistence_frames": 1,
        "timestep_fs": 0.5, "font_family": "Arial", "font_path": None,
    })()
    assert run_analysis(args)["analyzed_frames"] == 2
