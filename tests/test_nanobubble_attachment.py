from pathlib import Path

import numpy as np

from molsimflow.postprocess.nanobubble_attachment import (
    SelectedFrame,
    analyze_frame,
    first_persistent_contact,
    molecule_centers,
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
    rows = [{"bubble_contact_n2_count": value, "step": step} for step, value in enumerate([0, 2, 2, 1])]
    assert first_persistent_contact(rows, minimum=2, persistence=2)["step"] == 1
