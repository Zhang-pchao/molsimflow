import numpy as np

from molsimflow.postprocess.model_validation import (
    coordinate_coverage_status,
    force_error_metrics,
    nearest_coordinate_rmsd,
    relative_energy_errors,
)


def test_force_and_relative_energy_metrics_accept_explicit_regions_and_groups():
    reference_forces = np.zeros((3, 3))
    predicted_forces = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])

    metrics = force_error_metrics(reference_forces, predicted_forces, atom_indices=[0, 2])
    errors = relative_energy_errors(
        [10.0, 12.0, 20.0, 25.0],
        [30.0, 33.0, 40.0, 43.0],
        groups=["a", "a", "b", "b"],
    )

    assert metrics.mean == 3.5
    assert metrics.rms == np.sqrt(14.5)
    assert metrics.maximum == 5.0
    assert np.allclose(errors, [0.0, 1.0, 0.0, -2.0])


def test_nearest_coordinate_rmsd_uses_explicit_periodic_boxes_and_zero_based_subset():
    query = np.asarray([[9.9, 0.0, 0.0], [5.0, 5.0, 5.0]])
    candidates = np.asarray(
        [
            [[4.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
            [[0.1, 0.0, 0.0], [8.0, 8.0, 8.0]],
        ]
    )

    nearest = nearest_coordinate_rmsd(
        query,
        candidates,
        box_lengths=np.asarray([10.0, 10.0, 10.0]),
        atom_indices=[0],
    )

    assert nearest.index == 1
    assert np.isclose(nearest.rmsd, 0.2 / np.sqrt(3.0))
    assert coordinate_coverage_status(1.0e-9) == "exact_coordinate_match"
