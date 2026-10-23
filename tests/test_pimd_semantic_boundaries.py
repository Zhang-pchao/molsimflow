"""Analytic guards for independent observables, estimators and path samples."""

import numpy as np
from molsimflow.postprocess.path_contraction import contract_coordinates
from molsimflow.postprocess.pimd.kde import weighted_log_kde_1d, weighted_log_kde_2d
from molsimflow.postprocess.pimd.weights import normalized_log_weights
from molsimflow.postprocess.pimd.estimators import bead_density_estimators


def test_joint_density_keeps_correlated_frame_pairs():
    observations = np.array([[-1.0, -1.0], [1.0, 1.0]])
    logs = np.log([0.25, 0.75])
    grid = np.array([-1.0, 1.0])
    joint = weighted_log_kde_2d(observations, logs, grid, grid, [0.2, 0.2])
    marginal = weighted_log_kde_1d(observations[:, 0], logs, grid, 0.2)
    assert joint[0, 1] < marginal[0] + marginal[1] - 20
    np.testing.assert_allclose(joint[1, 1] - joint[0, 0], np.log(3.0), atol=1e-12)


def test_nonlinear_contraction_endpoints_are_observables():
    paths = np.array(
        [[[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0]], [[5.0, 0.0, 0.0]]]]
    )
    cv = lambda positions: positions[..., 0] ** 2

    def q(alpha):
        return np.array([cv(contract_coordinates(path, alpha)).mean() for path in paths])

    np.testing.assert_array_equal(q(1.0), cv(paths).mean(axis=(1, 2)))
    np.testing.assert_array_equal(q(0.0), cv(paths.mean(axis=1))[:, 0])
    assert not np.allclose(q(0.5), 0.5 * (q(0.0) + q(1.0)))


def test_single_bead_collapses_all_four_constructions():
    samples = np.array([-1.0, 0.2, 2.0])
    weights = normalized_log_weights(np.array([-1000.0, 0.4, -0.3]))
    grid = np.linspace(-3.0, 4.0, 19)
    density = weighted_log_kde_1d(samples, weights, grid, 0.3)
    result = bead_density_estimators(density[None], 0.6)
    expected = -0.6 * density
    expected -= expected.min()
    for name in ("probability_mean", "free_energy_mean"):
        np.testing.assert_array_equal(result[name], expected)
    # Arithmetic and contracted CVs of a one-bead path equal the same CV.
    for alpha in (0.0, 0.5, 1.0):
        path_cv = np.array(
            [contract_coordinates(np.array([[[x, 0.0, 0.0]]]), alpha)[0, 0, 0] for x in samples]
        )
        np.testing.assert_array_equal(weighted_log_kde_1d(path_cv, weights, grid, 0.3), density)
