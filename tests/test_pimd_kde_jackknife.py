"""KDE frame-jackknife checks against direct Gaussian sums."""

import numpy as np
import pytest

from molsimflow.postprocess.pimd_reweight import quantum_kde_block_jackknife


@pytest.mark.parametrize("dimensions", [1, 2])
def test_kde_jackknife_matches_direct_kernel_oracle(dimensions):
    values = np.array(
        [
            [[-0.8, 0.1], [-0.5, 0.4]],
            [[0.2, -0.3], [0.4, 0.1]],
            [[0.7, 0.6], [0.3, 0.8]],
            [[-0.2, -0.7], [0.1, -0.2]],
            [[0.5, -0.1], [0.8, 0.3]],
            [[-0.6, 0.7], [-0.3, 0.5]],
        ]
    )[:, :, :dimensions]
    logw = np.log([1, 2, 3, 1, 4, 2])
    grids = [np.array([-0.4, 0.0, 0.6]), np.array([-0.5, 0.2])][:dimensions]
    bandwidth = np.array([0.4, 0.7])[:dimensions]
    ref = [1, 0][:dimensions]
    points = np.column_stack([g.ravel() for g in np.meshgrid(*grids)])
    ref_flat = 1

    def direct(keep):
        kernel = np.exp(
            -0.5
            * np.sum(
                ((points[:, None, None, :] - values[keep][None, :, :, :]) / bandwidth) ** 2, axis=-1
            )
        )
        density = np.sum(kernel * np.exp(logw[keep])[None, :, None], axis=(1, 2))
        return -0.7 * np.log(density / density[ref_flat])

    options = {
        "kbt": 0.7, "block_frames": 2, "reference_grid_index": ref, "relative_density_support": 1e-12
    }
    result = quantum_kde_block_jackknife(values, logw, grids, bandwidth, **options)
    replicates = np.array([direct(np.arange(6) // 2 != b) for b in range(3)])
    expected = np.sqrt(2 / 3 * np.sum((replicates - replicates.mean(axis=0)) ** 2, axis=0))
    np.testing.assert_allclose(result["free_energy_difference"].ravel(), direct(np.ones(6, bool)))
    np.testing.assert_allclose(result["standard_error"].ravel(), expected, atol=1e-14)
    repeated = quantum_kde_block_jackknife(
        np.repeat(values, 3, axis=1), logw + 10000, grids, bandwidth, **options
    )
    np.testing.assert_allclose(result["standard_error"], repeated["standard_error"], atol=1e-12)


def test_support_loss_is_explicit():
    values = np.array([-2, -2, 0, 0, 0, 0], dtype=float)[:, None, None]
    options = {"kbt": 1.0, "block_frames": 2, "reference_grid_index": [1], "relative_density_support": 1e-3}
    result = quantum_kde_block_jackknife(values, np.zeros(6), [[-2, 0]], [0.2], **options)
    assert not result["support"][0]
    assert np.isnan(result["standard_error"][0])
    options["reference_grid_index"] = [0]
    with pytest.raises(ValueError, match="reference grid point"):
        quantum_kde_block_jackknife(values, np.zeros(6), [[-2, 0]], [0.2], **options)


@pytest.mark.parametrize(
    "change",
    [
        {"block_frames": 0},
        {"block_frames": True},
        {"block_frames": 4},
        {"reference_grid_index": [2]},
        {"reference_grid_index": None},
        {"reference_grid_index": [True]},
        {"relative_density_support": 0},
        {"kbt": float("nan")},
    ],
)
def test_invalid_kde_uncertainty_arguments(change):
    options = {"kbt": 1.0, "block_frames": 2, "reference_grid_index": [0], "relative_density_support": 1e-8}
    options.update(change)
    with pytest.raises(ValueError):
        quantum_kde_block_jackknife(np.zeros((6, 1, 1)), np.zeros(6), [[0, 1]], [0.5], **options)
