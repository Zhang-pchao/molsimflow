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


def test_markov_chain_kde_block_length_calibration(record_property):
    """Check weighted KDE uncertainty against known exponential time correlation."""
    rng = np.random.default_rng(782431)
    frames = 2048
    replicas = 96
    rho = 0.9
    flip_probability = (1 - rho) / 2
    state_weight = 2.0
    bandwidth = 0.5
    estimates = []
    short_variances = []
    long_variances = []
    for _ in range(replicas):
        # A stationary symmetric two-state Markov chain has Cov(X_0,X_l)=rho^l/4.
        initial = rng.integers(2)
        flips = rng.binomial(1, flip_probability, size=frames - 1)
        states = np.concatenate(([initial], (initial + np.cumsum(flips)) % 2))
        values = states.astype(float)[:, None, None]
        log_weights = states * np.log(state_weight)
        results = [
            quantum_kde_block_jackknife(
                values, log_weights, [[0.0, 1.0]], [bandwidth],
                kbt=1.0, block_frames=size, reference_grid_index=[0],
                relative_density_support=1e-8,
            )
            for size in (16, 128)
        ]
        estimates.append(results[1]["free_energy_difference"][1])
        short_variances.append(results[0]["standard_error"][1] ** 2)
        long_variances.append(results[1]["standard_error"][1] ** 2)

    # At empirical state fraction p, the KDE density ratio is
    # [k*(1-p)+a*p] / [(1-p)+k*a*p], with k the cross-state Gaussian kernel.
    cross_kernel = np.exp(-0.5 / bandwidth**2)
    numerator = (cross_kernel + state_weight) / 2
    denominator = (1 + cross_kernel * state_weight) / 2
    truth = -np.log(numerator / denominator)
    derivative = -(
        (state_weight - cross_kernel) / numerator
        - (cross_kernel * state_weight - 1) / denominator
    )
    lags = np.arange(1, frames)
    variance_fraction = (1 + 2 * np.sum((1 - lags / frames) * rho**lags)) / (4 * frames)
    theoretical_variance = derivative**2 * variance_fraction
    empirical_variance = np.var(estimates, ddof=1)
    short_variance = np.mean(short_variances)
    long_variance = np.mean(long_variances)
    for name, value in {
        "theoretical_variance": theoretical_variance,
        "empirical_variance": empirical_variance,
        "short_block_variance": short_variance,
        "long_block_variance": long_variance,
        "mean_fes_difference": np.mean(estimates),
        "target_fes_difference": truth,
    }.items():
        record_property(name, float(value))

    # The delta method is asymptotic; 96 replicas give about 15% variance MC error.
    # Predeclared 40% envelopes allow finite-chain and finite-ensemble variation.
    assert 0.6 < empirical_variance / theoretical_variance < 1.4
    assert 0.6 < long_variance / theoretical_variance < 1.4
    assert 0.6 < long_variance / empirical_variance < 1.4
    assert short_variance < 0.85 * long_variance
    assert abs(np.mean(estimates) - truth) < 4 * np.sqrt(theoretical_variance / replicas)
