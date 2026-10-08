"""Independent count-based checks of whole-frame FES jackknife uncertainty."""

import numpy as np
import pytest

from molsimflow.postprocess.pimd_fes import quantum_fes_block_jackknife_1d

VALUES = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], dtype=float)[:, None]
EDGES = [-0.5, 0.5, 2.5]


def estimate(values=VALUES, weights=None, block_size=4, reference_bin=0):
    if weights is None:
        weights = np.zeros(len(values))
    return quantum_fes_block_jackknife_1d(
        values,
        weights,
        EDGES,
        kbt=1.0,
        block_size=block_size,
        reference_bin=reference_bin,
    )


def test_count_oracle_with_unequal_bin_widths():
    result = estimate()
    # Four deleted-block counts; bin widths are 1 and 2.
    ratios = np.array([6 / 2 / 6, 4 / 2 / 8, 5 / 2 / 7, 6 / 2 / 6])
    estimates = -np.log(ratios)
    expected_se = np.sqrt(3 / 4 * np.sum((estimates - estimates.mean()) ** 2))
    assert result["free_energy_difference"][1] == pytest.approx(-np.log(7 / 2 / 9))
    np.testing.assert_allclose(result["leave_one_block_out"][:, 1], estimates)
    assert result["standard_error"][1] == pytest.approx(expected_se)
    assert result["standard_error"][0] == 0


def test_repeated_beads_do_not_increase_sample_size():
    single = estimate()
    duplicate = estimate(np.repeat(VALUES, 8, axis=1))
    np.testing.assert_allclose(single["standard_error"], duplicate["standard_error"])


def test_correlated_frame_repetition_requires_matching_block_length():
    original = estimate()
    repeated = np.repeat(VALUES, 5, axis=0)
    clustered = estimate(repeated, block_size=20)
    np.testing.assert_allclose(original["standard_error"], clustered["standard_error"])
    naive = estimate(repeated, block_size=1)
    assert naive["standard_error"][1] < clustered["standard_error"][1]


def test_weights_are_renormalized_after_dominant_block_deletion():
    weights = np.repeat([10000.0, 0.0, 0.0, 0.0], 4)
    result = estimate(weights=weights)
    assert np.isfinite(result["standard_error"]).all()
    shifted = estimate(weights=weights + 10000)
    np.testing.assert_allclose(result["standard_error"], shifted["standard_error"])


def test_bin_losing_support_has_unknown_uncertainty():
    values = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=float)[:, None]
    result = estimate(values)
    assert not result["support"][1] and np.isnan(result["standard_error"][1])
    with pytest.raises(ValueError, match="reference bin loses support"):
        estimate(values, reference_bin=1)


@pytest.mark.parametrize("size", [0, True, 1.5, 3, 16])
def test_invalid_block_layout_is_rejected(size):
    with pytest.raises(ValueError):
        estimate(block_size=size)


@pytest.mark.parametrize("reference", [-1, 2, True, 0.5])
def test_invalid_reference_is_rejected(reference):
    with pytest.raises(ValueError):
        estimate(reference_bin=reference)

@pytest.mark.parametrize("state_log_weight", [0.0, np.log(2.0)])
def test_repeated_bernoulli_ensemble_calibrates_block_variance(state_log_weight):
    """Compare ensemble and jackknife variance with a Bernoulli delta-method oracle."""
    rng = np.random.default_rng(20260910)
    independent_frames = 256
    probability = 0.35
    repetition = 8
    estimates = []
    variances = []
    for _ in range(128):
        independent = rng.binomial(1, probability, size=independent_frames)
        values = np.repeat(independent, repetition).astype(float)
        result = quantum_fes_block_jackknife_1d(
            values[:, None],
            state_log_weight * values,
            [-0.5, 0.5, 1.5],
            kbt=1.0,
            block_size=16 * repetition,
            reference_bin=0,
        )
        estimates.append(result["free_energy_difference"][1])
        variances.append(result["standard_error"][1] ** 2)

    # F = -log(exp(a)*p/(1-p)); dF/dp = -1/(p*(1-p)).
    # Repeated observations are fully correlated; N counts original draws only.
    expected_variance = 1 / (independent_frames * probability * (1 - probability))
    empirical_variance = np.var(estimates, ddof=1)
    mean_variance = np.mean(variances)
    # 128 replicates give about 13% relative MC error for a variance estimate.
    # These predeclared 40% envelopes test calibration, not exact finite-N equality.
    assert 0.6 < empirical_variance / expected_variance < 1.4
    assert 0.6 < mean_variance / expected_variance < 1.4
    assert 0.6 < mean_variance / empirical_variance < 1.4
    true_difference = -state_log_weight - np.log(probability / (1 - probability))
    assert abs(np.mean(estimates) - true_difference) < 4 * np.sqrt(
        expected_variance / len(estimates)
    )
