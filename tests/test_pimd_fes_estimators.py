"""Analytical contracts for the maintained bead-density FES estimators."""

from itertools import product

import numpy as np
import pytest

from molsimflow.postprocess.pimd_fes import (
    assemble_bead_frames,
    bead_density_estimators,
    frame_log_weights,
    total_bias_energy,
    quantum_fes_1d,
    quantum_histogram_masses,
)


def test_symmetric_beads_recover_uniform_probability_and_nonzero_diagnostic():
    # Two differently sampled beads have the same pooled uniform distribution.
    result = bead_density_estimators(np.log([[0.9, 0.1], [0.1, 0.9]]), kbt=2.0)
    np.testing.assert_allclose(result["log_probability_mean"], -np.log(2))
    np.testing.assert_allclose(result["raw_probability_mean"], 2 * np.log(2))
    np.testing.assert_allclose(result["probability_mean"], 0, atol=1e-15)
    np.testing.assert_allclose(result["free_energy_mean_diagnostic"], 2 * np.log(5 / 3))
    # Independently zeroing the diagnostic would hide this finite-sampling gap.
    assert result["zero_reference"] == pytest.approx(2 * np.log(2))


def test_free_energy_mean_primary_changes_only_the_shared_zero_reference():
    logs = np.log([[0.9, 0.1], [0.1, 0.9]])
    probability_primary = bead_density_estimators(logs, kbt=2.0)
    free_energy_primary = bead_density_estimators(
        logs, kbt=2.0, primary_estimator="free_energy_mean"
    )
    assert np.min(free_energy_primary["free_energy_mean"]) == pytest.approx(0.0)
    assert not np.allclose(
        free_energy_primary["probability_mean"],
        free_energy_primary["free_energy_mean"],
    )
    np.testing.assert_allclose(
        free_energy_primary["free_energy_mean"]
        - free_energy_primary["probability_mean"],
        probability_primary["free_energy_mean"]
        - probability_primary["probability_mean"],
    )


def test_weighted_histogram_has_one_path_weight_and_accounts_for_bin_widths():
    result = quantum_fes_1d(
        [[0.25, 0.25, 2.5], [0.25, 2.5, 2.5], [2.5, 2.5, 2.5], [0.25, 0.25, 0.25]],
        np.log([1, 2, 3, 4]),
        [0, 1, 4],
        kbt=1,
    )
    # Weighted counts are 16 and 14 out of 30; the second bin is 3 times wider.
    np.testing.assert_allclose(result["raw_probability_mean"], -np.log([16 / 30, 14 / 90]))
    np.testing.assert_allclose(result["probability_mean"], [0, np.log(24 / 7)])


@pytest.mark.parametrize("delta", [50.0, 1000.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_histogram_log_accumulation_preserves_rare_supported_bins(delta, reverse):
    samples = np.array([[0.25], [1.25]])
    log_weights = np.array([0.0, -delta])
    if reverse:
        samples = samples[::-1]
        log_weights = log_weights[::-1]
    result = quantum_fes_1d(samples, log_weights, [0.0, 1.0, 2.0], kbt=1.0)
    np.testing.assert_array_equal(result["probability_support"], [True, True])
    np.testing.assert_allclose(result["probability_mean"], [0.0, delta])


def test_histogram_log_accumulation_keeps_empty_bins_unsupported():
    result = quantum_fes_1d(
        [[0.25], [2.25]], [0.0, -1000.0], [0.0, 1.0, 2.0, 4.0], kbt=1.0
    )
    np.testing.assert_array_equal(
        result["probability_support"], [True, False, True]
    )
    assert np.isposinf(result["probability_mean"][1])


def test_histogram_log_accumulation_handles_unequal_bin_widths_at_extreme_range():
    result = quantum_fes_1d(
        [[0.25], [2.5]], [0.0, -50.0], [0.0, 1.0, 4.0], kbt=1.0
    )
    np.testing.assert_allclose(result["probability_mean"], [0.0, 50.0 + np.log(3.0)])


def test_histogram_conditional_and_direct_log_masses_match_at_extreme_range():
    samples = np.array([[0.25], [1.25], [0.25], [1.25]])
    result = quantum_histogram_masses(
        samples,
        [0.0, -50.0, -1000.0, -1050.0],
        [0.0, 1.0, 2.0],
        conditioning=[0.0, 1.0, 0.0, 1.0],
        conditioning_edges=[-0.5, 0.5, 1.5],
    )
    np.testing.assert_allclose(result["log_direct"], result["log_conditional"])


@pytest.mark.parametrize("beads", [1, 3, 7, 32])
def test_bead_permutation_and_replication_leave_estimators_unchanged(beads):
    rng = np.random.default_rng(123)
    density = rng.uniform(0.01, 1.0, size=(beads, 2, 4))
    density /= density.sum(axis=(1, 2), keepdims=True)
    logs = np.log(density)
    reference = bead_density_estimators(logs, kbt=0.8)
    permuted = bead_density_estimators(logs[rng.permutation(beads)], kbt=0.8)
    repeated = bead_density_estimators(np.repeat(logs, 3, axis=0), kbt=0.8)
    for key in reference:
        np.testing.assert_allclose(permuted[key], reference[key], atol=1e-14)
        np.testing.assert_allclose(repeated[key], reference[key], atol=1e-14)


@pytest.mark.parametrize("primary", ["probability_mean", "free_energy_mean"])
def test_single_bead_matches_ordinary_free_energy(primary):
    result = bead_density_estimators(
        np.log([[0.2, 0.8]]), kbt=0.5, primary_estimator=primary
    )
    np.testing.assert_allclose(result["probability_mean"], [0.5 * np.log(4), 0])
    np.testing.assert_allclose(result["free_energy_mean_diagnostic"], result["probability_mean"])


def test_primary_keeps_partial_support_and_diagnostic_requires_every_bead():
    result = bead_density_estimators([[0, -np.inf, -np.inf], [-np.inf, 0, -np.inf]], kbt=1)
    np.testing.assert_array_equal(result["probability_mean"], [0, 0, np.inf])
    np.testing.assert_array_equal(result["probability_support"], [True, True, False])
    assert np.isposinf(result["free_energy_mean_diagnostic"]).all()
    assert not result["common_support"].any()


def test_log_density_estimator_preserves_finite_remote_tails():
    result = bead_density_estimators([[-1000, -1001], [-1000, -1001]], kbt=1)
    np.testing.assert_allclose(result["probability_mean"], [0, 1])
    np.testing.assert_allclose(result["raw_probability_mean"], [1000, 1001])


def test_diagnostic_avoids_intermediate_sum_overflow():
    result = bead_density_estimators(np.full((4, 1), -1e308), kbt=1)
    np.testing.assert_array_equal(result["raw_free_energy_mean_diagnostic"], [1e308])
    np.testing.assert_array_equal(result["probability_mean"], [0])


@pytest.mark.parametrize("logs", [[], [0, 0], np.zeros((0, 2)), np.zeros((2, 0)), [[np.nan]], [[np.inf]], [[-np.inf]]])
def test_invalid_or_empty_log_density_is_rejected(logs):
    with pytest.raises(ValueError):
        bead_density_estimators(logs, kbt=1)


@pytest.mark.parametrize("kbt", [0, -1, np.nan, np.inf])
def test_invalid_thermal_energy_is_rejected(kbt):
    with pytest.raises(ValueError, match="kBT"):
        bead_density_estimators([[0]], kbt=kbt)


@pytest.mark.parametrize("beads", [2.5, 2.0, True, np.bool_(True), 0, -1])
def test_assembly_rejects_noninteger_or_nonpositive_bead_count(beads):
    with pytest.raises(ValueError, match="positive integer"):
        assemble_bead_frames([0, 0], [0, 1], [0.1, 0.2], expected_beads=beads)


def test_assembly_accepts_numpy_integer_bead_count():
    _, _, values = assemble_bead_frames([0, 0], [0, 1], [0.1, 0.2], expected_beads=np.int64(2))
    np.testing.assert_array_equal(values, [[0.1, 0.2]])


def test_histogram_legacy_keys_are_explicit_aliases_only():
    result = quantum_fes_1d([[0, 1], [1, 0]], [0, 0], [-0.5, 0.5, 1.5], kbt=1)
    assert result["eq8"] is result["probability_mean"]
    assert result["eq10"] is result["free_energy_mean_diagnostic"]
    assert result["logmean_diagnostic"] is result["free_energy_mean_diagnostic"]
    assert result["support"] is result["common_support"]


def _finite_canonical_ensemble(beads):
    """Enumerate coupled cyclic paths with a nonlinear bead observable."""
    coordinates = np.asarray(list(product([-1.0, 0.0, 2.0], repeat=beads)))
    bead_cv = coordinates**2
    spring = np.mean((coordinates - np.roll(coordinates, 1, axis=1)) ** 2, axis=1)
    energy = 0.3 * np.mean(coordinates**2, axis=1) + 0.2 * spring
    local_wall = 0.35 * np.maximum(bead_cv - 0.5, 0) ** 2

    def potential(cv):
        return 0.6 * (cv - 0.9) ** 2 + 0.2 * cv

    energies = {
        "centroid_coord": potential(np.mean(coordinates, axis=1) ** 2),
        "bead_mean": potential(np.mean(bead_cv, axis=1)),
        "bead_density_shared": np.mean(potential(bead_cv), axis=1),
    }
    # Distinct biased ensembles must recover the same physical bead marginal.
    assert not np.allclose(energies["centroid_coord"], energies["bead_mean"])
    assert not np.allclose(energies["bead_mean"], energies["bead_density_shared"])
    return bead_cv, energy, local_wall, potential(bead_cv), energies


@pytest.mark.parametrize("beads", [2, 3])
@pytest.mark.parametrize("mode", ["centroid_coord", "bead_mean", "bead_density_shared"])
def test_exact_biased_canonical_ensemble_recovers_unbiased_bead_marginal(beads, mode):
    bead_cv, energy, local_wall, local_bias, bias_energies = _finite_canonical_ensemble(beads)
    kbt = 0.7
    edges = np.array([-0.5, 0.5, 1.5, 4.5])
    unbiased_probability = np.exp(-energy / kbt)
    unbiased_probability /= unbiased_probability.sum()
    # Independent reference: enumerate every path and bead using the known
    # unbiased canonical probabilities, without any estimator library helper.
    expected_mass = np.histogram(
        bead_cv.ravel(), bins=edges,
        weights=np.repeat(unbiased_probability / beads, beads),
    )[0]
    expected_fes = -kbt * np.log(expected_mass / np.diff(edges))

    physical_bias = bias_energies[mode] + np.mean(local_wall, axis=1)
    biased_probability = np.exp(-(energy + physical_bias) / kbt)
    biased_probability /= biased_probability.sum()
    if mode == "bead_density_shared":
        removed_bias = total_bias_energy(mode, bead_bias_energies=local_bias + local_wall)
    else:
        removed_bias = total_bias_energy(mode, sampling_bias_energy=physical_bias)
    correction = frame_log_weights("fixed_bias", bias_energy=removed_bias, kbt=kbt)
    # Each enumerated state represents its exact biased sampling probability.
    # This deterministic quadrature avoids random Monte Carlo tolerances.
    result = quantum_fes_1d(
        bead_cv, np.log(biased_probability) + correction, edges, kbt=kbt,
    )
    np.testing.assert_allclose(result["raw_probability_mean"], expected_fes, atol=2e-14)
    np.testing.assert_allclose(result["probability_mean"], expected_fes - expected_fes.min(), atol=2e-14)


@pytest.mark.parametrize("beads", [2, 3])
def test_averaging_local_exponentials_is_not_a_complete_path_weight(beads):
    bead_cv, energy, local_wall, local_bias, _ = _finite_canonical_ensemble(beads)
    kbt = 0.7
    edges = np.array([-0.5, 0.5, 1.5, 4.5])
    local_energy = local_bias + local_wall
    physical_bias = np.mean(local_energy, axis=1)
    biased_probability = np.exp(-(energy + physical_bias) / kbt)
    biased_probability /= biased_probability.sum()
    unbiased_probability = np.exp(-energy / kbt)
    unbiased_probability /= unbiased_probability.sum()
    expected_mass = np.histogram(
        bead_cv.ravel(), bins=edges,
        weights=np.repeat(unbiased_probability / beads, beads),
    )[0]
    # A tempting alternative exp(beta U_b) averaged over beads does not
    # remove the path potential mean(U_b), even under exact equilibrium.
    incorrect_weights = biased_probability * np.mean(np.exp(local_energy / kbt), axis=1)
    incorrect_weights /= incorrect_weights.sum()
    incorrect_mass = np.histogram(
        bead_cv.ravel(), bins=edges,
        weights=np.repeat(incorrect_weights / beads, beads),
    )[0]
    assert np.max(np.abs(incorrect_mass - expected_mass)) > 0.05
