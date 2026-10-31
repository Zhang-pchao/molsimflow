"""Analytic tests for equilibrium umbrella algebra and support handling."""
import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist

import numpy as np

from molsimflow.postprocess.umbrella_wham import (
    fit_harmonic_umbrellas,
    harmonic_bias_matrix,
    histogram_pmf,
    load_window,
    solve_wham,
)


def gaussian_windows(counts):
    centers = np.linspace(-2, 2, len(counts))
    spring, base_spring, kbt = 2.0, 1.0, 1.0
    samples = []
    normal = NormalDist()
    for center, count in zip(centers, counts):
        mean = spring * center / (base_spring + spring)
        sigma = np.sqrt(kbt / (base_spring + spring))
        values = [normal.inv_cdf((i + 0.5) / count) for i in range(count)]
        samples.append(mean + sigma * np.array(values))
    return samples, centers, np.full(len(counts), spring), kbt


class UmbrellaWhamTests(unittest.TestCase):
    def test_analytic_gaussian_free_energy_offsets_with_unequal_counts(self):
        samples, centers, k, kbt = gaussian_windows([601, 901, 1201, 751, 1051])
        result = fit_harmonic_umbrellas(samples, centers, k, kbt)
        # Integrating exp[-(x^2 + k(x-c)^2)/(2kBT)] gives these offsets.
        expected = 0.5 * k / (1 + k) * centers**2 / kbt
        expected -= expected[0]
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.free_energies, expected, atol=0.002)
        self.assertAlmostEqual(float(np.exp(result.log_weights).sum()), 1.0)
        np.testing.assert_allclose(result.overlap_matrix.sum(axis=1), 1, atol=1e-8)

    def test_reconstructs_known_unbiased_gaussian_pmf(self):
        samples, centers, k, kbt = gaussian_windows([2001] * 5)
        result = fit_harmonic_umbrellas(samples, centers, k, kbt)
        q = np.concatenate(samples)
        edges = np.arange(-6.05, 6.15, 0.1)
        profile = histogram_pmf(q, result.log_weights, edges, kbt, reference_coordinate=0)
        core = abs(profile.centers) < 1.8
        target = 0.5 * profile.centers[core] ** 2
        np.testing.assert_allclose(profile.free_energy[core], target, atol=0.025)

    def test_large_bias_energy_offset_does_not_change_weights(self):
        samples, centers, k, kbt = gaussian_windows([101] * 3)
        u = harmonic_bias_matrix(np.concatenate(samples), centers, k) / kbt
        first = solve_wham(u, [101] * 3)
        second = solve_wham(u + 1000, [101] * 3)
        self.assertTrue(second.converged)
        np.testing.assert_allclose(first.log_weights, second.log_weights, atol=1e-7)

    def test_energy_unit_rescaling_is_invariant(self):
        samples, centers, k, kbt = gaussian_windows([101] * 3)
        a = fit_harmonic_umbrellas(samples, centers, k, kbt)
        b = fit_harmonic_umbrellas(samples, centers, k * 96.485, kbt * 96.485)
        np.testing.assert_allclose(a.log_weights, b.log_weights, atol=1e-9)

    def test_single_window_removes_its_bias(self):
        u = np.array([[0.0, 1.0, 2.0]])
        result = solve_wham(u, [3])
        expected = np.exp(u[0]) / np.exp(u[0]).sum()
        np.testing.assert_allclose(np.exp(result.log_weights), expected)

    def test_unobserved_bin_is_not_filled_and_reference_requires_support(self):
        q = np.array([0.0, 0.2, 2.0])
        profile = histogram_pmf(q, np.zeros(3), np.array([-0.5, 0.5, 1.5, 2.5]), 1)
        self.assertTrue(np.isinf(profile.free_energy[1]))
        self.assertEqual(profile.raw_counts[1], 0)
        self.assertEqual(profile.density[1], 0)
        with self.assertRaises(ValueError):
            histogram_pmf(q, np.zeros(3), np.array([-0.5, 0.5, 1.5, 2.5]), 1, reference_coordinate=1)

    def test_histogram_nonuniform_widths_and_final_edge(self):
        p = histogram_pmf(np.array([0.0, 1.0, 3.0]), np.zeros(3), np.array([0, 1, 3]), 1)
        np.testing.assert_allclose(p.raw_counts, [1, 2])
        self.assertAlmostEqual(float(np.dot(p.density, [1, 2])), 1)
        with self.assertRaises(ValueError):
            histogram_pmf(np.array([-1.0]), np.zeros(1), np.array([0, 1]), 1)

    def test_invalid_counts_biases_and_nonconvergence_are_explicit(self):
        with self.assertRaises(ValueError):
            solve_wham(np.zeros((2, 4)), [1, 1])
        with self.assertRaises(ValueError):
            solve_wham(np.array([[np.nan]]), [1])
        u = np.random.default_rng(123).normal(size=(5, 501)) * 8
        result = solve_wham(u, [10, 17, 30, 143, 301], max_iterations=1, tolerance=1e-14)
        self.assertFalse(result.converged)

    def test_colvar_selection_factor_half_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'COLVAR'
            path.write_text('#! FIELDS time dz.z umb.bias\n0 1 0\n10 2 2\n10.05 3 8\n11 4 18\n')
            t, q = load_window(path, 1, 4, discard_time=10)
            np.testing.assert_allclose(t, [10.05, 11])
            np.testing.assert_allclose(q, [3, 4])
            with self.assertRaises(ValueError):
                load_window(path, 1, 8, discard_time=10)
            path.write_text('#! FIELDS time dz.z umb.bias\n11 1 0\n11 1 0\n')
            with self.assertRaises(ValueError):
                load_window(path, 1, 4, discard_time=10)


if __name__ == '__main__':
    unittest.main()
