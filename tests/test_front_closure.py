"""Tests for the column-level closure statistics.

These tests pin the definitions that decide whether the chemical pattern
controls sustained advance. They check the statistics against cases
with a known answer, and they check that the guard against a sparse
tail actually damps a sparse column.
"""

from __future__ import annotations

import numpy as np
import pytest

from molsimflow.postprocess.front_closure import (
    area_normalised_rate,
    closure_test,
    column_selectivity,
    envelope_leading_edge,
    exposure_normalised_preference,
    leading_edge_advance,
    net_displacement,
    permutation_p_value,
    seam_depth,
    shrink_ratio_toward_unity,
    spearman_rank_correlation,
    travel_ordered_pattern_match,
)


def test_spearman_detects_a_monotone_relation():
    x = np.arange(10, dtype=float)
    assert spearman_rank_correlation(x, 2.0 * x + 1.0) == pytest.approx(1.0)
    assert spearman_rank_correlation(x, -3.0 * x) == pytest.approx(-1.0)


def test_spearman_is_undefined_for_too_few_pairs():
    assert np.isnan(spearman_rank_correlation([1.0, 2.0], [1.0, 2.0]))


def test_spearman_is_undefined_for_a_constant_series():
    assert np.isnan(spearman_rank_correlation(np.ones(10), np.arange(10.0)))


def test_spearman_drops_non_finite_pairs():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    y = np.array([1.0, 3.0, 2.0, 4.0, 5.0])
    # the four finite pairs are not perfectly ordered, but they are
    # clearly positively related
    assert spearman_rank_correlation(x, y) == pytest.approx(0.8)


def test_permutation_p_is_large_for_a_random_relation():
    rng = np.random.default_rng(7)
    a = rng.normal(size=40)
    b = rng.normal(size=40)
    rho = spearman_rank_correlation(a, b)
    assert permutation_p_value(rho, a, b, n_permutations=2000) > 0.05


def test_permutation_p_is_small_for_a_strong_relation():
    a = np.arange(40, dtype=float)
    b = 3.0 * a + 0.5
    rho = spearman_rank_correlation(a, b)
    assert permutation_p_value(rho, a, b, n_permutations=2000) < 0.01


def test_area_normalised_rate_removes_the_site_count_factor():
    # the same contact rate per site, different numbers of sites
    hits = np.array([60.0, 30.0])
    area = np.array([10.0, 5.0])
    rates = area_normalised_rate(hits, area)
    assert rates[0] == pytest.approx(6.0)
    assert rates[1] == pytest.approx(6.0)


def test_area_normalised_rate_is_nan_without_sites():
    rates = area_normalised_rate(np.array([3.0, 0.0]), np.array([2.0, 0.0]))
    assert rates[0] == pytest.approx(1.5)
    assert np.isnan(rates[1])


def test_shrinkage_pulls_a_sparse_ratio_towards_one():
    # a ratio of ten backed by a single observation
    sparse = shrink_ratio_toward_unity(10.0, support=1.0, pseudo_count=10.0)
    assert sparse < 2.0
    # the same ratio backed by a hundred observations is nearly kept
    strong = shrink_ratio_toward_unity(10.0, support=100.0, pseudo_count=10.0)
    assert strong == pytest.approx(10.0 ** (100.0 / 110.0))
    assert strong > 8.0


def test_shrinkage_is_symmetric_in_log_space():
    # a ratio of ten and its inverse are damped by the same factor
    up = shrink_ratio_toward_unity(10.0, support=1.0)
    down = shrink_ratio_toward_unity(0.1, support=1.0)
    assert up * down == pytest.approx(1.0)


def test_shrinkage_leaves_a_ratio_near_one_alone():
    assert shrink_ratio_toward_unity(1.0, support=1.0) == pytest.approx(1.0)


def test_column_selectivity_is_undefined_without_the_denominator_motif():
    hits = np.array([10.0, 5.0, 0.0])
    area = np.array([3.0, 3.0, 0.0])
    assert np.isnan(column_selectivity(hits, area, 0, 2))


def test_column_selectivity_uses_area_not_raw_counts():
    # identical contact rate per site, so no preference
    hits = np.array([60.0, 30.0])
    area = np.array([10.0, 5.0])
    assert column_selectivity(hits, area, 0, 1) == pytest.approx(1.0)


def test_column_selectivity_damps_a_sparse_denominator():
    # the numerator is seen often, the denominator once
    hits = np.array([1000.0, 1.0])
    area = np.array([10.0, 3.0])
    stabilised = column_selectivity(hits, area, 0, 1, pseudo_count=10.0)
    raw = (hits[0] / area[0]) / (hits[1] / area[1])
    assert raw == pytest.approx(300.0)
    # the log-space shrinkage keeps it a ratio but far closer to one
    assert stabilised == pytest.approx(300.0 ** (1.0 / 11.0))
    assert stabilised < 30.0


def test_closure_test_reports_the_number_of_usable_columns():
    pref = [1.0, 2.0, 3.0, 4.0, 5.0]
    adv = [0.1, 0.4, 0.3, 0.8, 0.6]
    out = closure_test(pref, adv, n_permutations=500)
    assert out["n_columns"] == 5
    assert np.isfinite(out["spearman_rho"])
    assert np.isfinite(out["permutation_p"])


def test_seam_depth_measures_a_distance_not_a_column_index():
    # three columns starting 1.53 A from the seam, 2.4706 A apart
    x = [75.47, 77.94, 80.41]
    assert seam_depth(x, +1, 73.94) == pytest.approx(80.41 - 73.94)
    assert seam_depth(x, +1, 73.94) == pytest.approx(6.47, abs=0.01)


def test_seam_depth_is_signed_by_the_direction_of_travel():
    # a face the front moves out of is measured the other way round
    x = [55.71, 60.65, 73.00]
    assert seam_depth(x, -1, 73.94) == pytest.approx(73.94 - 55.71)


def test_net_displacement_is_not_the_span():
    # the front explores out and comes back
    x = [10.0, 30.0, 12.0]
    assert net_displacement(x) == pytest.approx(2.0)
    assert (max(x) - min(x)) == pytest.approx(20.0)


def test_travel_ordered_pattern_match_accepts_a_mirrored_pattern():
    # the arrival face counts up from the seam, the departure face down
    a = ["mixed", "dual_oh", "mixed"]
    b = ["mixed", "dual_oh", "mixed"]
    out = travel_ordered_pattern_match(a, b, [1.5, 4.0, 6.5], [3.4, 5.9, 8.4])
    assert out["identical"] is True
    assert out["n_matched"] == 3
    assert out["n_mismatch"] == 0
    assert out["depth_a"] == pytest.approx(6.5)
    assert out["depth_b"] == pytest.approx(8.4)


def test_travel_ordered_pattern_match_reports_the_first_mismatch():
    a = ["mixed", "dual_oh", "mixed"]
    b = ["mixed", "mixed", "dual_oh"]
    out = travel_ordered_pattern_match(a, b, [1.5, 4.0, 6.5], [3.4, 5.9, 8.4])
    assert out["identical"] is False
    assert out["n_mismatch"] == 2
    assert out["mismatch_at"] == 1


def test_exposure_normalised_preference_removes_the_exposure():
    # the same chemistry seen for twice as long has twice the raw rate,
    # but the preference is unchanged
    short = exposure_normalised_preference(100.0, 10.0, 50.0, 10.0, 100.0)
    long = exposure_normalised_preference(200.0, 10.0, 100.0, 10.0, 200.0)
    assert short == pytest.approx(long)
    assert short == pytest.approx(2.0)


def test_exposure_normalised_preference_is_undefined_without_exposure():
    assert np.isnan(exposure_normalised_preference(
        100.0, 10.0, 50.0, 10.0, 0.0))
    assert np.isnan(exposure_normalised_preference(
        100.0, 0.0, 50.0, 10.0, 100.0))


# --- envelope leading edge -------------------------------------------------

def test_leading_edge_finds_the_rightmost_covered_site():
    # Four sites, three of them under the envelope (signed distance <= 0).
    sd = np.array([[-1.0, -1.0, -1.0, 5.0]])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    mask = np.array([True, True, True, True])
    edge = envelope_leading_edge(sd, x, mask, min_run=3, gap=3.0)
    assert edge[0] == pytest.approx(2.0)


def test_leading_edge_requires_a_minimum_run():
    # A single covered site is not a front.
    sd = np.array([[-1.0, 5.0, 5.0, 5.0]])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    mask = np.ones(4, dtype=bool)
    edge = envelope_leading_edge(sd, x, mask, min_run=3, gap=3.0)
    assert np.isnan(edge[0])


def test_leading_edge_ignores_an_isolated_outlier():
    # Sites 0..2 form a run; site 6 is covered but far away. The gap
    # guard must stop the edge from jumping to the outlier.
    sd = np.array([[-1.0, -1.0, -1.0, 5.0, 5.0, 5.0, -1.0]])
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mask = np.ones(7, dtype=bool)
    edge = envelope_leading_edge(sd, x, mask, min_run=3, gap=3.0)
    assert edge[0] == pytest.approx(2.0)


def test_leading_edge_respects_the_face_mask():
    # The rightmost covered site lies on the other face and must be
    # excluded from this face's edge.
    sd = np.array([[-1.0, -1.0, -1.0, -1.0]])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    mask = np.array([True, True, True, False])
    edge = envelope_leading_edge(sd, x, mask, min_run=3, gap=3.0)
    assert edge[0] == pytest.approx(2.0)


def test_leading_edge_is_nan_when_nothing_is_covered():
    sd = np.array([[1.0, 2.0, 3.0]])
    x = np.array([0.0, 1.0, 2.0])
    mask = np.ones(3, dtype=bool)
    edge = envelope_leading_edge(sd, x, mask, min_run=2)
    assert np.isnan(edge[0])


def test_leading_edge_handles_an_empty_face():
    sd = np.zeros((2, 4))
    x = np.array([0.0, 1.0, 2.0, 3.0])
    mask = np.zeros(4, dtype=bool)
    edge = envelope_leading_edge(sd, x, mask)
    assert np.all(np.isnan(edge))


def test_leading_edge_advance_uses_the_first_and_last_valid_frames():
    # A gap in the middle must not reset the reference points.
    edge = np.array([10.0, np.nan, 14.0, np.nan, 18.0])
    assert leading_edge_advance(edge, seam_x=4.0, direction=1) == pytest.approx(8.0)


def test_leading_edge_advance_sign_follows_the_direction():
    edge = np.array([10.0, 6.0])
    # Towards increasing x the depth from the seam grows from 6 to 4
    # ... which is a retreat, so the advance is negative.
    assert leading_edge_advance(edge, seam_x=4.0, direction=1) == pytest.approx(-4.0)
    # Towards decreasing x the depth from the seam grows from -6 to -2,
    # so the front came closer to the seam: the advance is positive.
    assert leading_edge_advance(edge, seam_x=4.0, direction=-1) == pytest.approx(4.0)


def test_leading_edge_advance_needs_at_least_two_valid_frames():
    assert np.isnan(leading_edge_advance(np.array([1.0, np.nan]), seam_x=0.0))


def test_leading_edge_rejects_a_mismatched_x_array():
    sd = np.zeros((1, 4))
    with pytest.raises(ValueError):
        envelope_leading_edge(sd, np.zeros(3), np.ones(4, dtype=bool))
