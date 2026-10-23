"""Tests for frame-grouped bead heterogeneity and structural proxies."""

import numpy as np
import pytest

from molsimflow.postprocess.pimd_route_diagnostics import (
    contiguous_episodes,
    distance_frame_metrics,
    nearest_oxygen_proxy,
    read_plumed_selected,
    regional_weight_support,
)


def test_distance_fractions_and_deep_split_keep_frame_grain():
    values = np.array([[1.0, 2.4, 2.6, 3.2], [2.4, 2.45, 2.5, 2.55]])
    result = distance_frame_metrics(values, thresholds=[2.25, 2.5, 2.75], deep_bounds=(2.0, 3.0))
    np.testing.assert_allclose(result.fraction_ge, [[0.75, 0.5, 0.25], [1.0, 0.5, 0.0]])
    np.testing.assert_array_equal(result.straddles, [[True, True, True], [False, True, False]])
    np.testing.assert_array_equal(result.deep_split, [True, False])
    np.testing.assert_allclose(result.quantiles[:, 2], [2.5, 2.475])
    assert result.mean.shape == (2,)


def test_episodes_break_at_unsampled_gap_and_inactive_frame():
    result = contiguous_episodes(
        [0, 4000, 8000, 16000, 20000],
        [True, True, False, True, True],
        stride=4000,
    )
    assert result == [
        {"first_step": 0, "last_step": 4000, "frames": 2},
        {"first_step": 16000, "last_step": 20000, "frames": 2},
    ]
    assert contiguous_episodes([0, 8000], [True, True], stride=4000) == [
        {"first_step": 0, "last_step": 0, "frames": 1},
        {"first_step": 8000, "last_step": 8000, "frames": 1},
    ]


def test_region_support_uses_one_weight_per_path():
    result = regional_weight_support([0, 0.5, 1], np.log([1, 2, 1]), [0, 1, 1])
    assert result["frames_with_bead_support"] == 2
    assert result["weighted_probability_mass"] == pytest.approx(0.5)
    assert result["contribution_kish_ess"] == pytest.approx(2)
    assert result["maximum_contribution_share"] == pytest.approx(0.5)
    assert result["block_probability_masses"] == pytest.approx([0, 0.5])
    empty = regional_weight_support([0, 0], [0, 0], [0, 1])
    assert empty["weighted_probability_mass"] == 0
    assert empty["contribution_kish_ess"] == 0


def test_nearest_oxygen_proxy_pair_and_periodic_invariance():
    xyz = np.array(
        [
            [1.0, 1, 1],
            [6.0, 1, 1],
            [1.9, 1, 1],
            [5.1, 1, 1],
            [6.9, 1, 1],
            [7.2, 1, 1],
        ]
    )
    types = np.array([2, 2, 1, 1, 1, 1])
    original = nearest_oxygen_proxy(xyz, types, np.full(3, 20.0))
    assert original["single_pair_proxy"]
    assert not original["neutral_proxy"]
    assert original["pair_oo_A"] == pytest.approx(5.0)
    assert original["nearest_oxygen_counts"] == [1, 3]
    wrapped = xyz.copy()
    wrapped[[0, 2], 0] += 20
    shifted = nearest_oxygen_proxy(wrapped, types, np.full(3, 20.0))
    assert shifted["nearest_oxygen_counts"] == original["nearest_oxygen_counts"]
    assert shifted["pair_oo_A"] == pytest.approx(original["pair_oo_A"])
    assert shifted["minimum_sharing_gap_A"] == pytest.approx(original["minimum_sharing_gap_A"])


def test_invalid_arrays_fail_closed():
    with pytest.raises(ValueError):
        distance_frame_metrics(np.array([[1, np.nan]]), thresholds=[2.5], deep_bounds=(2, 3))
    with pytest.raises(ValueError):
        regional_weight_support([1.1], [0], [0])
    with pytest.raises(ValueError):
        nearest_oxygen_proxy(np.zeros((2, 3)), np.array([2, 1]), [10, 10, 10])


def test_selected_plumed_reader_aligns_and_fails_on_missing_time(tmp_path):
    table = tmp_path / "COLVAR"
    table.write_text(
        "#! FIELDS time iondistance opes.bias\n0 0.1 -0.5\n1000 2.4 -0.3\n2000 2.6 -0.2\n"
    )
    data = read_plumed_selected(table, [0, 2000], ["iondistance", "opes.bias"])
    np.testing.assert_allclose(data["iondistance"], [0.1, 2.6])
    with pytest.raises(ValueError, match="incomplete"):
        read_plumed_selected(table, [0, 1500], ["iondistance"])


def test_selected_reader_rejects_header_drift_and_duplicate_times(tmp_path):
    import pytest

    path = tmp_path / "COLVAR"
    path.write_text("#! FIELDS time x\n0 1\n#! FIELDS time y\n1 2\n")
    with pytest.raises(ValueError, match="FIELDS"):
        read_plumed_selected(path, [0, 1], ["x"])
    path.write_text("#! FIELDS time x\n0 1\n0 2\n1 3\n")
    with pytest.raises(ValueError, match="increasing"):
        read_plumed_selected(path, [0, 1], ["x"])
