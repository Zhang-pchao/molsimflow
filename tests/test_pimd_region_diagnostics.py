"""Region diagnostics must distinguish invisible activity from its disappearance."""

import numpy as np
import pytest

from molsimflow.postprocess.pimd.region_diagnostics import (
    fixed_bead_coverage,
    membership_crossings,
    regional_activity,
    sphere_weights,
)


def test_escape_retains_global_positive_and_negative_activity():
    xyz = np.array([[[0.1, 0, 0], [-0.1, 0, 0]], [[2.0, 0, 0], [2.2, 0, 0]]])
    _, weights = sphere_weights(xyz, np.zeros(3), core_radius=0.5, outer_radius=1.0)
    activity = regional_activity(np.array([[3.0, 1.0], [3.0, 1.0]]), weights, reference=2)
    for sign in ("positive", "negative"):
        np.testing.assert_array_equal(activity[f"local_{sign}"], [1, 0])
        np.testing.assert_array_equal(activity[f"outside_{sign}"], [0, 1])
        np.testing.assert_array_equal(activity[f"global_{sign}"], [1, 1])
    # Opposite defects are visible separately even when their net signed sum is zero.
    assert sum(activity[f"global_{s}"][0] for s in ("positive", "negative")) == 2


def test_complete_path_axes_complement_and_permutation():
    rng = np.random.default_rng(81)
    occupancy = 2 + rng.normal(size=(4, 8, 5))
    weights = rng.random((4, 8, 5))
    expected = regional_activity(occupancy, weights, reference=[2] * 5)
    perm = [3, 1, 4, 2, 0]
    actual = regional_activity(occupancy[..., perm], weights[..., perm], reference=2)
    for key in expected:
        assert actual[key].shape == (4, 8)
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-14)
    for sign in ("positive", "negative"):
        np.testing.assert_allclose(
            expected[f"local_{sign}"] + expected[f"outside_{sign}"], expected[f"global_{sign}"]
        )


def test_periodic_images_moving_anchor_and_endpoint_weights():
    xyz = np.array([[[0, 0, 0], [1.0, 0, 0], [2, 0, 0], [3, 0, 0]]])
    d, w = sphere_weights(xyz, [0, 0, 0], core_radius=1, outer_radius=2, box_lengths=[10, 10, 10])
    np.testing.assert_array_equal(w, [[1, 1, 0, 0]])
    shifted = xyz + [10, -10, 20]
    dp, wp = sphere_weights(
        shifted, [0, 0, 0], core_radius=1, outer_radius=2, box_lengths=[10, 10, 10]
    )
    np.testing.assert_allclose(dp, d)
    np.testing.assert_allclose(wp, w)
    _, moved = sphere_weights(xyz + 4, np.array([[4, 4, 4]]), core_radius=1, outer_radius=2)
    np.testing.assert_array_equal(moved, w)


def test_fixed_bead_coverage_incoming_classical_atom_and_empty_region():
    weights = np.array([[1, 0], [1, 0.5], [0, 0]])
    result = fixed_bead_coverage(weights, [4, 1], required_beads=4)
    np.testing.assert_array_equal(result["low_resolution_atoms"], [0, 1, 0])
    np.testing.assert_array_equal(result["low_resolution_weight"], [0, 0.5, 0])
    np.testing.assert_array_equal(result["support_present"], [True, True, False])


def test_membership_crossings_keep_atom_ids_and_do_not_bridge_missing_frames():
    events = membership_crossings(
        [0, 10, 30, 40],
        [7, 11],
        np.array([[0.2, 1.2], [1.2, 0.2], [0.2, 1.2], [1.2, 0.2]]),
        radius=1,
        stride=10,
    )
    assert [(e["step"], e["atom_id"], e["direction"]) for e in events] == [
        (10, 7, "outward"),
        (10, 11, "inward"),
        (40, 7, "outward"),
        (40, 11, "inward"),
    ]


@pytest.mark.parametrize("counts", [[1.5, 4], [0, 4], [True, False], [4]])
def test_invalid_bead_counts(counts):
    with pytest.raises(ValueError):
        fixed_bead_coverage(np.ones((2, 2)), counts, required_beads=4)


@pytest.mark.parametrize("weights", [[-0.1, 1], [0, 1.1], [np.nan, 1]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        regional_activity([2, 2], weights, reference=2)


def test_invalid_geometry_and_crossing_contracts():
    for kw in (
        {"core_radius": 1, "outer_radius": 1},
        {"core_radius": 0, "outer_radius": 2, "box_lengths": [3, 3, 3]},
        {"core_radius": 0, "outer_radius": 1, "box_lengths": np.eye(3)},
    ):
        with pytest.raises(ValueError):
            sphere_weights(np.zeros((2, 3)), np.zeros(3), **kw)
    with pytest.raises(ValueError):
        membership_crossings([0, 10], [1, 1], np.ones((2, 2)), radius=1, stride=10)
    with pytest.raises(ValueError):
        membership_crossings([0, 0], [1], np.ones((2, 1)), radius=1, stride=10)


@pytest.mark.parametrize("kind", ["translation", "relay", "ionization"])
def test_activity_change_balance_does_not_confuse_relay_with_creation(kind):
    from molsimflow.postprocess.pimd.region_diagnostics import activity_change_terms

    old = np.array([3.0, 2.0, 1.0])
    new = old.copy()
    w0, w1 = np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    if kind == "translation":
        w1 = np.zeros(3)
    elif kind == "relay":
        new = np.array([2.0, 3.0, 1.0])
    else:
        old = np.full(3, 2.0)
    values = activity_change_terms(old, new, w0, w1, reference=2)
    for sign in ("positive", "negative"):
        assert values[f"{sign}_local_change"] == pytest.approx(
            values[f"{sign}_occupancy_change"] + values[f"{sign}_spatial_change"]
        )
    if kind == "translation":
        assert values["positive_occupancy_change"] == 0
        assert values["positive_spatial_change"] == -1
    if kind == "relay":
        assert values["positive_occupancy_change"] == -1
        assert values["positive_spatial_change"] == 0
    assert values["positive_global_change"] == (1 if kind == "ionization" else 0)


def test_adaptive_coverage_endpoints_transition_and_empty_support():
    from molsimflow.postprocess.pimd.region_diagnostics import adaptive_amplitude_coverage

    weights = np.array([[1, 0.5, 0.25], [0, 0, 0]])
    result = adaptive_amplitude_coverage(weights, [[1, 0.5, 0], [0, 0, 0]])
    np.testing.assert_array_equal(result["support_present"], [True, False])
    np.testing.assert_array_equal(result["reduced_amplitude_atoms"], [2, 0])
    np.testing.assert_allclose(result["reduced_amplitude_weight"], [0.75, 0])
    np.testing.assert_allclose(result["amplitude_deficit_weight"], [0.5, 0])


def test_adaptive_coverage_preserves_axes_species_and_permutation():
    from molsimflow.postprocess.pimd.region_diagnostics import adaptive_amplitude_coverage

    weights = np.array([[[1, 0.2, 0], [0.5, 0.4, 0.8]], [[0, 0, 0], [0.1, 1, 1]]])
    amp = np.array([[[1, 0.5, 0], [1, 0.5, 0]], [[1, 0.2, 0], [1, 0.2, 0]]])
    result = adaptive_amplitude_coverage(weights, amp)
    perm = [2, 0, 1]
    permuted = adaptive_amplitude_coverage(weights[..., perm], amp[..., perm])
    left = adaptive_amplitude_coverage(weights[..., :1], amp[..., :1])
    right = adaptive_amplitude_coverage(weights[..., 1:], amp[..., 1:])
    for key in result:
        assert result[key].shape == (2, 2)
        np.testing.assert_allclose(result[key], permuted[key])
        if key != "support_present":
            np.testing.assert_allclose(result[key], left[key] + right[key])


def test_adaptive_coverage_tolerance_only_changes_classification():
    from molsimflow.postprocess.pimd.region_diagnostics import adaptive_amplitude_coverage

    exact = adaptive_amplitude_coverage([1, 1], [1, 1 - 1e-13], amplitude_tolerance=0)
    tolerant = adaptive_amplitude_coverage([1, 1], [1, 1 - 1e-13])
    assert exact["reduced_amplitude_atoms"] == 1
    assert tolerant["reduced_amplitude_atoms"] == 0
    assert exact["amplitude_deficit_weight"] == tolerant["amplitude_deficit_weight"]


@pytest.mark.parametrize(
    "weights,amp,tol",
    [
        ([1, 1], [4, 1], 0),
        ([1, 1], [1, -0.1], 0),
        ([1, 1], [1, np.nan], 0),
        ([[1, 1]], [1, 1], 0),
        ([], [], 0),
        ([1, 1.1], [1, 1], 0),
        ([1, np.nan], [1, 1], 0),
        ([1, 1], [1, 1], -0.1),
        ([1, 1], [1, 1], 1),
        ([1, 1], [1, 1], np.nan),
    ],
)
def test_adaptive_coverage_rejects_counts_broadcast_and_invalid_inputs(weights, amp, tol):
    from molsimflow.postprocess.pimd.region_diagnostics import adaptive_amplitude_coverage

    with pytest.raises(ValueError):
        adaptive_amplitude_coverage(weights, amp, amplitude_tolerance=tol)


def test_general_slab_ignores_transverse_coordinates_and_matches_moving_reference():
    from molsimflow.postprocess.pimd.region_diagnostics import region_weights

    x = np.array([[[9, -8, 0.5], [1, 3, 1.5], [-2, 4, 2.5]]])
    d, w = region_weights(x, [0, 0, 0], core_radius=1, outer_radius=2, geometry="slab_z")
    np.testing.assert_allclose(d, [[0.5, 1.5, 2.5]])
    np.testing.assert_allclose(w, [[1, 0.5, 0]])
    shifted = region_weights(
        x + 7,
        np.array([[7, 7, 7]]),
        core_radius=1,
        outer_radius=2,
        geometry="slab_z",
        box_lengths=[2, 3, 10],
    )
    np.testing.assert_allclose(shifted[1], w)
    with pytest.raises(ValueError):
        region_weights(x, [0, 0, 0], core_radius=1, outer_radius=2, geometry="cylinder")


def test_normalization_competitor_outside_geometric_region_is_required():
    from molsimflow.postprocess.pimd.region_diagnostics import (
        adaptive_amplitude_coverage,
        voronoi_dependency_support,
    )

    # Center 2 has zero CV weight, but competes for the assigned atom near center 1.
    centers = np.array([[0, 0, 0], [1.1, 0, 0], [4, 0, 0]])
    assigned = np.array([[0.5, 0, 0], [4.1, 0, 0]])
    masks = voronoi_dependency_support(
        centers, assigned, [1, 0, 0], pair_switch=0.7, pair_cutoff=1.2, chunk_size=1
    )
    np.testing.assert_array_equal(masks["assigned_support"], [True, False])
    np.testing.assert_array_equal(masks["normalization_centers"], [True, True, False])
    assert not masks["missing_unswitched_assignments"].any()
    geometric = adaptive_amplitude_coverage([1, 0, 0], [1, 0.5, 0])
    chemical = adaptive_amplitude_coverage(masks["normalization_centers"], [1, 0.5, 0])
    assert geometric["reduced_amplitude_atoms"] == 0
    assert chemical["reduced_amplitude_atoms"] == 1


def test_dependency_support_axes_periodic_images_chunking_and_missing_center():
    from molsimflow.postprocess.pimd.region_diagnostics import voronoi_dependency_support

    centers = np.array([[[[0.1, 0, 0], [1.2, 0, 0]]]])
    assigned = np.array([[[[9.8, 0, 0], [0.6, 0, 0]]]])
    kwargs = {"pair_switch": 0.4, "pair_cutoff": 1.3, "box_lengths": [10, 10, 10]}
    a = voronoi_dependency_support(centers, assigned, np.ones((1, 1, 2)), chunk_size=1, **kwargs)
    b = voronoi_dependency_support(
        centers + 10, assigned, np.ones((1, 1, 2)), chunk_size=9, **kwargs
    )
    for key in a:
        assert a[key].shape == (1, 1, 2)
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(a["missing_unswitched_assignments"], [[[False, True]]])
    with pytest.raises(ValueError):
        voronoi_dependency_support(centers, assigned[0], [1, 1], **kwargs)
