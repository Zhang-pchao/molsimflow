"""Synthetic geometry tests for offline path moments and their limits."""

import numpy as np
import pytest

from molsimflow.postprocess.quantum_path import describe_quantum_path, unwrap_compact_path


BOX = np.full(3, 20.0)
TYPES = np.array([1, 1, 2])
PARAMETERS = dict(center_type=1, assigned_type=2, kappa=2.0, distance_kappa=3.0,
                  reference=0.5, environment_r0=2.0)


def path_fixture():
    path = np.repeat(np.array([[[-1.0, 0, 0], [1.0, 0, 0], [-0.7, 0.15, 0]]]), 3, axis=0)
    path[:, 2, 0] = [-0.7, -0.5, 0.2]
    return path


def describe(path, types=TYPES, box=BOX, **kwargs):
    return describe_quantum_path(path, types, box, **(PARAMETERS | kwargs))


def assert_same_descriptors(left, right, excluded=()):
    assert left.keys() == right.keys()
    for key in left.keys() - {"centroid_positions", *excluded}:
        np.testing.assert_allclose(left[key], right[key], rtol=1e-12, atol=2e-14, err_msg=key)


def test_one_bead_analytic_assignment_and_separate_distance_kappa():
    path = np.array([[[-1.0, 0, 0], [1.0, 0, 0], [-0.5, 0, 0]]])
    result = describe(path)
    defect = 1 / (1 + np.exp(-PARAMETERS["kappa"])) - 0.5
    distance_defect = 1 / (1 + np.exp(-PARAMETERS["distance_kappa"])) - 0.5
    np.testing.assert_allclose(result["defect_beads"], [[defect, -defect]])
    assert result["q_bead_mean"] == pytest.approx(2 * defect**2)
    assert result["q_centroid"] == pytest.approx(result["q_bead_mean"])
    assert result["mean_defect_square"] == pytest.approx(result["q_centroid"])
    assert result["distance_centroid"] == pytest.approx(2 * distance_defect**2)
    np.testing.assert_allclose(result["distance_beads"], result["distance_centroid"])
    np.testing.assert_allclose(result["oo_coordination_centroid"], [0.5, 0.5])
    assert result["oo_coordination_centroid_mean"] == pytest.approx(0.5)
    for key in ("v_occ", "q_bead_variance", "distance_bead_variance",
                "variance_identity_residual"):
        assert result[key] == 0


def test_variance_identity_does_not_replace_mean_defects_by_centroid_defects():
    result = describe(path_fixture())
    assert result["v_occ"] > 0
    assert result["q_bead_variance"] > 0
    assert result["q_bead_mean"] == pytest.approx(
        result["mean_defect_square"] + result["v_occ"], abs=1e-14
    )
    assert abs(result["variance_identity_residual"]) < 1e-14
    assert abs(result["mean_defect_square"] - result["q_centroid"]) > 1e-3
    assert abs(result["v_occ"] - (result["q_bead_mean"] - result["q_centroid"])) > 1e-3
    np.testing.assert_allclose(np.sum(result["defect_beads"], axis=1), 0, atol=1e-15)


def test_translation_and_rotation_in_geometry_clear_of_periodic_boundaries():
    path = path_fixture()
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    translated = path @ rotation.T + [3, 2, -1]
    original = describe(path)
    transformed = describe(translated)
    assert_same_descriptors(original, transformed)
    np.testing.assert_allclose(
        transformed["centroid_positions"], original["centroid_positions"] @ rotation.T + [3, 2, -1]
    )


def test_integer_periodic_image_shifts_and_compact_boundary_crossing():
    path = path_fixture()
    shifts = np.random.default_rng(18).integers(-3, 4, size=path.shape)
    assert_same_descriptors(describe(path), describe(path + shifts * BOX))
    crossing = np.array([[[9.8, 0, 0]], [[0.1, 0, 0]], [[0.2, 0, 0]]])
    unwrapped = unwrap_compact_path(crossing, [10, 10, 10])
    np.testing.assert_allclose(unwrapped[:, 0, 0], [9.8, 10.1, 10.2])
    np.testing.assert_allclose(np.mean(unwrapped, axis=0)[0, 0], (9.8 + 10.1 + 10.2) / 3)


def test_atom_reordering_is_equivariant_for_site_arrays():
    path = path_fixture()
    order = [2, 1, 0]
    original = describe(path)
    reordered = describe(path[:, order], TYPES[order])
    site_keys = ("defect_beads", "defect_centroid", "defect_bead_mean",
                 "oo_coordination_centroid")
    assert_same_descriptors(original, reordered, excluded=site_keys)
    for key in site_keys:
        np.testing.assert_allclose(reordered[key], np.asarray(original[key])[..., ::-1])
    np.testing.assert_allclose(
        reordered["centroid_positions"], original["centroid_positions"][order]
    )


@pytest.mark.parametrize("order", ([1, 2, 0], [2, 1, 0]))
def test_bead_cyclic_and_reversal_symmetries(order):
    original = describe(path_fixture())
    reordered = describe(path_fixture()[order])
    bead_keys = ("defect_beads", "q_beads", "distance_beads")
    assert_same_descriptors(original, reordered, excluded=bead_keys)
    for key in bead_keys:
        np.testing.assert_allclose(reordered[key], original[key][order])


def test_unit_conversion_scales_distance_but_preserves_dimensionless_moments():
    original = describe(path_fixture())
    scale = 0.1  # Angstrom to nm; inverse-length sharpness must scale inversely.
    converted = describe(path_fixture() * scale, box=BOX * scale,
                         kappa=PARAMETERS["kappa"] / scale,
                         distance_kappa=PARAMETERS["distance_kappa"] / scale,
                         environment_r0=PARAMETERS["environment_r0"] * scale)
    distance_keys = ("distance_beads", "distance_centroid", "distance_bead_mean",
                     "distance_bead_variance")
    assert_same_descriptors(original, converted, excluded=distance_keys)
    for key in distance_keys:
        power = 2 if key == "distance_bead_variance" else 1
        np.testing.assert_allclose(converted[key], np.asarray(original[key]) * scale**power)
    np.testing.assert_allclose(
        converted["centroid_positions"], original["centroid_positions"] * scale
    )


@pytest.mark.parametrize("path, box", [
    (np.zeros((0, 3, 3)), BOX), (np.zeros((2, 0, 3)), BOX),
    (np.zeros((2, 3)), BOX), (np.full((2, 3, 3), np.nan), BOX),
    (np.full((2, 3, 3), np.inf), BOX), (path_fixture(), [0, 20, 20]),
    (path_fixture(), [np.inf, 20, 20]), (path_fixture(), np.eye(3)),
    (path_fixture(), [np.nan, 20, 20]), (path_fixture(), [-1, 20, 20]),
])
def test_invalid_geometry_fails_closed(path, box):
    with pytest.raises(ValueError):
        unwrap_compact_path(path, box)


@pytest.mark.parametrize("offsets", ([0, 5], [0, -4, 4]))
def test_half_box_or_extended_path_has_no_admitted_centroid(offsets):
    path = np.zeros((len(offsets), 1, 3))
    path[:, 0, 0] = offsets
    with pytest.raises(ValueError, match="ambiguous path"):
        unwrap_compact_path(path, [10, 10, 10])


@pytest.mark.parametrize("types", ([1, 2], [1.0, 1.0, 2.0], [1, 1, 2.5],
                                  [1, 1, 0], [True, True, False], [1, 1, 1]))
def test_invalid_atom_types_fail_without_silent_cast(types):
    with pytest.raises(ValueError):
        describe(path_fixture(), np.asarray(types))


@pytest.mark.parametrize("kwargs", [
    {"center_type": 1.5}, {"assigned_type": True}, {"assigned_type": 1},
    {"center_type": 0}, {"center_type": 3}, {"kappa": 0}, {"kappa": np.inf},
    {"kappa": np.nan}, {"distance_kappa": -1}, {"distance_kappa": np.inf},
    {"environment_r0": 0}, {"environment_r0": np.nan}, {"reference": np.inf},
    {"reference": np.nan},
])
def test_invalid_parameters_fail_closed(kwargs):
    with pytest.raises(ValueError):
        describe(path_fixture(), **kwargs)


@pytest.mark.parametrize("overlap", ("center_assigned", "centers", "centroid_only"))
def test_zero_pair_distances_fail_in_beads_and_centroid(overlap):
    path = path_fixture()
    if overlap == "center_assigned":
        path[0, 2] = path[0, 0]
    elif overlap == "centers":
        path[1, 1] = path[1, 0]
    else:
        path[:, 2] = path[:, 0] + np.array([[-0.3, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    with pytest.raises(ValueError, match="distances must exceed eps"):
        describe(path)


def test_one_center_fails_group1_only_voronoi_distance_contract():
    with pytest.raises(ValueError, match="at least two center atoms"):
        describe(path_fixture()[:, [0, 2]], np.array([1, 2]))
