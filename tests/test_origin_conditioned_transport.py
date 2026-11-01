"""Analytic endpoint transport checks, separate from trajectory bookkeeping."""
import numpy as np
import pytest

from molsimflow.postprocess.interfacial_contact_renewal import origin_conditioned_transport


def run(x0, x1, before, after, shift=(0., 0., 0.), ids=None, states=3):
    ids = np.arange(len(x0), dtype=np.int64) if ids is None else np.asarray(ids)
    return origin_conditioned_transport(
        ids, ids, np.asarray(x0), np.asarray(x1), np.asarray(before),
        np.asarray(after), np.asarray(shift), state_count=states)


def test_translation_with_leavers_keeps_origin_support_and_zero_relative_motion():
    x0 = np.array([[0., 1., 2.], [2., 3., 4.], [-1., 2., 4.]])
    shift = np.array([4., -2., .5])
    q = run(x0, x0 + shift, [1, 1, 2], [0, 2, 0], shift)
    assert q["counts"].tolist() == [[0, 0, 0], [1, 0, 1], [1, 0, 0]]
    assert q["origin_support"].tolist() == [0, 2, 1]
    assert q["destination_support"].tolist() == [2, 0, 1]
    assert np.array_equal(q["lab_displacement_sum"].sum(axis=1)[1], 2 * shift)
    assert np.array_equal(q["lab_displacement_squared_sum"].sum(axis=1)[1], 2 * shift**2)
    assert not q["relative_displacement_sum"].any()
    assert not q["relative_displacement_squared_sum"].any()


def test_stationary_particles_moving_geometry_are_not_lab_transport():
    x = np.array([[1., 2., 3.], [2., 5., 7.]])
    shift = np.array([2., 0., 0.])
    q = run(x, x, [1, 1], [0, 2], shift)
    assert q["counts"].sum() == 2
    assert not q["lab_displacement_sum"].any()
    assert np.array_equal(q["relative_displacement_sum"].sum(axis=1)[1], -2 * shift)


def test_image_unwrapped_periodic_crossing_uses_true_lab_displacement():
    lengths = np.array([10., 10., 10.])
    wrapped0 = np.array([[9., 2., 3.]])
    wrapped1 = np.array([[1., 2., 3.]])
    images0 = np.array([[0, 0, 0]])
    images1 = np.array([[1, 0, 0]])
    q = run(wrapped0 + images0 * lengths, wrapped1 + images1 * lengths, [1], [0])
    assert np.array_equal(q["lab_displacement_sum"][1, 0], [2., 0., 0.])
    assert np.array_equal(q["lab_displacement_squared_sum"][1, 0], [4., 0., 0.])


def test_ordered_identity_permutation_and_common_translation_invariance():
    rng = np.random.default_rng(13)
    x0 = rng.normal(size=(12, 3))
    x1 = x0 + rng.normal(size=x0.shape)
    ids = np.arange(100, 112)
    b, a = rng.integers(0, 3, size=(2, 12))
    shift = np.array([.3, -.2, .1])
    q = run(x0, x1, b, a, shift, ids)
    order = rng.permutation(12)
    r = run(x0[order] + 100, x1[order] + 100, b[order], a[order], shift, ids[order])
    for key in q:
        assert np.allclose(q[key], r[key], rtol=1e-12, atol=1e-12)
    delta = x1 - x0
    assert np.allclose(q["lab_displacement_sum"].sum((0, 1)), delta.sum(0))
    assert np.allclose(q["lab_displacement_squared_sum"].sum((0, 1)), (delta**2).sum(0))
    assert np.allclose(q["relative_displacement_sum"].sum((0, 1)),
                       (delta - shift).sum(0))


def test_time_reversal_transposes_counts_reverses_moments_but_not_squares():
    x0 = np.array([[1., 2., 3.], [5., 4., 2.], [2., 1., 7.]])
    x1 = x0 + np.array([[1., 2., 0.], [0., -3., 1.], [-1., 0., 4.]])
    shift = np.array([.5, 0., 0.])
    q = run(x0, x1, [0, 1, 1], [2, 0, 2], shift)
    r = run(x1, x0, [2, 0, 2], [0, 1, 1], -shift)
    assert np.array_equal(q["counts"], r["counts"].T)
    for name in ("lab", "relative"):
        assert np.allclose(q[name + "_displacement_sum"],
                           -r[name + "_displacement_sum"].transpose(1, 0, 2))
        assert np.allclose(q[name + "_displacement_squared_sum"],
                           r[name + "_displacement_squared_sum"].transpose(1, 0, 2))


def test_empty_particles_preserve_explicit_state_schema_and_zero_support():
    q = run(np.empty((0, 3)), np.empty((0, 3)), np.empty(0, int), np.empty(0, int))
    assert q["counts"].shape == (3, 3)
    assert q["lab_displacement_sum"].shape == (3, 3, 3)
    for value in q.values():
        assert np.isfinite(value).all() and not value.any()


@pytest.mark.parametrize("ids0,ids1", [
    ([1, 1], [1, 1]), ([1, 2], [2, 1]), ([1, 2], [1]),
    ([1., 2.], [1., 2.]), ([-1, 2], [-1, 2]), ([[1, 2]], [[1, 2]]),
])
def test_inconsistent_or_duplicate_identities_rejected(ids0, ids1):
    with pytest.raises(ValueError, match="IDs"):
        origin_conditioned_transport(
            np.asarray(ids0), np.asarray(ids1), np.zeros((2, 3)), np.zeros((2, 3)),
            np.array([1, 1]), np.array([1, 1]), np.zeros(3), state_count=3)


@pytest.mark.parametrize("before,after", [
    ([-1, 1], [1, 1]), ([1, 1], [1, 3]), ([1., 1.], [1, 1]),
    ([True, True], [1, 1]), ([1], [1, 1]), ([[1, 1]], [1, 1]),
])
def test_invalid_states_rejected(before, after):
    with pytest.raises(ValueError, match="integer state"):
        run(np.zeros((2, 3)), np.zeros((2, 3)), before, after)


@pytest.mark.parametrize("count", [0, -1, 2.5, True])
def test_invalid_state_count_rejected(count):
    with pytest.raises(ValueError, match="state_count"):
        run(np.zeros((1, 3)), np.zeros((1, 3)), [0], [0], states=count)


@pytest.mark.parametrize("x0,x1,shift", [
    (np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(3)),
    (np.zeros((2, 3)), np.zeros((1, 3)), np.zeros(3)),
    (np.full((2, 3), np.nan), np.zeros((2, 3)), np.zeros(3)),
    (np.zeros((2, 3)), np.full((2, 3), np.inf), np.zeros(3)),
    (np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2)),
    (np.zeros((2, 3)), np.zeros((2, 3)), np.full(3, np.nan)),
    (np.zeros((2, 3), dtype=complex), np.zeros((2, 3)), np.zeros(3)),
])
def test_invalid_coordinates_rejected(x0, x1, shift):
    with pytest.raises(ValueError):
        run(x0, x1, [1, 1], [0, 2], shift)


def test_finite_but_overflowing_moments_rejected():
    with pytest.raises(ValueError, match="overflow"):
        run(np.zeros((1, 3)), np.full((1, 3), 1e200), [1], [1])
