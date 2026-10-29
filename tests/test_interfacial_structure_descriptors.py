"""Synthetic invariances and support guards for spatial mark descriptors."""
import numpy as np
import pytest
from molsimflow.postprocess.interfacial_structure_descriptors import (
    periodic_site_edges, directional_mark_contrast,
)


def fixture():
    xy = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    edges, vectors = periodic_site_edges(xy, [5, 5], 1.1)
    return xy, edges, vectors


def test_directional_binary_stripes():
    _, e, v = fixture()
    q = directional_mark_contrast([0, 1, 0, 1], e, v)
    assert q["parallel"] == 1
    assert q["transverse"] == 0
    assert q["isotropic"] == .5
    assert q["contrast"] == 1
    assert q["positive_edges"] == 4


def test_axis_reversal_and_exchange():
    _, e, v = fixture()
    marks = [0, 1, 0, 1]
    q = directional_mark_contrast(marks, e, v)
    reverse = directional_mark_contrast(marks, e, v, axis=(-1, 0))
    swap = directional_mark_contrast(marks, e, v, axis=(0, 1))
    assert reverse["contrast"] == q["contrast"]
    assert swap["contrast"] == -q["contrast"]


def test_periodic_translation_and_rotation():
    xy, e, v = fixture()
    e2, v2 = periodic_site_edges(xy + [4.5, 9.5], [5, 5], 1.1)
    np.testing.assert_array_equal(e, e2)
    np.testing.assert_allclose(v, v2)
    rot = np.array([[0, -1], [1, 0]])
    q = directional_mark_contrast([0, 1, 0, 1], e, v)
    qr = directional_mark_contrast([0, 1, 0, 1], e, v @ rot.T, axis=rot @ [1, 0])
    assert q["contrast"] == qr["contrast"]


def test_permutation_invariance():
    xy, e, v = fixture()
    z = np.array([.1, .3, .4, .9])
    perm = [3, 1, 0, 2]
    e2, v2 = periodic_site_edges(xy[perm], [5, 5], 1.1)
    a = directional_mark_contrast(z, e, v)
    b = directional_mark_contrast(z[perm], e2, v2)
    for key in ["parallel", "transverse", "isotropic", "contrast"]:
        np.testing.assert_allclose(a[key], b[key])


def test_channel_batch_and_constant_marks():
    _, e, v = fixture()
    z = np.array([[0, 1, 0, 1], [1, 1, 1, 1]])
    q = directional_mark_contrast(z, e, v)
    np.testing.assert_allclose(q["contrast"], [1, 0])
    np.testing.assert_allclose(q["isotropic"], [.5, 0])


def test_unsupported_is_not_zero():
    q = directional_mark_contrast([0, 1], np.empty((0, 2), int),
                                  np.empty((0, 2)))
    assert q["positive_edges"] == 0
    assert np.isnan(q["contrast"])
    q = directional_mark_contrast([0, 1], np.array([[0, 1]]), [[1, 0]])
    assert q["parallel"] == 1 and np.isnan(q["transverse"])
    assert np.isnan(q["contrast"])


def test_weights_and_mark_scale():
    _, e, v = fixture()
    z = [0, 1, 0, 1]
    q = directional_mark_contrast(z, e, v, weights=np.ones(4)*7)
    assert q["contrast"] == 1
    q2 = directional_mark_contrast(np.asarray(z)*3 + 2, e, v)
    assert q2["contrast"] == 9
    q0 = directional_mark_contrast(z, e, v, weights=np.zeros(4))
    assert np.isnan(q0["isotropic"])


@pytest.mark.parametrize("xy,box,cut", [
    ([[0, 0], [5, 0]], [5, 5], 1),
    ([[0, 0]], [5, 5], 2.5),
    ([[np.nan, 0]], [5, 5], 1),
    ([[0, 0]], [0, 5], 1),
])
def test_bad_geometry(xy, box, cut):
    with pytest.raises(ValueError):
        periodic_site_edges(xy, box, cut)


@pytest.mark.parametrize("edges,vectors", [
    ([[1, 0]], [[1, 0]]), ([[0, 2]], [[1, 0]]),
    ([[0, 1]], [[0, 0]]), ([[0, 1], [0, 1]], [[1, 0], [1, 0]]),
    ([[0., 1.]], [[1, 0]]),
])
def test_bad_edges(edges, vectors):
    with pytest.raises(ValueError):
        directional_mark_contrast([0, 1], np.asarray(edges), vectors)


def test_empty_site_geometry():
    e, v = periodic_site_edges(np.empty((0, 2)), [5, 5], 1)
    assert e.shape == v.shape == (0, 2)


def test_spatial_arrangement_is_not_composition():
    _, e, v = fixture()
    across = directional_mark_contrast([0, 1, 0, 1], e, v)
    along = directional_mark_contrast([0, 0, 1, 1], e, v)
    assert across["contrast"] == -along["contrast"] == 1
    assert across["isotropic"] == along["isotropic"]
