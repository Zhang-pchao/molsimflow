import numpy as np
import pytest
from molsimflow.postprocess.surface_exposure import (
    probe_accessibility, local_cover_depth, local_label_fraction,
)

BOX = np.array([[0., 20.], [0., 20.], [0., 30.]])
UP = np.array([[0., 0., 1.]])
EMPTY = np.empty((0, 3))


def test_upward_probe_reproduces_exposed_and_occluded_geometry():
    sites = np.array([[2., 2., 5.], [10., 10., 5.]])
    heavy = np.vstack([sites, [10., 10., 8.]])
    r = probe_accessibility(sites, heavy, EMPTY, BOX, directions=UP)
    np.testing.assert_array_equal(r.accessible[:, 0], [True, False])
    np.testing.assert_allclose(r.heavy_distance_A[:, 0], [2.5, .5])


def test_bound_hydrogen_can_occlude_an_otherwise_open_probe():
    s = np.array([[3., 3., 5.]])
    r = probe_accessibility(s, s, np.array([[3., 3., 7.8]]), BOX, directions=UP)
    assert not r.accessible[0, 0]
    np.testing.assert_allclose(r.hydrogen_distance_A, [[.3]])


def test_periodic_boundary_and_translation_invariance():
    s = np.array([[19.8, 2., 5.]])
    h = np.vstack([s, [.2, 2., 7.5]])
    r = probe_accessibility(s, h, EMPTY, BOX, directions=UP)
    shift = np.array([13., -27., 42.])
    t = probe_accessibility(s + shift, h + shift, EMPTY, BOX + shift[:, None],
                            directions=UP)
    np.testing.assert_allclose(r.clearance_A, t.clearance_A)
    assert not r.accessible[0, 0]
    q = probe_accessibility(s, h, EMPTY, BOX, directions=UP,
                            periodic=(False, False, False))
    assert q.accessible[0, 0]


def test_nonperiodic_z_does_not_create_false_opposite_face_cover():
    s = np.array([[2., 2., 29.]])
    h = np.vstack([s, [2., 2., 1.]])
    assert not probe_accessibility(s, h, EMPTY, BOX, directions=UP).accessible[0, 0]
    assert probe_accessibility(s, h, EMPTY, BOX, directions=UP,
                               periodic=(True, True, False)).accessible[0, 0]


def test_direction_sensitivity_kept_separate_from_upward_probe():
    s = np.array([[5., 5., 5.]])
    h = np.vstack([s, [5., 5., 7.5]])
    d = np.array([[0., 0., 1.], [1., 0., 0.]])
    np.testing.assert_array_equal(
        probe_accessibility(s, h, EMPTY, BOX, directions=d).accessible, [[False, True]])


def test_local_cover_and_periodic_patch_count():
    s = np.array([[.2, 3., 5.], [19.8, 3., 7.], [10., 10., 6.]])
    np.testing.assert_allclose(local_cover_depth(s, s, BOX, lateral_radius_A=1.), [2., 0., 0.])
    fraction, count = local_label_fraction(s, np.array([True, False, True]), BOX,
                                          lateral_radius_A=1.)
    np.testing.assert_allclose(fraction, [.5, .5, 1.])
    np.testing.assert_array_equal(count, [2, 2, 1])


@pytest.mark.parametrize('kwargs', [dict(offset_A=0), dict(heavy_exclusion_A=-1),
                                    dict(periodic=(True, True)),
                                    dict(directions=np.array([[0., 0., 2.]]))])
def test_invalid_probe_contract_rejected(kwargs):
    s = np.array([[5., 5., 5.]])
    options = dict(directions=UP);options.update(kwargs)
    with pytest.raises(ValueError):probe_accessibility(s, s, EMPTY, BOX, **options)


def test_duplicate_projected_sites_rejected():
    s = np.array([[2., 2., 3.], [2., 2., 5.]])
    with pytest.raises(ValueError):
        local_label_fraction(s, np.array([True, False]), BOX, lateral_radius_A=2.)
