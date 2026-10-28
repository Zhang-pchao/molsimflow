import numpy as np
import pytest
from molsimflow.postprocess.interfacial_contact_geometry import surface_grid
from molsimflow.postprocess.interfacial_contact_fields import (
    sampled_window_means, periodic_contact_field, periodic_mask_labels,
)


def chart():
    xy = np.stack(np.meshgrid(np.arange(.5, 12), np.arange(.5, 10),
                             indexing="ij"), axis=-1).reshape(-1, 2)
    return surface_grid(xy, np.array([[0., 12.], [0., 10.], [0., 20.]]), 1.)


def test_window_does_not_cross_gap_or_epoch():
    t = np.r_[np.arange(8), np.arange(9, 17)] * 10
    s = np.r_[np.zeros(4), np.ones(12)]
    i, m, n, d = sampled_window_means(t[:, None], t, 10, s, 1)
    assert i.tolist() == [1, 2, 5, 6, 9, 10, 11, 12, 13, 14]
    assert np.array_equal(m[:, 0], t[i])
    assert np.all(n == 3) and np.all(d == 0)


@pytest.mark.parametrize("offset", range(4))
def test_decimation_keeps_gaps_visible_and_reports_time_offset(offset):
    t = np.arange(25) * 10
    v = np.stack([t, 2*t], axis=1)
    i, m, n, d = sampled_window_means(v, t, 10, np.zeros(25), 4,
                                      centers=np.array([12]), stride=4, offset=offset)
    rows = np.arange(8, 17)
    rows = rows[rows % 4 == offset]
    assert np.array_equal(m[0], v[rows].mean(0))
    assert n[0] == len(rows) and d[0] == t[rows].mean()-t[12]
    s = np.zeros(25); s[11:] = 1
    assert len(sampled_window_means(v, t, 10, s, 4, centers=np.array([12]),
                                    stride=4, offset=offset)[0]) == 0


def test_insufficient_support_returns_shaped_empty():
    i, m, n, d = sampled_window_means(np.ones((2, 3)), np.array([0, 10]),
                                     10, np.zeros(2), 4)
    assert m.shape == (0, 3) and len(i) == len(n) == len(d) == 0


def test_spatial_mean_and_periodic_translation():
    g = chart()
    v = np.zeros((12, 10)); v[10:, 3:6] = 1; v[:2, 3:6] = 1
    f = periodic_contact_field(g, v.ravel(), 1.4)
    assert np.isclose(f.mean(), v.mean())
    q = periodic_contact_field(g, np.roll(v, (4, 3), (0, 1)).ravel(), 1.4)
    assert np.allclose(q, np.roll(f, (4, 3), (0, 1)))
    assert np.allclose(periodic_contact_field(g, np.full((2, 120), .3), 2), .3)


def test_components_merge_across_seam_and_preserve_hole_and_satellite():
    mask = np.zeros((12, 10), bool)
    mask[[0, 1, 10, 11], 2:8] = True
    mask[0, 4:6] = False
    mask[5, 5] = True
    labels, sizes = periodic_mask_labels(mask)
    assert sorted(sizes.tolist()) == [1, 22]
    assert labels[0, 3] == labels[11, 3]
    assert labels[0, 4] == 0
    assert sizes.sum() == mask.sum()
    assert periodic_mask_labels(np.zeros((4, 5), bool))[1].size == 0
    assert periodic_mask_labels(np.ones((4, 5), bool))[1].tolist() == [20]


@pytest.mark.parametrize("value", [-.01, 1.01, np.nan])
def test_invalid_occupancy_rejected(value):
    with pytest.raises(ValueError):
        periodic_contact_field(chart(), np.full(120, value), 1)


def test_invalid_times_rejected():
    with pytest.raises(ValueError):
        sampled_window_means(np.ones((3, 2)), np.array([0, 2, 1]), 1, np.zeros(3), 1)
