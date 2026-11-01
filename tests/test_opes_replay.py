"""Analytic and compression invariants for recorded OPES history replay."""
import numpy as np
import pytest
from molsimflow.postprocess.opes_replay import NonperiodicOPESReplay


def model(kernels=None, threshold=1):
    if kernels is None:
        kernels = np.array([[0., 0., .4, .2, 2.]])
    return NonperiodicOPESReplay(kernels, 2., 20., 1e-8, 5., threshold, .025)


def moments(k):
    return (k[:, -1].sum(), (k[:, -1, None]*k[:, :2]).sum(0),
            (k[:, -1, None]*(k[:, :2]**2+k[:, 2:4]**2)).sum(0))


def test_gradient_and_cutoff():
    m = model()
    x = np.array([.3, .1])
    e, g = m.evaluate(x)
    for i in range(2):
        d = np.eye(2)[i]*1e-6
        numeric = (m.evaluate(x+d)[0]-m.evaluate(x-d)[0])/2e-6
        assert g[i] == pytest.approx(numeric, abs=1e-9)
    outside, gradient = m.evaluate(np.array([10., 10.]))
    assert outside == pytest.approx(.025*(1-1/20)*np.log(1e-8))
    np.testing.assert_array_equal(gradient, [0., 0.])
    assert np.isfinite(e)


def test_recursive_merge_preserves_moments_and_weights():
    m = model(np.array([[0., 0., 1., 1., 1.], [1.1, 0., 1., 1., 1.]]))
    new = np.array([.8, 0., .5, .5, 2.])
    expected = moments(np.vstack((m.kernels, new)))
    m.add(new[:2], new[2:4], new[4], np.log(3.))
    assert len(m.kernels) == 1
    for a, b in zip(moments(m.kernels), expected):
        np.testing.assert_allclose(a, b, atol=1e-14)
    assert m.sum_weights == pytest.approx(5.)


def test_empty_start_and_disabled_compression():
    m = model(np.empty((0, 5)), threshold=0)
    assert m.normalization() == 1.
    assert np.isfinite(m.evaluate(np.array([0., 0.]))[0])
    for _ in range(2):
        m.add(np.zeros(2), np.ones(2), 1., 0.)
    assert len(m.kernels) == 2


@pytest.mark.parametrize("sigma,height,log_weight", [([-1., 1.], 1., 0.),
                                                    ([1., 1.], -1., 0.),
                                                    ([1., 1.], 1., np.nan)])
def test_reject_invalid_deposition(sigma, height, log_weight):
    m = model()
    before = m.kernels.copy()
    with pytest.raises(ValueError):
        m.add(np.zeros(2), np.array(sigma), height, log_weight)
    np.testing.assert_array_equal(m.kernels, before)
