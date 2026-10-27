import numpy as np
import pytest
from molsimflow.postprocess.interfacial_gas_kinematics import (
    quantile_layer_labels, lagged_state_statistics, fit_rigid_motion,
    coarse_velocity_field,
)


def test_layers_preserve_identity_and_outside():
    z = np.arange(10.)
    member = np.arange(10) != 4
    s = quantile_layer_labels(z, member)
    assert s[4] == -1 and s[0] == 0 and s[-1] == 2
    assert (s == 0).sum() == (s == 1).sum() == (s == 2).sum() == 3


def test_exchange_distinguishes_return_survival_and_cluster_loss():
    s = np.array([[0, 0, 0], [1, 0, -1], [0, 2, -1]])
    r = lagged_state_statistics(s, 2)
    assert r["return_fraction"] == 1 / 3
    assert r["sampled_survival_fraction"] == 0
    assert r["terminal_fractions"][-1] == 1 / 3


def test_persistent_population():
    r = lagged_state_statistics(np.zeros((8, 5), dtype=int), 3)
    assert r["return_fraction"] == r["sampled_survival_fraction"] == 1
    assert r["origin_count"] == 25


def test_rigid_rotation_with_translation_and_shift():
    x = np.random.default_rng(3).normal(size=(50, 3))
    w = np.array([.4, .2, -.1])
    v = np.array([3., 2., 1.]) + np.cross(w, x - x.mean(0))
    r = fit_rigid_motion(x + 1000, v)
    assert np.allclose(r["translation"], [3, 2, 1])
    assert np.allclose(r["omega"], w)
    assert r["rotation_energy_fraction"] == pytest.approx(1)
    assert r["residual_rms"] < 1e-10


def test_translation_is_not_rotation():
    x = np.random.default_rng(2).normal(size=(50, 3))
    r = fit_rigid_motion(x, np.tile([5., 0, 0], (50, 1)))
    assert np.allclose(r["omega"], 0)
    assert r["rotation_energy_fraction"] == 0


def test_field_conservation_and_empty_bins():
    p = np.array([[.2, 1, .2], [.3, 2, .3], [1.5, 4, 1.5]])
    v = np.array([[1., 2, 3], [4., 5, 6], [7., 8, 9]])
    n, r, u = coarse_velocity_field(p, v, [0, 1, 2], [0, 1, 2])
    assert n.sum() == 3
    assert np.allclose(r.sum((0, 1)), p.sum(0))
    assert np.allclose(u.sum((0, 1)), v.sum(0))
    assert n[0, 1] == 0


@pytest.mark.parametrize("bad", [0, 3, .5])
def test_invalid_lag_rejected(bad):
    with pytest.raises(ValueError):
        lagged_state_statistics(np.zeros((3, 2), int), bad)


def test_degenerate_motion_and_layers_rejected():
    with pytest.raises(ValueError):
        fit_rigid_motion(np.zeros((3, 3)), np.ones((3, 3)))
    with pytest.raises(ValueError):
        quantile_layer_labels(np.zeros(4), np.ones(4, bool))
