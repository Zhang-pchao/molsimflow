import numpy as np
import pytest
from molsimflow.postprocess.interfacial_contact_events import (
    periodic_exterior_void, complete_window_centers, saved_contact_states,
)
def test_contact_ring_and_internal_hole():
    m = np.zeros((15, 15), bool);m[4:11, 4:11] = True;m[6:9, 6:9] = False
    e = periodic_exterior_void(m)
    assert e[0, 0] and not e[7, 7] and not e[m].any()
    assert np.array_equal(e, np.roll(periodic_exterior_void(np.roll(m, 7, axis=0)), -7, axis=0))
def test_periodic_crossing_finite_void_not_winding():
    m = np.ones((12, 12), bool);m[np.ix_([11, 0, 1], [11, 0, 1])] = False
    assert not periodic_exterior_void(m).any()
def test_winding_channel_and_extremes():
    m = np.ones((12, 12), bool);m[:, 3:6] = False
    assert np.array_equal(periodic_exterior_void(m), ~m)
    assert periodic_exterior_void(np.zeros((8, 8), bool)).all()
    assert not periodic_exterior_void(np.ones((8, 8), bool)).any()
def test_gain_loss_and_stable_states():
    c = np.zeros((15, 4), bool);c[7:, 0] = True;c[:7, 1] = True;c[:, 3] = True
    out = saved_contact_states(c, np.arange(15)*10, 10, persistence=3)
    assert out[7].tolist() == [1, -1, 2, 3]
    assert not out[:3].any()
    assert (out[6, :2] == 4).all()
def test_missing_time_not_joined():
    steps = np.r_[np.arange(8), np.arange(10, 18)] * 10
    c = np.zeros((16, 1), bool);c[8:] = True
    labels = saved_contact_states(c, steps, 10, persistence=3)
    assert labels[5:11].tolist() == [[0]]*6
    v = complete_window_centers(steps, 10, 2)
    assert not v[6:10].any() and v[2]
@pytest.mark.parametrize("bad", [np.zeros((3, 3), int), np.zeros((2, 3), bool), np.zeros(3, bool)])
def test_invalid_masks(bad):
    with pytest.raises(ValueError):periodic_exterior_void(bad)
def test_invalid_temporal_inputs():
    with pytest.raises(ValueError):complete_window_centers(np.array([0, 10, 10]), 10, 1)
    with pytest.raises(ValueError):saved_contact_states(np.zeros((5, 2)), np.arange(5), 1)

def test_diagonal_periodic_channel():
    m = np.ones((12, 12), bool)
    for i in range(12):
        m[i, i] = False
        m[(i+1)%12, i] = False
    assert np.array_equal(periodic_exterior_void(m), ~m)
    shifted = np.roll(np.roll(m, 4, axis=0), 7, axis=1)
    assert np.array_equal(periodic_exterior_void(shifted), ~shifted)
