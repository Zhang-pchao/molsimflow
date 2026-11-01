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


from molsimflow.postprocess.interfacial_contact_events import persistent_state_intervals

def test_interval_confirmation_and_long_unknown():
    states = np.array([1]*5 + [-1]*50 + [0]*5)
    runs, events = persistent_state_intervals(states, np.arange(60)*10, 10)
    assert (events[0].left, events[0].right, events[0].confirmed) == (4, 55, 59)
    assert (events[0].before, events[0].after) == (1, 0)
    assert len(runs) == 2

def test_same_state_updates_last_support_and_short_runs_stay_uncertain():
    s = np.array([1]*3 + [-1]*2 + [1]*3 + [0] + [-1]*2 + [0]*3)
    runs, e = persistent_state_intervals(s, np.arange(len(s)), 1, persistence=3)
    assert len(runs) == 3 and len(e) == 1
    assert (e[0].left, e[0].right, e[0].confirmed) == (7, 11, 13)

@pytest.mark.parametrize("gap,epoch", [(True, False), (False, True)])
def test_interval_never_crosses_native_or_explicit_boundary(gap, epoch):
    s = np.array([1]*3 + [0]*3); t = np.arange(6)*10
    if gap:t[3:] += 10
    segments = np.array([0]*3 + [int(epoch)]*3)
    _, e = persistent_state_intervals(s, t, 10, persistence=3, segments=segments)
    assert not e

def test_censored_runs_and_empty_series():
    runs, e = persistent_state_intervals(np.ones(6, int), np.arange(6), 1)
    assert len(runs) == 1 and not e
    assert persistent_state_intervals(np.array([], int), np.array([], int), 1) == ([], [])
    assert not persistent_state_intervals(np.ones(4, int), np.arange(4), 1)[0]

@pytest.mark.parametrize("s,t,k", [
    ([0,2], [0,1], 1), ([0,1], [0,0], 1), ([0,1], [0,1], 0),
    ([0.,1.], [0,1], 1),
])
def test_invalid_interval_input(s,t,k):
    with pytest.raises(ValueError):
        persistent_state_intervals(np.array(s), np.array(t), 1, persistence=k)
