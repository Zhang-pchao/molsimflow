import numpy as np
import pytest
from molsimflow.postprocess.surface_state_persistence import confirm_core_states


def run(core, dwell=2, times=None, segments=None):
    n = len(core)
    return confirm_core_states(np.array(core), np.arange(n) if times is None else times,
                               ["s"] * n if segments is None else segments, 1, dwell)


def test_six_saved_frames_are_required_for_five_time_units():
    assert run([1] * 6, dwell=5).tolist() == [-1, -1, -1, -1, -1, 1]


def test_brief_outward_recrossing_does_not_confirm_exit():
    assert run([1, 1, 1, 0, 0, 1]).tolist() == [-1, -1, 1, 1, 1, 1]


def test_confirmed_exit_is_causal_not_backdated():
    assert run([1, 1, 1, 0, 0, 0]).tolist() == [-1, -1, 1, 1, 1, 0]


def test_neutral_band_retains_memory_but_breaks_pending_dwell():
    assert run([1, 1, 1, 0, -1, 0, 0, 0]).tolist() == [-1, -1, 1, 1, 1, 1, 1, 0]


@pytest.mark.parametrize("kind", ["missing", "source", "time"])
def test_unknown_data_never_bridge_confirmation(kind):
    core = [1] * 7
    times = np.arange(7, dtype=float)
    segments = ["a"] * 7
    if kind == "missing":
        core[3] = -2
        expected = [-1, -1, 1, -1, -1, -1, 1]
    elif kind == "source":
        segments[3:] = ["b"] * 4
        expected = [-1, -1, 1, -1, -1, 1, 1]
    else:
        times[3:] += 1
        expected = [-1, -1, 1, -1, -1, 1, 1]
    assert run(core, times=times, segments=segments).tolist() == expected


def test_vectorized_objects_have_independent_histories():
    c = np.array([[1, 0], [1, -1], [1, 0], [0, 0], [0, 0]])
    result = run(c)
    assert result[:, 0].tolist() == [-1, -1, 1, 1, 1]
    assert result[:, 1].tolist() == [-1, -1, -1, -1, 0]


def test_two_occupant_exchange_is_not_net_depletion():
    core = np.array([[1, 0]] * 3 + [[0, 1]] * 3)
    committed = run(core)
    assert np.all((committed[2:] == 1).sum(axis=1) == 1)


def test_appending_a_future_return_never_changes_the_past():
    c = [1] * 3 + [0] * 3
    assert np.array_equal(run(c), run(c + [1] * 3)[:len(c)])


@pytest.mark.parametrize("times", [[0, 0], [0, float("nan")]])
def test_invalid_times_rejected(times):
    with pytest.raises(ValueError):
        run([1, 1], times=times)


def test_empty_trace_and_zero_dwell():
    assert run([]).shape == (0,)
    assert run([1, 0], dwell=0).tolist() == [1, 0]
