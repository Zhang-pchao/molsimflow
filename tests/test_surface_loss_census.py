import numpy as np
import pytest
from molsimflow.postprocess.surface_loss_census import prospective_count_losses


def census(w, times=None, segments=None, **kwargs):
    w = np.asarray(w)[:, None]
    parameters = dict(expected_dt=1., history_span=2., search_span=3.,
                      minimum_initial_count=5, minimum_loss=3, fractional_loss=.3,
                      history_absolute_tolerance=2, history_fractional_tolerance=.2)
    parameters.update(kwargs)
    return prospective_count_losses(
        w, np.arange(len(w)) if times is None else times,
        np.zeros(len(w)) if segments is None else segments, **parameters)


def test_identity_turnover_without_net_count_loss_is_not_a_loss():
    result = census([10] * 8)
    assert result.eligible[2:, 0].all()
    assert (result.first_loss_index == -1).all()


def test_first_observed_threshold_crossing_and_no_future_selection():
    result = census([10, 10, 10, 9, 8, 7, 10])
    assert result.eligible[2, 0]
    assert result.first_loss_index[2, 0] == 5
    assert not result.eligible[:2].any()


def test_reset_boundary_censors_before_loss_and_past_history_after_reset():
    result = census([10, 10, 10, 7, 7, 7], segments=[0, 0, 0, 1, 1, 1])
    assert result.eligible[2, 0]
    assert result.available_followup_frames[2] == 0
    assert result.first_loss_index[2, 0] == -1
    assert not result.eligible[3:5].any()
    assert result.eligible[5, 0]


def test_missing_frame_is_not_bridged():
    result = census([10, 10, 10, 7, 7, 7], times=[0, 1, 2, 4, 5, 6])
    assert result.available_followup_frames[2] == 0
    assert result.first_loss_index[2, 0] == -1


def test_shortened_followup_can_observe_loss_but_not_full_window_absence():
    result = census([10, 10, 10, 7])
    assert result.first_loss_index[2, 0] == 3
    assert result.available_followup_frames[2] == 1
    no_loss = census([10, 10, 10, 9])
    assert no_loss.first_loss_index[2, 0] == -1
    assert no_loss.available_followup_frames[2] < 3


def test_fractional_threshold_is_based_on_initial_population():
    result = census([20, 20, 20, 17, 15, 14])
    assert result.first_loss_index[2, 0] == 5


def test_fluctuating_past_fails_eligibility_even_if_future_decreases():
    result = census([5, 10, 10, 7])
    assert not result.eligible[2, 0]
    assert result.first_loss_index[2, 0] == -1


@pytest.mark.parametrize("w", [[10, -1, 7], [10, np.nan, 7], [10, 9.5, 7]])
def test_invalid_counts_rejected(w):
    with pytest.raises(ValueError):
        census(w)
