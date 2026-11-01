import numpy as np
import pytest
from molsimflow.postprocess.surface_water_renewal import (
    cohort_position_states, observed_cohort_history, site_restoration,
    periodic_grid_components, sampled_support_runs,
)


def test_periodic_lateral_and_vertical_are_distinct():
    xy = np.array([[9.8, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    state = cohort_position_states(xy, np.array([1.0, 1.0, 4.0, 9.0]),
                                   np.array([0., 0.]), np.array([10., 10.]), 1.)
    assert state.tolist() == [0, 1, 2, 3]


def test_rough_surface_height_follows_new_position():
    xy = np.array([[0., 0.], [4., 0.]])
    z = np.array([1., 5.])
    new_local_top = np.array([0., 4.])
    state = cohort_position_states(xy, z - new_local_top, [0., 0.], [20., 20.], 2.)
    assert state.tolist() == [0, 1]
    # Using the old patch top would falsely classify the second water as outer.
    assert cohort_position_states(xy, z, [0., 0.], [20., 20.], 2.).tolist() == [0, 2]


def test_endpoint_retention_is_not_continuous_residence():
    r = observed_cohort_history([0., 1., 2.], ["a"] * 3,
                                np.array([[0, 0], [1, 2], [0, 3]]), 1.)
    assert r["in_original_patch_layer"] == 1
    assert r["observed_return_anytime"] == 1
    assert r["observed_layer_exit"] == 1
    assert r["bulk_like"] == 1


@pytest.mark.parametrize("times,segments", [
    ([0., 1., 3.], ["a"] * 3),
    ([0., 1., 2.], ["a", "a", "b"]),
])
def test_gap_and_source_switch_censor(times, segments):
    assert observed_cohort_history(times, segments, np.zeros((3, 2), int), 1.)["status"].startswith("CENSORED")


def test_site_structure_survives_water_identity_change():
    old = {(10, 1), (11, 2), (12, 3)}
    now = {(20, 1), (11, 2), (21, 2)}
    r = site_restoration(old, now)
    assert (r["same_water_same_site"], r["new_water_same_site_only"], r["site_unrestored"]) == (1, 1, 1)
    assert r["same_site_old_and_new"] == 1
    perm = {10: 110, 11: 111, 12: 112, 20: 120, 21: 121}
    assert site_restoration({(perm[w], s) for w, s in old},
                            {(perm[w], s) for w, s in now}) == r


def test_periodic_grid_connectivity_and_diagonals():
    a = np.zeros((5, 5), bool)
    a[0, 0] = a[-1, 0] = True
    assert len(np.unique(periodic_grid_components(a)[a])) == 1
    a = np.eye(5, dtype=bool)
    assert len(np.unique(periodic_grid_components(a, 4)[a])) == 5
    assert len(np.unique(periodic_grid_components(a, 8)[a])) == 1


def test_sampled_runs_censor_seams_and_gaps():
    r = sampled_support_runs(np.ones(5, bool), [0., 1., 2., 4., 5.], ["a", "a", "b", "b", "b"], 1.)
    assert [x["frames"] for x in r] == [2, 1, 2]
    assert [x["sampled_span_ps"] for x in r] == [1., 0., 1.]
    assert all(x["left_censored"] and x["right_censored"] for x in r)


def test_ambiguous_height_is_not_a_known_departure():
    r = observed_cohort_history([0., 1., 2.], ["a"] * 3, np.array([[0], [4], [0]]), 1.)
    assert r["observed_return_anytime"] == 0
    assert r["any_ambiguous"] == 1
