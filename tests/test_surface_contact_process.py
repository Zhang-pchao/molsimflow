import numpy as np
import pytest
from molsimflow.postprocess.surface_contact_process import (
    capsule_membership, supported_cell_counts, sampled_episodes, establishment_outcome,
)


def test_periodic_capsule_and_translation():
    points = np.array([[.1, 5], [9.9, 5], [5, 5], [.1, 6.]])
    bonds = np.array([[[9.7, 5], [.3, 5]]])
    a = capsule_membership(points, bonds, np.array([10, 10]), .4)
    b = capsule_membership(points + [4, 1], bonds + [4, 1], np.array([10, 10]), .4)
    assert set(zip(*a)) == {(0, 0), (1, 0)} == set(zip(*b))


def test_one_molecule_is_not_multiple_supporters():
    r = supported_cell_counts(np.array([0, 1, 2, 3, 0]), np.zeros(5, int), np.array([0, 0, 1, 1]), 2, 1)
    np.testing.assert_array_equal(r['distinct_molecules'], [1, 1])
    np.testing.assert_array_equal(r['union_samples'], [2, 2])
    assert r['fractional_molecules'].sum() == 1
    assert r['distinct_molecules'].max() < 3


def test_union_and_counts_differ():
    r = supported_cell_counts(np.array([0, 0, 0]), np.array([0, 1, 2]), np.array([0]), 1, 3)
    assert r['union_samples'][0] == 1
    assert r['distinct_molecules'][0] == 3
    assert r['fractional_molecules'][0] == 3


def test_empty_membership():
    r = supported_cell_counts(np.array([],int), np.array([],int), np.array([0,1]), 2, 300)
    assert r['union_samples'].sum() == 0
    assert r['fractional_molecules'].sum() == 0


def test_turnover_does_not_terminate_spatial_contact():
    original_ids = [{1,2,3}, {4,5,6}, {7,8,9}]
    spatial = [int(len(x) >= 3) for x in original_ids]
    e = sampled_episodes(spatial, [0,10,20], ['a']*3, 10)
    assert len(e) == 1 and e[0]['sampled_span'] == 20
    assert e[0]['left_censored'] and e[0]['right_censored']
    assert not (original_ids[0] & original_ids[-1])


def test_seam_gap_and_unknown_do_not_bridge_contact():
    e = sampled_episodes([0,1,1,1,-1,1,1], [0,10,20,30,40,50,70], ['a','a','a','b','b','b','b'], 10)
    assert [r['samples'] for r in e] == [2,1,1,1]
    assert e[0]['left_censored'] is False
    assert e[0]['end_reason'] == 'source_or_gap'
    assert e[1]['end_reason'] == 'unknown'


def test_sampled_confirmation_not_backdated():
    r=establishment_outcome([0,1,1,1,0], [0,10,20,30,40], ['a']*5, 0, 40, 20, 10)
    assert r['outcome']==1 and r['first_index']==1 and r['confirmed_index']==3
    assert establishment_outcome([0,1,1], [0,10,20], ['a']*3, 0, 20, 20, 10)['outcome']==0


def test_initial_contact_and_censoring():
    assert establishment_outcome([1,1,1], [0,10,20], ['a']*3, 0, 20, 20, 10)['status']=='NOT_AT_RISK'
    assert establishment_outcome([0,0,-1], [0,10,20], ['a']*3, 0, 20, 20, 10)['outcome'] is None
    assert establishment_outcome([0,1,1], [0,10,20], ['a','a','b'], 0, 20, 20, 10)['outcome'] is None


def test_invalid_states_rejected():
    with pytest.raises(ValueError): sampled_episodes([2], [0], ['a'], 10)
    with pytest.raises(ValueError): capsule_membership(np.array([[np.nan,0]]),np.zeros((1,2,2)),np.ones(2)*10,1)
