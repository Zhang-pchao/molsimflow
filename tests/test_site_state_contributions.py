import numpy as np
import pytest
from molsimflow.postprocess.site_state_contributions import aggregate_site_observations,sampled_site_history

def test_identity_multiplicity_and_whole_population_weight():
    a=aggregate_site_observations([[7,7,9],[9,7,9]],[[1,1,1],[1,0,1]],[[1,1,0],[1,0,1]],[4,5])
    np.testing.assert_array_equal(a.site_ids,[7,9]);np.testing.assert_array_equal(a.state,[[1,0],[-1,1]])
    np.testing.assert_array_equal(a.molecule_count,[[2,1],[0,2]])
    np.testing.assert_allclose(a.frame_weight,[[.5,.25],[0,.4]])
    assert np.isclose(a.frame_weight.sum(axis=1).mean(),(.75+.4)/2)

def test_conflicting_same_site_state_rejected():
    with pytest.raises(ValueError):aggregate_site_observations([[7,7]],[[1,1]],[[0,1]],[2])

def test_censored_gaps_never_create_changes():
    a=np.array([[0,0],[1,-1],[0,1],[2,2],[1,0],[0,0]])
    h=sampled_site_history(a,np.arange(6)*.5,.5)
    np.testing.assert_array_equal(h['site_class'],[2,2]);np.testing.assert_array_equal(h['forward'],[1,0]);np.testing.assert_array_equal(h['backward'],[2,0])
    np.testing.assert_array_equal(h['reversal'],[True,False]);np.testing.assert_array_equal(h['max_changes_in_run'],[2,0])

def test_time_and_source_breaks():
    h=sampled_site_history(np.array([[0],[1],[0],[1]]),[0,.5,2,2.5],.5,['a','b','b','b'])
    assert h['forward'][0]==1 and h['backward'][0]==0 and not h['reversal'][0]

def test_single_state_and_unresolved_classes():
    h=sampled_site_history(np.array([[0,1,2],[0,-1,2],[2,1,-1]]),[0,.5,1],.5)
    np.testing.assert_array_equal(h['site_class'],[0,1,3]);assert not h['reversal'].any()

def test_empty_population_and_bad_normalization():
    a=aggregate_site_observations([[7]],[[0]],[[-1]],[3]);assert a.state.shape==(1,0)
    with pytest.raises(ValueError):aggregate_site_observations([[7,7]],[[1,1]],[[0,0]],[1])
