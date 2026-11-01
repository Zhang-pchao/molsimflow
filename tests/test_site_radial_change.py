import numpy as np
import pytest
from molsimflow.postprocess.site_radial_change import radial_core_labels,segmented_radial_excursions

def test_clear_requires_all_solution_oxygen_and_boundary_is_strict():
    a=radial_core_labels([3.2,4,4,4,3.8],[9,2.9,3.8,3.81,9],[1,1,1,1,1])
    np.testing.assert_array_equal(a,[1,0,0,2,0])

def test_qualification_change_cannot_fake_clearance():
    a=radial_core_labels([3,4],[5,3],[1,1])
    np.testing.assert_array_equal(a,[1,0])

def test_hits_returns_reverse_and_last_origin():
    a=np.array([1,1,0,1,0,2,0,1])[:,None]
    r=segmented_radial_excursions(a,np.arange(8)*.5,['a']*8)
    assert [(x['start'],x['end'],x['outcome']) for x in r]==[(1,3,'return'),(3,5,'hit'),(5,7,'hit')]

def test_invalid_and_source_gaps_censor_without_bridging():
    a=np.array([1,0,-1,2,0,1,0,2])[:,None]
    r=segmented_radial_excursions(a,np.arange(8)*.5,['a']*7+['b'])
    assert [(x['start'],x['end'],x['outcome']) for x in r]==[(0,1,'censored'),(3,5,'hit'),(5,6,'censored')]

def test_initial_buffer_is_not_an_origin_and_time_gap_breaks():
    a=np.array([0,0,2,0,1])[:,None]
    r=segmented_radial_excursions(a,[0,.5,1,1.5,3],['a']*5)
    assert len(r)==1 and r[0]['outcome']=='censored' and r[0]['end']==3

def test_no_observations_and_bad_radius():
    assert segmented_radial_excursions(np.array([[-1],[-1]]),[0,.5],['a','a'])==[]
    with pytest.raises(ValueError):radial_core_labels([3],[4],[True],near=4,far=3)
