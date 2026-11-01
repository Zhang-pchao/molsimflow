import numpy as np
import pytest
from molsimflow.postprocess.interfacial_contact_coordination import lane_boundary_motion
def chart():
    return np.broadcast_to(np.arange(20)[:,None],(20,6)).astype(float)
def test_translation_and_dilation_have_distinct_coordination():
    b=np.zeros((20,6),bool);b[5:11]=True
    a=np.roll(b,2,axis=0)
    z=lane_boundary_motion(b,a,chart())
    assert z['coordinated_forward_fraction']==1
    assert z['front_mean']==pytest.approx(2)
    assert z['rear_mean']==pytest.approx(2)
    a=b.copy();a[3:13]=True
    z=lane_boundary_motion(b,a,chart())
    assert z['coordinated_forward_fraction']==0
    assert z['expansion_fraction']==1
    assert z['translation_mean']==pytest.approx(0)
def test_missing_lanes_not_zero_motion():
    b=np.zeros((20,6),bool);a=b.copy();a[5:11]=True
    z=lane_boundary_motion(b,a,chart())
    assert z['supported_lanes']==0 and z['appeared_lanes']==6
    assert np.isnan(z['front_mean'])
def test_chart_translation_invariance():
    b=np.zeros((20,6),bool);b[5:11]=True;a=np.roll(b,2,axis=0)
    assert lane_boundary_motion(b,a,chart())==lane_boundary_motion(b,a,chart()+200)
def test_invalid_chart_rejected():
    with pytest.raises(ValueError):lane_boundary_motion(np.ones((2,2),bool),np.ones((2,2),bool),np.full((2,2),np.nan))
