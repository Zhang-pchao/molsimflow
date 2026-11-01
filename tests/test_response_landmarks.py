import numpy as np
from molsimflow.postprocess.response_landmarks import first_crossing,signed_progress,endpoint_description

def test_linear_interpolation():
    r=first_crossing([0,10,20],[0,.4,1],.5)
    assert r["status"]=="OBSERVED"
    np.testing.assert_allclose(r["time"],10+10/6)
def test_decreasing_response_and_overshoot():
    np.testing.assert_allclose(signed_progress([10,5,0,-2],10,0),[0,.5,1,1.2])
def test_left_censored():
    assert first_crossing([0,1],[.8,.9])["status"]=="LEFT_CENSORED"
def test_missing_values_not_bridged():
    r=first_crossing([0,1,2],[0,np.nan,1])
    assert r["time"] is None and r["status"]=="GAP_CENSORED"
def test_not_reached_and_constant():
    assert first_crossing([0,1],[0,.2])["status"]=="NOT_REACHED"
    assert np.isnan(signed_progress([1,1],1,1)).all()
def test_first_crossing_keeps_recrossing():
    assert first_crossing([0,1,2,3],[0,.8,.1,1])["time"]==.625
def test_endpoints_are_not_standard_errors():
    r=endpoint_description([0,2,9,11],2)
    assert r["delta"]==9
    np.testing.assert_allclose(r["endpoint_sd"],np.sqrt(2))
