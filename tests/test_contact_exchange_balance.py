import numpy as np
from molsimflow.postprocess.contact_exchange_balance import (
    contact_exchange,population_states,fit_contact_probabilities,predict_contact_increment)

def test_contact_growth_and_component_transfer_are_separate():
    states=np.array([[2,3,3,0,1],[3,2,0,3,1]])
    got=contact_exchange(states)
    np.testing.assert_array_equal(got["gain"],[1])
    np.testing.assert_array_equal(got["loss"],[1])
    np.testing.assert_array_equal(got["member_in"],[1])
    np.testing.assert_array_equal(got["member_out"],[1])
    np.testing.assert_array_equal(got["delta"],[0])
    assert got["flow"].sum()==5

def test_accepted_contact_mask_is_not_rethresholded():
    main=np.array([[1,1,1,0],[1,1,1,0]],bool)
    contact=np.array([[1,0,0,0],[0,1,0,0]],bool)
    # The existing contact classification is authoritative, even after storage rounding.
    distance=np.array([[4.0000001,6.,12.,2.],[4.5,4.,13.,2.]])
    states=population_states(main,contact,distance)
    np.testing.assert_array_equal(states,[[3,2,1,0],[2,3,1,0]])

def test_unknown_exposure_is_not_zero_probability():
    f=np.zeros((2,4,4),int);f[:,0,0]=10
    fit=fit_contact_probabilities(f,near_only=True)
    assert fit["p_on"] is None and fit["p_off"] is None
    assert np.isnan(predict_contact_increment(np.array([[10,0,0,0]]),fit)).all()

def test_conditional_probabilities_and_zero_net_drift():
    f=np.zeros((1,4,4),int)
    f[0,2,2]=80;f[0,2,3]=20
    f[0,3,2]=20;f[0,3,3]=30
    fit=fit_contact_probabilities(f,near_only=True)
    assert fit["p_on"]==.2 and fit["p_off"]==.4
    np.testing.assert_allclose(predict_contact_increment(np.array([[0,0,100,50]]),fit),0.)
