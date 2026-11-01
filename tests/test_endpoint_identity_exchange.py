import numpy as np
import pytest
from molsimflow.postprocess.interfacial_contact_renewal import endpoint_identity_exchange

def test_equal_occupation_can_hide_complete_replacement():
    q=endpoint_identity_exchange(np.array([[True],[False]]),np.array([[False],[True]]))
    assert q['net'][0]==0 and q['gross'][0]==2
    assert q['retained'][0]==0 and q['cancellation_fraction'][0]==1
    assert q['replacement_fraction'][0]==1

def test_stationary_membership():
    b=np.array([[True,False],[False,True],[True,True]])
    q=endpoint_identity_exchange(b,b)
    assert np.array_equal(q['retained'],[2,2])
    assert np.array_equal(q['retained_fraction'],[1,1])
    assert np.isnan(q['cancellation_fraction']).all()

def test_empty_region_is_missing_fraction_not_zero_retention():
    q=endpoint_identity_exchange(np.zeros((0,2),bool),np.zeros((0,2),bool))
    assert np.array_equal(q['gross'],[0,0])
    assert np.isnan(q['retained_fraction']).all()
    assert np.isnan(q['replacement_fraction']).all()

def test_random_closure_time_reversal_and_common_identity_permutation():
    rng=np.random.default_rng(17)
    b,a=rng.random((2,75,4))>.6
    q=endpoint_identity_exchange(b,a);r=endpoint_identity_exchange(a,b)
    p=rng.permutation(75);s=endpoint_identity_exchange(b[p],a[p])
    assert np.array_equal(q['before'],q['retained']+q['lost'])
    assert np.array_equal(q['after'],q['retained']+q['gained'])
    assert np.array_equal(q['after']-q['before'],q['net'])
    assert np.array_equal(q['lost'],r['gained'])
    for k in q: assert np.allclose(q[k],s[k],equal_nan=True)

@pytest.mark.parametrize('b,a',[(np.array([True]),np.array([True])),
    (np.ones((2,1)),np.ones((2,1))),
    (np.ones((2,1),bool),np.ones((3,1),bool)),
    (np.ones((2,0),bool),np.ones((2,0),bool))])
def test_bad_memberships_rejected(b,a):
    with pytest.raises(ValueError): endpoint_identity_exchange(b,a)
