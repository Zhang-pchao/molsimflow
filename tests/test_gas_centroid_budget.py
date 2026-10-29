import numpy as np
import pytest
from molsimflow.postprocess.interfacial_gas_transport import moving_membership_centroid_budget as budget

def test_translation_without_exchange():
    x=np.array([[0.,1.],[2.,3.],[5.,4.]])
    b=np.array([True,True,False]);q=budget(x,x+[3,-2],b,b)
    np.testing.assert_allclose(q["centroid_change"],[3,-2])
    np.testing.assert_allclose(q["retained_displacement"],[3,-2])
    np.testing.assert_allclose(q["joined_contribution"],0)
    np.testing.assert_allclose(q["departed_contribution"],0)

def test_membership_moves_center_without_particle_motion():
    x=np.arange(3.)[:,None];b=np.array([True,True,False]);a=np.array([False,True,True])
    q=budget(x,x,b,a)
    np.testing.assert_allclose(q["centroid_change"],1)
    np.testing.assert_allclose(q["retained_displacement"],0)
    np.testing.assert_allclose(q["joined_contribution"],.75)
    np.testing.assert_allclose(q["departed_contribution"],.25)

def test_variable_population_and_translation_invariance():
    x=np.array([[-2.,0.],[0.,1.],[5.,2.]])
    y=x+[[1.,2.],[3.,1.],[-1.,-2.]]
    b=np.array([True,True,False]);a=np.ones(3,dtype=bool)
    q=budget(x,y,b,a);r=budget(x+1200,y+1200,b,a)
    assert q["count_residual"]==0 and q["joined"]==1
    for k in ["centroid_change","retained_displacement","joined_contribution","departed_contribution"]:
        np.testing.assert_allclose(q[k],r[k],atol=1e-12)
    np.testing.assert_allclose(q["centroid_residual"],0,atol=1e-12)

def test_exchange_can_cancel_motion():
    x=np.array([[0.],[2.],[10.]])
    y=np.array([[0.],[3.],[2.]])
    b=np.array([False,True,True]);a=np.array([True,True,False])
    q=budget(x,y,b,a)
    assert q["retained_displacement"][0]>0 and q["centroid_change"][0]<0
    np.testing.assert_allclose(q["centroid_residual"],0,atol=1e-12)

def test_permutation_and_random_first_moments():
    rng=np.random.default_rng(473)
    for _ in range(40):
        x=rng.normal(size=(20,3));y=x+rng.normal(size=x.shape)
        b=rng.random(20)>.3;a=rng.random(20)>.3
        q=budget(x,y,b,a);order=rng.permutation(20);r=budget(x[order],y[order],b[order],a[order])
        for k in ["centroid_change","retained_displacement","joined_contribution","departed_contribution"]:
            np.testing.assert_allclose(q[k],r[k],atol=1e-12)
        np.testing.assert_allclose(q["centroid_residual"],0,atol=1e-12)

@pytest.mark.parametrize("before,after",[(False,True),(True,False),(False,False)])
def test_empty_support_is_explicit(before,after):
    q=budget(np.zeros((2,1)),np.ones((2,1)),np.full(2,before),np.full(2,after))
    assert not q["centroid_valid"] and np.isnan(q["centroid_change"]).all()

@pytest.mark.parametrize("bad",["shape","nan","mask_shape","integer_mask"])
def test_invalid_inputs(bad):
    x=np.zeros((3,2));y=x.copy();b=np.ones(3,dtype=bool);a=b.copy()
    if bad=="shape":y=np.zeros((2,2))
    elif bad=="nan":x[0,0]=np.nan
    elif bad=="mask_shape":a=np.ones(2,dtype=bool)
    else:b=b.astype(int)
    with pytest.raises(ValueError):budget(x,y,b,a)

def test_step_and_endpoint_budgets_close_but_terms_need_not_agree():
    x=np.array([[0.],[2.],[4.]])
    y=x+1;z=y+1
    b=np.array([True,True,False]);a=np.array([False,True,True])
    q0=budget(x,y,b,a);q1=budget(y,z,a,b);q=budget(x,z,b,b)
    np.testing.assert_allclose(q0["centroid_change"]+q1["centroid_change"],q["centroid_change"])
    assert not np.allclose(q0["retained_displacement"]+q1["retained_displacement"],q["retained_displacement"])
