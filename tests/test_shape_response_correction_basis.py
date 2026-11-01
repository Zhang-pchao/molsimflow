"""Checks for one-dimensional prescribed forcing corrections."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from molsimflow.postprocess.shape_relaxation import profile_shape_response, driven_shape_response

def test_constant_basis_preserves_existing_fit():
    t=np.arange(0.,200.,10);g=24-0.03*t;h=driven_shape_response(t,g+3,30,29)
    train=t<120
    a,_=profile_shape_response(t,g,h,train,[0,10,30,90],additive_offset=True)
    b,_=profile_shape_response(t,g,h,train,[0,10,30,90],additive_offset=True,offset_basis=np.ones_like(t))
    assert a['tau']==b['tau']
    assert a['offset']==b['offset']
    np.testing.assert_array_equal(a['prediction'],b['prediction'])

def test_scaled_basis_recovery_and_independent_ode():
    t=np.arange(0.,200.,10);g=24-0.03*t;length=25+2*np.sin(t/60)
    h=driven_shape_response(t,g+0.12*length,30,29)
    best,_=profile_shape_response(t,g,h,t<120,[0,10,30,90],additive_offset=True,offset_basis=length)
    assert best['tau']==30
    assert best['offset']==pytest.approx(.12)
    solution=solve_ivp(lambda x,y:(np.interp(x,t,g+.12*length)-y)/30,
                      [t[0],t[-1]],[29],t_eval=t,rtol=1e-10,atol=1e-11,max_step=.1)
    np.testing.assert_allclose(best['prediction'],solution.y[0],atol=1e-7)

def test_scaled_fit_ignores_held_out_observations():
    t=np.arange(0.,200.,10);g=24-.03*t;length=25+2*np.sin(t/60)
    h=driven_shape_response(t,g+.12*length,30,29);train=t<120
    a,_=profile_shape_response(t,g,h,train,[0,10,30,90],additive_offset=True,offset_basis=length)
    h[~train]+=1000
    b,_=profile_shape_response(t,g,h,train,[0,10,30,90],additive_offset=True,offset_basis=length)
    assert a['tau']==b['tau'] and a['offset']==b['offset']
    np.testing.assert_array_equal(a['prediction'],b['prediction'])

def test_invalid_correction_basis_rejected():
    t=np.arange(6.);g=t+5;h=g+2
    for basis in [np.ones(3),np.full(6,np.nan),np.zeros(6)]:
        with pytest.raises(ValueError):
            profile_shape_response(t,g,h,np.ones(6,bool),[0,1],additive_offset=True,offset_basis=basis)
    with pytest.raises(ValueError):
        profile_shape_response(t,g,h,np.ones(6,bool),[0,1],offset_basis=np.ones(6))
