"""Independent analytic and leakage checks for contact-driven geometry."""
import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from molsimflow.postprocess.shape_relaxation import (
 cap_from_area_volume, lower_profile_footprint, density_component_profile,
 driven_shape_response, profile_shape_response)

def test_cap_quantiles_against_numerical_volume():
    a,h=12.,19.
    volume=np.pi*h*(3*a*a+h*h)/6
    out=cap_from_area_volume(np.pi*a*a,volume)
    assert out['cap_height']==pytest.approx(h)
    center=(h*h-a*a)/(2*h)
    for probability,key in [(.05,'q05'),(.5,'q50'),(.95,'q95')]:
        integral=quad(lambda z:np.pi*(a*a+2*center*z-z*z),0,out[key])[0]
        assert integral/volume==pytest.approx(probability)

def test_lower_footprint_ignores_top():
    z=np.arange(.5,30,.5); area=np.pi*np.maximum(144+5*z-z*z,0)
    before=lower_profile_footprint(z,area)
    area[z>9]=123456
    after=lower_profile_footprint(z,area)
    assert before==after
    assert before['area']==pytest.approx(np.pi*144)
    assert not lower_profile_footprint(z,np.zeros_like(z))['available']

def test_density_volume_independent_of_cap_height():
    d=np.zeros((20,20,20));d[3:13,4:14,5:15]=2
    p=density_component_profile(d,np.arange(20),[1,1,1],1)
    assert p['volume']==1000
    assert np.sum(p['areas'])==1000
    assert p['density_mass_fraction']==1
    assert not p['boundary_touch']

def test_driven_response_matches_ode():
    t=np.array([0.,3.,8.,15.,30.]);g=np.array([25.,24.,20.,19.,18.])
    tau=9.;pred=driven_shape_response(t,g,tau,30.)
    numerical=solve_ivp(lambda x,y:(np.interp(x,t,g)-y)/tau,
                        [0,30],[30.],t_eval=t,rtol=1e-10,atol=1e-11,max_step=.05).y[0]
    np.testing.assert_allclose(pred,numerical,atol=1e-7)
    np.testing.assert_array_equal(driven_shape_response(t,g,0,30),g)

def test_causal_integration_and_training_ignore_future_outcome():
    t=np.arange(0,220.,10);g=25-8*(1-np.exp(-t/35))
    h=driven_shape_response(t,g,40,27.)
    train=t<120
    best,_=profile_shape_response(t,g,h,train,[0,10,40,100])
    changed=h.copy();changed[~train]=1e8
    other,_=profile_shape_response(t,g,changed,train,[0,10,40,100])
    assert best['tau']==other['tau']==40
    assert best['train_mse']==other['train_mse']
    future_g=g.copy();future_g[t>120]=1e8
    np.testing.assert_array_equal(driven_shape_response(t,g,40,27)[t<=120],
                                  driven_shape_response(t,future_g,40,27)[t<=120])

def test_common_offset_not_hidden_in_tau():
    t=np.arange(0,220.,10);g=22-6*(1-np.exp(-t/30))
    h=driven_shape_response(t,g+2,40,27.)
    best,_=profile_shape_response(t,g,h,t<120,[0,10,40,100],additive_offset=True)
    assert best['tau']==40
    assert best['offset']==pytest.approx(2.)

def test_invalid_inputs():
    with pytest.raises(ValueError):cap_from_area_volume(0,20)
    with pytest.raises(ValueError):driven_shape_response([1,0],[2,3],4,5)
