import numpy as np
import pytest
from molsimflow.postprocess.shape_relaxation import moment_axes,constant_volume_radius,relaxation_curve,fit_relaxation
from molsimflow.postprocess.axisymmetric_contact_angle import fit_axisymmetric_circle

def test_circle_gas_side_convention():
    z=np.linspace(2,28,50);R=20.;zc=10.
    p=np.c_[np.sqrt(R*R-(z-zc)**2),z]
    f=fit_axisymmetric_circle(p)
    assert f['dense_phase_contact_angle_deg']==pytest.approx(120)
    assert f['fit_rmse_A']<1e-12

def test_relaxation_recovery():
    t=np.linspace(0,1000,51);y=relaxation_curve(t,12.,30.,270.)
    f=fit_relaxation(t,y)
    assert f['tau_ps']==pytest.approx(270,rel=1e-6)
    assert f['rmse']<1e-7

def test_moment_geometry_covariance_and_volume():
    p=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
    a,c,v=moment_axes(p)
    b,d,w=moment_axes(p*np.array([2,2,.25])+[3,5,7])
    assert b/a==pytest.approx(2)
    assert d/c==pytest.approx(.25)
    assert w==pytest.approx(v)
    assert constant_volume_radius(d,a,c)==pytest.approx(b)

def test_invalid_axis():
    with pytest.raises(ValueError):constant_volume_radius([0],1,1)
