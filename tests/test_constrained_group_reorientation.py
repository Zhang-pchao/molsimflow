import numpy as np
import pytest
from molsimflow.postprocess.constrained_group_reorientation import reorient_group_with_energy_compensation as reorient

def sample():
    rng=np.random.default_rng(281)
    x=rng.normal(size=(15,3));v=rng.normal(size=(15,3));m=rng.uniform(3,20,15)
    axis=np.array([.3,-.2,.8]);axis/=np.linalg.norm(axis)
    ref=np.array([.1,-.2,.05]);wanted=float(v[0]@axis)+.2
    return x,v,m,axis,ref,wanted

def test_local_and_global_physical_invariants():
    x,v,m,a,ref,t=sample();r=reorient(v,m,x,0,a,t,ref);w=r.velocities;d=w-v
    assert abs(w[0]@a-t)<1e-12
    assert abs(np.sum((w[0]-ref)**2)-np.sum((v[0]-ref)**2))<1e-12
    np.testing.assert_allclose(np.sum(m[:,None]*d,axis=0),0,atol=3e-12)
    np.testing.assert_allclose(np.sum(np.cross(x,m[:,None]*d),axis=0),0,atol=3e-12)
    for frame in [np.zeros(3),ref,np.array([1.,2.,-3.])]:
        assert abs(np.sum(m[:,None]*((w-frame)**2-(v-frame)**2)))<4e-12

def test_identity_and_target_round_trip():
    x,v,m,a,ref,t=sample();old=float(v[0]@a)
    w=reorient(v,m,x,0,a,old,ref).velocities
    np.testing.assert_allclose(w,v,atol=1e-12)
    w=reorient(v,m,x,0,a,t,ref).velocities
    z=reorient(w,m,x,0,a,old,ref).velocities
    np.testing.assert_allclose(z,v,atol=2e-12)

def test_galilean_translation_rotation_covariance():
    x,v,m,a,ref,t=sample();w=reorient(v,m,x,0,a,t,ref).velocities
    boost=np.array([2.,3.,-1.])
    z=reorient(v+boost,m,x+[30.,-20.,10.],0,a,t+boost@a,ref+boost).velocities
    np.testing.assert_allclose(z-boost,w,atol=3e-12)
    Q=np.array([[0,1,0],[0,0,1],[1,0,0]])
    z=reorient(v@Q,m,x@Q,0,a@Q,t,ref@Q).velocities
    np.testing.assert_allclose(z,w@Q,atol=3e-12)

def test_rigid_group_internal_velocities_and_energy():
    x,v,m,a,ref,t=sample();w=reorient(v,m,x,0,a,t,ref).velocities
    ma=.7*m[0];mb=.3*m[0];q=np.array([.4,.2,-.3])
    va=v[0]+q;vb=v[0]-ma/mb*q;d=w[0]-v[0]
    np.testing.assert_allclose(va+d-vb-d,va-vb,atol=1e-12)
    old=ma*np.sum((va-v[0])**2)+mb*np.sum((vb-v[0])**2)
    new=ma*np.sum((va+d-w[0])**2)+mb*np.sum((vb+d-w[0])**2)
    assert abs(old-new)<1e-12

def test_infeasible_projection_and_invalid_input():
    x,v,m,a,ref,t=sample()
    with pytest.raises(ValueError,match='envelope'):
        reorient(v,m,x,0,a,ref@a+np.linalg.norm(v[0]-ref)+1,ref)
    with pytest.raises(ValueError,match='unit'):
        reorient(v,m,x,0,2*a,t,ref)
    with pytest.raises(ValueError,match='positive'):
        reorient(v,-m,x,0,a,t,ref)

def test_compensation_nearest_point_against_independent_optimizer():
    from scipy.optimize import minimize
    x,v,m,a,ref,t=sample();out=reorient(v,m,x,0,a,t,ref).velocities
    p0=np.sum(m[:,None]*v,axis=0);l0=np.sum(np.cross(x,m[:,None]*v),axis=0);k0=np.sum(m[:,None]*v*v)
    def full(z):return np.vstack([out[0],z.reshape(-1,3)])
    def constraints(z):
        w=full(z)
        return np.r_[np.sum(m[:,None]*w,axis=0)-p0,np.sum(np.cross(x,m[:,None]*w),axis=0)-l0,np.sum(m[:,None]*w*w)-k0]
    def cost(z):return .5*np.sum(m[1:,None]*(z.reshape(-1,3)-v[1:])**2)
    fit=minimize(cost,v[1:].ravel(),method='SLSQP',constraints={'type':'eq','fun':constraints},
                 options={'ftol':1e-11,'maxiter':200})
    assert fit.success,fit.message
    assert np.max(abs(constraints(fit.x)))<1e-8
    assert abs(cost(fit.x)-cost(out[1:].ravel()))<1e-8

def test_relative_nearest_atom_speed_reversal_at_fixed_local_energy():
    x,v,m,a,ref,t=sample();site_projection=.2;internal_projection=.1
    before=float(v[0]@a)+internal_projection-site_projection
    requested=2*(site_projection-internal_projection)-float(v[0]@a)
    out=reorient(v,m,x,0,a,requested,ref).velocities
    after=float(out[0]@a)+internal_projection-site_projection
    assert abs(before+after)<1e-12
    assert abs(np.linalg.norm(out[0]-ref)-np.linalg.norm(v[0]-ref))<1e-12
