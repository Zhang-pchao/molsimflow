import numpy as np
import pytest
from molsimflow.postprocess.constrained_velocity_reflection import reflect_linear_velocity_mode as reflect

def sample():
    rng=np.random.default_rng(222)
    x=rng.normal(size=(12,3));v=rng.normal(size=(12,3));m=rng.uniform(1,30,12)
    axis=np.array([.3,.8,-.2]);axis/=np.linalg.norm(axis)
    bond=x[0]-x[1];bond/=np.linalg.norm(bond);tangent=axis-(axis@bond)*bond
    target=np.zeros_like(v);target[0]=-tangent;target[1]=tangent
    extra=np.zeros((1,12,3));extra[0,0]=bond;extra[0,1]=-bond
    return x,v,m,target,extra

def test_energy_momentum_angular_stretch_and_mode():
    x,v,m,t,c=sample();r=reflect(v,m,x,t,c);d=r.velocities-v
    np.testing.assert_allclose(np.sum(m[:,None]*d,axis=0),0,atol=1e-12)
    np.testing.assert_allclose(np.sum(np.cross(x,m[:,None]*d),axis=0),0,atol=1e-12)
    np.testing.assert_allclose(np.sum(m[:,None]*(r.velocities**2-v**2)),0,atol=1e-12)
    np.testing.assert_allclose(np.sum(c*d[None],axis=(1,2)),0,atol=1e-12)
    assert abs(r.mode_after+r.mode_before)<1e-12 and r.constraint_rank==7

def test_double_reflection():
    x,v,m,t,c=sample();a=reflect(v,m,x,t,c);b=reflect(a.velocities,m,x,t,c)
    np.testing.assert_allclose(b.velocities,v,atol=1e-12)

def test_translation_and_galilean_covariance():
    x,v,m,t,c=sample();a=reflect(v,m,x,t,c)
    boost=np.array([2.,-3.,1.]);b=reflect(v+boost,m,x+[20,-30,15],t,c)
    np.testing.assert_allclose(b.velocities-boost,a.velocities,atol=1e-12)

def test_rotational_covariance():
    x,v,m,t,c=sample();a=reflect(v,m,x,t,c)
    Q=np.array([[0,1,0],[0,0,1],[1,0,0]])
    b=reflect(v@Q,m,x@Q,t@Q,c@Q)
    np.testing.assert_allclose(b.velocities,a.velocities@Q,atol=1e-12)

def test_group_increment_preserves_internal_kinetic_and_angular_motion():
    x,v,m,t,c=sample();r=reflect(v,m,x,t,c)
    ma,mb=.8*m[0],.2*m[0]
    xa=x[0]+[.1,.2,-.1];xb=x[0]-ma/mb*(xa-x[0])
    va=v[0]+[.3,-.2,.1];vb=v[0]-ma/mb*(va-v[0])
    dv=r.velocities[0]-v[0]
    np.testing.assert_allclose((va+dv)-(vb+dv),va-vb)
    old=ma*np.dot(va,va)+mb*np.dot(vb,vb)-m[0]*np.dot(v[0],v[0])
    new=ma*np.dot(va+dv,va+dv)+mb*np.dot(vb+dv,vb+dv)-m[0]*np.dot(r.velocities[0],r.velocities[0])
    assert abs(old-new)<1e-12
    oldL=ma*np.cross(xa-x[0],va-v[0])+mb*np.cross(xb-x[0],vb-v[0])
    newL=ma*np.cross(xa-x[0],va+dv-r.velocities[0])+mb*np.cross(xb-x[0],vb+dv-r.velocities[0])
    np.testing.assert_allclose(oldL,newL,atol=1e-12)

def test_constraint_scaling_redundancy_and_order():
    x,v,m,t,c=sample();a=reflect(v,m,x,t,c)
    b=reflect(v,m,x,t,np.concatenate([c*1e10,c*1e-10,c]))
    np.testing.assert_allclose(a.velocities,b.velocities,atol=1e-12)

def test_reject_zero_constrained_or_bad_inputs():
    x,v,m,t,c=sample()
    with pytest.raises(ValueError,match='constrained'):reflect(v,m,x,np.zeros_like(v),c)
    with pytest.raises(ValueError,match='constrained'):reflect(v,m,x,np.column_stack([m,np.zeros((12,2))]),c)
    with pytest.raises(ValueError,match='positive'):reflect(v,-m,x,t,c)
    with pytest.raises(ValueError,match='shape'):reflect(v,m,x,t,np.ones((12,3)))
