import numpy as np
import pytest
from molsimflow.postprocess.momentum_reflection import reflect_group_relative_velocity as reflect

def test_unequal_mass_momentum_energy_and_relative_modes():
 rng=np.random.default_rng(1701);v=rng.normal(size=(50,3));m=rng.uniform(.5,30,50)
 mask=np.arange(50)<3;axis=np.array([.3,-.5,.2]);u=axis/np.linalg.norm(axis)
 w=reflect(v,m,mask,axis)
 np.testing.assert_allclose((m[:,None]*w).sum(0),(m[:,None]*v).sum(0),atol=2e-13)
 assert abs(np.sum(m[:,None]*w*w)-np.sum(m[:,None]*v*v))<1e-12
 def rel(x):return np.average(x[mask],axis=0,weights=m[mask])-np.average(x[~mask],axis=0,weights=m[~mask])
 a,b=rel(v),rel(w)
 np.testing.assert_allclose(b,a-2*(a@u)*u,atol=1e-14)
 for sel in [mask,~mask]:
  np.testing.assert_allclose(w[sel]-w[sel][0],v[sel]-v[sel][0],atol=1e-14)

def test_involution_and_galilean_rotation_covariance():
 rng=np.random.default_rng(1702);v=rng.normal(size=(12,3));m=rng.uniform(1,10,12);mask=np.arange(12)<2;axis=rng.normal(size=3)
 w=reflect(v,m,mask,axis)
 np.testing.assert_allclose(reflect(w,m,mask,axis),v,atol=2e-14)
 boost=np.array([4,-2,7.]);np.testing.assert_allclose(reflect(v+boost,m,mask,axis),w+boost,atol=1e-14)
 Q,_=np.linalg.qr(rng.normal(size=(3,3)))
 np.testing.assert_allclose(reflect(v@Q,m,mask,axis@Q),w@Q,atol=1e-14)

def test_two_particles_analytic_exchange():
 v=np.array([[3.,2,1],[-1.,2,1]])
 w=reflect(v,np.ones(2),np.array([True,False]),np.array([1.,0,0]))
 np.testing.assert_array_equal(w,v[::-1])

def test_reject_invalid_groups_axis_and_mass():
 v=np.ones((4,3));m=np.ones(4);mask=np.array([True,False,False,False])
 for bad in [np.zeros(4,bool),np.ones(4,bool),np.ones(4,int)]:
  with pytest.raises(ValueError):reflect(v,m,bad,np.array([1.,0,0]))
 with pytest.raises(ValueError):reflect(v,m,mask,np.zeros(3))
 with pytest.raises(ValueError):reflect(v,-m,mask,np.ones(3))

from molsimflow.postprocess.momentum_reflection import reflect_group_mode_with_angular_constraint as constrained

def test_constrained_reflection_preserves_P_L_KE_and_involution():
 rng=np.random.default_rng(1703);r=rng.normal(size=(30,3));v=rng.normal(size=(30,3));m=rng.uniform(1,30,30);mask=np.arange(30)<2;axis=np.array([.2,.3,.9])
 w,d=constrained(v,m,mask,axis,r)
 assert d['constraint_rank']==6 and abs(d['mode_coordinate_before']+d['mode_coordinate_after'])<1e-13
 np.testing.assert_allclose((m[:,None]*w).sum(0),(m[:,None]*v).sum(0),atol=2e-13)
 np.testing.assert_allclose(np.cross(r,m[:,None]*w).sum(0),np.cross(r,m[:,None]*v).sum(0),atol=2e-13)
 assert abs(np.sum(m[:,None]*(w*w-v*v)))<1e-12
 back,_=constrained(w,m,mask,axis,r);np.testing.assert_allclose(back,v,atol=1e-14)
 Q,_=np.linalg.qr(rng.normal(size=(3,3)));boost=np.array([3,-2,7])
 other,_=constrained(v@Q+boost,m,mask,axis@Q,r@Q+7)
 np.testing.assert_allclose(other,w@Q+boost,atol=1e-13)

def test_constrained_radial_two_body_limit():
 v=np.array([[3.,2,1],[-1.,2,1]]);r=np.array([[1.,0,0],[-1.,0,0]])
 w,d=constrained(v,np.ones(2),np.array([True,False]),np.array([1.,0,0]),r)
 np.testing.assert_allclose(w,v[::-1],atol=1e-14);assert d['constraint_rank']==5
 with pytest.raises(ValueError):constrained(v,np.ones(2),np.array([True,False]),np.array([0.,1,0]),r)
