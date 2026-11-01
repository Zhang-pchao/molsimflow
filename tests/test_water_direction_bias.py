"""Numerical and symmetry controls for a conservative molecular direction bias."""
import numpy as np
import pytest
from molsimflow.postprocess.water_direction_bias import evaluate,reference_distances,donor_coordinate,schedule

def geometry():
    return np.array([[0,0,0],[2.8,0,0],[1.91,.40,0],[3.35,.72,.20],[-1.5,.2,0],[-1,1.8,0],[1,3.7,.2],[1.2,4.7,.4]],float),np.array([30.,31.,32.])

def test_gradient_and_reactions():
    r,L=geometry();d=reference_distances(r,L)+np.linspace(-.1,.1,9)
    energy,force,_=evaluate(r,L,.5,1.,2.,d,.7)
    error=0.
    for i in range(8):
        for k in range(3):
            a=r.copy();b=r.copy();a[i,k]+=1e-6;b[i,k]-=1e-6
            fd=-(evaluate(a,L,.5,1.,2.,d,.7)[0]-evaluate(b,L,.5,1.,2.,d,.7)[0])/2e-6
            error=max(error,abs(fd-force[i,k]))
    assert error<2e-9
    assert np.linalg.norm(force.sum(0))<1e-12
    assert np.linalg.norm(np.cross(r,force).sum(0))<1e-12

def test_symmetries_and_zero():
    r,L=geometry();d=reference_distances(r,L)
    E,F,z=evaluate(r,L,.5,1.,2.,d)
    shifted=r+np.array([3.5,-9.6,13.]);shifted[2]+=L;shifted[5]-=2*L
    e,f,_=evaluate(shifted,L,.5,1.,2.,d)
    assert abs(e-E)<1e-13 and np.max(abs(f-F))<1e-12
    swap=r.copy();swap[[2,3]]=swap[[3,2]];e,f,_=evaluate(swap,L,.5,1.,2.,d);f[[2,3]]=f[[3,2]]
    assert abs(e-E)<1e-13 and np.max(abs(f-F))<1e-12
    for arm in ['aligned','away','sham']:
        e,f,_=evaluate(r,L,references=d,**schedule(2.,z['q'],arm))
        assert e==0 and np.array_equal(f,np.zeros((8,3)))

def test_release_work_and_invalid_parameters():
    r,L=geometry();d=reference_distances(r,L);q=donor_coordinate(r,L)[0]
    times=np.r_[np.linspace(0,1.999,300),2.];energies=[evaluate(r,L,references=d,**schedule(t,q,'away'))[0] for t in times]
    increments=np.diff(energies)
    assert abs(increments.sum())<1e-14
    assert abs(increments[-1]+energies[-2])<1e-14
    with pytest.raises(ValueError):evaluate(r,L,0.,-1.,2.,d)
    with pytest.raises(ValueError):schedule(-.1,q,'away')
