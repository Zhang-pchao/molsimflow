"""Numerical and symmetry tests for the full four-atom azimuth force."""
import numpy as np
import pytest
from molsimflow.postprocess.local_azimuth import periodic_azimuth_bias as bias

BASE=np.array([[.9,.2,1.9],[0.,0.,1.6],[0.,0.,0.],[1.5,.1,-.4]])
BOX=np.array([30.,31.,32.])
def test_full_gradient_and_symmetries():
    rng=np.random.default_rng(81071)
    for _ in range(30):
        r=BASE+rng.normal(0,.035,(4,3));ref=rng.uniform(-8,8)
        a=bias(r,BOX,ref,.03)
        numeric=np.zeros((4,3));eps=1e-6
        for i in range(4):
            for k in range(3):
                u=r.copy();v=r.copy();u[i,k]+=eps;v[i,k]-=eps
                numeric[i,k]=-(bias(u,BOX,ref,.03).energy-bias(v,BOX,ref,.03).energy)/(2*eps)
        np.testing.assert_allclose(a.forces,numeric,atol=2e-10,rtol=2e-7)
        np.testing.assert_allclose(a.forces.sum(0),0,atol=1e-15)
        np.testing.assert_allclose(np.cross(a.local_positions,a.forces).sum(0),0,atol=1e-15)
        wrapped=(r+rng.normal(0,40,(1,3)))%BOX
        b=bias(wrapped,BOX,ref,.03)
        np.testing.assert_allclose(a.forces,b.forces,atol=1e-14)
        assert abs(a.energy-b.energy)<1e-14
        Q,_=np.linalg.qr(rng.normal(size=(3,3)));Q[:,0]*=np.linalg.det(Q)
        c=bias(r@Q,BOX,ref,.03)
        np.testing.assert_allclose(a.forces@Q,c.forces,atol=1e-14)
        assert abs(a.energy-c.energy)<1e-14
        np.testing.assert_allclose(a.forces,bias(r,BOX,ref+2*np.pi,.03).forces,atol=1e-14)

def test_zero_bias_and_branch_continuity():
    a=bias(BASE,BOX,0,0)
    np.testing.assert_array_equal(a.forces,0)
    assert a.energy==0
    r=BASE.copy();r[0]=r[1]+[-.9,0,.3]
    plus=r.copy();minus=r.copy();plus[0,1]=1e-8;minus[0,1]=-1e-8
    np.testing.assert_allclose(bias(plus,BOX,.2,.03).forces,bias(minus,BOX,.2,.03).forces,atol=2e-9)

@pytest.mark.parametrize("kind",["collinear_reference","collinear_H","long_edge","nan","box"])
def test_invalid_geometry_rejected(kind):
    r=BASE.copy();box=BOX.copy()
    if kind=="collinear_reference":r[3]=[0,0,-1.6]
    elif kind=="collinear_H":r[0]=r[1]+[0,0,.96]
    elif kind=="long_edge":r[0]+=10
    elif kind=="nan":r[0,0]=np.nan
    elif kind=="box":box[0]=4
    with pytest.raises(ValueError):bias(r,box,.3,.03)
