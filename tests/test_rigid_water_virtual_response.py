import numpy as np
import pytest
from molsimflow.postprocess.rigid_water_virtual_response import virtual_map, generalized_forces
R = np.array([[9.8,2.,3.],[.55,2.2,3.1],[9.4,2.8,3.], [4.,5.,6.],[4.8,5.1,6.],[3.7,5.9,6.]])
L = np.full(3,10.)
A = np.array([.3,.4,np.sqrt(.75)])
E = np.array([0.,1.,0.])


def test_exact_map_jacobian_and_periodic_rigidity():
    z = np.array([.15,.03,-.04]);m,j = virtual_map(R,L,A,E,z)
    eps = 1e-6
    for i in range(3):
        dz = np.eye(3)[i]*eps
        fd = (virtual_map(R,L,A,E,z+dz)[0]-virtual_map(R,L,A,E,z-dz)[0])/(2*eps)
        np.testing.assert_allclose(fd,j[i],atol=1e-9,rtol=0)
    def bonds(r):
        v=r[1:3]-r[0];v-=np.rint(v/L)*L;return v@v.T
    np.testing.assert_allclose(bonds(m),bonds(R),atol=1e-14)
    np.testing.assert_allclose(m[3:]-m[3],R[3:]-R[3],atol=1e-14)


def test_virtual_work_sign_and_hessian_reciprocity_on_cartesian_potential():
    rng = np.random.default_rng(103);b=rng.normal(size=(18,18));k=b.T@b
    def eg(z):
        m,j=virtual_map(R,L,A,E,z);v=(m-R).ravel();f=-(k@v+1).reshape(6,3)
        return .5*v@k@v+v.sum(),generalized_forces(f,j)
    z=np.array([.04,.01,-.02]);eps=1e-5;hessian=np.empty((3,3))
    for i in range(3):
        dz=np.eye(3)[i]*eps;ep,gp=eg(z+dz);em,gm=eg(z-dz)
        assert -(ep-em)/(2*eps)==pytest.approx(eg(z)[1][i],abs=2e-8)
        hessian[:,i]=-(gp-gm)/(2*eps)
    np.testing.assert_allclose(hessian,hessian.T,atol=2e-8,rtol=0)


def test_invalid_axis_and_nonfinite_force_rejected():
    with pytest.raises(ValueError):virtual_map(R,L,2*A,E,np.zeros(3))
    _,j=virtual_map(R,L,A,E,np.zeros(3))
    with pytest.raises(ValueError):generalized_forces(np.full((6,3),np.nan),j)
