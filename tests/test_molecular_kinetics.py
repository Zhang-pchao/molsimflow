import numpy as np
import pytest
from molsimflow.postprocess.molecular_kinetics import (
    KB, MVV2E, kinetic_partition, diatomic_modes, native_image_translation)

def fixture():
    p=np.array([[[0,0,0],[1.1,0,0]],[[0,3,0],[0,4.1,0]]],float)
    v=np.array([[[1,2,3],[-1,3,1]],[[2,2,1],[4,1,2]]],float)
    return p,v,np.array([[14,15],[12,16]],float),np.full(3,10.)

def test_energy_closure_boost_pair_order_and_image_invariance():
    p,v,m,L=fixture();a=diatomic_modes(p,v,m,L)
    assert abs(a["closure_eV"])<1e-12
    assert a["lab_eV"]==pytest.approx(.5*MVV2E*np.sum(m[:,:,None]*v*v))
    shifted=p.copy();shifted[0,1]+=L
    variations=[(p,v+np.array([3,-2,1]),m),(shifted,v,m),(p[:,::-1],v[:,::-1],m[:,::-1]),
                (p[::-1],v[::-1],m[::-1])]
    for pp,vv,mm in variations:
        b=diatomic_modes(pp,vv,mm,L)
        for key in ["translation_K","rotation_K","radial_K","min_bond_A","max_bond_A"]:
            assert b[key]==pytest.approx(a[key],abs=1e-10)

def test_analytic_radial_and_rotational_modes():
    p=np.array([[[0,0,0],[1,0,0]],[[0,3,0],[1,3,0]]],float)
    v=np.array([[[-2,0,0],[2,0,0]],[[-2,0,0],[2,0,0]]],float)
    m=np.full((2,2),14.)
    a=diatomic_modes(p,v,m,np.full(3,10.))
    assert a["radial_eV"]==pytest.approx(112*MVV2E)
    assert a["rotation_eV"]==pytest.approx(0)
    assert a["translation_eV"]==pytest.approx(0)
    b=diatomic_modes(p,v[:,:,[1,0,2]],m,np.full(3,10.))
    assert b["rotation_eV"]==pytest.approx(a["radial_eV"])
    assert b["radial_eV"]==pytest.approx(0)

def test_temperature_dof_and_com():
    v=np.ones((2,3));m=np.array([2.,2.])
    a=kinetic_partition(m,v,remove_com=False,degrees_of_freedom=6)
    assert a["temperature_K"]==pytest.approx(2*a["lab_KE_eV"]/(6*KB))
    assert kinetic_partition(m,v,remove_com=True)["temperature_K"]==0

def test_native_image_translation_preserves_deformation():
    ref=np.array([[1,2,3],[2,4,6],[3,6,9]],float);L=np.full(3,10.)
    unwrapped=ref+np.array([4,-12,23]);unwrapped[0,0]+=.2
    im=np.floor(unwrapped/L);p=unwrapped-im*L
    shift,res=native_image_translation(p,im,ref,L)
    np.testing.assert_allclose(shift,[4,-12,23])
    assert res[0,0]==pytest.approx(.2)
    s2,r2=native_image_translation(p+L,im-1,ref,L)
    np.testing.assert_allclose(s2,shift);np.testing.assert_allclose(r2,res,atol=1e-14)
    with pytest.raises(ValueError):native_image_translation(p,im+.2,ref,L)

def test_invalid_inputs():
    p,v,m,L=fixture()
    with pytest.raises(ValueError):diatomic_modes(p,v,m*0,L)
    with pytest.raises(ValueError):diatomic_modes(p,v,m,L*0)
    p[:,1]=p[:,0]
    with pytest.raises(ValueError):diatomic_modes(p,v,m,L)
