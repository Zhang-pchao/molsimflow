import numpy as np
import pytest
from molsimflow.postprocess.interfacial_gas_kinematics import fit_affine_motion

@pytest.fixture
def xyz():
    return np.random.default_rng(42).normal(size=(80,3))

@pytest.mark.parametrize("gradient", [
    np.zeros((3,3)), np.diag([.2,.2,.2]),
    np.array([[0,0,2],[0,0,0],[-2,0,0]]),
    np.array([[0,0,2],[0,0,0],[0,0,0]])])
def test_exact_translation_expansion_rotation_and_simple_shear(xyz, gradient):
    velocity=xyz @ gradient.T + [2,3,-4]
    r=fit_affine_motion(xyz,velocity)
    np.testing.assert_allclose(r["gradient"],gradient,atol=1e-12)
    np.testing.assert_allclose(r["strain"]+r["rotation"],gradient,atol=1e-12)
    assert r["residual_rms"]<1e-12 and r["rank"]==3

def test_simple_shear_has_both_rotation_and_strain(xyz):
    v=np.zeros_like(xyz);v[:,0]=2*xyz[:,2]
    r=fit_affine_motion(xyz,v)
    assert r["rotation"][0,2]==pytest.approx(1)
    assert r["strain"][0,2]==pytest.approx(1)
    assert r["gradient"][2,0]==pytest.approx(0,abs=1e-12)

def test_translation_permutation_and_velocity_offset_invariance(xyz):
    rng=np.random.default_rng(99);v=rng.normal(size=xyz.shape)
    a=fit_affine_motion(xyz,v)
    order=rng.permutation(len(xyz));b=fit_affine_motion(xyz[order]+[20,-30,40],v[order]+[8,1,2])
    for key in ["gradient","strain","rotation","residual_rms"]:
        np.testing.assert_allclose(a[key],b[key],atol=1e-12)
    np.testing.assert_allclose(b["translation"]-a["translation"],[8,1,2])

def test_degenerate_and_ill_conditioned_geometry(xyz):
    bad=xyz.copy();bad[:,2]=0
    with pytest.raises(ValueError):fit_affine_motion(bad,xyz)
    bad=xyz.copy();bad[:,2]*=1e-10
    with pytest.raises(ValueError):fit_affine_motion(bad,xyz)

def test_nonfinite_and_bad_condition_limit(xyz):
    with pytest.raises(ValueError):fit_affine_motion(xyz,np.full_like(xyz,np.nan))
    with pytest.raises(ValueError):fit_affine_motion(xyz,xyz,condition_limit=1)

