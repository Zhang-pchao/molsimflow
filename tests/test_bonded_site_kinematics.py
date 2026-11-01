import numpy as np
import pytest
from molsimflow.postprocess.bonded_site_kinematics import decompose_bonded_site_distance as split
BOX=np.array([20.,20.,20.])
def test_point_only_radial_motion():
    d=split([3,0,0],[4,0,0],[1,0,0],[1,0,0],[0,0,0],[0,0,0],BOX)
    np.testing.assert_allclose(d.components,[1,0,0])
def test_site_stretch_only():
    d=split([3,0,0],[3,0,0],[1,0,0],[2,0,0],[0,0,0],[0,0,0],BOX)
    np.testing.assert_allclose(d.components,[0,0,-1])
def test_site_rotation_only():
    d=split([3,0,0],[3,0,0],[1,0,0],[0,1,0],[0,0,0],[0,0,0],BOX)
    np.testing.assert_allclose(d.components,[0,np.sqrt(10)-2,0])
    assert d.site_rotation_angle_deg==90
def test_common_motion_and_lattice_translation_invariance():
    p=np.array([[3.,.4,0],[3.5,.6,.3],[1.,0,0],[1.1,.3,.1],[0,0,0],[.1,0,0]])
    a=split(*p,BOX)
    p[[1,3,5]] += [2,-1,3]
    p += 2*BOX
    b=split(*p,BOX)
    np.testing.assert_allclose(a.components,b.components,atol=1e-13)
def test_reverse_time_and_random_closure():
    rng=np.random.default_rng(11)
    for _ in range(100):
        c=rng.normal(size=(2,3))*.1
        s=c+[1,0,0]+rng.normal(size=(2,3))*.1
        p=c+[3,0,0]+rng.normal(size=(2,3))*.1
        a=split(p[0],p[1],s[0],s[1],c[0],c[1],BOX)
        b=split(p[1],p[0],s[1],s[0],c[1],c[0],BOX)
        np.testing.assert_allclose(a.components,-b.components,atol=1e-13)
        assert a.closure_error<1e-13
def test_pbc_ambiguity_and_invalid_bond_rejected():
    with pytest.raises(ValueError,match="offsets"):
        split([10.9,0,0],[11.1,0,0],[1,0,0],[1,0,0],[0,0,0],[0,0,0],BOX)
    with pytest.raises(ValueError,match="bond"):
        split([3,0,0],[3,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],BOX)
