import numpy as np
import pytest
from molsimflow.postprocess.contact_kinematics import (
    diatomic_snapshot, decompose_contact, ImageAlignmentError)

BOX = np.full(3, 40.)
BASE = np.array([[[4.,0,0],[5.,0,0]],[[8.,2.,0],[9.,2.,0]]])

def snap(a=BASE, o=None, s=None, box=BOX):
    return diatomic_snapshot(a, np.zeros(3) if o is None else o,
                            np.zeros(3) if s is None else s, box, 2.)

def test_rigid_cohort_translation_and_stationary_site():
    d=decompose_contact(snap(),snap(BASE+[.6,0,0]),0,0,BOX)
    np.testing.assert_allclose(d.projected,[.6,0,0,0,0],atol=1e-14)

def test_one_molecule_and_site_have_expected_signed_terms():
    a=BASE.copy();a[0,:,0]+=.6
    d=decompose_contact(snap(),snap(a),0,0,BOX)
    np.testing.assert_allclose(d.projected,[.3,.3,0,0,0],atol=1e-14)
    d=decompose_contact(snap(),snap(o=np.array([.4,0,0])),0,0,BOX)
    np.testing.assert_allclose(d.projected,[0,0,0,0,-.4],atol=1e-14)

def test_pure_rotation_and_stretch_close_to_direct_distance():
    rot=BASE.copy();rot[0]=[[4.5,-.5,0],[4.5,.5,0]]
    d=decompose_contact(snap(),snap(rot),0,0,BOX)
    np.testing.assert_allclose(d.projected[[0,1,3,4]],0,atol=1e-14)
    assert abs(d.projected[2]-(np.sqrt(4.5**2+.5**2)-4))<1e-14
    ext=BASE.copy();ext[0]=[[3.8,0,0],[5.2,0,0]]
    d=decompose_contact(snap(),snap(ext),0,0,BOX)
    np.testing.assert_allclose(d.projected,[0,0,0,-.2,0],atol=1e-14)

def test_translation_invariance_and_reference_cancellation():
    move=np.array([17.,-21.,9.])
    a=BASE.copy();a[0]+=[.1,.3,-.1]
    d=decompose_contact(snap(),snap(a),0,1,BOX)
    e=decompose_contact(snap(BASE+move,move,move),snap(a+move,move,move),0,1,BOX)
    np.testing.assert_allclose(d.projected,e.projected,atol=1e-13)
    f=decompose_contact(snap(),snap(s=np.array([.2,0,0])),0,0,BOX)
    np.testing.assert_allclose(f.projected,[-.2,0,0,0,.2],atol=1e-14)

def test_boundary_reconstruction_and_fixed_image_crossing():
    box=np.full(3,10.)
    a=np.array([[[9.7,1,1],[.7,1,1]]])
    b=np.array([[[10.2,1,1],[1.2,1,1]]])
    o=np.array([1.5,1,1]);s=np.array([1.5,1,0])
    p=diatomic_snapshot(a,o,s,box,2);c=diatomic_snapshot(b,o,s,box,2)
    np.testing.assert_allclose(p.bonds,[[1,0,0]],atol=1e-14)
    d=decompose_contact(p,c,0,0,box)
    np.testing.assert_allclose(d.projected,[-.5,0,0,0,0],atol=1e-14)
    c=diatomic_snapshot(a+[-5.,0.,0.],o,s,box,2)
    with pytest.raises(ImageAlignmentError):decompose_contact(p,c,0,0,box)

def test_random_motion_closure_both_atom_identities():
    rng=np.random.default_rng(7612)
    for _ in range(300):
        centers=rng.uniform(4,8,(5,3))
        bonds=rng.normal(size=(5,3));bonds/=np.linalg.norm(bonds,axis=1)[:,None]
        a=centers[:,None,:]+np.array([-.5,.5])[None,:,None]*bonds[:,None,:]
        b=a+rng.normal(0,.07,a.shape)
        p=snap(a);c=snap(b,rng.normal(0,.1,3),rng.normal(0,.1,3))
        for atom in (0,1):
            d=decompose_contact(p,c,2,atom,BOX)
            direct=np.linalg.norm(b[2,atom]-c.site)-np.linalg.norm(a[2,atom]-p.site)
            assert abs(d.projected.sum()-direct)<2e-14

def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):diatomic_snapshot(np.zeros((1,2,3)),np.zeros(3),np.zeros(3),BOX,2)
    with pytest.raises(ValueError):snap(BASE*np.nan)
