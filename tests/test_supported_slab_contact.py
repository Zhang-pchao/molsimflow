import numpy as np
from molsimflow.postprocess.supported_slab_contact import same_face_slab_contact

BOX=np.array([[0.,40.],[0.,40.],[0.,40.]])
SOLID=np.array([[x,10.,z] for x in (10.,12.,14.) for z in (9.,11.)])
def pairs(c):
    c=np.asarray(c,float)
    return np.stack([c+[-.5,0,0],c+[.5,0,0]],axis=1).reshape(-1,3)

def test_connected_same_face_contact():
    a=same_face_slab_contact(pairs([[10,10,13],[12,10,13],[14,10,13]]),SOLID,SOLID,BOX)
    assert a.supported and a.contact.sum()==3
    np.testing.assert_array_equal(a.component_size,[3,3,3])
    np.testing.assert_array_equal(a.face,[1,1,1])

def test_opposite_faces_do_not_combine_support():
    a=same_face_slab_contact(pairs([[10,10,12.5],[11,10,12.5],[10,10,7.5],[11,10,7.5]]),SOLID,SOLID,BOX)
    assert a.main.sum()==4 and a.contact.sum()==4 and not a.supported
    np.testing.assert_array_equal(a.component_size,[2,2,2,2])

def test_translation_and_periodic_seam_preserve_faces():
    n=pairs([[10,10,13],[12,10,13],[14,10,13]])
    a=same_face_slab_contact(n,SOLID,SOLID,BOX)
    shift=np.array([37.,-12.,29.])
    b=same_face_slab_contact((n+shift)%40,(SOLID+shift)%40,(SOLID+shift)%40,BOX)
    np.testing.assert_allclose(a.distance,b.distance,atol=1e-12)
    np.testing.assert_array_equal(a.face,b.face)
    assert b.supported
    movedbox=BOX+np.array([[-5],[-7],[13]])
    c=same_face_slab_contact(n+[-5,-7,13],SOLID+[-5,-7,13],SOLID+[-5,-7,13],movedbox)
    np.testing.assert_allclose(a.distance,c.distance,atol=1e-12)
    np.testing.assert_array_equal(a.face,c.face)

def test_noncontact_bridge_does_not_join_separate_touch_groups():
    s=np.array([[x,10,z] for x in (5.,15.) for z in (9.,11.)])
    n=pairs([[5,10,13],[5.5,10,13],[15,10,13],[15.5,10,13],[10,10,15]])
    a=same_face_slab_contact(n,s,s,BOX)
    assert a.main.sum()==5 and a.contact.sum()==4 and not a.supported
    assert sorted(a.component_size.tolist())==[0,2,2,2,2]

def test_single_touch_and_absent_contact():
    a=same_face_slab_contact(pairs([[10,10,13]]),SOLID,SOLID,BOX)
    assert a.contact.sum()==1 and not a.supported
    b=same_face_slab_contact(pairs([[10,10,25],[12,10,25],[14,10,25]]),SOLID,SOLID,BOX)
    assert b.contact.sum()==0 and not b.supported


def test_ppf_matches_ppp_away_from_z_seam():
    n=pairs([[10,10,13],[12,10,13],[14,10,13]])
    periodic=same_face_slab_contact(n,SOLID,SOLID,BOX)
    open_z=same_face_slab_contact(n,SOLID,SOLID,BOX,periodic_z=False)
    np.testing.assert_allclose(open_z.distance,periodic.distance)
    np.testing.assert_array_equal(open_z.contact,periodic.contact)
    np.testing.assert_array_equal(open_z.component_size,periodic.component_size)
    assert open_z.supported


def test_ppf_does_not_contact_through_z_boundary():
    solid=np.array([[10,10,38],[12,10,38],[14,10,38],[10,10,36]])
    n=pairs([[10,10,1],[12,10,1],[14,10,1]])
    periodic=same_face_slab_contact(n,solid,solid,BOX)
    open_z=same_face_slab_contact(n,solid,solid,BOX,periodic_z=False)
    assert periodic.contact.sum()==3
    assert open_z.contact.sum()==0
    assert open_z.distance.min()>30


def test_ppf_clusters_do_not_join_through_z_boundary():
    solid=np.array([[10,10,19],[12,10,19],[14,10,21],[10,10,21]])
    n=pairs([[10,10,1],[12,10,1],[10,10,39],[12,10,39]])
    a=same_face_slab_contact(n,solid,solid,BOX,periodic_z=True)
    b=same_face_slab_contact(n,solid,solid,BOX,periodic_z=False)
    assert a.main.sum()==4 and b.main.sum()==2


def test_ppf_translation_and_declared_box_preserve_observer():
    n=pairs([[10,10,13],[12,10,13],[14,10,13]])
    a=same_face_slab_contact(n,SOLID,SOLID,BOX,periodic_z=False)
    shift=np.array([37.,-12.,29.])
    box=BOX.copy();box[2]+=29.
    moved=n+shift;moved[:,:2]%=40
    slab=SOLID+shift;slab[:,:2]%=40
    b=same_face_slab_contact(moved,slab,slab,box,periodic_z=False)
    np.testing.assert_allclose(a.distance,b.distance,atol=1e-12)
    np.testing.assert_array_equal(a.face,b.face)
    assert b.supported
