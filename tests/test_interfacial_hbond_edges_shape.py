import numpy as np
import pytest
from molsimflow.postprocess.interfacial_contact_renewal import (
    water_surface_hbond_edges, water_surface_hbond_degrees, shared_water_site_bridges)
from molsimflow.postprocess.interfacial_water_hbond import (
    hydrogen_bond_pair_edges, water_water_hbond_count)
from molsimflow.postprocess.interfacial_gas_kinematics import planar_shape_tensor

BOX=np.array([[0.,20.]]*3)
def test_directed_edges_reproduce_counts_and_periodic_identity():
    w=np.array([[19.,5,5],[10,10,10]])
    s=np.array([[1.,5,5],[17,5,5]])
    wo=[np.array([[1.,0,0],[-1,0,0]]),np.array([[0.,1,0]])]
    so=[np.array([[-1.,0,0]]),np.array([[1.,0,0]])]
    e=water_surface_hbond_edges(w,wo,s,so,BOX)
    d,a=water_surface_hbond_degrees(w,wo,s,so,BOX)
    assert e.tolist()==[[0,0,0],[0,0,1],[0,1,0],[0,1,1]]
    assert np.array_equal(np.bincount(e[e[:,2]==0,0],minlength=2),d)
    assert np.array_equal(np.bincount(e[e[:,2]==1,0],minlength=2),a)
    shifted=water_surface_hbond_edges(w+20,wo,s+20,so,BOX)
    assert np.array_equal(shifted,e)
    pairs,count=shared_water_site_bridges(e)
    assert pairs.tolist()==[[0,1]] and count.tolist()==[1]

def test_bridge_counts_collapse_duplicate_directions():
    e=np.array([[7,2,0],[7,2,1],[7,5,0],[7,5,0],[9,2,1],[9,5,1],[9,8,0]])
    pairs,count=shared_water_site_bridges(e)
    assert pairs.tolist()==[[2,5],[2,8],[5,8]]
    assert count.tolist()==[2,1,1]

def test_pair_edges_count_parity_and_no_double_count():
    x=np.array([[19.,5,5],[1.,5,5],[10,10,10]])
    v=[np.array([[1.,0,0]]),np.array([[-1.,0,0]]),np.empty((0,3))]
    e=hydrogen_bond_pair_edges(x,v,BOX)
    assert e.tolist()==[[0,1]]
    assert len(e)==water_water_hbond_count(x,v,BOX,oo_cutoff=3.5,angle_cutoff_deg=30)

def test_empty_edges_keep_column_schema():
    z=np.empty((0,3))
    assert water_surface_hbond_edges(z,[],z,[],BOX).shape==(0,3)
    assert hydrogen_bond_pair_edges(z,[],BOX).shape==(0,2)
    assert shared_water_site_bridges(np.empty((0,3),int))[0].shape==(0,2)

@pytest.mark.parametrize('bad',[np.array([[1,2,3]]),np.array([[1.,2.,0.]]),np.array([[-1,2,0]])])
def test_bad_bridge_rows_rejected(bad):
    with pytest.raises(ValueError):shared_water_site_bridges(bad)

def test_invalid_hbond_geometry_rejected():
    with pytest.raises(ValueError):
        hydrogen_bond_pair_edges(np.zeros((1,3)),[np.zeros((1,3))],BOX)
    with pytest.raises(ValueError):
        water_surface_hbond_edges(np.zeros((1,3)),[],np.empty((0,3)),[],BOX)

def test_rotated_planar_shape_and_translation():
    a=np.linspace(0,2*np.pi,120,endpoint=False)
    xy=np.column_stack([2*np.cos(a),5*np.sin(a)])
    q=planar_shape_tensor(xy)
    assert np.isclose(q['aspect_ratio'],2.5)
    assert np.isclose(q['y_variance_fraction'],25/29)
    assert abs(q['principal_axis'][1])>1-1e-12
    angle=.43;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    p=planar_shape_tensor(xy@r.T+np.array([100,-30]))
    assert np.allclose(q['eigenvalues'],p['eigenvalues'])
    assert np.isclose(abs(np.dot(p['principal_axis'],r@np.array([0,1]))),1)

def test_isotropic_axis_unidentified_and_zero_extent_rejected():
    q=planar_shape_tensor(np.array([[1.,0],[0,1],[-1,0],[0,-1]]))
    assert np.isnan(q['principal_axis']).all()
    assert q['anisotropy']==0
    with pytest.raises(ValueError):planar_shape_tensor(np.zeros((5,2)))
