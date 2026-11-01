import numpy as np
import pytest
from molsimflow.postprocess.contact_water_geometry import (
    segment_geometry,contacting_atom,corridor_minima,corridor_category)

def test_projection_and_endpoint_planes():
    g=segment_geometry([4,0,0],[[2,1,0],[-1,0,0],[5,0,0],[0,2,0],[4,2,0]])
    np.testing.assert_allclose(g.fraction,[.5,-.25,1.25,0,1])
    np.testing.assert_allclose(g.perpendicular,[1,0,0,2,2])

def test_pbc_atom_selection_and_tie():
    v,i=contacting_atom(np.array([19.,0,0]),[[1,0,0],[2,0,0]],[2,3],[20,20,20])
    np.testing.assert_allclose(v,[2,0,0]);assert i==2
    v,i=contacting_atom(np.zeros(3),[[1,0,0],[-1,0,0]],[9,3],[20,20,20])
    assert i==3

def test_all_water_search_is_not_nearest_water_only():
    m,ids=corridor_minima([[0,0,0]],[[4,0,0]],
        [[-1,0,0],[2,1.4,0],[2,.4,0],[0,0,0],[4,0,0]],
        [11,12,13,14,15],[1,1,0,1,1],[20,20,20],2.)
    np.testing.assert_allclose(m,[[1.4,.4]]);np.testing.assert_array_equal(ids,[[12,13]])
    np.testing.assert_array_equal(corridor_category(m[:,0],m[:,1],1),[2])
    np.testing.assert_array_equal(corridor_category(m[:,0],m[:,1],1.5),[1])

def test_periodic_shift_and_permutation():
    s=np.array([[19.,2,2]]);a=np.array([[4.,0,0]])
    w=np.array([[1.,3,2],[1.,2.5,2], [5.,2,2]])
    args=(np.array([5,3,7]),np.array([True,False,True]),np.array([20.,20,20]))
    m,ids=corridor_minima(s,a,w,*args,2)
    n,ni=corridor_minima(s+20,a,w+np.array([20,-20,40]),*args,2)
    np.testing.assert_allclose(m,n);np.testing.assert_array_equal(ids,ni)

def test_search_envelope_includes_cylinder_corners():
    m,ids=corridor_minima([[0,0,0]],[[4,0,0]],[[3.99,1.99,0]], [3],[1],[30,30,30],2)
    assert m[0,0]==pytest.approx(1.99) and ids[0,0]==3
    assert np.isinf(m[0,1])

def test_empty_and_invalid():
    m,ids=corridor_minima([[0,0,0]],[[4,0,0]],np.empty((0,3)),[],[],[20,20,20],2)
    assert np.isinf(m).all() and (ids==-1).all()
    with pytest.raises(ValueError): segment_geometry([0,0,0],[[1,1,1]])
    with pytest.raises(ValueError): corridor_minima([[0,0,0]],[[10,0,0]],[[1,0,0]],[3],[1],[20,20,20],2)
