import numpy as np
import pytest
from molsimflow.postprocess.local_pair_coordination import compact_switch, pair_coordination


def test_quintic_boundaries_derivative_and_force_bound():
    r=np.array([0, 2.8, 2.8+1e-8, 3.4, 4.0-1e-8, 4.0, 9.])
    s,d=compact_switch(r,2.8,4.0)
    assert np.all((0<=s)&(s<=1)) and np.all(d<=0)
    assert np.allclose(s[[0,1,5,6]],[1,1,0,0])
    assert np.allclose(d[[0,1,5,6]],0)
    assert np.max(np.abs(d)) <= 1.875/1.2+1e-12
    h=1e-6
    rp=np.linspace(2.8001,3.9999,100)
    _,deriv=compact_switch(rp,2.8,4.0)
    fd=(compact_switch(rp+h,2.8,4.0)[0]-compact_switch(rp-h,2.8,4.0)[0])/(2*h)
    assert np.max(np.abs(fd-deriv))<4e-9


def test_all_cartesian_derivatives_include_anchor_and_unequal_weights():
    points=np.array([[3.2,.3,.5],[18.1,1.4,.8],[1.1,1.2,1.3],[9,8,7.]])
    anchor=np.array([.4,.2,.1]);box=np.array([20,21,22.]);weights=np.array([1,.5,2,1])
    def val(p,a):return pair_coordination(p,a,box,2.0,4.0,weights).value
    q=pair_coordination(points,anchor,box,2.,4.,weights)
    eps=1e-5
    for i in range(len(points)):
        for j in range(3):
            p=points.copy();m=points.copy();p[i,j]+=eps;m[i,j]-=eps
            assert abs((val(p,anchor)-val(m,anchor))/(2*eps)-q.point_gradients[i,j])<1e-8
    for j in range(3):
        p=anchor.copy();m=anchor.copy();p[j]+=eps;m[j]-=eps
        assert abs((val(points,p)-val(points,m))/(2*eps)-q.anchor_gradient[j])<1e-8
    assert np.linalg.norm(q.anchor_gradient+q.point_gradients.sum(axis=0))<1e-14


def test_translation_wrapping_atom_exchange_and_empty_selection():
    rng=np.random.default_rng(231)
    box=np.array([20,21,22.]);points=rng.uniform(0,20,(60,3));anchor=np.array([19,2,3.])
    q=pair_coordination(points,anchor,box,2.,4.)
    shift=np.array([2.4,-9.3,28.7]);order=rng.permutation(len(points))
    other=pair_coordination((points[order]+shift)%box,(anchor+shift)%box,box,2.,4.)
    assert abs(q.value-other.value)<1e-12
    assert np.allclose(q.point_gradients[order],other.point_gradients,atol=1e-12)
    assert np.allclose(q.anchor_gradient,other.anchor_gradient,atol=1e-12)
    empty=pair_coordination(np.empty((0,3)),anchor,box,2.,4.)
    assert empty.value==0 and empty.point_gradients.shape==(0,3)
    assert np.array_equal(empty.anchor_gradient,np.zeros(3))


def test_zero_distance_and_minimum_image_boundary_are_regular():
    q=pair_coordination(np.array([[0,0,0],[10,0,0]]),np.zeros(3),np.array([20,20,20]),0,4)
    assert q.value==1 and np.array_equal(q.point_gradients,np.zeros((2,3)))
    with pytest.raises(ValueError):
        pair_coordination(np.zeros((1,3)),np.zeros(3),np.full(3,8),0,4)
    for inner,outer in [(-1,3),(3,3),(4,3),(0,float('nan'))]:
        with pytest.raises(ValueError): compact_switch(np.array([1]),inner,outer)

from molsimflow.postprocess.local_pair_coordination import soft_minimum_distance


def test_soft_minimum_force_derivatives_and_dry_state_reach():
    box=np.array([24.,26.,28.]);anchor=np.array([.2,.3,.4])
    points=np.array([[5.8,.5,.8],[22.,3.,2.],[9.1,1.,.5],[12.,12.,12.]])
    q=soft_minimum_distance(points,anchor,box)
    eps=1e-5
    for i in range(len(points)):
        for j in range(3):
            p=points.copy();m=points.copy();p[i,j]+=eps;m[i,j]-=eps
            fd=(soft_minimum_distance(p,anchor,box).value-soft_minimum_distance(m,anchor,box).value)/(2*eps)
            assert abs(fd-q.point_gradients[i,j])<1e-8
    for j in range(3):
        p=anchor.copy();m=anchor.copy();p[j]+=eps;m[j]-=eps
        fd=(soft_minimum_distance(points,p,box).value-soft_minimum_distance(points,m,box).value)/(2*eps)
        assert abs(fd-q.anchor_gradient[j])<1e-8
    dry=soft_minimum_distance(np.array([[5.5,0,0.]]),np.zeros(3),box)
    assert 5.4<dry.value<5.6 and np.linalg.norm(dry.point_gradients[0])>.99
    assert np.linalg.norm(q.anchor_gradient+q.point_gradients.sum(axis=0))<1e-14


def test_soft_minimum_identity_wrap_empty_and_coincident_regularization():
    box=np.array([24.,26.,28.]);anchor=np.array([23.,1.,2.])
    points=np.array([[2.,1.,2.],[21.,3.,2.],[9.,1.,2.]])
    q=soft_minimum_distance(points,anchor,box)
    shift=np.array([5.,-8.,31.])
    other=soft_minimum_distance((points[::-1]+shift)%box,(anchor+shift)%box,box)
    assert np.isclose(q.value,other.value,atol=1e-12)
    assert np.allclose(q.point_gradients[::-1],other.point_gradients,atol=1e-12)
    for pts in [np.empty((0,3)),np.array([[12.,13.,14.]])]:
        empty=soft_minimum_distance(pts,np.zeros(3),box)
        assert empty.value==10 and np.all(empty.point_gradients==0)
    coincident=soft_minimum_distance(np.zeros((1,3)),np.zeros(3),box)
    assert np.isfinite(coincident.value) and np.all(coincident.point_gradients==0)
