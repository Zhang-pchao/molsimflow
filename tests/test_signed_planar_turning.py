"""Synthetic interpretation guards for sampled planar turning."""
import numpy as np
import pytest
from molsimflow.postprocess.interfacial_gas_kinematics import signed_planar_turning

def path(theta, radius=10.):
    x=np.zeros((len(theta),1,3))
    x[:,0,0]=radius*np.cos(theta);x[:,0,2]=radius*np.sin(theta)
    return x
def run(p, **kw):
    n=len(p)
    return signed_planar_turning(p,np.arange(n,dtype=int),np.zeros(n,int),
                                 np.ones(p.shape[:2],bool),expected_step=1,**kw)
def test_full_turn_and_direction():
    theta=np.linspace(0,2*np.pi,33)
    q=run(path(theta))
    assert q['valid_interval'].all()
    assert np.isclose(np.sum(q['angle_increment']),2*np.pi)
    assert np.isclose(np.sum(run(path(-theta))['angle_increment']),-2*np.pi)
def test_common_lab_translation_removed_by_supplied_center():
    p=path(np.linspace(0,1,12));translation=np.arange(12)[:,None,None]*np.array([2.,3.,4.])
    assert np.allclose(run((p+translation)-translation)['angle_increment'],run(p)['angle_increment'])
def test_stationary_relative_cloud():
    q=run(np.repeat(np.array([[[8.,3.,4.],[9.,-2.,3.]]]),6,axis=0))
    assert np.all(q['angle_increment']==0)
def test_open_shear_path_is_not_a_full_turn():
    # A positive rotation fit in shear is not a complete revolution.
    p=np.zeros((21,1,3));p[:,0,0]=np.linspace(-8,8,21);p[:,0,2]=8
    theta=run(p)['angle_increment']
    assert np.all(theta<0) and abs(np.sum(theta))<np.pi
def test_axis_crossing_radius_mask():
    p=path(np.linspace(0,.3,4));p[1,0]=[.1,0,.1]
    q=run(p)
    assert q['valid_interval'][:,0].tolist()==[False,False,True]
    assert np.isnan(q['angle_increment'][:2]).all()
def test_cadence_epoch_and_departure_masks():
    p=path(np.linspace(0,.6,7));step=np.array([0,1,2,4,5,6,7]);epoch=np.array([0,0,0,0,1,1,1])
    member=np.ones((7,1),bool);member[5]=False
    q=signed_planar_turning(p,step,epoch,member,expected_step=1)
    assert q['valid_interval'][:,0].tolist()==[True,True,False,False,False,False]
def test_large_sampled_angle_rejected():
    q=run(path(np.array([0.,2.])))
    assert not q['valid_interval'].any()
    assert np.isnan(q['angle_increment']).all()
def test_missing_positions_and_invalid_inputs():
    p=path(np.linspace(0,.2,3));p[1]=np.nan
    member=np.array([[True],[False],[True]])
    q=signed_planar_turning(p,np.arange(3),np.zeros(3,int),member,expected_step=1)
    assert not q['valid_interval'].any()
    with pytest.raises(ValueError):run(p)
    with pytest.raises(ValueError):run(path(np.arange(3)),minimum_radius=0)
    with pytest.raises(ValueError):run(path(np.arange(3)),axes=(0,0))
def test_counter_rotating_identities_cancel():
    theta=np.linspace(0,2*np.pi,33)
    p=np.concatenate([path(theta),path(-theta)],axis=1)
    q=run(p)['angle_increment']
    assert np.isclose(q.sum(),0)
    assert np.isclose(np.abs(q).sum(),4*np.pi)
