import numpy as np
import pytest
from molsimflow.postprocess.interfacial_contact_renewal import (
    site_transition_metrics, surface_region_masks, water_surface_hbond_degrees)
from molsimflow.postprocess.interfacial_water_hbond import water_surface_hbond_counts
from molsimflow.postprocess.surface_proton_transfer import assign_hydrogens_to_oxygen_or_carbon


def test_weighted_transitions_distinguish_preexisting_water():
    r = site_transition_metrics([1,1,0,0],[0,0,1,0],[0,1,1,0],[1,1,0,1],[2,3,5,7])
    assert r == dict(gas_lost_area=5.,gas_gained_area=5.,gas_retained_area=0.,gas_union_area=10.,lost_then_wet_area=5.,lost_and_water_gained_area=2.,gained_and_water_lost_area=5.)

@pytest.mark.parametrize("weights", [[-1], [np.nan], [[1]]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError): site_transition_metrics([1],[1],[0],[0],weights)


def test_region_pbc_face_ownership():
    bounds=np.array([[0,100],[0,100],[0,100.]])
    sites=np.array([[98,50,10],[98,50,90.]])
    gas=np.array([[94,50,15],[2,50,15.]])
    pts=np.array([[1,50,12],[1,50,8],[1,50,88],[1,50,92.]])
    masks, nearest=surface_region_masks(pts,gas,sites,['top_left','bottom_left'],bounds)
    assert masks['top','front'].tolist()==[True,False,False,False]
    assert masks['bottom','front'].tolist()==[False,False,True,False]
    shifted,_=surface_region_masks(pts+100,gas+100,sites+100,['top_left','bottom_left'],bounds)
    for k in masks: np.testing.assert_array_equal(masks[k],shifted[k])


def test_directed_hb_parity_and_periodic_neighbor():
    b=np.array([[0,10],[0,10],[0,10.]])
    w=np.array([[9,5,5.],[5,5,5.]])
    s=np.array([[1,5,5.]])
    wo=[np.array([[1,0,0.]]),np.array([[0,1,0.]])]
    so=[np.array([[-1,0,0.]])]
    d,a=water_surface_hbond_degrees(w,wo,s,so,b)
    assert d.tolist()==[1,0] and a.tolist()==[1,0]
    assert (d.sum(),a.sum())==water_surface_hbond_counts(w,wo,s,so,b,oo_cutoff=3.5,angle_cutoff_deg=30)


def test_empty_surface():
    d,a=water_surface_hbond_degrees(np.zeros((1,3)),[np.zeros((0,3))],np.zeros((0,3)),[],np.array([[0,10]]*3))
    assert (d+a).sum()==0


def test_unique_hydrogen_owner_oxygen_carbon_competition():
    a=assign_hydrogens_to_oxygen_or_carbon(np.array([1]),np.array([[0,0,0.]]),np.array([2]),np.array([[1.5,0,0.]]),np.array([3]),np.array([[1,0,0.]]),np.array([[0,10.]]*3),oh_cutoff_A=1.3,ch_cutoff_A=1.3)
    assert a.owner_ids.tolist()==[2]
