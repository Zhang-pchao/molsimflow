"""Tests for physical support, source seams, and endpoint-versus-survival meaning."""
import numpy as np
import pytest
from molsimflow.postprocess.surface_contact_memory import (
    supported_material_regions, region_recontact_curve,
)

def test_endpoint_return_is_not_continuous_survival():
    support = np.array([[1, 0], [0, 1], [1, 0]], dtype=bool)
    rows = region_recontact_curve(support, [0, .5, 1], ["a"]*3, [0, 1, 2], frame_dt=.5)
    assert rows[0]["mean_original_region_fraction"] == 1
    assert rows[1]["mean_original_region_fraction"] == 0
    assert rows[2]["mean_original_region_fraction"] == 1

def test_all_origins_equal_weight_and_empty_endpoints():
    support = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
    row = region_recontact_curve(support, [0, 1, 2], ["a"]*3, [1], frame_dt=1)[0]
    assert row["n_origins"] == 2
    assert row["mean_original_region_fraction"] == .25

def test_epoch_labels_do_not_rejoin_across_seam():
    s = np.ones((3, 2), dtype=bool)
    row = region_recontact_curve(s, [0, 1, 2], ["a", "b", "a"], [2], frame_dt=1)[0]
    assert row["n_origins"] == 0
    assert row["mean_original_region_fraction"] is None

@pytest.mark.parametrize("times,valid", [
    ([0, 1, 3], [True]*3), ([0, 1, 2], [True, False, True]),
])
def test_gaps_and_unknowns_censor_entire_pair(times, valid):
    row = region_recontact_curve(np.ones((3, 1), bool), times, ["a"]*3, [2],
                                 frame_dt=1, valid=valid)[0]
    assert row["n_origins"] == 0

def test_permutation_of_region_identity_is_invariant():
    s = np.array([[1,0,1], [0,1,1], [1,1,0]], dtype=bool)
    args = ([0,1,2], ["a"]*3, [0,1,2])
    assert region_recontact_curve(s,*args,frame_dt=1) == region_recontact_curve(s[:,[2,0,1]],*args,frame_dt=1)

def test_periodic_local_support_and_main_cluster_only():
    box = np.array([[0,40],[0,40],[0,40]], float)
    gas = np.array([[.1,2,3],[39.5,2,3],[.5,2.5,3],[20,20,3]], float)
    out = supported_material_regions(gas, np.ones(4)*3.8, np.ones(4)*2,
                                     np.array([[0,2],[20,20]]), box)
    assert out.supported.tolist() == [[False,False],[True,False],[True,False]]
    assert out.main.tolist() == [True,True,True,False]
    shifted = supported_material_regions(gas+np.array([40,0,0]), np.ones(4)*3.8,
                        np.ones(4)*2, np.array([[0,2],[20,20]]), box)
    assert np.array_equal(out.supported,shifted.supported)

def test_nearby_but_disconnected_molecules_do_not_supply_support():
    box = np.array([[0,40],[0,40],[0,40]], float)
    # The global gas cluster is connected through high molecules, but near
    # contact candidates are separated beyond the local connection cutoff.
    gas = np.array([[0,0,3],[6,0,3],[3,0,7]], float)
    out = supported_material_regions(gas, np.array([3.,3.,6.]), np.array([2.,2.,6.]),
                                     np.array([[3,0]]), box)
    assert not out.supported.any()

def test_missing_geometry_and_nonboolean_support_are_rejected():
    with pytest.raises(ValueError,match="Missing geometry"):
        supported_material_regions(np.array([[0.,0.,np.nan]]),np.ones(1),np.ones(1),
                                   np.array([[0.,0.]]),np.array([[0,20]]*3))
    with pytest.raises(ValueError,match="boolean"):
        region_recontact_curve(np.ones((2,1)),[0,1],["a","a"],[1],frame_dt=1)

def test_empty_initial_state_has_no_denominator():
    row = region_recontact_curve(np.zeros((3,2),bool),[0,1,2],["a"]*3,[1],frame_dt=1)[0]
    assert row["n_origins"] == 0
    assert row["mean_original_region_fraction"] is None
