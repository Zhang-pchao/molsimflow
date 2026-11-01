import numpy as np
import pytest
from molsimflow.postprocess.surface_contact_memory import supported_material_regions
from molsimflow.postprocess.surface_contact_loss import diagnose_region_support_losses


def test_periodic_counts_include_isolated_near_gas_and_shell_boundaries():
    xyz=np.array([[.1,2,3],[39.5,2,3],[.5,2.5,3],[20,20,3]],float)
    out=supported_material_regions(xyz,np.array([4.,4.,4.,4.1]),np.full(4,2.),np.array([[0,2],[20,20],[30,30]]),np.array([[0,40]]*3))
    assert out.near_counts.tolist()==[3,1,0]
    assert out.main_near_counts.tolist()==[3,0,0]
    assert out.shell_counts[1].tolist()==[3,0,0]
    assert out.shell_counts[2].tolist()==[3,1,0]
    assert out.main_shell_counts[2].tolist()==[3,0,0]
    assert out.supported[1].tolist()==[True,False,False]


def test_exclusive_diagnostic_categories_are_not_all_empty():
    s=np.array([[1]*5,[0]*5],bool)
    near=np.array([[3]*5,[0,2,2,2,3]])
    shell=np.array([[3]*5,[0,0,1,2,3]])
    main=np.array([[3]*5,[0,0,0,2,3]])
    r=diagnose_region_support_losses(s,near,shell,main,[0,.5],['a','a'],frame_dt=.5)
    assert [x['diagnostic'] for x in r]==['EMPTY_NEAR_REGION','NEAR_GAS_NO_SHELL_CONTACT','SHELL_CONTACT_OUTSIDE_MAIN','MAIN_CONTACT_BELOW_SUPPORT','MAIN_CONTACT_DISCONNECTED']
    assert sum(x['near_gas_present'] for x in r)==4


def test_only_adjacent_losses_not_seams_gaps_or_repeated_labels():
    s=np.array([[1],[0],[1],[0],[1],[0]],bool);n=np.full(s.shape,3)
    r=diagnose_region_support_losses(s,n,n,n,[0,.5,1,2,2.5,3],['a','a','a','a','b','a'],frame_dt=.5)
    assert [(x['start_index'],x['end_index']) for x in r]==[(0,1)]


def test_unknown_and_inconsistent_counts_rejected():
    s=np.array([[1],[0]],bool);n=np.array([[3.],[np.nan]])
    with pytest.raises(ValueError):diagnose_region_support_losses(s,n,n,n,[0,1],['a','a'],frame_dt=1)
    with pytest.raises(ValueError):diagnose_region_support_losses(s,[[3],[1]],[[3],[2]],[[3],[2]],[0,1],['a','a'],frame_dt=1)


def test_no_losses_has_explicit_empty_result():
    s=np.ones((2,2),bool);n=np.full(s.shape,3)
    assert diagnose_region_support_losses(s,n,n,n,[0,1],['a','a'],frame_dt=1)==[]
