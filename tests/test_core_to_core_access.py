import numpy as np
from molsimflow.postprocess.core_to_core_access import core_labels, inventory, first_passages, excursions

def test_return_does_not_reset_first_passage():
    a = np.array([1,0,1,0,2])[:,None]
    assert first_passages(a)[0] == dict(molecule=0,start=0,end=4,arrived=True,initial=True)
    assert [r["outcome"] for r in excursions(a)] == ["return","hit"]

def test_censor_no_origin_and_late_origin():
    a = np.array([[0,0,0],[1,0,0],[0,0,1],[0,2,0]])
    r = first_passages(a)
    assert len(r)==2 and all(not x["arrived"] for x in r)
    assert [x["start"] for x in r]==[1,2]
    assert [x["outcome"] for x in excursions(a)]==["censored","censored"]

def test_direct_jump_and_reverse():
    a = np.array([1,2,0,2,1])[:,None]
    assert excursions(a)[0]["end"] == 1
    assert [r["outcome"] for r in excursions(a,2,1)]==["return","hit"]

def test_inventory_is_not_gross_crossing_rate():
    dense = np.array([0,2,0,2,2])[:,None]
    d = inventory(dense); coarse = inventory(dense[::2])
    assert d["entries"].sum()==2 and coarse["entries"].sum()==1
    assert not np.any(d["residual"]) and not np.any(coarse["residual"])
    assert d["count"][-1]==coarse["count"][-1]

def test_water_origin_and_disjoint_geometry():
    a=np.array([1,0,1,0,2])[:,None]
    ok=np.array([False,False,True,True,True])[:,None]
    assert first_passages(a,ok)[0]["start"]==2
    b=dict(r_lo_A=32,r_hi_A=36,h_lo_A=8.5,h_hi_A=20.5)
    t=dict(radius_A=10,h_lo_A=.5,h_hi_A=8.5)
    assert core_labels(np.array([[33,0,10],[5,0,3],[15,0,10]]),b,t).tolist()==[1,2,0]
