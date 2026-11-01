import numpy as np
import pytest
from molsimflow.postprocess.gas_component_lineage import (
    gas_components,component_overlap,inventory_states,scaffold_translation)
from molsimflow.postprocess.droplet_translation_diagnostics import periodic_connectivity_winding

B=np.array([[0.,10.]]*3)
def test_edge_crossing_is_not_winding():
    x=np.array([[.2,1,1],[9.8,1,1],[5,5,5]])
    c=gas_components(x,B,1)
    assert c.labels.tolist()==[0,0,2]
    assert not len(c.cycle_translations)
    wrapped=(x+[3,4,6])%10
    np.testing.assert_array_equal(c.labels,gas_components(wrapped,B,1).labels)

def test_cycle_winding_matches_existing_module():
    x=np.array([[i,1,1] for i in range(10)],float)
    c=gas_components(x,B,1.1)
    assert np.any(c.cycle_translations[:,1]!=0)
    assert periodic_connectivity_winding(x,B,1.1,periodic=(True,True,True))==(True,False,False)

def test_overlap_preserves_all_members_in_merger_and_split():
    before=np.array([0,0,2,2,4]);after=np.array([0,0,0,3,3])
    o=component_overlap(before,after)
    assert o.tolist()==[[0,0,2],[2,0,1],[2,3,1],[4,3,1]]
    for k in np.unique(before):assert o[o[:,0]==k,2].sum()==sum(before==k)
    for k in np.unique(after):assert o[o[:,1]==k,2].sum()==sum(after==k)

def test_exhaustive_inventory():
    s=inventory_states([1,1,0,0],[1,0,0,0],[3,4,8,12])
    assert s.tolist()==[0,1,2,3]
    with pytest.raises(ValueError):inventory_states([0],[1],[3])

def test_scaffold_shift_uses_material_ids():
    ref=np.array([[9.9,1,1],[2,4,5],[5,6,8]])
    cur=(ref+[.3,.2,-.4])%10
    shift,res=scaffold_translation(cur,ref,B)
    np.testing.assert_allclose(shift,[.3,.2,-.4],atol=1e-12)
    np.testing.assert_allclose(res,0,atol=1e-12)
