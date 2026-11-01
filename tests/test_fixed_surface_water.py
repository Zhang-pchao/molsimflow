import numpy as np
from molsimflow.postprocess.fixed_surface_water import oxygen_patch_members,occupation_change

def test_fixed_patch_pbc_and_vertical_bounds():
 x=np.array([[9.9,0,0],[0,0,3],[0,0,-.1],[4.1,0,1]])
 assert oxygen_patch_members(x,[0,0],0,[10,10]).tolist()==[0]

def test_reaction_is_not_oxygen_exit():
 r=occupation_change([1,2],[2,2],[1,2],[1,3])
 assert r['delta_O']==0 and r['O_left']==0
 assert r['delta_H2']==-2 and r['H2_retained_category_change']==-2

def test_exchange_identity_balance():
 r=occupation_change([1,2],[2,1],[2,3],[2,2])
 assert r['delta_O']==0 and r['O_left']==r['O_entered']==1
 assert r['H2_spatial_change']==0 and r['H2_retained_category_change']==1
