import numpy as np
from molsimflow.postprocess.water_column_balance import column_membership,ordered_column_balance
L=np.array([4.,4.,20.]);S=np.array([2,2]);top=np.zeros(4);support=np.ones(4,bool)
def balance(old,new,q0,q1,g0,g1,s0=support,s1=support):
 return ordered_column_balance(np.array(old,float),np.array(new,float),q0,q1,top,top,np.full(4,g0),np.full(4,g1),s0,s1,L,S)
def test_stationary_water_swept_out_is_not_particle_exit():
 _,m,d=balance([[1,1,3]],[[1,1,3]],[1],[1],5,3)
 assert d[:,0].tolist()==[0,0,-1,0]
def test_chemical_loss_without_motion():
 _,m,d=balance([[1,1,2]],[[1,1,2]],[1],[0],5,5)
 assert d[:,0].tolist()==[0,0,0,-1]
def test_lateral_cell_transfer_and_vertical_exit():
 _,m,d=balance([[1,1,2]],[[3,1,6]],[1],[1],5,5)
 assert d[0,0]==-1 and d[0,2]==1 and d[1,2]==-1
def test_missing_envelope_is_not_zero_water():
 common,m,d=balance([[1,1,2]],[[1,1,2]],[1],[1],5,5,s1=np.zeros(4,bool))
 assert not common.any() and not d.any()
def test_periodic_wrapping():
 assert column_membership(np.array([[-3.,1.,2.]]),[1],top,np.full(4,5),support,L,S).tolist()==[0]
