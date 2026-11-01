import numpy as np
import pytest
from molsimflow.postprocess.surface_approach_paths import approach_paths

def run(d,m=None):
 d=np.asarray(d,float)[:,None]
 return approach_paths(d,np.ones_like(d,bool) if m is None else np.asarray(m,bool)[:,None],np.arange(len(d)))

def test_contact_return_and_nonoverlap():
 r=run([9,7,3,5,9,7,9])
 assert [(x['start'],x['end'],x['outcome']) for x in r]==[(0,2,'contact'),(4,6,'return')]

def test_membership_and_right_censor():
 r=run([9,7,6,9,7],[1,1,0,1,1])
 assert [x['outcome'] for x in r]==['membership_censored','window_censored']

def test_initial_inner_and_new_main_do_not_invent_origin():
 assert run([3,5,7])==[]
 assert run([9,7,3],[0,1,1])==[]

def test_one_interval_contact_and_exact_boundaries():
 assert run([8.5,4])[0]['outcome']=='contact'
 assert run([8.5,7,8.5])[0]['outcome']=='return'

def test_irregular_or_missing_rejected():
 with pytest.raises(ValueError):approach_paths([[9],[7]],[[1],[1]],[0,0])
 with pytest.raises(ValueError):run([9,float('nan')])
