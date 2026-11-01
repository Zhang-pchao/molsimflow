import numpy as np
from molsimflow.postprocess.prospective_contact_regions import (
 uncontacted_front_candidates,complete_window_indices,event_family_labels,temporal_family_allocation)
def test_nonadvancing_regions_are_registered_without_outcomes():
 grid=np.array([[0,0],[7,0],[13,0],[19,0]],float)
 eligible,d=uncontacted_front_candidates(grid,[[0,0]],[[0,0]],[100,100])
 assert eligible.tolist()==[False,True,False,False]
def test_periodic_candidates():
 eligible,_=uncontacted_front_candidates([[95,0],[90,0]],[[1,0]],[[1,0]],[100,100])
 assert eligible.tolist()==[False,True]
def test_complete_windows_reject_missing_and_boundaries():
 steps=np.arange(0,100001,200)
 assert complete_window_indices(steps,40000)==(100,400)
 assert complete_window_indices(steps,0) is None
 assert complete_window_indices(steps[steps!=50000],40000) is None
def test_transitive_cascade_is_not_randomly_split():
 anchors=np.array([40000,80000,120000])
 labels=event_family_labels(anchors,np.array([[0,0],[8,0],[16,0]]),[{1},{1,2},{2}],[100,100])
 assert len(set(labels))==1
 assert temporal_family_allocation(anchors,labels,100000).tolist()==['purged']*3
def test_separate_seed_and_family_holdout():
 anchors=np.array([40000,200000])
 labels=event_family_labels(anchors,np.zeros((2,2)),[{1},{2}],[100,100])
 assert len(set(labels))==2
 assert temporal_family_allocation(anchors,labels,120000).tolist()==['discovery','holdout']
