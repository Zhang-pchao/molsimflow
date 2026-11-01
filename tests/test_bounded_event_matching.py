from molsimflow.postprocess.bounded_event_matching import *
def row(**kwargs):
 r={k:1. for k in G_FIELDS};r.update({k:0 for k in C_FIELDS})
 r.update(branch='mixed275_A',face='1',Y=0.)
 r.update(kwargs);return r
def test_branch_and_composition_cannot_be_relaxed():
 scale={k:1. for k in G_FIELDS}
 assert greedy_caliper_pairs([row(),row(branch='mixed275_B')],scale)==[]
 assert greedy_caliper_pairs([row(),row(OH_count=1)],scale)==[]
 assert len(greedy_caliper_pairs([row(),row(OH_count=1)],scale,composition=False))==1
def test_zero_sd_requires_equality():
 scale={k:0. for k in G_FIELDS}
 assert len(greedy_caliper_pairs([row(),row()],scale))==1
 assert not greedy_caliper_pairs([row(),row(volume_A3=1.01)],scale)
def test_matching_does_not_read_outcomes():
 scale={k:1. for k in G_FIELDS}
 a=[row(Y=0),row(Y=1),row(Y=.3),row(Y=.7)]
 expected=greedy_caliper_pairs(a,scale)
 for r in a:r['Y']=100
 assert greedy_caliper_pairs(a,scale)==expected
def test_tight_speed_caliper_and_no_holdout_rank_refit():
 scale={k:1. for k in G_FIELDS}
 assert not greedy_caliper_pairs([row(),row(approach_A_ps=1.03)],scale)
 assert frozen_ecdf([0,1],[2,-1]).tolist()==[1.,0.]
