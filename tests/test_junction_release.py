"""Regression tests for junction geometry, lineage and incremental admission."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from molsimflow.io.lammps_dump import LammpsDumpFrame, write_lammps_dump_frame
from molsimflow.postprocess.junction_release_metrics import frame_metrics, site_area_weights
from molsimflow.postprocess.junction_release_events import assemble_series, contact_state
from molsimflow.postprocess.trajectory_feature_cache import FeatureCache
from molsimflow.postprocess.junction_release import run_contract


def fixture():
    xyz = np.array([[5,10,10],[15,10,10],[5,10,2],[15,10,2],
                    [4.45,10,12],[5.55,10,12],[4.45,11,12],[5.55,11,12],
                    [10,10,6],[5,10,14]], dtype=float)
    types = np.array([7,7,7,7,3,3,3,3,8,2])
    ref = dict(types=types,lengths=np.array([20.,20.,30.]),nitrogen_ids=np.arange(5,9),
               site_ids=np.arange(1,5),site_groups=np.array(['top_left','top_right','bottom_left','bottom_right']),
               site_weights_1A=np.full(4,200.),site_weights_2A=np.full(4,200.),
               fixed_ids=np.array([9]),fixed_positions=xyz[[8]],silicon_ids=np.array([9]),waterO_ids=np.array([10]))
    params = dict(cluster_cutoff_A=5.5,contact_cutoffs_A=[3.5,4.,4.5],internal_seam_x_A=10.,
                  water_neighbor_cutoff_A=3.5,stride_steps=10000,timestep_ps=.0005,
                  hold_start_step=0,minimum_membership_overlap=.8,minimum_main_n2=2,
                  gap_screen_A=5.,persistence_ps=100.)
    return xyz,ref,params


def make_frame(xyz,ref,step=0):
    fields=('id','type','x','y','z','vx','vy','vz','v_sicn')
    rows=tuple(tuple(map(str,[i+1,ref['types'][i],*p,0.,0.,0.,4.])) for i,p in enumerate(xyz))
    return LammpsDumpFrame(0,step,np.column_stack((np.zeros(3),ref['lengths'])),
                          'ITEM: BOX BOUNDS pp pp pp',fields,rows)


def test_four_faces_and_z_periodic_contact():
    xyz,ref,p=fixture()
    row=frame_metrics(make_frame(xyz,ref),ref,p,Path('synthetic'))
    assert row['top_left_contacts_4p0']==2
    assert row['top_right_contacts_4p0']==0
    assert row['bottom_left_contacts_4p0']==0
    assert row['top_left_site_area_4p0_A2']==200.
    xyz[4:8,2]=29.9
    row=frame_metrics(make_frame(xyz,ref),ref,p,Path('synthetic'))
    assert row['bottom_left_contacts_4p0']==2
    assert row['top_left_contacts_4p0']==0
    assert contact_state(row,2,5.)=='BOTTOM_FACE_ATTACHED'


def test_site_weights_partition_periodic_area():
    weights=site_area_weights(np.array([[.1,4.],[5.1,4.]]),np.array([10.,8.,30.]),1.)
    assert weights.sum()==pytest.approx(80.)
    assert np.allclose(weights,[40.,40.])


def test_identity_and_nonfinite_rejected():
    xyz,ref,p=fixture();bad=copy.deepcopy(ref);bad['types'][0]=8
    with pytest.raises(ValueError,match='types'):
        frame_metrics(make_frame(xyz,ref),bad,p,Path('synthetic'))
    xyz[5,1]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        frame_metrics(make_frame(xyz,ref),ref,p,Path('synthetic'))


def detached_rows():
    xyz,ref,p=fixture();row=frame_metrics(make_frame(xyz,ref),ref,p,Path('synthetic'))
    for group in ['top_left','top_right','bottom_left','bottom_right']:
        row[group+'_contacts_4p0']=0
    row['min_bubble_surface_distance_A']=6.
    row['_gas_bits']='00';row['_water_bits']='00';row['_site_count']=4
    return [dict(copy.deepcopy(row),step=i*10000,reseed_boundary=False) for i in range(23)],p


def test_persistence_requires_time_and_resets_at_reseed():
    rows,p=detached_rows();series,episodes=assemble_series(rows,p)
    assert episodes[0]['persistent_screen']
    rows[20]['reseed_boundary']=True
    series,episodes=assemble_series(rows,p)
    assert not any(e['persistent_screen'] for e in episodes)
    assert series[20]['bubble_vx_interval_A_ps'] is None


def test_missing_frame_and_identity_jump_reset():
    rows,p=detached_rows();rows.pop(10)
    _,episodes=assemble_series(rows,p)
    assert not any(e['persistent_screen'] for e in episodes)
    rows,p=detached_rows();rows[10]['_members']=[100,101]
    series,_=assemble_series(rows,p)
    assert not series[10]['temporal_interval_valid']
    with pytest.raises(ValueError,match='Duplicate'):
        assemble_series([rows[0],rows[0]],p)


def test_cache_rejects_changed_definition_or_same_step_data(tmp_path):
    c=FeatureCache(tmp_path/'c.db','version-a');c.put('source',10,'a',{'value':2})
    c.put('source',10,'a',{'value':2})
    with pytest.raises(ValueError,match='changed'):
        c.put('source',10,'b',{'value':2})
    c.close()
    with pytest.raises(ValueError,match='identity changed'):
        FeatureCache(tmp_path/'c.db','version-b')


def test_incremental_append_reuses_frames_and_verifies_boundary(tmp_path):
    xyz,ref,p=fixture();p['stride_steps']=1
    source=tmp_path/'frames.dump'
    with source.open('w') as stream:
        for step in range(2):write_lammps_dump_frame(stream,make_frame(xyz,ref,step))
    reference=tmp_path/'reference.npz';np.savez_compressed(reference,**ref)
    contract=dict(schema_version=1,mode='production',analysis_key='test',case_id='test',label='Synthetic',
                  reference_npz=str(reference),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  parameters=p,first_step=0,last_step=1,sources=[dict(path=str(source),source_id='test-source',
                  segment_id='s0',inode=source.stat().st_ino,admitted_size=source.stat().st_size,
                  head_bytes=32,head_sha256=hashlib.sha256(source.read_bytes()[:32]).hexdigest(),
                  minimum_step=0,maximum_step=1,native_start_step=0,reseed=False)])
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract))
    first=run_contract(path,tmp_path/'out');assert first['computed_frames']==2
    replay=run_contract(path,tmp_path/'out');assert replay['computed_frames']==0
    with source.open('a') as stream:write_lammps_dump_frame(stream,make_frame(xyz,ref,2))
    contract['last_step']=2;contract['sources'][0]['maximum_step']=2
    path.write_text(json.dumps(contract))
    added=run_contract(path,tmp_path/'out')
    assert added['computed_frames']==1 and added['reused_frames']==2
    assert added['revalidated_cached_boundary_frames']==1
