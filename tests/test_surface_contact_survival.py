import numpy as np
import pytest
from molsimflow.postprocess.surface_contact_survival import sampled_region_survival


def test_leave_and_return_differs_from_survival():
    s=np.array([[1,1],[1,0],[1,1]],bool)
    r=sampled_region_survival(s,[0,1,2],['a']*3,[0,1,2],frame_dt=1)
    assert r[2]['mean_original_region_fraction']==1
    assert r[2]['sampled_uninterrupted_fraction']==.5
    assert r[2]['interrupted_then_recontacted_fraction']==.5
    assert r[0]['sampled_uninterrupted_fraction']==1


def test_origin_weights_and_empty_endpoints():
    s=np.array([[1,1],[1,0],[0,0]],bool)
    r=sampled_region_survival(s,[0,1,2],['a']*3,[1],frame_dt=1)[0]
    assert r['n_origins']==2
    assert r['sampled_uninterrupted_fraction']==.25


def test_seams_gaps_invalid_and_repeated_epoch():
    s=np.ones((5,2),bool)
    r=sampled_region_survival(s,[0,1,2,4,5],['a','a','b','a','a'],[1,2],frame_dt=1,valid=[1,1,1,0,1])
    assert r[0]['n_origins']==1
    assert r[1]['n_origins']==0
    assert r[1]['sampled_uninterrupted_fraction'] is None


def test_matches_brute_force_every_saved_frame():
    rng=np.random.default_rng(31);s=rng.random((23,7))<.7
    t=np.arange(23,dtype=float);t[15:]+=1
    e=np.array(['a']*10+['b']*13);valid=np.ones(23,bool);valid[5]=False
    rows=sampled_region_survival(s,t,e,np.arange(26),frame_dt=1,valid=valid)
    for row in rows:
        lag=row['lag_steps'];v=[]
        for a in range(len(s)-lag):
            b=a+lag
            if not valid[a:b+1].all() or len(set(e[a:b+1]))!=1 or not np.allclose(np.diff(t[a:b+1]),1) or not s[a].any():continue
            v.append(s[a:b+1].all(axis=0).sum()/s[a].sum())
        assert row['n_origins']==len(v)
        assert row['sampled_uninterrupted_fraction']==pytest.approx(np.mean(v)) if v else row['sampled_uninterrupted_fraction'] is None


def test_invalid_input_rejected():
    with pytest.raises(ValueError):sampled_region_survival(np.ones((3,2)),[0,1,2],['a']*3,[1],frame_dt=1)
    with pytest.raises(ValueError):sampled_region_survival(np.ones((3,2),bool),[0,1,2],['a']*3,[1],frame_dt=np.nan)
