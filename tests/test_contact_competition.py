import numpy as np
import pytest
from molsimflow.postprocess.interfacial_contact_geometry import (
    SurfaceGrid, prescribed_contact_shift, site_weighted_load)
from molsimflow.postprocess.interfacial_contact_renewal import sampled_run_support
from molsimflow.postprocess.interfacial_velocity_profile import height_velocity_moments


def test_shift_fractional_periodic_holes():
    m=np.zeros((9,7),bool);m[1:6,1:5]=True;m[3,2]=False;m[8,6]=True
    for d in [0,0.3,2.4,-1.6,18]:
        r=prescribed_contact_shift(m,np.array([2.,3.]),d)
        assert abs(r['area_residual']) < 1e-12
        assert r['gained_area']==pytest.approx(r['lost_area'])
        assert np.all((r['shifted']>=0)&(r['shifted']<=1))
        shifted=prescribed_contact_shift(np.roll(m,4,axis=0),np.array([2.,3.]),d)
        assert np.allclose(shifted['gain'],np.roll(r['gain'],4,axis=0))
    assert not prescribed_contact_shift(m,np.ones(2),0)['gain'].any()
    assert np.array_equal(prescribed_contact_shift(m,np.ones(2),2)['shifted'],
                          np.roll(m,2,axis=0))


def test_uniform_strip_analytical_area():
    m=np.zeros((10,4),bool);m[2:6,:]=True
    r=prescribed_contact_shift(m,np.array([1.,2.]),0.4)
    assert r['gained_area']==pytest.approx(3.2)


def test_site_load_bounds_and_no_duplicate_ownership():
    owner=np.array([[0,0],[1,1]])
    g=SurfaceGrid(np.array([[0,2],[0,2],[0,1]]),np.ones(2),
                  np.zeros((2,2,2)),owner,2)
    for w in [np.array([[1,0],[.5,.5]]),np.zeros((2,2)),np.ones((2,2))]:
        r=site_weighted_load(w,g,np.array([4.,8.]))
        assert r['lower']<=r['equivalent']<=r['upper']
    assert site_weighted_load(np.ones((2,2)),g,np.array([4.,8.]))['equivalent']==12


def test_runs_require_elapsed_span_and_reset_at_gaps():
    t=np.array([0,5,10,15,20,30,35,40,45,50])
    s=np.ones((10,2),bool);s[2,1]=False
    r=sampled_run_support(s,t,span_steps=20,cadence_steps=5)
    assert np.flatnonzero(r[:,0]).tolist()==[4,9]
    assert np.flatnonzero(r[:,1]).tolist()==[9]


def test_identity_change_does_not_clear_spatial_occupancy():
    counts=np.ones((5,1),int)
    assert not sampled_run_support(counts==0,np.arange(5)*5,
                                   span_steps=20,cadence_steps=5).any()


def test_profile_moments_partition_and_empty_bins():
    h=np.array([-.1,0,1,2,3,4.]);v=np.arange(18.).reshape(6,3)
    a=height_velocity_moments(h,v,np.array([0.,1,2,3,4]))
    assert np.array_equal(a['count'],[1,1,1,1])
    assert np.allclose(a['velocity_sum'].sum(0),v[1:5].sum(0))
    assert a['excluded_count']==2
    x=height_velocity_moments(h,v,np.array([0.,1,2,3,4]),np.arange(6)%2==0)
    y=height_velocity_moments(h,v,np.array([0.,1,2,3,4]),np.arange(6)%2==1)
    assert np.array_equal(x['count']+y['count'],a['count'])
    assert np.allclose(x['velocity_sum']+y['velocity_sum'],a['velocity_sum'])


@pytest.mark.parametrize("bad",[np.array([0.,0.]),np.array([2.,1.])])
def test_profile_rejects_invalid_edges(bad):
    with pytest.raises(ValueError):
        height_velocity_moments(np.zeros(1),np.zeros((1,3)),bad)


def test_shift_rejects_nonboolean():
    with pytest.raises(ValueError):
        prescribed_contact_shift(np.ones((2,2)),np.ones(2),1)
