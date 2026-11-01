import numpy as np
from molsimflow.postprocess.transient_flow_observers import (
    periodic_density,density_shift,velocity_moments,contact_envelope,coherent_center)
def test_density_translation_wrap_and_permutation():
    rng=np.random.default_rng(918)
    x=rng.normal(58,3,500)
    for spacing in [.5,1.]:
        a=periodic_density(x,64,spacing)
        for shift in [-7.3,8.7,31.]:
            b=periodic_density((x+shift)[::-1]+64,64,spacing)
            r=density_shift(a,b,64)
            assert abs(r['shift_A']-shift)<.06
            assert r['correlation']>.99
def test_velocity_galilean_invariance():
    rng=np.random.default_rng(12)
    m=rng.uniform(1,16,1000);v=rng.normal(0,5,(1000,3))
    a=velocity_moments(m,v);b=velocity_moments(m,v+[9,-3,2])
    assert np.isclose(a['kinetic_T_K'],b['kinetic_T_K'])
    assert np.isclose(b['vx_A_ps']-a['vx_A_ps'],9)
def test_edges_separate_translation_from_deformation():
    x=np.linspace(-8,8,101);d=np.full(len(x),3.7)
    a=contact_envelope(x,d,4);b=contact_envelope(x+3,d,4)
    c=contact_envelope(1.5*x,d,4)
    assert np.isclose(b['midpoint_A']-a['midpoint_A'],3)
    assert np.isclose(b['halfwidth_A'],a['halfwidth_A'])
    assert np.isclose(c['midpoint_A'],a['midpoint_A'])
    assert np.isclose(c['halfwidth_A'],1.5*a['halfwidth_A'])
def test_center_wrap_and_rigid_reference_change():
    L=np.array([60,60,60])
    x=np.array([[59,1,20],[1,1,20],[0,2,20.]])
    a,res=coherent_center(x,np.array([60,1,20]),L)
    b,resb=coherent_center((x+[7,8,9])%L,np.array([67,9,29]),L)
    assert np.allclose(b-a,[7,8,9])
    assert np.allclose(res,resb)
