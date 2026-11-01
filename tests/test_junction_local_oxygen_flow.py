import numpy as np
import pytest
from molsimflow.postprocess.junction_local_oxygen_flow import oxygen_regions


def example(shift=0.):
    bounds=np.array([[0.,100.],[0.,100.],[0.,100.]])
    sites=np.array([[40.,50.,20.],[60.,50.,20.],[40.,50.,10.],[60.,50.,10.]])+shift
    groups=np.array(['top_left','top_right','bottom_left','bottom_right'])
    gas=np.array([[40.,48.,25.],[60.,52.,25.],[45.,49.,25.],[55.,51.,25.]])+shift
    oxygen=np.array([[59.,50.,22.],[59.,50.,8.]])+shift
    velocities=np.array([[.1,.2,.3],[-.1,-.2,-.3]])
    return oxygen_regions(oxygen,velocities,gas,sites,groups,bounds,minimum_oxygen=1)


def test_faces_and_units():
    rows=example();top=next(r for r in rows if r['face']=='top' and r['region']=='front');bottom=next(r for r in rows if r['face']=='bottom' and r['region']=='front')
    assert top['original_oxygen_count']==bottom['original_oxygen_count']==1
    assert top['oxygen_vx_mps']==pytest.approx(10.)
    assert bottom['oxygen_vx_mps']==pytest.approx(-10.)
    assert top['right_oxygen_count']==1


def test_full_periodic_translation():
    assert example()==example(100.)


def test_empty_region_is_missing_not_zero():
    rows=example();rear=next(r for r in rows if r['face']=='top' and r['region']=='rear')
    assert rear['status']=='INSUFFICIENT_OCCUPANCY' and rear['oxygen_vx_mps'] is None
