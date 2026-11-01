import numpy as np
import pytest
from molsimflow.postprocess.projected_gap_profile import projected_gap_columns,gap_cdf

def observe(gas,water,**kw):
    return projected_gap_columns(np.array(gas,float).reshape(-1,3),np.array(water,float).reshape(-1,3),[[2.,2.]], [0.], [[0,20],[0,20],[0,100]],[1,1],[0,0],**kw)

def test_far_bubble_is_retained_and_oxygen_between():
    a=observe([[2,2,20],[3,2,21],[2,3,22]],[[2,2,1],[2,2,10],[2,2,30]])
    assert a.projected[0] and a.supported[0] and a.gap[0]==22
    assert a.interposed_count[0]==2

def test_isolated_tail_does_not_define_envelope():
    a=observe([[2,2,3],[2,2,20],[3,2,21],[2,3,22]],[])
    assert a.gap[0]==22

def test_unsupported_columns_remain_in_denominator():
    total,wet,empty,unknown=gap_cdf([4,np.nan,8],[0,0,2],[1,1,1],[1,0,1],[4,8])
    np.testing.assert_allclose(total,[1/3,2/3]);np.testing.assert_allclose(wet,[0,1/3])
    np.testing.assert_allclose(wet+empty,total);assert unknown==pytest.approx(1/3)

def test_strict_oxygen_clearance():
    a=observe([[2,2,3],[3,2,3],[2,3,3]],[[2,2,.5],[2,2,2.5],[2,2,1.5]])
    assert a.interposed_count[0]==1

def test_xy_pbc_and_translation():
    gas=np.array([[19,2,10],[.5,2,10],[1,2,10]])
    args=(np.array([[.5,2,2]]),np.array([[0,2.]]),np.array([0.]),np.array([[0,20],[0,20],[0,100]]),[1,1],[0,0])
    a=projected_gap_columns(gas,*args)
    b=projected_gap_columns(gas+np.array([20,0,0]),*args)
    assert a.supported[0] and a.gap[0]==10
    np.testing.assert_array_equal(a.interposed_count,b.interposed_count)

def test_ppf_does_not_connect_opposite_z_faces():
    args=([],[[2,2]], [0.], [[0,20],[0,20],[0,20]],[1,1],[0,0])
    gas=np.array([[2,2,1],[2,2,18],[2,3,19.]])
    a=projected_gap_columns(gas,np.empty((0,3)),*args[1:],periodic_z=True)
    b=projected_gap_columns(gas,np.empty((0,3)),*args[1:],periodic_z=False)
    assert a.supported[0] and not b.supported[0]

def test_missing_geometry_rejected():
    with pytest.raises(ValueError):observe([[np.nan,0,1]],[])

def test_zero_envelope_is_unresolved():
    a=observe([[2,2,0],[3,2,0],[2,3,0]],[])
    assert a.projected[0] and not a.supported[0] and np.isnan(a.gap[0])
