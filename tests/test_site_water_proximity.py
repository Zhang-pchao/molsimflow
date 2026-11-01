import numpy as np
import pytest
from molsimflow.postprocess.site_water_proximity import site_water_proximity,water_presence_category
B=np.array([[0.,20.]]*3)

def test_pbc_qualified_ambiguity_and_identity():
    result=site_water_proximity([[.1,1,1],[10,1,1]],[[19.8,1,1],[10.1,1,1],[10.2,1,1]],[20,21,22],[2,2,1],[False,True,False],B)
    np.testing.assert_allclose(result.qualified_distance,[.3,9.8])
    np.testing.assert_array_equal(result.qualified_id,[20,20])
    np.testing.assert_array_equal(water_presence_category(result.qualified_distance,result.other_distance),[1,2])

def test_presence_priority_boundary_empty_and_translation():
    np.testing.assert_array_equal(water_presence_category([3.5,4,4],[1,3.5,4]),[1,2,0])
    a=site_water_proximity([[0,0,0]],[],[],[],[],B)
    assert np.isinf(a.qualified_distance).all() and a.qualified_id[0]==-1
    assert water_presence_category(a.qualified_distance,a.other_distance)[0]==0
    a=site_water_proximity([[.1,1,1]],[[19.8,1,1]],[7],[2],[False],B)
    b=site_water_proximity([[20.1,21,21]],[[39.8,21,21]],[7],[2],[False],B)
    np.testing.assert_allclose(a.qualified_distance,b.qualified_distance)

def test_duplicate_solution_identity_rejected():
    with pytest.raises(ValueError):site_water_proximity([[0,0,0]],[[1,0,0],[2,0,0]],[7,7],[2,2],[False,False],B)
