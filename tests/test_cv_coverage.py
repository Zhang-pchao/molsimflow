import numpy as np
import pytest
from molsimflow.postprocess.cv_coverage import path_grid_coverage, compare_grid_coverage


def test_complete_paths_not_correlated_beads_and_threshold_time():
    x=np.array([[[.1,.1],[.2,.2]],[[.3,.3],[1.2,.1]],[[1.1,.1],[1.4,.2]]])
    r=path_grid_coverage(x,[.1,.2,.3],[1,1],minimum_paths=2,
                         block_width=.2,progress_times=[0,.1,.2,.3])
    np.testing.assert_array_equal(r.cells,[[0,0],[1,0]])
    np.testing.assert_array_equal(r.path_counts,[2,2])
    np.testing.assert_allclose(r.threshold_times,[.2,.3])
    np.testing.assert_array_equal(r.growth_counts,[0,0,1,2])
    np.testing.assert_array_equal(r.block_counts,[1,2])


def test_common_exclusive_empty_and_grid_guard():
    a=path_grid_coverage([[.1,.1],[1.1,.1]],[1,2],[1,1],minimum_paths=1)
    b=path_grid_coverage([[1.1,.1],[2.1,.1]],[1,2],[1,1],minimum_paths=1)
    assert compare_grid_coverage(a,b)==dict(first=2,second=2,common=1,only_first=1,
        only_second=1,union=3,second_over_first=1.,jaccard=1/3)
    e=path_grid_coverage(np.empty((0,16,2)),[],[1,1],minimum_paths=1,progress_times=[0,1])
    np.testing.assert_array_equal(e.growth_counts,[0,0])
    assert compare_grid_coverage(e,e)['jaccard'] is None
    shifted=path_grid_coverage([[.1,.1]],[1],[1,1],origin_fraction=[.5,0])
    with pytest.raises(ValueError):compare_grid_coverage(a,shifted)


def test_negative_coordinates_boundaries_and_validation():
    r=path_grid_coverage([[-.001,0],[0,0],[.5,.5]],[0,1,2],[.5,.5],minimum_paths=1)
    np.testing.assert_array_equal(r.cells,[[-1,0],[0,0],[1,1]])
    for times in ([1,1],[2,1]):
        with pytest.raises(ValueError):path_grid_coverage([[0,0],[0,0]],times,[1,1])
    with pytest.raises(ValueError):path_grid_coverage([[np.nan,0]],[1],[1,1])
    with pytest.raises(ValueError):path_grid_coverage([[0,0]],[1],[0,1])
