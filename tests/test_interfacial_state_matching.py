import itertools
import numpy as np
import pytest
from molsimflow.postprocess.interfacial_state_matching import match_caliper_states


def test_cardinality_wins_over_greedy_nearest_pair():
    r = match_caliper_states([[0], [.8]], [[.7], [1.6]], [.9])
    assert r["pairs"].tolist() == [[0, 0], [1, 1]]
    assert r["eligible_pair_count"] == 3


def test_empty_and_zero_support():
    r = match_caliper_states(np.empty((0, 2)), [[1, 2]], [1, 1])
    assert r["pairs"].shape == (0, 2)
    assert r["unmatched_right"].tolist() == [0]
    q = match_caliper_states([[0, 0]], [[2, 0]], [1, 1])
    assert len(q["pairs"]) == 0 and q["eligible_pair_count"] == 0


def test_boundaries_and_no_replacement():
    r = match_caliper_states([[0], [0], [3]], [[1]], [1])
    assert len(r["pairs"]) == 1
    assert r["left_candidate_counts"].tolist() == [1, 1, 0]
    assert len(r["unmatched_left"]) == 2
    assert r["normalized_gaps"].tolist() == [[1.]]


def test_units_and_permutations_preserve_optimum():
    a=np.array([[0., 2], [1, 3], [6, 1]])
    b=np.array([[.1, 2.2], [1.2, 3.1], [8, 2]])
    r=match_caliper_states(a,b,[.5,.5])
    s=match_caliper_states(a[[2,0,1]]*[10,2]+[4,-9],b[[1,2,0]]*[10,2]+[4,-9],[5,1])
    assert len(r["pairs"]) == len(s["pairs"]) == 2
    assert r["total_squared_cost"] == pytest.approx(s["total_squared_cost"])


def test_small_random_optima_against_exhaustive_search():
    rng=np.random.default_rng(38)
    for _ in range(30):
        a,b=rng.random((3,2)),rng.random((3,2))
        r=match_caliper_states(a,b,[.45,.6])
        gap=abs(a[:,None]-b[None,:])/[.45,.6]
        allowed=np.all(gap<=1,axis=2)
        best=(-1,np.inf)
        for assign in itertools.product(range(-1,3),repeat=3):
            used=[j for j in assign if j>=0]
            if len(set(used))!=len(used):continue
            pairs=[(i,j)for i,j in enumerate(assign)if j>=0]
            if any(not allowed[i,j]for i,j in pairs):continue
            cost=sum(float((gap[i,j]**2).sum())for i,j in pairs)
            if len(pairs)>best[0]or(len(pairs)==best[0]and cost<best[1]):best=(len(pairs),cost)
        assert len(r["pairs"])==best[0]
        assert r["total_squared_cost"]==pytest.approx(best[1])


@pytest.mark.parametrize("left,right,calipers",[
    ([[np.nan]],[[0]],[1]), ([[0]],[[1]],[0]),
    ([[0,1]],[[1]],[1]), ([[0]],[[1]],[-1])])
def test_invalid_contract(left,right,calipers):
    with pytest.raises(ValueError):match_caliper_states(left,right,calipers)
