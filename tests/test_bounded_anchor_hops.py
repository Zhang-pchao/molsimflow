import numpy as np
import pytest
from molsimflow.postprocess.contact_graph import bounded_anchor_hops

def test_chain_bound_and_disconnected():
    r = bounded_anchor_hops(6, np.array([[0,1],[1,2],[2,3],[4,5]]), [0])
    np.testing.assert_array_equal(r["distance"], [0,1,2,-1,-1,-1])
    np.testing.assert_array_equal(r["parent"], [0,0,1,-1,-1,-1])
    np.testing.assert_array_equal(r["anchor"], [0,0,0,-1,-1,-1])

def test_deterministic_ties_duplicates_cycles_and_path():
    edges = np.array([[0,2],[1,2],[2,3],[3,0],[0,2],[4,4]])
    a = bounded_anchor_hops(5, edges, [1,0,1], 3)
    b = bounded_anchor_hops(5, edges[::-1,::-1], [0,1], 3)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert a["anchor"][2] == 0
    for node in np.flatnonzero(a["distance"] > 0):
        parent = a["parent"][node]
        assert a["distance"][parent] == a["distance"][node] - 1
        assert any(set(e) == {node,parent} for e in edges)

def test_empty_and_zero_depth():
    r = bounded_anchor_hops(0, [], [])
    assert all(x.size == 0 for x in r.values())
    np.testing.assert_array_equal(bounded_anchor_hops(3, [[0,1]], [], 2)["distance"], [-1]*3)
    np.testing.assert_array_equal(bounded_anchor_hops(3, [[0,1]], [0], 0)["distance"], [0,-1,-1])

@pytest.mark.parametrize("n,e,a,h", [
    (-1,[],[],2), (2.0,[],[],2), (2,[[0,2]],[0],2),
    (2,[[0.0,1.0]],[0],2), (2,[[0,1]],[-1],2),
    (2,[[0,1]],[0],-1), (2,[[0,1]],[0],True),
    (2,[[0,1]],[0.5],2), (2,[0,1],[0],2),
])
def test_invalid_inputs(n,e,a,h):
    with pytest.raises(ValueError):
        bounded_anchor_hops(n,e,a,h)

def test_matches_exhaustive_shortest_lengths():
    rng=np.random.default_rng(59)
    for _ in range(15):
        n=15;edge=np.argwhere(np.triu(rng.random((n,n)) < .13,1))
        dist=np.full((n,n),n+1);np.fill_diagonal(dist,0)
        for a,b in edge:dist[a,b]=dist[b,a]=1
        for k in range(n):dist=np.minimum(dist,dist[:,k,None]+dist[None,k,:])
        expected=dist[:,[2,7]].min(1);expected[expected>3]=-1
        np.testing.assert_array_equal(bounded_anchor_hops(n,edge,[2,7],3)["distance"],expected)
