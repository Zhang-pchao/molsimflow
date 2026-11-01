import numpy as np
from molsimflow.postprocess.gas_solid_identity import nearest_solid_identity


def test_periodic_bruteforce_and_permuted_solids():
    rng = np.random.default_rng(7)
    box = np.array([[0., 10.], [0., 12.], [0., 15.]])
    lengths = box[:, 1]
    s = rng.random((31, 3))*lengths
    g = rng.random((20, 3))*lengths
    ids = np.arange(1,32);types=ids%3
    d, winner, typ, ties = nearest_solid_identity(g,s,ids,types,box)
    delta=g[:,None,:]-s[None,:,:];delta-=np.rint(delta/lengths)*lengths
    all_d=np.linalg.norm(delta,axis=-1).reshape(10,2,31)
    flat=all_d.reshape(10,-1);arg=flat.argmin(1)
    np.testing.assert_allclose(d,flat.min(1));np.testing.assert_array_equal(winner,ids[arg%31])
    perm=rng.permutation(31)
    shifted=nearest_solid_identity(g+lengths,s[perm],ids[perm],types[perm],box)
    np.testing.assert_allclose(shifted[0],d);np.testing.assert_array_equal(shifted[1],winner)


def test_solid_and_pair_ties_choose_smallest_id():
    box=np.array([[0.,10.]]*3)
    g=np.array([[0.,0.,0.],[0.,0.,0.]])
    s=np.array([[1.,0.,0.],[9.,0.,0.]])
    d,ids,types,ties=nearest_solid_identity(g,s,[9,2],[7,2],box)
    np.testing.assert_allclose(d,[1.]);assert ids[0]==2 and types[0]==2 and ties[0]
