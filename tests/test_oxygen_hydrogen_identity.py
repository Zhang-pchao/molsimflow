import numpy as np
import pytest
from molsimflow.postprocess.oxygen_hydrogen_identity import oxygen_hydrogen_ownership,oxygen_h_category
B=np.array([[0.,10.]]*3)


def test_oc_competition_pbc_and_unassigned():
    a=oxygen_hydrogen_ownership([[.1,0,0],[5.9,5,5],[3,3,3]],[1,2,3],[[9.8,0,0]],[10],[[5,5,5]],[20],B)
    np.testing.assert_array_equal(a.owner_id,[10,20,-1]);np.testing.assert_array_equal(a.oxygen_h_count,[1]);assert not a.oxygen_ambiguous.any()


def test_oo_and_oc_ambiguity_marks_alternative_oxygen():
    a=oxygen_hydrogen_ownership([[1,1,1]],[1],[[0,1,1],[2.05,1,1]],[10,11],[[1,1.98,1]],[20],B)
    assert a.owner_id[0]==20;np.testing.assert_array_equal(a.oxygen_h_count,[0,0]);assert a.oxygen_ambiguous.all()
    np.testing.assert_array_equal(oxygen_h_category(a.oxygen_h_count,a.oxygen_ambiguous),[3,3])


def test_counts_zero_one_many_and_exact_oc_tie():
    a=oxygen_hydrogen_ownership([[1,0,0],[0,1,0]],[1,2],[[0,0,0],[5,5,5]],[10,11],[[2,0,0]],[20],B)
    assert a.owner_id[0]==10 and a.oxygen_h_count[0]==2
    np.testing.assert_array_equal(oxygen_h_category([0,1,2,4],[False]*4),[0,1,2,2])
    assert a.oxygen_ambiguous[0]


def test_duplicate_hydrogen_rejected():
    with pytest.raises(ValueError):oxygen_hydrogen_ownership([[1,0,0],[1,0,0]],[1,1],[[0,0,0]],[10],[],[],B)

def test_open_z_rejects_artificial_seam_owner():
    bounds=np.array([[0.,10.],[0.,10.],[0.,20.]])
    args=(np.array([[2.,2.,.2]]),[1],np.array([[2.,2.,19.8],[2.,2.,3.]]),[2,3],np.empty((0,3)),[],bounds)
    periodic=oxygen_hydrogen_ownership(*args)
    opened=oxygen_hydrogen_ownership(*args,periodic_z=False)
    assert periodic.owner_id[0]==2 and opened.owner_id[0]==-1
    assert not opened.oxygen_ambiguous.any()

def test_open_z_preserves_real_nearby_ownership():
    bounds=np.array([[0.,10.],[0.,10.],[0.,20.]])
    args=(np.array([[.1,2.,19.1]]),[1],np.array([[9.9,2.,19.8],[2.,2.,3.]]),[2,3],np.empty((0,3)),[],bounds)
    opened=oxygen_hydrogen_ownership(*args,periodic_z=False)
    assert opened.owner_id[0]==2 and opened.oxygen_h_count.tolist()==[1,0]
