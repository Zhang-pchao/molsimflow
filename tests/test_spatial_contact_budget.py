import numpy as np
import pytest
from molsimflow.postprocess.interfacial_contact_renewal import spatial_contact_budget
from molsimflow.postprocess.contact_graph import components_with_isolated_nodes


def test_translation_and_periodic_reference():
    # Already unwrapped across a box edge; never use [9,0,1] here.
    r=spatial_contact_budget([1,1,0],[0,1,1],[2,2,2],[[9],[10],[11]])
    assert r['area_change']==0
    np.testing.assert_allclose(r['centroid_change'],[1])
    np.testing.assert_allclose(r['gain_centroid_change'],[.75])
    np.testing.assert_allclose(r['loss_centroid_change'],[.25])
    np.testing.assert_allclose(r['centroid_residual'],0,atol=1e-12)


def test_rear_retraction_moves_center_without_forward_gain():
    r=spatial_contact_budget([1,1,1],[0,1,1],[1,1,1],[[0],[1],[2]])
    assert r['gained_area']==0 and r['area_change']==-1
    np.testing.assert_allclose(r['centroid_change'],[.5])
    np.testing.assert_allclose(r['loss_centroid_change'],[.5])


def test_symmetric_exchange_is_not_translation():
    r=spatial_contact_budget([0,1,1,0],[1,0,0,1],[1]*4,[[-2],[-1],[1],[2]])
    assert r['gained_area']==r['lost_area']==2
    np.testing.assert_allclose(r['centroid_change'],0)


def test_random_weighted_multidimensional_identity_and_origin_shift():
    rng=np.random.default_rng(12);x=rng.normal(size=(100,2));w=rng.uniform(0,3,100)
    b=rng.random(100)>.4;a=rng.random(100)>.4
    r=spatial_contact_budget(b,a,w,x);q=spatial_contact_budget(b,a,w,x+[200,-30])
    assert abs(r['area_residual'])<1e-12
    np.testing.assert_allclose(r['first_moment_residual'],0,atol=1e-12)
    np.testing.assert_allclose(r['centroid_residual'],0,atol=1e-12)
    np.testing.assert_allclose(r['gain_centroid_change'],q['gain_centroid_change'],atol=1e-12)


def test_empty_contact_keeps_area_but_not_fictitious_centroid():
    r=spatial_contact_budget([1],[0],[2],[[3,4]])
    assert not r['centroid_valid'] and r['area_change']==-2
    assert np.isnan(r['centroid_change']).all()


def test_invalid_coordinates():
    with pytest.raises(ValueError):spatial_contact_budget([1],[1],[1],[[np.nan]])


def test_explicit_isolates_and_deterministic_components():
    assert components_with_isolated_nodes(['c','a','b','d'],[('b','a')])==[['a','b'],['c'],['d']]
    assert components_with_isolated_nodes([],[])==[]


@pytest.mark.parametrize('nodes,edges', [(['a','a'],[]),(['a'],[('a','b')]),(['a'],[('a','a')])])
def test_invalid_graph_identity(nodes,edges):
    with pytest.raises(ValueError):components_with_isolated_nodes(nodes,edges)
