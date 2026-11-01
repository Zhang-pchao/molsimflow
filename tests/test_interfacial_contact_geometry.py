import numpy as np
import pytest

from molsimflow.postprocess.interfacial_contact_geometry import (
    contact_geometry, endpoint_occupation_budget, geometry_from_mask,
    orientation_weights, region_weights, surface_grid,
)


def chart(spacing=1., shift=(0., 0., 0.)):
    shift = np.array(shift)
    bounds = np.array([[0., 24.], [0., 20.], [0., 40.]]) + shift[:, None]
    xy = np.stack(np.meshgrid(np.arange(.5, 24, 2), np.arange(.5, 20, 2),
                             indexing="ij"), axis=-1).reshape(-1, 2) + shift[:2]
    return surface_grid(xy, bounds, spacing)


def rectangle(g):
    x, y = np.moveaxis(g.centers-g.bounds[:2, 0], -1, 0)
    return (x >= 5) & (x < 15) & (y >= 5) & (y < 15)


def test_ownership_area_and_equal_material_area():
    g = chart()
    assert np.isclose(g.cell_area*g.owner.size, 480)
    assert np.isclose(g.cell_area*np.sum(rectangle(g)), 100)
    assert g.owner.min() == 0 and g.owner.max() == g.site_count-1


def test_holes_concavity_and_disconnected_contact_are_preserved():
    g = chart()
    mask = rectangle(g)
    mask[8:12, 8:12] = False
    mask[12:15, 12:15] = False
    mask[20:22, 1:3] = True
    a = geometry_from_mask(g, mask, np.zeros(g.site_count))
    assert np.array_equal(a.footprint, mask)
    assert (a.signed_distance[mask] < 0).all()
    assert (a.signed_distance[~mask] > 0).all()
    assert a.signed_distance[9, 9] > 0


@pytest.mark.parametrize("shift", [(11, 0), (0, 13), (11, 13)])
def test_periodic_both_seams(shift):
    g = chart()
    mask = rectangle(g)
    a = geometry_from_mask(g, mask, np.zeros(g.site_count))
    b = geometry_from_mask(g, np.roll(mask, shift, axis=(0, 1)), np.zeros(g.site_count))
    assert np.allclose(b.signed_distance, np.roll(a.signed_distance, shift, axis=(0, 1)))
    assert np.allclose(b.normal, np.roll(a.normal, shift, axis=(0, 1)))


def test_common_frame_translation():
    g, h = chart(), chart(shift=(3.1, -12.4, 7.))
    a = geometry_from_mask(g, rectangle(g), np.zeros(g.site_count))
    b = geometry_from_mask(h, rectangle(h), np.full(h.site_count, 7.))
    p = np.array([[17., 10., 2.], [3., 8., 1.], [8., 9., 1.]])
    assert np.allclose(region_weights(a, p), region_weights(b, p+[3.1, -12.4, 7.]))


def test_stationary_particles_moving_geometry():
    g = chart()
    mask = rectangle(g)
    a = geometry_from_mask(g, mask, np.zeros(g.site_count))
    b = geometry_from_mask(g, np.roll(mask, 3, axis=0), np.zeros(g.site_count))
    p = np.array([[16., 10., 1.], [21., 10., 1.]])
    x, y = region_weights(a, p), region_weights(b, p)
    q = endpoint_occupation_budget(x, x, y, y)
    assert np.allclose(q["particle"], 0)
    assert np.allclose(q["geometry"], q["delta"])


def test_stationary_geometry_moving_particles_and_identity_exchange():
    g = chart()
    a = geometry_from_mask(g, rectangle(g), np.zeros(g.site_count))
    p = np.array([[16., 10., 1.], [10., 10., 1.]])
    x, y = region_weights(a, p), region_weights(a, p+np.array([4., 0., 0.]))
    q = endpoint_occupation_budget(x, y, x, y)
    assert np.allclose(q["geometry"], 0)
    assert np.allclose(q["particle"], q["delta"])
    # A count balance cannot detect turnover when identities swap occupancy.
    q = endpoint_occupation_budget(x, x[::-1], x, x[::-1])
    assert np.allclose(q["delta"], 0)
    assert not np.array_equal(x, x[::-1])


def test_general_weighted_budget_closure_and_endpoint_symmetry():
    rng = np.random.default_rng(8)
    a, b, c, d = rng.random((4, 31, 4))
    q, r = endpoint_occupation_budget(a,b,c,d), endpoint_occupation_budget(d,c,b,a)
    assert np.max(abs(q["residual"])) < 1e-13
    assert np.allclose(q["particle"], -r["particle"])
    assert np.allclose(q["geometry"], -r["geometry"])


@pytest.mark.parametrize("value,status", [(False,"EMPTY_CONTACT"), (True,"FULL_CONTACT")])
def test_undefined_contact(value, status):
    g = chart()
    a = geometry_from_mask(g, np.full(g.owner.shape,value), np.zeros(g.site_count))
    assert a.status == status and np.isnan(a.signed_distance).all()
    with pytest.raises(ValueError, match="undefined"):
        region_weights(a, np.zeros((1,3)))


def test_grid_refinement_rectangle():
    areas = []
    for spacing in (2., 1., .5):
        g = chart(spacing)
        mask = rectangle(g)
        a = geometry_from_mask(g, mask, np.zeros(g.site_count))
        areas.append(mask.sum()*g.cell_area)
        # Outside at x=18, y=10: distance to opposite cell center is 3+dx/2.
        index = np.floor(np.array([18., 10.])/g.spacing).astype(int)
        assert abs(a.signed_distance[tuple(index)]-3.) <= 1.5*g.spacing[0]
    assert max(areas)-min(areas) < 1e-10


def test_contact_uses_gas_and_local_height():
    g = chart()
    sites = np.column_stack([g.centers.reshape(-1,2)[::4][:g.site_count],
                              np.zeros(g.site_count)])
    a = contact_geometry(g, sites, np.empty((0,3)), 4.)
    assert a.status == "EMPTY_CONTACT"
    a = contact_geometry(g, sites, sites[[0]]+[0,0,2], 3.)
    assert a.footprint.any() and not a.footprint.all()
    low = geometry_from_mask(g, rectangle(g), np.zeros(g.site_count))
    high = geometry_from_mask(g, rectangle(g), np.ones(g.site_count)*4)
    p = np.array([[17., 10., 2.]])
    assert region_weights(low,p).sum() == pytest.approx(1)
    assert region_weights(high,p).sum() == 0


def test_normal_partition_and_input_validation():
    w = orientation_weights(np.array([[1.,0.],[-1.,0.],[0.,-1.],[0.,0.]]))
    assert np.array_equal(w, np.eye(4))
    with pytest.raises(ValueError):
        surface_grid(np.zeros((0,2)), np.array([[0,1]]*3), 1)
    with pytest.raises(ValueError):
        geometry_from_mask(chart(), np.zeros(chart().owner.shape), np.zeros(chart().site_count))
    with pytest.raises(ValueError):
        endpoint_occupation_budget(np.ones((2,1)),np.ones((3,1)),np.ones((2,1)),np.ones((2,1)))
