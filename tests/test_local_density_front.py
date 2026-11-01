import numpy as np
import pytest
from molsimflow.postprocess.local_density_front import (
    DensityGrid, radial_interface_geometry, normal_displacements,water_density_grid)

def sphere_grid():
    x=np.arange(-24.,24.1,.5);z=np.arange(-3.,12.1,.5)
    X,Y,Z=np.meshgrid(x,x,z,indexing="ij")
    # Gas interior is low water density. Center below the solid plane.
    radius=np.sqrt(X*X+Y*Y+(Z+10)**2)
    rho=.033/(1+np.exp(-(radius-20)/.35))
    return DensityGrid(x,x,z,rho,np.array([.5,.5,.5]))

def test_local_spherical_normal_and_angle_convention():
    g=sphere_grid()
    a=radial_interface_geometry(g,np.full(len(g.z),.033),(2.,4.),0.)
    assert a["complete_closed_boundary"]
    expected=np.degrees(np.arccos((a["measurement_z"]+10)/20))
    np.testing.assert_allclose(a["angle_g_deg"],expected,atol=.5)
    assert np.ptp(a["angle_g_deg"])<.4

def test_water_only_does_not_create_gas_contact_angle():
    g=sphere_grid()
    g=DensityGrid(g.x,g.y,g.z,np.full_like(g.density,.033),g.spacing)
    a=radial_interface_geometry(g,np.full(len(g.z),.033),(2.,4.),0.)
    assert not a["valid"].any() and not a["complete_closed_boundary"]

def test_normal_motion_separates_expansion_and_translation():
    phi=np.arange(128)*2*np.pi/128
    normals=np.column_stack((np.cos(phi),np.sin(phi)));old=10*normals
    expansion=normal_displacements(old,normals,11*normals)
    np.testing.assert_allclose(expansion,1.,atol=1e-12)
    shift=normal_displacements(old,normals,old+[.1,0])
    np.testing.assert_allclose(shift,.1*np.cos(phi),atol=.003)

def test_open_contour_cannot_silently_bridge_missing_data():
    xy=np.array([[0.,0.],[1.,0.],[1.,1.],[np.nan,1.]])
    with pytest.raises(ValueError,match="Complete"):
        normal_displacements(xy,np.tile([1.,0.],(4,1)),xy)

def test_density_periodic_translation_invariance():
    rng=np.random.default_rng(8);x=rng.uniform(0,10,(1000,3))
    a=water_density_grid([x],np.array([5.,5.]),np.array([10.,10.]),np.arange(-2,12.5,.5))
    moved=x.copy();moved[:,:2]=(moved[:,:2]+[8.,3.])%10
    b=water_density_grid([moved],np.array([3.,8.]),np.array([10.,10.]),np.arange(-2,12.5,.5))
    np.testing.assert_allclose(a.density,b.density,atol=1e-12)


def test_nonaxisymmetric_ellipsoid_preserves_azimuthal_angle_variation():
    x=np.arange(-25.,25.1,.5);z=np.arange(-3.,12.1,.5)
    X,Y,Z=np.meshgrid(x,x,z,indexing="ij")
    aa,bb,cc,zc=22.,13.,20.,-8.
    level=X*X/aa**2+Y*Y/bb**2+(Z-zc)**2/cc**2-1.
    rho=.033/(1+np.exp(-level/.025))
    grid=DensityGrid(x,x,z,rho,np.array([.5,.5,.5]))
    got=radial_interface_geometry(grid,np.full(len(z),.033),(2.,4.),0.)
    assert got["complete_closed_boundary"]
    phi=got["phi"];zz=got["measurement_z"]
    r=np.sqrt((1-(zz-zc)**2/cc**2)/(np.cos(phi)**2/aa**2+np.sin(phi)**2/bb**2))
    gradient=np.column_stack((r*np.cos(phi)/aa**2,r*np.sin(phi)/bb**2,
                              np.full(len(phi),(zz-zc)/cc**2)))
    expected=np.degrees(np.arccos(gradient[:,2]/np.linalg.norm(gradient,axis=1)))
    np.testing.assert_allclose(got["angle_g_deg"],expected,atol=1.)
    assert np.ptp(got["angle_g_deg"])>10.
