import numpy as np
from molsimflow.postprocess.curved_density_front import (
    curved_patch_from_radii, partial_normal_displacements)

def analytic_radii(a=35.,b=28.,c=32.,zc=-8.):
    p=np.arange(128)*2*np.pi/128;z=np.arange(4.,12.01,.5)
    r=np.sqrt((1-(z[:,None]-zc)**2/c**2)/(np.cos(p)**2/a**2+np.sin(p)**2/b**2))
    return r,z,p,(a,b,c,zc)

def test_ellipsoid_extrapolation_retains_local_angle_variation():
    r,z,p,(a,b,c,zc)=analytic_radii()
    got=curved_patch_from_radii(r,z,p,0.,45.)
    assert got['valid'].all()
    ph=got['phi'];rad=np.sqrt((1-zc**2/c**2)/(np.cos(ph)**2/a**2+np.sin(ph)**2/b**2))
    grad=np.column_stack([rad*np.cos(ph)/a**2,rad*np.sin(ph)/b**2,np.full(len(ph),-zc/c**2)])
    expected=np.degrees(np.arccos(grad[:,2]/np.linalg.norm(grad,axis=1)))
    np.testing.assert_allclose(got['angle_g_deg'],expected,atol=1.)
    np.testing.assert_allclose(np.linalg.norm(got['boundary_xy'],axis=1),rad,atol=.5)
    assert np.ptp(got['angle_g_deg'])>.8*np.ptp(expected)

def test_missing_angular_column_is_not_filled_by_polynomial():
    r,z,p,_=analytic_radii();r[:,0]=np.nan
    got=curved_patch_from_radii(r,z,p,0.,45.)
    assert not got['valid'][0]
    assert got['valid'][16]

def test_patch_excludes_missing_height_or_excessive_residual():
    r,z,p,_=analytic_radii();r[5,:]=np.nan
    assert not curved_patch_from_radii(r,z,p,0.,45.)['valid'].any()
    r,z,p,_=analytic_radii();r+=2.*((-1.)**np.arange(len(z)))[:,None]
    assert not curved_patch_from_radii(r,z,p,0.,45.)['valid'].any()

def test_local_open_contour_motion_does_not_bridge_gap():
    ph=np.arange(1280)*2*np.pi/1280;n=np.column_stack([np.cos(ph),np.sin(ph)])
    old=20*n;new=old+np.array([.2,0.]);new[0]=np.nan
    d=partial_normal_displacements(old,n,new)
    assert np.isnan(d[0])
    assert np.isfinite(d[640])
    exact=.2*np.cos(ph)+np.sqrt(20.**2-.2**2*np.sin(ph)**2)-20.
    np.testing.assert_allclose(d[10:1270],exact[10:1270],atol=.0001)

def test_translation_does_not_create_spreading_everywhere():
    ph=np.arange(1280)*2*np.pi/1280;n=np.column_stack([np.cos(ph),np.sin(ph)])
    d=partial_normal_displacements(20*n,n,20*n+[.2,0.])
    assert d[0]>.19 and d[640]<-.19
    exact=.2*np.cos(ph)+np.sqrt(20.**2-.2**2*np.sin(ph)**2)-20.
    np.testing.assert_allclose(d,exact,atol=.0001)


def test_full_density_wrapper_recovers_spherical_reference_angle():
    from molsimflow.postprocess.local_density_front import DensityGrid
    from molsimflow.postprocess.curved_density_front import curved_patch_geometry
    x=np.arange(-36.,36.1,.5);z=np.arange(-1.,15.1,.5)
    X,Y,Z=np.meshgrid(x,x,z,indexing='ij')
    distance=np.sqrt(X*X+Y*Y+(Z+8.)**2)
    rho=.033/(1+np.exp(-(distance-30.)/.35))
    grid=DensityGrid(x,x,z,rho,np.array([.5,.5,.5]))
    got=curved_patch_geometry(grid,np.full(len(z),.033),(4.,12.),0.,targets=128)
    assert got['valid'].all()
    np.testing.assert_allclose(got['angle_g_deg'],np.degrees(np.arccos(8./30)),atol=1.)

def test_reference_plane_translation_preserves_curved_observation():
    r,z,p,_=analytic_radii()
    a=curved_patch_from_radii(r,z,p,0.,45.)
    b=curved_patch_from_radii(r,z+27,p,27.,45.)
    np.testing.assert_allclose(a['boundary_xy'],b['boundary_xy'],atol=1e-12)
    np.testing.assert_allclose(a['angle_g_deg'],b['angle_g_deg'],atol=1e-12)
