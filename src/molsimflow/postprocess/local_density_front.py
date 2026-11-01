"""Local density-front geometry without an axisymmetric shape assumption.

Angles are apparent gas-side tangent angles at the supplied fit band.
The extrapolated boundary is not proof of molecular gas-solid contact.
Callers must independently qualify contact, topology and solid registration.
"""
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

@dataclass(frozen=True)
class DensityGrid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    density: np.ndarray
    spacing: np.ndarray

def water_density_grid(water_frames: Sequence[np.ndarray], center_xy: np.ndarray,
                       lengths_xy: np.ndarray, z_edges: np.ndarray, *,
                       spacing: float = 0.5, sigma: float = 1.0) -> DensityGrid:
    """Average in a fixed solid-frame grid with periodic XY and open Z."""
    if not len(water_frames) or spacing <= 0 or sigma <= 0:
        raise ValueError("Nonempty frames and positive observation scales required")
    lengths=np.asarray(lengths_xy,float)
    center=np.asarray(center_xy,float)
    ze=np.asarray(z_edges,float)
    if lengths.shape!=(2,) or center.shape!=(2,) or np.any(lengths<=0):
        raise ValueError("Two positive XY lengths and a finite center required")
    dz=np.diff(ze)
    if len(dz)<2 or np.any(dz<=0) or not np.allclose(dz,dz[0]):
        raise ValueError("Uniform increasing Z edges required")
    n=np.maximum(2,np.rint(lengths/spacing).astype(int))
    xe=np.linspace(-lengths[0]/2,lengths[0]/2,n[0]+1)
    ye=np.linspace(-lengths[1]/2,lengths[1]/2,n[1]+1)
    delta=np.array([lengths[0]/n[0],lengths[1]/n[1],dz[0]])
    hist=np.zeros((n[0],n[1],len(ze)-1),np.float64)
    for frame in water_frames:
        xyz=np.asarray(frame,float).copy()
        if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
            raise ValueError("Finite water coordinates required")
        xyz[:,:2]=(xyz[:,:2]-center+lengths/2)%lengths-lengths/2
        hist+=np.histogramdd(xyz,bins=(xe,ye,ze))[0]
    hist/=len(water_frames)*np.prod(delta)
    density=gaussian_filter(hist,sigma/delta,mode=("wrap","wrap","constant"))
    return DensityGrid((xe[1:]+xe[:-1])/2,(ye[1:]+ye[:-1])/2,
                       (ze[1:]+ze[:-1])/2,density,delta)

def radial_interface_geometry(grid: DensityGrid, bulk_profile: np.ndarray,
                              fit_band: tuple[float,float], reference_z: float,
                              *, sectors: int = 32) -> dict:
    """Extract local tangents of a star-shaped low-density domain.

    Each azimuth is independent; no circle or common radius is fitted.
    Non-star-shaped/center-wet/missing crossings produce invalid sectors.
    The 3D normal includes the azimuthal slope of the interface.
    """
    bulk=np.asarray(bulk_profile,float)
    if bulk.shape!=grid.z.shape or sectors<8:
        raise ValueError("Aligned bulk profile and at least eight sectors required")
    zi=np.flatnonzero((grid.z>=fit_band[0])&(grid.z<=fit_band[1]))
    if len(zi)<4:
        raise ValueError("At least four Z planes required for local tangent fit")
    phi=np.arange(sectors)*2*np.pi/sectors
    rr=np.arange(0.,min(max(grid.x),max(grid.y))-2*max(grid.spacing[:2]),
                 min(grid.spacing[:2])/2)
    if len(rr)<8:raise ValueError("Insufficient radial domain")
    radii=np.full((len(zi),sectors),np.nan)
    for iz,k in enumerate(zi):
        if not np.isfinite(bulk[k]) or bulk[k]<=0:continue
        xy=np.stack((np.cos(phi)[:,None]*rr,np.sin(phi)[:,None]*rr),axis=-1)
        coords=np.stack(((xy[:,:,0]-grid.x[0])/grid.spacing[0],
                         (xy[:,:,1]-grid.y[0])/grid.spacing[1]))
        values=map_coordinates(grid.density[:,:,k],coords,order=1,mode="nearest")/bulk[k]-.5
        for j,v in enumerate(values):
            if not np.isfinite(v).all() or v[0]>=0:continue
            crossed=np.flatnonzero((v[:-1]<0)&(v[1:]>=0))
            if len(crossed)==0:continue
            h=int(crossed[0])
            # Reject a crossing that immediately returns to a cavity.
            tail=v[h+1:min(len(v),h+1+max(2,int(1./(rr[1]-rr[0]))))]
            if len(tail)<2 or np.any(tail<0):continue
            radii[iz,j]=rr[h]-v[h]*(rr[h+1]-rr[h])/(v[h+1]-v[h])
    valid=np.isfinite(radii).all(axis=0)
    slope=np.full(sectors,np.nan);r0=slope.copy();rmid=slope.copy();rmse=slope.copy()
    zz=grid.z[zi];zmid=float(zz.mean())
    for j in np.flatnonzero(valid):
        slope[j],intercept=np.polyfit(zz-reference_z,radii[:,j],1)
        r0[j]=intercept;rmid[j]=intercept+slope[j]*(zmid-reference_z)
        rmse[j]=np.sqrt(np.mean((radii[:,j]-(slope[j]*(zz-reference_z)+intercept))**2))
    valid &= (r0>0)&(r0<rr[-1])&(rmid>0)
    dphi=2*np.pi/sectors
    rphi=(np.roll(rmid,-1)-np.roll(rmid,1))/(2*dphi)
    q=rphi/rmid
    nz=-slope/np.sqrt(1.+q*q+slope*slope)
    angles=np.degrees(np.arccos(np.clip(nz,-1.,1.)))
    valid &= np.isfinite(angles)&np.roll(valid,1)&np.roll(valid,-1)
    er=np.column_stack((np.cos(phi),np.sin(phi)))
    ep=np.column_stack((-np.sin(phi),np.cos(phi)))
    q0=(np.roll(r0,-1)-np.roll(r0,1))/(2*dphi*r0)
    normal=er-q0[:,None]*ep
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    xy=r0[:,None]*er
    angles[~valid]=np.nan;xy[~valid]=np.nan;normal[~valid]=np.nan
    return dict(phi=phi,radii_by_z=radii,fit_z=zz,reference_z=float(reference_z),
                measurement_z=zmid,angle_g_deg=angles,boundary_xy=xy,
                normal_xy=normal,valid=valid,radial_fit_rmse=rmse,
                complete_closed_boundary=bool(valid.all()))

def normal_displacements(previous_xy: np.ndarray, previous_normal: np.ndarray,
                         current_xy: np.ndarray) -> np.ndarray:
    """Intersect each previous in-plane normal with a closed current polygon.

    Positive displacement is along the supplied outward gas normal.
    Inputs must already share a material frame and periodic image.
    This is a geometric displacement, not a force or pinning observer.
    """
    old=np.asarray(previous_xy,float);normal=np.asarray(previous_normal,float)
    new=np.asarray(current_xy,float)
    if old.ndim!=2 or old.shape[1]!=2 or normal.shape!=old.shape or new.ndim!=2 or new.shape[1]!=2:
        raise ValueError("Aligned planar boundaries and normals required")
    if not np.isfinite(old).all() or not np.isfinite(normal).all() or not np.isfinite(new).all():
        raise ValueError("Complete finite contours required; do not bridge missing sectors")
    if not np.allclose(np.linalg.norm(normal,axis=1),1.,atol=1e-6):
        raise ValueError("Unit outward normals required")
    edge=np.roll(new,-1,axis=0)-new
    result=np.full(len(old),np.nan)
    for i,(p,n) in enumerate(zip(old,normal)):
        # p+t*n = q+u*edge; solve in the local normal-tangent basis.
        tangent=np.array([-n[1],n[0]])
        denom=edge@tangent
        good=np.abs(denom)>1e-12
        u=np.full(len(edge),np.nan);u[good]=((p-new)@tangent)[good]/denom[good]
        good&=(u>=-1e-10)&(u<=1+1e-10)
        if np.any(good):
            hits=new[good]+u[good,None]*edge[good]
            distance=(hits-p)@n
            result[i]=distance[np.argmin(np.abs(distance))]
    return result
