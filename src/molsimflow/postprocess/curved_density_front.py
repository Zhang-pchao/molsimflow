"""Curved local interface patches on the existing density-front observer.

This is a geometric observation, not a Young angle, physical pinning test,
constitutive law or independent microscopic surface-tension measurement.
"""
import numpy as np
from molsimflow.postprocess.local_density_front import (
    DensityGrid, radial_interface_geometry)

def curved_patch_from_radii(radii, z, phi, reference_z, max_radius, *,
                            halfwidth=8.0, targets=32):
    """Fit independent quadratic patches r(phi,z)**2, retaining azimuthal slope.

    At least 80% of each selected patch must have real radial crossings.
    Fits never bridge a completely missing angular column or vertical plane.
    The returned angle is extrapolated to reference_z, with fit residuals
    and validity explicit; physical qualification belongs to the caller.
    """
    rr=np.asarray(radii,float);zz=np.asarray(z,float);pp=np.asarray(phi,float)
    if rr.shape!=(len(zz),len(pp)) or len(zz)<8 or len(pp)%targets or halfwidth<=0:
        raise ValueError("Aligned radii, >=8 heights and evenly sampled target rays required")
    if not np.all(np.diff(zz)>0) or max_radius<=0 or not np.isfinite(reference_z):
        raise ValueError("Ordered finite heights and positive radial domain required")
    step=len(pp)//targets;indices=np.arange(0,len(pp),step)
    theta=np.full(targets,np.nan);xy=np.full((targets,2),np.nan);normal=xy.copy()
    residual=np.full(targets,np.nan);coverage=residual.copy();condition=residual.copy()
    valid=np.zeros(targets,bool);coeff=np.full((targets,6),np.nan)
    zmid=float(zz.mean());zscale=float(np.ptp(zz)/2)
    for jj,j in enumerate(indices):
        center=rr[:,j];finite=np.isfinite(center)
        if finite.sum()<8:continue
        scale=float(np.median(center[finite]))
        if scale<=0:continue
        dp=(pp-pp[j]+np.pi)%(2*np.pi)-np.pi
        angular=np.abs(dp)*scale<=halfwidth+1e-10
        patch=rr[:,angular];mask=np.isfinite(patch);coverage[jj]=mask.mean()
        if angular.sum()<3 or coverage[jj]<.8 or np.any(mask.sum(axis=0)==0) or np.any(mask.sum(axis=1)==0):continue
        uu=np.broadcast_to(dp[angular][None,:]*scale/halfwidth,patch.shape)
        vv=np.broadcast_to((zz[:,None]-zmid)/zscale,patch.shape)
        u,v=uu[mask],vv[mask]
        X=np.column_stack([np.ones(len(u)),u,v,u*u,u*v,v*v])
        b,_,rank,sing=np.linalg.lstsq(X,patch[mask]**2,rcond=None)
        if rank!=6:continue
        condition[jj]=sing[0]/sing[-1]
        if condition[jj]>1e6:continue
        coeff[jj]=b
        fitted_squared=X@b
        if np.any(fitted_squared<=0):continue
        residual[jj]=np.sqrt(np.mean((np.sqrt(fitted_squared)-patch[mask])**2))
        v0=(reference_z-zmid)/zscale
        r_squared=b[0]+b[2]*v0+b[5]*v0*v0
        if r_squared<=0:continue
        r0=np.sqrt(r_squared)
        rz=(b[2]+2*b[5]*v0)/(zscale*2*r0)
        rphi=(b[1]+b[4]*v0)*scale/(halfwidth*2*r0)
        if not (0<r0<max_radius) or residual[jj]>1.0:continue
        qp=rphi/r0
        nz=-rz/np.sqrt(1+qp*qp+rz*rz)
        angle=np.degrees(np.arccos(np.clip(nz,-1.,1.)))
        er=np.array([np.cos(pp[j]),np.sin(pp[j])])
        et=np.array([-np.sin(pp[j]),np.cos(pp[j])])
        n=er-qp*et;n/=np.linalg.norm(n)
        theta[jj]=angle;xy[jj]=r0*er;normal[jj]=n;valid[jj]=True
    return dict(phi=pp[indices],angle_g_deg=theta,boundary_xy=xy,normal_xy=normal,
                valid=valid,radial_fit_rmse=residual,finite_crossing_fraction=coverage,
                condition_number=condition,coefficients=coeff,fit_z=zz,
                reference_z=float(reference_z),complete_closed_boundary=bool(valid.all()))

def curved_patch_geometry(grid: DensityGrid, bulk_profile, fit_band, reference_z, *,
                          halfwidth=8.0, rays=128, targets=32):
    """Reuse the accepted radial crossing extractor, without a common radius."""
    raw=radial_interface_geometry(grid,bulk_profile,fit_band,reference_z,sectors=rays)
    bound=min(max(grid.x),max(grid.y))-2*max(grid.spacing[:2])
    return curved_patch_from_radii(raw["radii_by_z"],raw["fit_z"],raw["phi"],
                                   reference_z,bound,halfwidth=halfwidth,targets=targets)

def partial_normal_displacements(previous_xy,previous_normal,current_xy,*,
                                  max_distance=8.0):
    """Intersect only adjacent finite target edges, never closing a gap.

    Missing old points or no local intersection return NaN, not zero motion.
    """
    old=np.asarray(previous_xy,float);normal=np.asarray(previous_normal,float)
    new=np.asarray(current_xy,float)
    if old.ndim!=2 or old.shape[1]!=2 or normal.shape!=old.shape or new.ndim!=2 or new.shape[1]!=2 or max_distance<=0:
        raise ValueError("Planar contours and a positive local range required")
    following=np.roll(new,-1,axis=0)
    good_edges=np.isfinite(new).all(axis=1)&np.isfinite(following).all(axis=1)
    result=np.full(len(old),np.nan)
    edge=following-new
    for i,(point,n) in enumerate(zip(old,normal)):
        if not np.isfinite(point).all() or not np.isfinite(n).all():continue
        if not np.isclose(np.linalg.norm(n),1.,atol=1e-6):raise ValueError("Unit normals required")
        tangent=np.array([-n[1],n[0]])
        denom=edge@tangent;ok=good_edges&(np.abs(denom)>1e-12)
        frac=np.full(len(new),np.nan)
        frac[ok]=((point-new)@tangent)[ok]/denom[ok]
        ok&=(frac>=-1e-10)&(frac<=1+1e-10)
        if ok.any():
            hits=new[ok]+frac[ok,None]*edge[ok]
            dist=(hits-point)@n;dist=dist[np.abs(dist)<=max_distance]
            if len(dist):result[i]=dist[np.argmin(np.abs(dist))]
    return result
