"""Whole-main-cluster projected lower envelopes and interposed solution oxygen."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from molsimflow.postprocess.surface_contact_memory import _component_sizes

@dataclass(frozen=True)
class GapColumns:
    projected: np.ndarray
    supported: np.ndarray
    gap: np.ndarray
    interposed_count: np.ndarray

def projected_gap_columns(gas_xyz, solution_xyz, centers, top, bounds, shape,
                          origin, *, periodic_z=True, radius=4., support=3,
                          cluster_cutoff=5.5, clearance=.5):
    """Use all heights; third-lowest connected main-N2 COM defines the envelope.

    gas_xyz must contain only the inherited main cluster. The calling analysis
    retains unsupported projected columns in its denominator. Solution atoms
    are origin-tagged O, not an assumed current molecular-water assignment.
    """
    gas=np.asarray(gas_xyz,float);water=np.asarray(solution_xyz,float)
    centers=np.asarray(centers,float);top=np.asarray(top,float)
    bounds=np.asarray(bounds,float);shape=np.asarray(shape,int);origin=np.asarray(origin,float)
    if gas.ndim!=2 or gas.shape[1]!=3 or water.ndim!=2 or water.shape[1]!=3:
        raise ValueError("Expected (N,3) coordinates")
    if bounds.shape!=(3,2) or np.any(np.diff(bounds,axis=1)<=0):
        raise ValueError("Positive orthorhombic bounds required")
    if shape.shape!=(2,) or np.any(shape<=0) or int(np.prod(shape))!=len(top) or centers.shape!=(len(top),2) or origin.shape!=(2,):
        raise ValueError("Aligned fixed-grid geometry required")
    if support<1 or radius<=0 or cluster_cutoff<=0 or clearance<0:
        raise ValueError("Invalid positive support and geometric parameters")
    if not all(np.isfinite(x).all() for x in [gas,water,centers,top,bounds,origin]):
        raise ValueError("Missing coordinates cannot imply an empty gap")
    L=np.diff(bounds,axis=1).ravel()
    neighbors=cKDTree(gas[:,:2]%L[:2],boxsize=L[:2]).query_ball_point(centers%L[:2],radius) if len(gas) else [[] for _ in top]
    projected=np.array([bool(n) for n in neighbors]);supported=np.zeros(len(top),bool);gap=np.full(len(top),np.nan)
    effective=bounds.copy()
    if not periodic_z and len(gas):
        lo,hi=gas[:,2].min(),gas[:,2].max();pad=max(1.,hi-lo)+cluster_cutoff
        effective[2]=[lo-pad,hi+pad]
    for j,nn in enumerate(neighbors):
        ix=np.asarray(nn,int)
        ix=ix[gas[ix,2]>=top[j]]
        if len(ix)<support:continue
        sizes=_component_sizes(gas[ix],effective,cluster_cutoff)
        keep=ix[sizes>=support]
        if len(keep)<support:continue
        gap[j]=np.partition(gas[keep,2],support-1)[support-1]-top[j]
        if gap[j]<=0:
            gap[j]=np.nan
            continue
        supported[j]=True
    ij=np.floor(((water[:,:2]-origin)%L[:2])/L[:2]*shape).astype(int)
    ij=np.minimum(ij,shape-1);bins=ij[:,0]*shape[1]+ij[:,1]
    height=water[:,2]-top[bins]
    between=supported[bins]&(height>clearance)&(height<gap[bins]-clearance)
    counts=np.bincount(bins[between],minlength=len(top)).astype(np.int32)
    return GapColumns(projected,supported,gap,counts)

def gap_cdf(gap,interposed_count,projected,supported,edges):
    """Additive wet/empty cumulative area fractions with full-projection denominator."""
    gap=np.asarray(gap,float);counts=np.asarray(interposed_count)
    projected=np.asarray(projected,bool);supported=np.asarray(supported,bool);edges=np.asarray(edges,float)
    if not (gap.shape==counts.shape==projected.shape==supported.shape) or gap.ndim!=1:
        raise ValueError("Aligned column arrays required")
    if np.any(supported&~projected) or np.any(~np.isfinite(gap[supported])) or np.any(counts<0):
        raise ValueError("Invalid support or interposed counts")
    n=int(projected.sum())
    if n==0:raise ValueError("No main-cluster projection; cannot normalize")
    total=np.searchsorted(np.sort(gap[supported]),edges,side='right')/n
    wet=np.searchsorted(np.sort(gap[supported&(counts>0)]),edges,side='right')/n
    return total,wet,total-wet,1-float(supported.sum())/n
