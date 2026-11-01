"""Periodic spatial geometry around a solid-site to contacting-atom segment."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from molsimflow.io.lammps_dump import minimum_image_vectors

@dataclass(frozen=True)
class SegmentGeometry:
    fraction: np.ndarray
    perpendicular: np.ndarray
    radial: np.ndarray

def segment_geometry(axis, relative):
    """Project relative vectors on a nonzero site-to-gas axis; no clipping."""
    axis=np.asarray(axis,float)
    relative=np.asarray(relative,float)
    if axis.shape != (3,) or relative.ndim != 2 or relative.shape[1] != 3:
        raise ValueError("Expected axis (3,) and relative (N,3)")
    if not np.isfinite(axis).all() or not np.isfinite(relative).all():
        raise ValueError("Finite coordinates required")
    norm2=float(axis@axis)
    if norm2<=0:
        raise ValueError("A nonzero segment is required")
    t=(relative@axis)/norm2
    rho=np.linalg.norm(relative-t[:,None]*axis,axis=1)
    return SegmentGeometry(t,rho,np.linalg.norm(relative,axis=1))

def contacting_atom(site, gas_pair, atom_ids, box):
    """Choose the closest N of the retained N2, breaking exact ties by atom ID."""
    vectors=minimum_image_vectors(np.asarray(gas_pair,float)-site,box)
    distances=np.linalg.norm(vectors,axis=1)
    ids=np.asarray(atom_ids,int)
    if vectors.shape != (2,3) or ids.shape != (2,) or len(set(ids))!=2:
        raise ValueError("One two-atom molecule with distinct IDs is required")
    winner=np.lexsort((ids,distances))[0]
    return vectors[winner],int(ids[winner])

def corridor_minima(sites, axes, solution, oxygen_ids, qualified, box, max_radius):
    """Find all-O minimum transverse distances strictly between endpoint planes.

    Returns separate qualified-water and other-O minima and identities. Search
    is exhaustive for every cylinder radius <= max_radius. A finite returned
    value above max_radius is not a globally minimal distance. Infinity means
    no candidate inside the local search sphere, not absence of bulk water.
    """
    sites=np.asarray(sites,float);axes=np.asarray(axes,float)
    solution=np.asarray(solution,float);ids=np.asarray(oxygen_ids,int)
    qualified=np.asarray(qualified,bool);box=np.asarray(box,float)
    if sites.shape!=axes.shape or sites.ndim!=2 or sites.shape[1]!=3:
        raise ValueError("Matching (N,3) sites and axes required")
    if solution.shape!=(len(ids),3) or qualified.shape!=ids.shape or len(np.unique(ids))!=len(ids):
        raise ValueError("Unique solution identities and qualification required")
    if box.shape!=(3,) or not np.isfinite(box).all() or np.any(box<=0) or max_radius<=0:
        raise ValueError("Positive finite box and radius required")
    if not all(np.isfinite(a).all() for a in (sites,axes,solution)):
        raise ValueError("Finite coordinates required")
    lengths=np.linalg.norm(axes,axis=1)
    search=np.sqrt(lengths**2+max_radius**2)
    if np.any(lengths<=0) or np.any(search>=box.min()/2):
        raise ValueError("Segments and search spheres must be nonzero and local")
    minima=np.full((len(sites),2),np.inf)
    winners=np.full((len(sites),2),-1,np.int64)
    if not len(solution):
        return minima,winners
    tree=cKDTree(solution%box,boxsize=box)
    for i,neighbors in enumerate(tree.query_ball_point(sites%box,search+1e-10)):
        j=np.asarray(neighbors,int)
        if not len(j):continue
        rel=minimum_image_vectors(solution[j]-sites[i],box)
        g=segment_geometry(axes[i],rel)
        interior=(g.fraction>0)&(g.fraction<1)
        for col,mask in enumerate((qualified[j],~qualified[j])):
            ix=np.flatnonzero(interior&mask)
            if len(ix):
                k=ix[np.lexsort((ids[j[ix]],g.perpendicular[ix]))[0]]
                minima[i,col]=g.perpendicular[k];winners[i,col]=ids[j[k]]
    return minima,winners

def corridor_category(qualified_rho,other_rho,radius):
    """0=no O in cylinder; 1=qualified water present; 2=other O only."""
    q,o=np.broadcast_arrays(qualified_rho,other_rho)
    return np.where(q<=radius,1,np.where(o<=radius,2,0)).astype(np.int8)
