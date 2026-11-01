"""Complete PPP gas components and ID-preserving overlap bookkeeping."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class GasComponents:
    labels: np.ndarray
    sizes: np.ndarray
    images: np.ndarray
    cycle_translations: np.ndarray

def gas_components(coordinates: np.ndarray, bounds: np.ndarray,
                   cutoff: float = 5.5) -> GasComponents:
    """Label components by smallest member index; retain nonzero cycle images.

    Input row identities must be stable across frames. Crossing an edge of the
    periodic box alone does not imply winding.
    """
    xyz=np.asarray(coordinates,float); bounds=np.asarray(bounds,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Finite nonempty N by 3 coordinates required")
    if bounds.shape!=(3,2) or np.any(np.diff(bounds,axis=1)<=0) or cutoff<=0:
        raise ValueError("Positive orthorhombic box and cutoff required")
    lengths=np.diff(bounds,axis=1).ravel(); pos=(xyz-bounds[:,0])%lengths
    pairs=cKDTree(pos,boxsize=lengths).query_pairs(cutoff,output_type='ndarray')
    neighbors=[[] for _ in pos]
    for i,j in pairs:
        jump=-np.rint((pos[j]-pos[i])/lengths).astype(int)
        neighbors[i].append((j,jump));neighbors[j].append((i,-jump))
    labels=np.full(len(pos),-1,int);images=np.zeros((len(pos),3),int)
    cycles=set()
    for seed in range(len(pos)):
        if labels[seed]>=0:continue
        labels[seed]=seed;stack=[seed]
        while stack:
            i=stack.pop()
            for j,jump in neighbors[i]:
                image=images[i]+jump
                if labels[j]<0:
                    labels[j]=seed;images[j]=image;stack.append(j)
                elif np.any(image!=images[j]):
                    v=image-images[j]
                    if v[np.flatnonzero(v)[0]]<0:v=-v
                    cycles.add((seed,*map(int,v)))
    sizes=np.bincount(labels,minlength=len(pos))
    return GasComponents(labels,sizes,images,np.asarray(sorted(cycles),int).reshape(-1,4))

def component_overlap(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Sparse complete overlap matrix as previous-label,current-label,count."""
    previous=np.asarray(previous);current=np.asarray(current)
    if previous.shape!=current.shape or previous.ndim!=1 or np.any(previous<0) or np.any(current<0):
        raise ValueError("Aligned nonnegative stable-identity labels required")
    pairs,counts=np.unique(np.column_stack((previous,current)),axis=0,return_counts=True)
    result=np.column_stack((pairs,counts))
    if result[:,2].sum()!=len(current):raise AssertionError("Inventory lost")
    return result

def inventory_states(main: np.ndarray, contact: np.ndarray,
                     distance: np.ndarray, near_cutoff: float=8.5) -> np.ndarray:
    """0=main-contact, 1=main-other, 2=nonmain-near, 3=remaining."""
    main=np.asarray(main,bool);contact=np.asarray(contact,bool);distance=np.asarray(distance,float)
    if main.shape!=contact.shape or main.shape!=distance.shape or np.any(contact&~main):
        raise ValueError("Aligned main-cluster contact required")
    if not np.isfinite(distance).all():raise ValueError("Finite distances required")
    result=np.full(main.shape,3,np.uint8)
    result[~main&(distance<=near_cutoff)]=2;result[main]=1;result[contact]=0
    return result

def scaffold_translation(current: np.ndarray, reference: np.ndarray,
                         bounds: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Minimum-image rigid translation and residual per fixed scaffold ID.

    This registers translation only. Deformation/rotation must be diagnosed
    from the returned residuals rather than silently fitted away.
    """
    current=np.asarray(current,float);reference=np.asarray(reference,float)
    if current.shape!=reference.shape or current.ndim!=2 or current.shape[1]!=3:
        raise ValueError("Aligned fixed scaffold IDs required")
    lengths=np.diff(bounds,axis=1).ravel()
    delta=current-reference;delta-=np.rint(delta/lengths)*lengths
    shift=np.median(delta,axis=0)
    return shift,np.linalg.norm(delta-shift,axis=1)
