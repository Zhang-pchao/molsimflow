"""Prospective patch registration and complete-window event families."""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

def periodic_xy_distances(left: np.ndarray, right: np.ndarray,
                          lengths: np.ndarray) -> np.ndarray:
    left=np.asarray(left,float).reshape(-1,2);right=np.asarray(right,float).reshape(-1,2)
    lengths=np.asarray(lengths,float)
    d=left[:,None,:]-right[None,:,:];d-=np.rint(d/lengths)*lengths
    return np.linalg.norm(d,axis=-1)

def uncontacted_front_candidates(grid_xy: np.ndarray, recent_contact_xy: np.ndarray,
                                supported_seed_xy: np.ndarray, lengths: np.ndarray,
                                *, patch_radius: float=6.,front_radius: float=12.) -> tuple[np.ndarray,np.ndarray]:
    """Use current/past contact only; no outcomes or future geometry accepted."""
    grid=np.asarray(grid_xy,float);recent=np.asarray(recent_contact_xy,float).reshape(-1,2)
    seed=np.asarray(supported_seed_xy,float).reshape(-1,2);lengths=np.asarray(lengths,float)
    nearest=np.full(len(grid),np.inf)
    if len(seed):nearest=cKDTree(seed%lengths,boxsize=lengths).query(grid%lengths)[0]
    clear=np.ones(len(grid),bool)
    if len(recent):clear=cKDTree(recent%lengths,boxsize=lengths).query(grid%lengths)[0]>patch_radius
    return clear&(nearest<=front_radius),nearest

def complete_window_indices(steps: np.ndarray, anchor: int, *, cadence: int=200,
                            pre: int=20000,post: int=40000) -> tuple[int,int] | None:
    """Reject missing frames and boundaries; never bridge a reset or a gap.

    Caller passes a single previously qualified continuous epoch.
    """
    steps=np.asarray(steps,dtype=np.int64)
    lo=np.searchsorted(steps,anchor-pre);hi=np.searchsorted(steps,anchor+post)
    if lo>=len(steps) or hi>=len(steps):return None
    if steps[lo]!=anchor-pre or steps[hi]!=anchor+post:return None
    if not np.all(np.diff(steps[lo:hi+1])==cadence):return None
    return int(lo),int(hi)

def event_family_labels(anchors: np.ndarray,centers: np.ndarray,seed_members: list[set[int]],
                        lengths: np.ndarray,*,pre: int=20000,post: int=40000,
                        patch_radius: float=6.) -> np.ndarray:
    """Join overlapping full windows/patches that share a seed molecule.

    Families are transitive: a continuous cascade crossing a nominal split is
    not fragmented into independent observations.
    """
    anchors=np.asarray(anchors,dtype=np.int64);centers=np.asarray(centers,float)
    n=len(anchors)
    if centers.shape!=(n,2) or len(seed_members)!=n:raise ValueError("Aligned candidates required")
    parent=np.arange(n)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    order=np.argsort(anchors,kind='stable')
    for ii,i in enumerate(order):
        for j in order[ii+1:]:
            if anchors[j]-anchors[i]>pre+post:break
            if not seed_members[i].intersection(seed_members[j]):continue
            d=centers[i]-centers[j];d-=np.rint(d/lengths)*lengths
            if np.linalg.norm(d)>2*patch_radius:continue
            x,y=find(int(i)),find(int(j))
            if x!=y:parent[max(x,y)]=min(x,y)
    return np.array([find(i) for i in range(n)])

def temporal_family_allocation(anchors: np.ndarray,families: np.ndarray,split: int,
                               *,pre: int=20000,post: int=40000) -> np.ndarray:
    """Purge an entire family if any member touches/crosses the split."""
    anchors=np.asarray(anchors);families=np.asarray(families)
    result=np.full(len(anchors),'purged',dtype='U9')
    for family in np.unique(families):
        mask=families==family
        if np.all(anchors[mask]+post<split):result[mask]='discovery'
        elif np.all(anchors[mask]-pre>split):result[mask]='holdout'
    return result
