"""Transverse-lane coordination of changes in a fixed contact chart."""
from __future__ import annotations
import numpy as np

def lane_boundary_motion(before, after, xchart, *, min_cells=3, threshold=1.0):
    """Compare q05/q95 X edges within fixed Y rows shared by two masks.

    Rows are spatial support, never independent statistical samples. The caller
    supplies one validated unwrapped X chart and handles periodic topology.
    Appearance/disappearance rows remain excluded and explicitly counted.
    """
    b,a=np.asarray(before),np.asarray(after);x=np.asarray(xchart,float)
    if b.dtype!=bool or a.dtype!=bool or b.ndim!=2 or a.shape!=b.shape or x.shape!=b.shape:
        raise ValueError("Matching 2D masks and X chart required")
    if not np.isfinite(x).all() or min_cells<1 or not np.isfinite(threshold) or threshold<=0:
        raise ValueError("Finite chart and positive support and threshold required")
    supported=(b.sum(0)>=min_cells)&(a.sum(0)>=min_cells)
    front=[];rear=[]
    for j in np.flatnonzero(supported):
        low0,high0=np.quantile(x[b[:,j],j],[.05,.95])
        low1,high1=np.quantile(x[a[:,j],j],[.05,.95])
        front.append(high1-high0);rear.append(low1-low0)
    f,r=np.asarray(front),np.asarray(rear)
    mean=lambda v:float(np.mean(v))if len(v)else np.nan
    return dict(supported_lanes=int(supported.sum()),union_lanes=int(((b.sum(0)>=min_cells)|(a.sum(0)>=min_cells)).sum()),
                appeared_lanes=int(((b.sum(0)<min_cells)&(a.sum(0)>=min_cells)).sum()),
                disappeared_lanes=int(((b.sum(0)>=min_cells)&(a.sum(0)<min_cells)).sum()),
                front_mean=mean(f),rear_mean=mean(r),
                front_advance_fraction=mean(f>threshold),rear_release_fraction=mean(r>threshold),
                coordinated_forward_fraction=mean((f>threshold)&(r>threshold)),
                expansion_fraction=mean((f>threshold)&(r< -threshold)),
                backward_fraction=mean((f< -threshold)&(r< -threshold)),
                stall_fraction=mean((abs(f)<=threshold)&(abs(r)<=threshold)),
                translation_mean=mean((f+r)/2),deformation_mean=mean(f-r))
