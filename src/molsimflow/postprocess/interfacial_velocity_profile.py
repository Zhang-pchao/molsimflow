"""Number-conserving instantaneous interfacial velocity profiles."""
from __future__ import annotations
import numpy as np


def height_velocity_moments(height: np.ndarray, velocity: np.ndarray,
                            edges: np.ndarray, selected: np.ndarray | None = None) -> dict:
    """Return counts and XYZ first/second moments in half-open height bins.

    Empty bins carry zero moments, not a defined zero velocity. Number-weighted
    means are sums/counts; averaging occupied-bin means is a different estimator.
    Heights and coordinate/frame choices are supplied explicitly by the caller.
    """
    h, v, e = [np.asarray(a, dtype=float) for a in (height, velocity, edges)]
    if (h.ndim != 1 or v.shape != (len(h), 3) or e.ndim != 1 or len(e)<2
            or not all(np.isfinite(a).all() for a in (h,v,e))
            or np.any(np.diff(e)<=0)):
        raise ValueError("Finite heights, N by 3 velocities and increasing edges required")
    keep = np.ones(len(h), dtype=bool) if selected is None else np.asarray(selected)
    if keep.dtype != bool or keep.shape != h.shape:
        raise ValueError("Selection must be a matching boolean vector")
    idx = np.searchsorted(e,h,side="right")-1
    valid = keep & (idx>=0) & (idx<len(e)-1)
    n = np.bincount(idx[valid], minlength=len(e)-1)
    sums = np.stack([np.bincount(idx[valid], weights=v[valid,j],
                                minlength=len(e)-1) for j in range(3)], axis=-1)
    squares = np.stack([np.bincount(idx[valid], weights=v[valid,j]**2,
                                   minlength=len(e)-1) for j in range(3)], axis=-1)
    return dict(count=n, velocity_sum=sums, velocity_squared_sum=squares,
                selected_count=int(keep.sum()), excluded_count=int((keep & ~valid).sum()))
