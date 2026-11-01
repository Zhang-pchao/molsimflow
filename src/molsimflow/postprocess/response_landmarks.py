"""Descriptive response crossings without monotonicization or gap bridging."""
from __future__ import annotations
import numpy as np

def first_crossing(time, progress, level=0.5):
    """Return first upward crossing; flag left censoring and unavailable gaps.

    A crossing is interpolated only between adjacent finite observations.
    A missing observation followed by an above-level value is interval censored.
    """
    t=np.asarray(time,float);p=np.asarray(progress,float)
    if t.ndim!=1 or p.shape!=t.shape or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError("Expected same-size vectors and finite increasing time")
    for i in range(len(t)):
        if not np.isfinite(p[i]) or p[i]<level:
            continue
        if i==0:
            return dict(time=float(t[i]),status="LEFT_CENSORED")
        if not np.isfinite(p[i-1]):
            return dict(time=None,status="GAP_CENSORED")
        frac=(level-p[i-1])/(p[i]-p[i-1])
        return dict(time=float(t[i-1]+frac*(t[i]-t[i-1])),status="OBSERVED")
    return dict(time=None,status="NOT_REACHED")

def signed_progress(values, start, end):
    """Normalize increasing or decreasing response; preserve overshoot."""
    x=np.asarray(values,float)
    if not np.isfinite([start,end]).all() or abs(end-start)<=1e-12:
        return np.full_like(x,np.nan)
    return (x-start)/(end-start)

def endpoint_description(values, bins):
    """Describe endpoint means and within-endpoint spread, not uncertainty."""
    x=np.asarray(values,float)
    if bins<2 or 2*bins>len(x):
        raise ValueError("Need at least two disjoint endpoint bins")
    a=x[:bins];b=x[-bins:]
    start=float(np.mean(a));end=float(np.mean(b))
    sd=float(np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2))
    return dict(start=start,end=end,delta=end-start,endpoint_sd=sd,
                amplitude_over_spread=abs(end-start)/sd if sd>0 else None)
