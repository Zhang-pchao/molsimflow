"""Identity-preserving natural approaches with explicit membership censoring."""
import numpy as np

def approach_paths(distance, main, times, *, inner=4., outer=8.5):
    """Record all exterior departures until contact, return, loss, or window end.

    A path begins at the last observed exterior frame, not at a chosen success.
    Existing inner/buffer occupants at the window start do not invent origins.
    No interpolation, persistence filter, or independence assumption is applied.
    """
    d=np.asarray(distance,float);m=np.asarray(main,bool);t=np.asarray(times,float)
    if d.ndim!=2 or d.shape!=m.shape or len(t)!=len(d) or len(t)<2:
        raise ValueError("Aligned regular frame-by-molecule arrays required")
    if not np.isfinite(d).all() or not np.isfinite(t).all() or not 0<inner<outer:
        raise ValueError("Finite geometry and ordered positive boundaries required")
    dt=np.diff(t)
    if np.any(dt<=0) or not np.allclose(dt,dt[0],atol=1e-8,rtol=0):
        raise ValueError("Regular source-contiguous sampling required")
    out=[]
    for j in range(d.shape[1]):
        start=None
        for k in range(1,len(t)):
            if start is None and m[k-1,j] and d[k-1,j]>=outer and m[k,j] and d[k,j]<outer:
                start=k-1
            if start is None:continue
            kind=None
            if not m[k,j]:kind='membership_censored'
            elif d[k,j]<=inner:kind='contact'
            elif d[k,j]>=outer:kind='return'
            elif k==len(t)-1:kind='window_censored'
            if kind:
                out.append(dict(molecule=j,start=start,end=k,outcome=kind,
                                origin_ps=float(t[start]),end_ps=float(t[k]),
                                duration_ps=float(t[k]-t[start]),
                                minimum_distance_A=float(d[start:k+1,j].min())))
                start=None
    return out
