"""Observed exterior visits and returns; samples never establish continuous paths."""
from __future__ import annotations
import numpy as np


def observed_exterior_return(contact, exterior, times, expected_dt):
    """Classify a complete fixed-horizon sampled path beginning in contact.

    The caller supplies one source-contiguous window and a frozen origin cohort.
    The contact and exterior regions must be disjoint. No observed visit is not
    proof that an inter-frame trajectory never entered the exterior region.
    """
    a=np.asarray(contact);b=np.asarray(exterior);t=np.asarray(times,float)
    if a.ndim!=1 or a.shape!=b.shape or a.shape!=t.shape or not len(a):
        raise ValueError('Aligned nonempty one-dimensional paths required')
    if not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():
        raise ValueError('Unknown states cannot be interpreted as absence')
    if not np.isfinite(t).all() or expected_dt<=0 or not np.allclose(np.diff(t),expected_dt,rtol=0,atol=1e-7):
        raise ValueError('A source-contiguous regular sampled window is required')
    a=a.astype(bool);b=b.astype(bool)
    if not a[0] or np.any(a&b):raise ValueError('Origin contact and disjoint regions required')
    lost=np.flatnonzero(~a);visits=np.flatnonzero(b)
    result=dict(observed_label_loss=bool(len(lost)),label_return_observed=bool(len(lost) and a[lost[0]+1:].any()),
        exterior_visit_observed=bool(len(visits)),exterior_return_observed=False,
        visit_lower_ps=None,visit_upper_ps=None,return_lower_ps=None,return_upper_ps=None,
        horizon_ps=float(t[-1]-t[0]),status='NO_EXTERIOR_VISIT_OBSERVED')
    if not len(visits):return result
    i=int(visits[0]);back=np.flatnonzero(a[i+1:])
    result.update(visit_lower_ps=float(t[i-1]-t[0]),visit_upper_ps=float(t[i]-t[0]),status='VISITED_RETURN_RIGHT_CENSORED')
    if len(back):
        j=i+1+int(back[0]);result.update(exterior_return_observed=True,return_lower_ps=float(t[j-1]-t[0]),return_upper_ps=float(t[j]-t[0]),status='VISITED_AND_RETURNED')
    return result
