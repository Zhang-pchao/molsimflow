"""Gap-aware radial core excursions, reusing the generic access observer."""
import numpy as np
from molsimflow.postprocess.core_to_core_access import excursions


def radial_core_labels(qualified_distance, other_distance, valid, near=3.2, far=3.8):
    """-1 invalid; 0 buffer; 1 qualified water near; 2 all solution O clear.

    Far requires every solution O, including unqualified/ambiguous ones, to
    lie beyond far. This prevents a change in water qualification alone from
    creating an apparent radial clearance. No metastable-state claim is made.
    """
    q=np.asarray(qualified_distance,float);o=np.asarray(other_distance,float);v=np.asarray(valid,bool)
    if q.shape!=o.shape or q.shape!=v.shape or not (0<=near<far):
        raise ValueError('Matching arrays and disjoint nonnegative radii required')
    if np.any(np.isnan(q[v])) or np.any(np.isnan(o[v])) or np.any(q[v]<0) or np.any(o[v]<0):
        raise ValueError('Valid observations require nonnegative distances without NaN')
    labels=np.zeros(q.shape,np.int8)
    labels[q<=near]=1;labels[(q>far)&(o>far)]=2;labels[~v]=-1
    return labels


def segmented_radial_excursions(labels,times,segments,expected_dt=.5):
    """Enumerate near/clear departures inside every contiguous valid record.

    Existing excursions define return, opposite-core hit and censoring. A
    segment is cut at invalid observations, cadence gaps or source changes.
    Intervals start at the last origin-core observation and end at the first
    return/hit, or the last valid observation if censored. A record initially
    in the buffer has no inferred origin. Endpoints can be shared by successive
    passages; unions of frame masks must avoid double counting those frames.
    """
    a=np.asarray(labels);t=np.asarray(times,float);seg=np.asarray(segments)
    if a.ndim!=2 or t.shape!=(len(a),) or seg.shape!=t.shape or not np.isin(a,[-1,0,1,2]).all():
        raise ValueError('Frame-by-site labels and matching time/source arrays required')
    if expected_dt<=0 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Positive cadence and increasing finite times required')
    edge=np.isclose(np.diff(t),expected_dt,rtol=0,atol=1e-8)&(seg[1:]==seg[:-1])
    records=[]
    for j in range(a.shape[1]):
        valid=a[:,j]>=0
        connected=valid[1:]&valid[:-1]&edge
        starts=np.flatnonzero(valid&np.r_[True,~connected])
        ends=np.flatnonzero(valid&np.r_[~connected,True])
        for start,end in zip(starts,ends):
            for origin,target in [(1,2),(2,1)]:
                for r in excursions(a[start:end+1,j:j+1],origin,target):
                    records.append(dict(site_column=j,run_start=int(start),run_end=int(end),start=int(start+r['start']),end=int(start+r['end']),origin=origin,target=target,outcome=r['outcome']))
    return sorted(records,key=lambda r:(r['site_column'],r['start'],r['end']))
