"""Identity-preserving site contributions and gap-aware sampled state changes."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SiteObservations:
    site_ids: np.ndarray
    state: np.ndarray
    molecule_count: np.ndarray
    frame_weight: np.ndarray


def aggregate_site_observations(site_id, eligible, state, denominator):
    """Group eligible molecules by exact site and frame, retaining multiplicity.

    States 0/1/2 denote two resolved environments and unresolved identity.
    Output -1 means no eligible observation, never an inferred physical state.
    Multiple molecules at one site must report the same environment.
    """
    ids=np.asarray(site_id);ok=np.asarray(eligible,bool);st=np.asarray(state)
    den=np.asarray(denominator,float)
    if ids.ndim!=2 or ids.shape!=ok.shape or ids.shape!=st.shape or den.shape!=(len(ids),):
        raise ValueError('Matching frame-by-molecule arrays and frame denominators required')
    if np.any(den<=0) or not np.isfinite(den).all() or np.any(den<ok.sum(axis=1)):
        raise ValueError('Positive whole-population denominators required')
    if not np.isin(st[ok],[0,1,2]).all() or np.any(ids[ok]<=0):
        raise ValueError('Valid eligible site identities and 0/1/2 states required')
    sites=np.unique(ids[ok]);states=np.full((len(ids),len(sites)),-1,np.int8)
    count=np.zeros(states.shape,np.int32)
    for i in range(len(ids)):
        present=np.flatnonzero(ok[i]);pos=np.searchsorted(sites,ids[i,present])
        low=np.full(len(sites),3,np.int8);high=np.full(len(sites),-1,np.int8)
        np.minimum.at(low,pos,st[i,present]);np.maximum.at(high,pos,st[i,present])
        np.add.at(count[i],pos,1)
        active=count[i]>0
        if np.any(low[active]!=high[active]):
            raise ValueError('Conflicting environment at the same site and frame')
        states[i,active]=low[active]
    return SiteObservations(sites,states,count,count/den[:,None])


def sampled_site_history(states,times,expected_dt,segments=None):
    """Classify visited environments and count changes on adjacent saved frames.

    Class 0=only environment 0 observed, 1=only 1, 2=both, 3=unresolved-only.
    Missing (-1), unresolved (2), cadence gaps and segment changes break runs.
    A reversal means two opposite changes inside one resolved observation run.
    No inference is made between saved frames or across a break. Single-type
    observation is not evidence of permanent site specialization.
    """
    a=np.asarray(states);t=np.asarray(times,float)
    seg=np.zeros(len(t),int) if segments is None else np.asarray(segments)
    if a.ndim!=2 or t.shape!=(len(a),) or seg.shape!=t.shape or not np.isin(a,[-1,0,1,2]).all():
        raise ValueError('Valid frame-by-site states and matching times/segments required')
    if not np.isfinite(t).all() or np.any(np.diff(t)<=0) or expected_dt<=0:
        raise ValueError('Strictly increasing finite times and positive cadence required')
    seen0=(a==0).any(axis=0);seen1=(a==1).any(axis=0)
    category=np.where(seen0&seen1,2,np.where(seen0,0,np.where(seen1,1,3))).astype(np.int8)
    n=a.shape[1];forward=np.zeros(n,int);backward=forward.copy();maxchanges=forward.copy();run=forward.copy();reversal=np.zeros(n,bool)
    for i in range(1,len(a)):
        cadence=bool(np.isclose(t[i]-t[i-1],expected_dt,rtol=0,atol=1e-8) and seg[i]==seg[i-1])
        connected=(a[i]>=0)&(a[i]<=1)&(a[i-1]>=0)&(a[i-1]<=1)&cadence
        changed=connected&(a[i]!=a[i-1])
        forward+=changed&(a[i-1]==0);backward+=changed&(a[i-1]==1)
        run=np.where(connected,run+changed,0)
        maxchanges=np.maximum(maxchanges,run);reversal|=run>=2
    return dict(site_class=category,forward=forward,backward=backward,reversal=reversal,max_changes_in_run=maxchanges)
