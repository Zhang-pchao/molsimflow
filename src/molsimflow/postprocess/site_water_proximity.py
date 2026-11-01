"""Same-site proximity to geometrically qualified solution water in PPP boxes."""
from dataclasses import dataclass
import numpy as np
from molsimflow.postprocess.constant_force_species_timeseries import _periodic_query

@dataclass(frozen=True)
class SiteWaterProximity:
    qualified_distance: np.ndarray
    other_distance: np.ndarray
    qualified_id: np.ndarray

def site_water_proximity(site_xyz, solution_xyz, solution_ids, h_count, ambiguous, bounds):
    """Find nearest two-H unambiguous and other solution O for each solid site.

    Counts and ambiguity must come from one unique-H, all-O/C assignment in
    the same frame. This is geometric proximity, not a hydrogen-bond test.
    """
    sites=np.asarray(site_xyz,float).reshape(-1,3)
    xyz=np.asarray(solution_xyz,float).reshape(-1,3)
    ids=np.asarray(solution_ids,np.int64)
    count=np.asarray(h_count);amb=np.asarray(ambiguous,bool)
    if not (len(xyz)==len(ids)==len(count)==len(amb)) or len(np.unique(ids))!=len(ids):
        raise ValueError('Matching unique solution oxygen identities required')
    if not np.isfinite(sites).all() or not np.isfinite(xyz).all():
        raise ValueError('Finite coordinates required')
    good=(count==2)&~amb
    near=np.full(len(sites),-1,dtype=np.int64)
    i,d=_periodic_query(sites,xyz[good],bounds,periodic_z=True)
    if good.any():near=ids[good][i]
    _,other=_periodic_query(sites,xyz[~good],bounds,periodic_z=True)
    return SiteWaterProximity(d,other,near)

def water_presence_category(qualified_distance, other_distance, cutoff=3.5):
    """0=no solution O nearby; 1=qualified water; 2=only unqualified O.

    Qualified water takes precedence even if another unqualified O is nearby.
    Code 2 is unresolved with respect to water; code 0 is not a dry-state label.
    """
    q,o=np.broadcast_arrays(qualified_distance,other_distance)
    return np.where(q<=cutoff,1,np.where(o<=cutoff,2,0)).astype(np.int8)
