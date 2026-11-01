"""Current geometric hydrogen ownership with explicit O/C ambiguity."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from molsimflow.postprocess.constant_force_species_timeseries import _periodic_query


@dataclass(frozen=True)
class HydrogenOwnership:
    owner_id: np.ndarray
    oxygen_h_count: np.ndarray
    oxygen_ambiguous: np.ndarray
    hydrogen_ambiguous: np.ndarray


def oxygen_hydrogen_ownership(h_xyz, h_ids, o_xyz, o_ids, c_xyz, c_ids, bounds,
                             *, cutoff=1.3, ambiguity_gap=0.1, periodic_z=True):
    """Assign each unique H to its closest O/C within cutoff with periodic XY and configurable Z.

    Preserve the established O-over-C preference at an exact O/C distance tie.
    An H is ambiguous when two candidate O/C atoms lie within cutoff and their
    distance gap is <= ambiguity_gap. Mark every O within that gap of the nearest
    candidate, including an O that lost the deterministic assignment. Ambiguous
    O classification takes precedence over its nominal H count.
    """
    h_xyz, o_xyz, c_xyz = [np.asarray(a, float).reshape(-1, 3) for a in (h_xyz, o_xyz, c_xyz)]
    h_ids, o_ids, c_ids = [np.asarray(a, np.int64) for a in (h_ids, o_ids, c_ids)]
    for xyz, ids in [(h_xyz,h_ids),(o_xyz,o_ids),(c_xyz,c_ids)]:
        if len(xyz)!=len(ids) or len(np.unique(ids))!=len(ids) or not np.isfinite(xyz).all():
            raise ValueError('Unique atom IDs and finite matching coordinates required')
    if not len(o_ids) or len(np.unique(np.concatenate((h_ids,o_ids,c_ids)))) != len(h_ids)+len(o_ids)+len(c_ids):
        raise ValueError('Nonempty oxygen and disjoint atom identities required')
    bounds=np.asarray(bounds,float).copy()
    if not periodic_z:
        # The empty Z seam exceeds twice the assignment cutoff, so no accepted
        # owner or ambiguity candidate can connect through that artificial seam.
        all_z=np.concatenate((h_xyz[:,2],o_xyz[:,2],c_xyz[:,2]))
        pad=max(1.,cutoff)+1.
        bounds[2]=[all_z.min()-pad,all_z.max()+pad]
    oi, od = _periodic_query(h_xyz, o_xyz, bounds, periodic_z=True)
    ci, cd = _periodic_query(h_xyz, c_xyz, bounds, periodic_z=True)
    to_o=(od<=cutoff)&((cd>cutoff)|(od<=cd))
    to_c=(cd<=cutoff)&((od>cutoff)|(cd<od))
    owner=np.full(len(h_ids),-1,dtype=np.int64)
    owner[to_o]=o_ids[oi[to_o]]
    if len(c_ids):owner[to_c]=c_ids[ci[to_c]]
    count=np.bincount(oi[to_o],minlength=len(o_ids))
    xyz=np.vstack((o_xyz,c_xyz));lengths=np.diff(bounds,axis=1).ravel()
    tree=cKDTree((xyz-bounds[:,0])%lengths,boxsize=lengths)
    distances,_=tree.query((h_xyz-bounds[:,0])%lengths,k=2)
    ambiguous=(distances[:,1]<=cutoff)&((distances[:,1]-distances[:,0])<=ambiguity_gap)
    affected=np.zeros(len(o_ids),dtype=bool)
    for j in np.flatnonzero(ambiguous):
        candidates=np.asarray(tree.query_ball_point((h_xyz[j]-bounds[:,0])%lengths,min(cutoff,distances[j,0]+ambiguity_gap)+1e-12),int)
        delta=xyz[candidates]-h_xyz[j];delta-=np.rint(delta/lengths)*lengths
        dd=np.linalg.norm(delta,axis=1)
        candidates=candidates[(dd<=cutoff)&(dd-distances[j,0]<=ambiguity_gap)]
        affected[candidates[candidates<len(o_ids)]]=True
    return HydrogenOwnership(owner,count,affected,ambiguous)


def oxygen_h_category(count, ambiguous):
    """0=no H, 1=one H, 2=two or more H, 3=ambiguous (takes precedence)."""
    result=np.minimum(np.asarray(count),2).astype(np.int8)
    result[np.asarray(ambiguous,bool)]=3
    return result
