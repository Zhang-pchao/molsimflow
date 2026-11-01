"""Energy- and momentum-preserving group-relative velocity reflection."""
from __future__ import annotations
import numpy as np

def reflect_group_relative_velocity(velocities: np.ndarray, masses: np.ndarray,
                                    selected: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Reverse one component of the selected/rest COM relative velocity.

    Both nonempty groups receive uniform velocity shifts. Total linear momentum,
    kinetic energy, tangential relative COM velocity, and within-group relative
    velocities are preserved. Positions are not inputs and angular momentum is
    not constrained. This is an imposed initial-condition intervention.
    """
    v=np.asarray(velocities,float);m=np.asarray(masses,float)
    mask=np.asarray(selected);axis=np.asarray(axis,float)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or m.shape!=(len(v),)
            or mask.shape!=(len(v),) or mask.dtype!=np.dtype(bool)
            or not mask.any() or mask.all() or np.any(m<=0)
            or axis.shape!=(3,) or not all(np.isfinite(x).all() for x in (v,m,axis))
            or np.linalg.norm(axis)==0):
        raise ValueError("Finite velocities, positive masses, two groups and a nonzero axis required")
    u=axis/np.linalg.norm(axis);ma,mb=m[mask].sum(),m[~mask].sum()
    va=np.average(v[mask],axis=0,weights=m[mask])
    vb=np.average(v[~mask],axis=0,weights=m[~mask])
    radial=float((va-vb)@u)
    result=v.copy()
    result[mask]-=2*mb/(ma+mb)*radial*u
    result[~mask]+=2*ma/(ma+mb)*radial*u
    return result

def reflect_group_mode_with_angular_constraint(
        velocities: np.ndarray, masses: np.ndarray, selected: np.ndarray,
        axis: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reflect the group-relative mode orthogonal to total P and orbital L.

    Inputs describe point COM velocities and positions, not internal rotations.
    Angular momentum is defined on the supplied continuous coordinate images.
    The constrained mode coordinate reverses; the raw approach velocity also
    contains the unchanged collective rotation and need not reverse exactly.
    """
    # Reuse the basic operator's validation; its result is not used.
    reflect_group_relative_velocity(velocities, masses, selected, axis)
    v=np.asarray(velocities,float);m=np.asarray(masses,float)
    mask=np.asarray(selected);u=np.asarray(axis,float);u=u/np.linalg.norm(u)
    pos=np.asarray(positions,float)
    if pos.shape!=v.shape or not np.isfinite(pos).all():
        raise ValueError("Finite COM positions matching velocity shape required")
    r=pos-np.average(pos,axis=0,weights=m);root=np.sqrt(m)
    C=np.zeros((6,3*len(m)))
    for i,(x,y,z) in enumerate(r):
        C[:3,3*i:3*i+3]=root[i]*np.eye(3)
        C[3:,3*i:3*i+3]=root[i]*np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    _,singular,vh=np.linalg.svd(C,full_matrices=False)
    rank=int(np.sum(singular>np.finfo(float).eps*max(C.shape)*singular[0]))
    target=np.where(mask,root/m[mask].sum(),-root/m[~mask].sum())[:,None]*u
    t=target.ravel()
    w=t-vh[:rank].T@(vh[:rank]@t)
    norm2=float(w@w)
    if norm2<1e-24:
        raise ValueError("Requested relative mode is entirely constrained")
    z=(root[:,None]*v).ravel()
    coordinate=float(w@z)
    zw=z-2*coordinate/norm2*w
    out=zw.reshape(v.shape)/root[:,None]
    return out,dict(constraint_rank=rank,mode_coordinate_before=coordinate,
                    mode_coordinate_after=float(w@zw),
                    raw_relative_before=float(t@z),raw_relative_after=float(t@zw),
                    mode_norm_squared=norm2,
                    constraint_residual=float(np.max(np.abs(C@(zw-z)))))
