"""Gas/group reorientation with local kinetic energy and global invariants held."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GroupReorientation:
    velocities: np.ndarray
    selected_projection_before: float
    selected_projection_after: float
    selected_relative_speed_before: float
    selected_relative_speed_after: float
    compensation_mass_weighted_norm: float
    constraint_rank: int


def reorient_group_with_energy_compensation(
    velocities: np.ndarray, masses: np.ndarray, positions: np.ndarray,
    selected: int, axis: np.ndarray, target_projection: float,
    reference_velocity: np.ndarray,
) -> GroupReorientation:
    """Reorient one group at fixed local speed; compensate other groups.

    The selected group uses the nearest velocity at its original speed relative
    to reference_velocity and requested laboratory-frame axis projection.
    Remaining groups take the nearest mass-weighted velocity satisfying the
    required total linear/orbital angular momentum and total kinetic energy.
    Positions must be fixed continuous images. Apply each COM increment to all
    constituent atoms to retain internal velocities.

    This is a specified two-stage intervention, not a global constrained
    minimum over selected rotations and compensation, nor an equilibrium map.
    Orbital angular momentum depends on the supplied image convention.
    """
    v=np.asarray(velocities,float);m=np.asarray(masses,float)
    x=np.asarray(positions,float);axis=np.asarray(axis,float)
    ref=np.asarray(reference_velocity,float)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or len(v)<4 or x.shape!=v.shape
            or m.shape!=(len(v),) or np.any(m<=0) or axis.shape!=(3,)
            or ref.shape!=(3,) or not isinstance(selected,(int,np.integer))
            or not 0<=selected<len(v) or not np.isfinite(target_projection)
            or not all(np.isfinite(a).all() for a in (v,m,x,axis,ref))):
        raise ValueError("Finite matching inputs, valid selected index and positive masses required")
    norm=float(np.linalg.norm(axis))
    if abs(norm-1)>1e-10:
        raise ValueError("Axis must be a unit vector")
    eps=np.finfo(float).eps
    w=v[selected]-ref;speed=float(np.linalg.norm(w))
    p=float(target_projection-ref@axis)
    tol=256*eps*max(1.,speed,abs(p))
    if abs(p)>speed+tol:
        raise ValueError("Requested selected projection exceeds fixed-local-energy envelope")
    p=float(np.clip(p,-speed,speed))
    transverse=w-(w@axis)*axis
    tn=float(np.linalg.norm(transverse))
    new_tn=float(np.sqrt(max(0.,(speed-p)*(speed+p))))
    if tn<=tol and new_tn>tol:
        raise ValueError("Degenerate selected transverse direction")
    selected_new=ref+p*axis+(new_tn/tn*transverse if tn>tol else 0.)
    keep=np.arange(len(v))!=selected
    r=x-np.average(x,weights=m,axis=0)
    A=np.zeros((6,3*len(v)))
    for i,(a,b,c) in enumerate(r):
        A[:3,3*i:3*i+3]=np.sqrt(m[i])*np.eye(3)
        A[3:,3*i:3*i+3]=np.sqrt(m[i])*np.array([[0,-c,b],[c,0,-a],[-b,a,0]])
    y=(np.sqrt(m)[:,None]*v).ravel()
    selected_columns=np.arange(3*selected,3*selected+3)
    remaining_columns=np.flatnonzero(np.repeat(keep,3))
    As=A[:,selected_columns];Ar=A[:,remaining_columns]
    ys=np.sqrt(m[selected])*selected_new
    rhs=A@y-As@ys
    scale=np.linalg.norm(Ar,axis=1)
    active=scale>0
    if np.any(np.abs(rhs[~active])>tol):
        raise ValueError("Remaining groups cannot compensate the requested momentum")
    C=Ar[active]/scale[active,None];b=rhs[active]/scale[active]
    u,s,vh=np.linalg.svd(C,full_matrices=False)
    rank=int(np.sum(s>eps*max(C.shape)*s[0]))
    basis=vh[:rank]
    fixed=basis.T@((u[:,:rank].T@b)/s[:rank])
    if np.max(np.abs(C@fixed-b))>512*eps*max(C.shape)*max(1.,np.max(np.abs(b))):
        raise ValueError("Incompatible momentum compensation constraints")
    yr=y[remaining_columns]
    free=yr-basis.T@(basis@yr)
    free_norm=float(np.linalg.norm(free))
    radius2=float(y@y-ys@ys-fixed@fixed)
    energy_tol=512*eps*max(1.,float(y@y))
    if radius2 < -energy_tol:
        raise ValueError("Insufficient kinetic energy for momentum compensation")
    radius=float(np.sqrt(max(0.,radius2)))
    if free_norm<=tol and radius>tol:
        raise ValueError("Degenerate compensation direction")
    compensated=fixed+(radius/free_norm*free if free_norm>tol else 0.)
    out=v.copy();out[selected]=selected_new
    out[keep]=compensated.reshape(-1,3)/np.sqrt(m[keep])[:,None]
    return GroupReorientation(
        out,float(v[selected]@axis),float(out[selected]@axis),speed,
        float(np.linalg.norm(out[selected]-ref)),
        float(np.linalg.norm(compensated-yr)),rank,
    )
