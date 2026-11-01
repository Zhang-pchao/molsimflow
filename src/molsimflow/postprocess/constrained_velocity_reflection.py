"""Constrained kinetic-energy preserving reflection of a linear velocity mode."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ConstrainedReflection:
    velocities: np.ndarray
    mode_before: float
    mode_after: float
    raw_before: float
    raw_after: float
    constraint_rank: int
    mode_norm_squared: float
    max_constraint_residual: float

def reflect_linear_velocity_mode(velocities: np.ndarray, masses: np.ndarray,
        positions: np.ndarray, target: np.ndarray,
        extra_constraints: np.ndarray | None = None) -> ConstrainedReflection:
    """Reflect the part of target·v orthogonal to total P, orbital L and extras.

    Positions are fixed, continuously imaged point/group COM positions. Target
    and extra rows are coefficients on physical velocities, with shapes (N,3)
    and (K,N,3). A rigid group may be represented by its COM and total mass;
    applying its output increment to every group atom preserves internal motion.
    Orbital angular momentum depends on the supplied coordinate images.
    """
    v=np.asarray(velocities,float);m=np.asarray(masses,float)
    x=np.asarray(positions,float);t=np.asarray(target,float)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or len(v)<2 or m.shape!=(len(v),)
            or x.shape!=v.shape or t.shape!=v.shape or np.any(m<=0)
            or not all(np.isfinite(a).all() for a in (v,m,x,t))):
        raise ValueError("Finite matching vectors and positive masses required")
    extras=np.empty((0,len(v),3)) if extra_constraints is None else np.asarray(extra_constraints,float)
    if extras.ndim!=3 or extras.shape[1:]!=v.shape or not np.isfinite(extras).all():
        raise ValueError("Extra constraints must have shape (K,N,3)")
    root=np.sqrt(m);r=x-np.average(x,weights=m,axis=0)
    C=np.zeros((6+len(extras),3*len(m)))
    for i,(a,b,c) in enumerate(r):
        C[:3,3*i:3*i+3]=root[i]*np.eye(3)
        C[3:6,3*i:3*i+3]=root[i]*np.array([[0,-c,b],[c,0,-a],[-b,a,0]])
    if len(extras):C[6:]=(extras/root[None,:,None]).reshape(len(extras),-1)
    norms=np.linalg.norm(C,axis=1);scaled=C[norms>0]/norms[norms>0,None]
    _,s,vh=np.linalg.svd(scaled,full_matrices=False)
    threshold=np.finfo(float).eps*max(scaled.shape)*s[0]
    rank=int(np.sum(s>threshold))
    physical_target=(t/root[:,None]).ravel()
    w=physical_target-vh[:rank].T@(vh[:rank]@physical_target)
    norm2=float(w@w)
    cutoff=64*np.finfo(float).eps*max(scaled.shape)*np.linalg.norm(physical_target)
    if np.linalg.norm(physical_target)==0 or norm2<=cutoff**2:
        raise ValueError("Requested mode is zero or entirely constrained")
    z=(root[:,None]*v).ravel();coordinate=float(w@z)
    changed=z-2*coordinate/norm2*w
    out=changed.reshape(v.shape)/root[:,None]
    return ConstrainedReflection(out,coordinate,float(w@changed),
        float(physical_target@z),float(physical_target@changed),rank,norm2,
        float(np.max(abs(C@(changed-z)))))
