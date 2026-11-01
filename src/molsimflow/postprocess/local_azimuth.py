"""Conservative four-atom azimuth restraint in a moving local reference frame.

Atom order is H, O, S, R. S->O defines z; S->R defines the projected x
reference. All four coordinates are differentiated. Only local edges in an
orthorhombic periodic box are supported; atom identity is a caller contract.
"""
from __future__ import annotations
from typing import NamedTuple
import numpy as np

class AzimuthBias(NamedTuple):
    energy: float
    forces: np.ndarray
    cosine: float
    sine: float
    edge_lengths: np.ndarray
    projections: np.ndarray
    local_positions: np.ndarray

def periodic_azimuth_bias(
    positions: np.ndarray, lengths: np.ndarray, reference: float, kappa: float,
    *, min_projection: float = 0.2, max_edge: float = 2.5,
) -> AzimuthBias:
    """Return U=kappa*(1-cos(phi-reference)) and negative Cartesian gradient.

    A degenerate azimuth is rejected rather than regularized into another
    potential. Edges must lie strictly below half the shortest cell length,
    avoiding minimum-image branch discontinuities. No chemistry is inferred.
    """
    r=np.asarray(positions,dtype=float);box=np.asarray(lengths,dtype=float)
    if r.shape!=(4,3) or box.shape!=(3,):
        raise ValueError("Expected four 3D positions and three box lengths.")
    if not all(np.isfinite(v).all() for v in (r,box,np.array([reference,kappa,min_projection,max_edge]))):
        raise ValueError("Inputs must be finite.")
    if min_projection<=0 or max_edge<=min_projection or kappa<0 or np.any(box<=2*max_edge):
        raise ValueError("Invalid bounds or local periodic support.")
    def mic(v): return v-np.rint(v/box)*box
    h=mic(r[0]-r[1]);b=mic(r[1]-r[2]);a=mic(r[3]-r[2])
    edges=np.array([np.linalg.norm(h),np.linalg.norm(b),np.linalg.norm(a)])
    if np.any(edges<=min_projection) or np.any(edges>=max_edge):
        raise ValueError("Local edge outside admitted domain.")
    z=b/edges[1];alpha=a@z;v=a-alpha*z
    nv=np.linalg.norm(v)
    if nv<=min_projection: raise ValueError("Degenerate reference projection.")
    x=v/nv;y=np.cross(z,x)
    p=h@x;q=h@y;n=np.hypot(p,q)
    if n<=min_projection: raise ValueError("Degenerate azimuth projection.")
    c0=np.cos(reference);s0=np.sin(reference);s=p*c0+q*s0
    gp=-kappa*(c0/n-s*p/n**3);gq=-kappa*(s0/n-s*q/n**3)
    gh=gp*x+gq*y
    gy=gq*h;gx=gp*h+np.cross(gy,z);gz=np.cross(x,gy)
    gv=(gx-x*(x@gx))/nv
    ga=gv-z*(gv@z)
    gz+=-alpha*gv-a*(gv@z)
    gb=(gz-z*(gz@z))/edges[1]
    gradients=np.stack([gh,gb-gh,-gb-ga,ga])
    local=np.stack([b+h,b,np.zeros(3),a])
    return AzimuthBias(float(kappa*(1-s/n)),-gradients,float(p/n),float(q/n),
                       edges,np.array([nv,n]),local)
