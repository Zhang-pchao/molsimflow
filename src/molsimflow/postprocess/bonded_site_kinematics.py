"""Finite distance changes resolved into point and bonded-site motion."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from molsimflow.io.lammps_dump import minimum_image_vectors

@dataclass(frozen=True)
class BondedSiteDistance:
    distances: np.ndarray
    components: np.ndarray
    secant: np.ndarray
    site_rotation_angle_deg: float
    site_lengths: np.ndarray
    closure_error: float

def decompose_bonded_site_distance(point0: np.ndarray, point1: np.ndarray,
        site0: np.ndarray, site1: np.ndarray,
        reference0: np.ndarray, reference1: np.ndarray,
        box: np.ndarray, tolerance: float = 1e-9) -> BondedSiteDistance:
    """Return point/reference, negative site rotation, negative site stretch.

    Input positions must be temporally unwrapped. A fixed minimum-image offset
    must describe both endpoints. Symmetric bond-length/direction splitting
    closes exactly; components are kinematics, not causal or energy fractions.
    """
    vectors = np.asarray([point0,point1,site0,site1,reference0,reference1],float)
    box=np.asarray(box,float)
    if (vectors.shape!=(6,3) or box.shape!=(3,) or np.any(box<=0)
            or not np.isfinite(vectors).all() or not np.isfinite(box).all()
            or not np.isfinite(tolerance) or tolerance<=0):
        raise ValueError("Invalid geometry, box, or tolerance")
    p0,p1,s0,s1,c0,c1=vectors
    def aligned(v0,v1):
        a=minimum_image_vectors(v0,box)
        b=v1-(v0-a)
        if np.max(abs(b-minimum_image_vectors(v1,box)))>tolerance:
            raise ValueError("Endpoints require different minimum-image offsets")
        return a,b
    r0,r1=aligned(p0-s0,p1-s1)
    b0,b1=aligned(s0-c0,s1-c1)
    d=np.linalg.norm([r0,r1],axis=1)
    lengths=np.linalg.norm([b0,b1],axis=1)
    if d.sum()<=tolerance or np.min(lengths)<=tolerance:
        raise ValueError("Undefined secant or site bond")
    l0,l1=lengths;u0,u1=b0/l0,b1/l1
    secant=(r0+r1)/d.sum()
    motions=np.array([(p1-c1)-(p0-c0),
                      -(l1+l0)/2*(u1-u0),
                      -(l1-l0)/2*(u1+u0)])
    components=motions@secant
    error=abs(float(components.sum()-(d[1]-d[0])))
    if error>tolerance or np.max(abs(motions.sum(0)-(r1-r0)))>tolerance:
        raise ValueError("Distance decomposition did not close")
    angle=float(np.degrees(np.arccos(np.clip(u0@u1,-1,1))))
    return BondedSiteDistance(d,components,secant,angle,lengths,error)
