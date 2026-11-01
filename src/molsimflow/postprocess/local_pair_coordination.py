"""Smooth local pair coordination for orthorhombic periodic cells.

These NumPy values and derivatives are an offline reference implementation.
They do not install a simulation-engine bias, provide a PMF, or define kinetics.
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np


class Coordination(NamedTuple):
    value: float
    point_gradients: np.ndarray
    anchor_gradient: np.ndarray


def compact_switch(
    distances: np.ndarray, inner: float, outer: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a C2 compact decreasing quintic and its radial derivative.

    The value is one at r <= inner, zero at r >= outer. The maximum
    derivative magnitude is 1.875 / (outer - inner).
    """
    r = np.asarray(distances, dtype=float)
    if not np.isfinite([inner, outer]).all() or not 0 <= inner < outer:
        raise ValueError("Require finite 0 <= inner < outer.")
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("Distances must be finite and nonnegative.")
    u = np.clip((r - inner) / (outer - inner), 0.0, 1.0)
    value = 1.0 - 10.0*u**3 + 15.0*u**4 - 6.0*u**5
    derivative = -30.0*u**2*(1.0-u)**2 / (outer-inner)
    return value, derivative


def pair_coordination(
    points: np.ndarray,
    anchor: np.ndarray,
    lengths: np.ndarray,
    inner: float,
    outer: float,
    weights: np.ndarray | None = None,
) -> Coordination:
    """Sum weighted compact switches and differentiate both pair endpoints.

    All three axes are periodic. The cutoff must be below half the shortest
    cell length, so minimum-image boundaries lie outside the switch support.
    Atoms are selected by the caller; no hard species or nearest-atom switch
    occurs inside this function. A bias U(q) has force -dU/dq times these
    gradients. The anchor reaction is required for zero net bias force.
    """
    xyz = np.asarray(points, dtype=float)
    a = np.asarray(anchor, dtype=float)
    box = np.asarray(lengths, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or a.shape != (3,) or box.shape != (3,):
        raise ValueError("Expected points (N,3), anchor (3,), lengths (3,).")
    if not all(np.isfinite(v).all() for v in (xyz, a, box)) or np.any(box <= 0):
        raise ValueError("Coordinates and positive cell lengths must be finite.")
    if outer >= 0.5*box.min():
        raise ValueError("Support must be below half the shortest box length.")
    w = np.ones(len(xyz)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(xyz),) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Weights must be finite, nonnegative, and match points.")
    delta = xyz-a
    delta -= np.rint(delta/box)*box
    radius = np.linalg.norm(delta, axis=1)
    value, derivative = compact_switch(radius, inner, outer)
    radial = np.divide(w*derivative, radius, out=np.zeros_like(radius), where=radius > 0)
    gradients = radial[:,None]*delta
    return Coordination(float(w@value), gradients, -gradients.sum(axis=0))

def soft_minimum_distance(
    points: np.ndarray,
    anchor: np.ndarray,
    lengths: np.ndarray,
    softness: float = 0.3,
    taper_inner: float = 8.0,
    cutoff: float = 10.0,
    regularizer: float = 0.05,
) -> Coordination:
    """Smooth identity-invariant nearest-distance proxy and both gradients.

    d = -a log[exp(-cutoff/a) + sum(s(r)*exp(-sqrt(r^2+e^2)/a))].
    s is the compact quintic from taper_inner to cutoff. A constant reservoir
    term prevents log(0); it does not represent a real atom. The distance is
    capped at cutoff when no atom is in support. The positive regularizer makes
    coincident-point gradients regular. This is not an exact minimum distance.
    """
    xyz = np.asarray(points, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    box = np.asarray(lengths, dtype=float)
    # Share shape, finite-value, radius and periodic-support validation.
    pair_coordination(xyz, anchor, box, taper_inner, cutoff)
    if not np.isfinite([softness, regularizer]).all() or min(softness, regularizer) <= 0:
        raise ValueError("Softness and regularizer must be finite and positive.")
    delta = xyz - anchor
    delta -= np.rint(delta / box) * box
    radius = np.linalg.norm(delta, axis=1)
    rho = np.sqrt(radius**2 + regularizer**2)
    switch, derivative = compact_switch(radius, taper_inner, cutoff)
    # Shift all exponents by a constant selected only for stable arithmetic.
    shift = min(cutoff, float(rho.min())) if len(rho) else cutoff
    exp_weight = np.exp((shift-rho)/softness)
    reservoir = np.exp((shift-cutoff)/softness)
    total = reservoir + float(switch @ exp_weight)
    value = shift-softness*np.log(total)
    ds_over_r = np.divide(derivative, radius, out=np.zeros_like(radius), where=radius > 0)
    radial = exp_weight*(switch/rho-softness*ds_over_r)/total
    gradients = radial[:,None]*delta
    return Coordination(float(value), gradients, -gradients.sum(axis=0))
