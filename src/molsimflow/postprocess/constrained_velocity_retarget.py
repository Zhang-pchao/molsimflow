"""Minimum-change velocity retargeting at fixed kinetic energy and momentum."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConstrainedRetarget:
    velocities: np.ndarray
    raw_before: float
    raw_after: float
    attainable_min: float
    attainable_max: float
    constraint_rank: int
    mass_weighted_change_norm: float
    max_constraint_residual: float


def retarget_linear_velocity_mode(
    velocities: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    target: np.ndarray,
    target_value: float,
    extra_constraints: np.ndarray | None = None,
) -> ConstrainedRetarget:
    """Set sum(target * v) with the smallest mass-weighted velocity change.

    Total linear momentum, orbital angular momentum about fixed continuously
    imaged positions, extra linear observables, and kinetic energy are held.
    Rigid molecular groups can be represented by their centers of mass; apply
    each group increment equally to its atoms to retain internal velocities.

    In mass-weighted coordinates the linear constraints fix an affine plane.
    Fixed kinetic energy gives a sphere inside that plane. The requested
    observable fixes one sphere coordinate. Keeping the direction of the
    remaining coordinates minimizes the distance to the original velocity.

    This is a deterministic intervention, not an equilibrium sampler or a
    volume-preserving/canonical transformation. Angular momentum is specific
    to the supplied periodic images. Infeasible and degenerate requests raise.
    """
    v = np.asarray(velocities, float)
    m = np.asarray(masses, float)
    x = np.asarray(positions, float)
    t = np.asarray(target, float)
    if (v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 2
            or m.shape != (len(v),) or x.shape != v.shape or t.shape != v.shape
            or np.any(m <= 0) or not np.isfinite(target_value)
            or not all(np.isfinite(a).all() for a in (v, m, x, t))):
        raise ValueError("Finite matching vectors, target and positive masses required")
    extras = (np.empty((0, len(v), 3)) if extra_constraints is None
              else np.asarray(extra_constraints, float))
    if extras.ndim != 3 or extras.shape[1:] != v.shape or not np.isfinite(extras).all():
        raise ValueError("Extra constraints must have shape (K,N,3)")
    root = np.sqrt(m)
    r = x - np.average(x, weights=m, axis=0)
    C = np.zeros((6 + len(extras), 3 * len(m)))
    for i, (a, b, c) in enumerate(r):
        C[:3, 3*i:3*i+3] = root[i] * np.eye(3)
        C[3:6, 3*i:3*i+3] = root[i] * np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    if len(extras):
        C[6:] = (extras / root[None, :, None]).reshape(len(extras), -1)
    norms = np.linalg.norm(C, axis=1)
    scaled = C[norms > 0] / norms[norms > 0, None]
    _, s, vh = np.linalg.svd(scaled, full_matrices=False)
    eps = np.finfo(float).eps
    rank = int(np.sum(s > eps * max(scaled.shape) * s[0]))
    rows = vh[:rank]
    y = (root[:, None] * v).ravel()
    fixed = rows.T @ (rows @ y)
    free = y - fixed
    observable = (t / root[:, None]).ravel()
    mode = observable - rows.T @ (rows @ observable)
    norm = float(np.linalg.norm(mode))
    tol_mode = 64 * eps * max(scaled.shape) * np.linalg.norm(observable)
    if norm <= tol_mode:
        raise ValueError("Requested mode is zero or entirely constrained")
    q = mode / norm
    radius = float(np.linalg.norm(free))
    center = float(observable @ fixed)
    lower, upper = center - norm*radius, center + norm*radius
    requested = float((target_value - center) / norm)
    tol = 256 * eps * max(1., radius, abs(requested))
    if abs(requested) > radius + tol:
        raise ValueError(f"Requested observable outside attainable interval [{lower}, {upper}]")
    requested = float(np.clip(requested, -radius, radius))
    old_coordinate = float(q @ free)
    transverse = free - old_coordinate*q
    transverse_norm = float(np.linalg.norm(transverse))
    new_transverse_norm = float(np.sqrt(max(0., (radius-requested)*(radius+requested))))
    if transverse_norm <= tol:
        if new_transverse_norm > tol:
            raise ValueError("Degenerate transverse direction; no unique nearest retarget")
        changed = fixed + requested*q
    else:
        changed = fixed + requested*q + new_transverse_norm/transverse_norm*transverse
    out = changed.reshape(v.shape) / root[:, None]
    return ConstrainedRetarget(
        out, float(observable @ y), float(observable @ changed), lower, upper,
        rank, float(np.linalg.norm(changed-y)),
        float(np.max(np.abs(C @ (changed-y)))),
    )
