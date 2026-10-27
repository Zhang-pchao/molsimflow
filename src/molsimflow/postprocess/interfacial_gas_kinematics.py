"""Identity-preserving gas-layer exchange and rigid-motion diagnostics.

Callers own molecule pairing, periodic unwrapping, cluster membership and units.
An instantaneous best-fit rotation of a fluid is not a rolling mechanism.
"""
from __future__ import annotations
import numpy as np


def quantile_layer_labels(z, membership, fractions=(1 / 3, 2 / 3)):
    """Label lower/middle/upper gas layers 0/1/2; nonmembers are -1.

    Dynamic layers describe position inside the gas, not solid contact.
    Ordered molecular identities must be retained across calls.
    """
    z = np.asarray(z, dtype=float)
    member = np.asarray(membership, dtype=bool)
    if z.ndim != 1 or z.shape != member.shape or member.sum() < 3:
        raise ValueError("Matching one-dimensional arrays and three members required")
    if not np.isfinite(z).all() or not 0 < fractions[0] < fractions[1] < 1:
        raise ValueError("Finite coordinates and ordered quantile fractions required")
    low, high = np.quantile(z[member], fractions)
    if high <= low:
        raise ValueError("Degenerate layer boundaries")
    labels = np.full(z.shape, -1, dtype=np.int8)
    labels[member] = np.digitize(z[member], [low, high])
    return labels


def lagged_state_statistics(states, lag, origin_state=0):
    """Pool molecule/time-origin counts at a positive integer frame lag.

    Return intermittent return and uninterrupted *sampled* survival.
    Departed cluster members remain in the denominator. The caller must
    provide a contiguous uniform time grid within one identity epoch.
    """
    s = np.asarray(states)
    if s.ndim != 2 or s.dtype.kind not in "iu" or not np.isin(s, [-1, 0, 1, 2]).all():
        raise ValueError("States must be integer time-by-molecule labels -1,0,1,2")
    if not isinstance(lag, (int, np.integer)) or not 0 < lag < len(s):
        raise ValueError("Lag must be a positive integer smaller than frame count")
    if origin_state not in (0, 1, 2):
        raise ValueError("Origin state must be 0, 1 or 2")
    initial = s[:-lag] == origin_state
    denominator = int(initial.sum())
    if not denominator:
        raise ValueError("No initial population")
    terminal = s[lag:]
    stay = initial.copy()
    for offset in range(1, lag + 1):
        stay &= s[offset:len(s) - lag + offset] == origin_state
    counts = {int(k): int((initial & (terminal == k)).sum()) for k in (-1, 0, 1, 2)}
    return dict(origin_count=denominator, terminal_counts=counts,
                return_fraction=counts[origin_state] / denominator,
                sampled_survival_fraction=int(stay.sum()) / denominator,
                terminal_fractions={k: v / denominator for k, v in counts.items()})


def fit_rigid_motion(positions, velocities):
    """Fit v=translation+omega cross r with equal molecular weights.

    Inputs use a common locally unwrapped frame. Units of omega are velocity
    units divided by length units. Thermal/fluctuating velocities remain in
    the residual; block and coarse-field checks are the caller's responsibility.
    """
    x, v = np.asarray(positions, float), np.asarray(velocities, float)
    if x.ndim != 2 or x.shape[1:] != (3,) or v.shape != x.shape or len(x) < 3:
        raise ValueError("Matching N-by-3 coordinates/velocities required")
    if not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite coordinates or velocities")
    r, u = x - x.mean(axis=0), v - v.mean(axis=0)
    inertia = np.eye(3) * np.sum(r * r) - r.T @ r
    if np.linalg.matrix_rank(inertia) < 3:
        raise ValueError("Rotation is not identifiable in degenerate geometry")
    omega = np.linalg.solve(inertia, np.cross(r, u).sum(axis=0))
    rotation = np.cross(omega, r)
    total = float(np.sum(u * u))
    residual = float(np.sum((u - rotation) ** 2))
    return dict(translation=v.mean(axis=0), omega=omega,
                rotation_energy_fraction=1 - residual / total if total > 1e-24 else 0.,
                residual_rms=float(np.sqrt(residual / len(x))))


def coarse_velocity_field(local_positions, relative_velocities, x_edges, z_edges):
    """Accumulate an XZ field over equal-weight molecular/frame samples.

    Return counts, coordinate sums and velocity sums; do not silently fill
    empty cells. This is a Y-projected field rather than a central Y slice.
    """
    p, v = np.asarray(local_positions, float), np.asarray(relative_velocities, float)
    if p.ndim != 2 or p.shape[1:] != (3,) or p.shape != v.shape:
        raise ValueError("Matching sample-by-3 arrays required")
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError("Finite samples required")
    edges = [np.asarray(x_edges, float), np.asarray(z_edges, float)]
    if any(e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all()
           or np.any(np.diff(e) <= 0) for e in edges):
        raise ValueError("Finite increasing bin edges required")
    xy = p[:, [0, 2]]
    count = np.histogram2d(*xy.T, bins=edges)[0]
    position = np.stack([np.histogram2d(*xy.T, bins=edges, weights=p[:, j])[0]
                         for j in range(3)], axis=-1)
    velocity = np.stack([np.histogram2d(*xy.T, bins=edges, weights=v[:, j])[0]
                         for j in range(3)], axis=-1)
    return count, position, velocity
