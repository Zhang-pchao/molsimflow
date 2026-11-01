"""Conservative local donor-orientation bias and matched distance restraints.

Coordinates are [site O, water O, water H, water H, anchor Si, anchor O, N, N].
This is a flexible-water potential, not a rigid-body integrator or a chemistry
model. Callers must guard topology and supply their own preparation schedule.
"""
from __future__ import annotations
import numpy as np

PAIR_INDICES = tuple((i, j) for i in (1, 6, 7) for j in (0, 4, 5))

def minimum_image(displacement: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return displacement - np.rint(displacement / lengths) * lengths

def donor_coordinate(positions: np.ndarray, lengths: np.ndarray, beta: float = 20.0):
    """Return normalized smooth maximum of both donor cosines and its gradient."""
    r = np.asarray(positions, dtype=float)
    L = np.asarray(lengths, dtype=float)
    if r.shape != (8, 3) or L.shape != (3,) or not np.all(np.isfinite(r)):
        raise ValueError('Expected eight finite Cartesian positions.')
    if not np.all(np.isfinite(L)) or np.any(L <= 0) or not np.isfinite(beta) or beta <= 0:
        raise ValueError('Invalid periodic cell or smoothing parameter.')
    h = minimum_image(r[2:4] - r[1], L)
    a = minimum_image(r[0] - r[1], L)
    lh = np.linalg.norm(h, axis=1)
    la = np.linalg.norm(a)
    if min(lh) <= 1e-8 or la <= 1e-8:
        raise ValueError('Degenerate direction.')
    u = h / lh[:, None]
    v = a / la
    c = u @ v
    e = np.exp(beta * (c - max(c)))
    weights = e / e.sum()
    q = max(c) + (np.log(e.sum()) - np.log(2.0)) / beta
    gradient = np.zeros((8, 3))
    for j in range(2):
        gh = weights[j] * (v - c[j] * u[j]) / lh[j]
        gs = weights[j] * (u[j] - c[j] * v) / la
        gradient[j+2] += gh
        gradient[0] += gs
        gradient[1] -= gh + gs
    return float(q), gradient, c

def reference_distances(positions: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(minimum_image(positions[i]-positions[j], lengths))
                     for i, j in PAIR_INDICES])

def evaluate(positions: np.ndarray, lengths: np.ndarray, target: float,
             orientation_k: float, distance_k: float, references: np.ndarray,
             amplitude: float = 1.0):
    """Return total potential, Cartesian force, and component diagnostics.

    Energies use the caller's units; with metal units k_orientation is eV and
    k_distance is eV/A^2. Reaction forces on all anchors are included.
    """
    r = np.asarray(positions, float); L = np.asarray(lengths, float)
    if not np.all(np.isfinite([target, orientation_k, distance_k, amplitude])):
        raise ValueError('Nonfinite potential parameter.')
    if orientation_k < 0 or distance_k < 0 or not 0 <= amplitude <= 1 or not -1 <= target <= 1:
        raise ValueError('Invalid potential parameter.')
    references = np.asarray(references, float)
    if references.shape != (9,) or np.any(references <= 0) or not np.all(np.isfinite(references)):
        raise ValueError('Expected nine positive reference distances.')
    q, grad, cosines = donor_coordinate(r, L)
    orient = .5 * orientation_k * (q-target)**2
    force = -orientation_k * (q-target) * grad
    common = 0.
    for (i,j),d0 in zip(PAIR_INDICES,references):
        dr = minimum_image(r[i]-r[j], L); d = np.linalg.norm(dr)
        if d <= 1e-8: raise ValueError('Degenerate restraint pair.')
        common += .5*distance_k*(d-d0)**2
        f = -distance_k*(d-d0)*dr/d
        force[i] += f; force[j] -= f
    return amplitude*(orient+common), amplitude*force, dict(q=q,cosines=cosines,
         orientation_energy=amplitude*orient,common_energy=amplitude*common)

def schedule(time_ps: float, initial_q: float, arm: str):
    """Frozen 2 ps construction followed by exact zero construction force.

    A discontinuity at release carries a protocol-work increment -U_before.
    The background model, drive, tether and bath are not part of this function.
    """
    if arm not in ('aligned','away','sham') or not np.isfinite(time_ps) or time_ps < 0:
        raise ValueError('Invalid schedule argument.')
    def smooth(z):
        z = np.clip(z, 0., 1.)
        return float(z*z*(3.-2.*z))
    if time_ps >= 2.: return dict(target=initial_q,orientation_k=0.,distance_k=0.,amplitude=0.)
    target = initial_q + ((.5-initial_q)*smooth(time_ps-.5) if arm=='away' else 0.)
    return dict(target=target,orientation_k=0. if arm=='sham' else 1.,distance_k=2.,amplitude=smooth(time_ps/.5))
