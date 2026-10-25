"""Geometric solvent-probe access and local chemistry on an explicit slab face.

These array kernels generalize upward test-point screening. They do not infer
adsorption free energies, contact angles, solvent-accessible area, or chemistry.
Callers supply current solid/H ownership and immutable site/material identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ProbeAccess:
    accessible: np.ndarray
    heavy_distance_A: np.ndarray
    hydrogen_distance_A: np.ndarray
    clearance_A: np.ndarray


def _xyz(value: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite N by 3 array")
    return a


def _box(bounds: np.ndarray, periodic: Sequence[bool]):
    b = np.asarray(bounds, dtype=float)
    p = np.asarray(periodic)
    if b.shape != (3, 2) or not np.isfinite(b).all() or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("bounds must contain three finite increasing intervals")
    if p.shape != (3,) or p.dtype.kind != 'b':
        raise ValueError("periodic must contain three explicit booleans")
    return b[:, 0], b[:, 1] - b[:, 0], p


def _wrap(xyz, origin, lengths, periodic):
    x = np.asarray(xyz, dtype=float) - origin
    x = x.copy()
    x[..., periodic] %= lengths[periodic]
    return x


def _nearest(points, atoms, origin, lengths, periodic):
    from scipy.spatial import cKDTree

    if not len(atoms):
        return np.full(len(points), np.inf)
    tree = cKDTree(_wrap(atoms, origin, lengths, periodic),
                   boxsize=np.where(periodic, lengths, 0.))
    return tree.query(_wrap(points, origin, lengths, periodic))[0]


def probe_accessibility(
    sites: np.ndarray,
    solid_heavy: np.ndarray,
    solid_hydrogen: np.ndarray,
    bounds: np.ndarray,
    *,
    directions: np.ndarray,
    offset_A: float = 2.5,
    heavy_exclusion_A: float = 2.,
    hydrogen_exclusion_A: float = 1.,
    periodic: Sequence[bool] = (True, True, True),
) -> ProbeAccess:
    """Test explicit unit-direction points around each site against solid atoms.

    Output shape is (number of sites, number of directions). The historical
    upward-O-center probe uses direction [0,0,1], offset 2.5 A, and heavy/H
    exclusion radii 2/1 A. Other directions test shielding sensitivity and do
    not represent an angularly integrated molecular insertion probability.
    """
    s = _xyz(sites, 'sites')
    heavy = _xyz(solid_heavy, 'solid_heavy')
    hydrogen = _xyz(solid_hydrogen, 'solid_hydrogen')
    d = _xyz(directions, 'directions')
    origin, lengths, p = _box(bounds, periodic)
    if not len(s) or not len(heavy) or not len(d):
        raise ValueError("Nonempty sites, solid heavy atoms and directions are required")
    if not np.allclose(np.linalg.norm(d, axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError("Probe directions must be unit vectors")
    settings = np.asarray([offset_A, heavy_exclusion_A, hydrogen_exclusion_A])
    if not np.isfinite(settings).all() or np.any(settings <= 0):
        raise ValueError("Probe offset and exclusion radii must be finite and positive")
    points = (s[:, None, :] + offset_A * d[None, :, :]).reshape(-1, 3)
    dh = _nearest(points, heavy, origin, lengths, p).reshape(len(s), len(d))
    dH = _nearest(points, hydrogen, origin, lengths, p).reshape(len(s), len(d))
    clearance = np.minimum(dh - heavy_exclusion_A, dH - hydrogen_exclusion_A)
    return ProbeAccess(clearance >= 0, dh, dH, clearance)


def local_cover_depth(
    sites: np.ndarray,
    solid_heavy: np.ndarray,
    bounds: np.ndarray,
    *,
    lateral_radius_A: float,
    periodic: Sequence[bool] = (True, True, True),
) -> np.ndarray:
    """Maximum positive heavy-atom height over a site within a periodic XY disk.

    This is a column-height descriptor, not proof of solvent inaccessibility.
    The selected slab face must be oriented toward +Z. Heights use the declared
    Z boundary convention; callers must avoid a slab spanning half the box.
    """
    from scipy.spatial import cKDTree

    s = _xyz(sites, 'sites')
    heavy = _xyz(solid_heavy, 'solid_heavy')
    origin, lengths, p = _box(bounds, periodic)
    if not len(heavy) or not np.isfinite(lateral_radius_A) or lateral_radius_A <= 0:
        raise ValueError("Nonempty solid and a positive finite radius are required")
    tree = cKDTree(_wrap(heavy, origin, lengths, p)[:, :2],
                   boxsize=np.where(p[:2], lengths[:2], 0.))
    neighbors = tree.query_ball_point(_wrap(s, origin, lengths, p)[:, :2], lateral_radius_A)
    result = np.zeros(len(s))
    for i, indices in enumerate(neighbors):
        dz = heavy[indices, 2] - s[i, 2]
        if p[2]:
            dz -= lengths[2] * np.rint(dz / lengths[2])
        result[i] = max(0., float(dz.max())) if len(dz) else 0.
    return result


def local_label_fraction(
    sites: np.ndarray,
    positive_labels: np.ndarray,
    bounds: np.ndarray,
    *,
    lateral_radius_A: float,
    periodic: Sequence[bool] = (True, True, True),
) -> tuple[np.ndarray, np.ndarray]:
    """Positive-label fraction and denominator in XY disks, including each site.

    Sites on different slab faces must be passed separately. This preserves
    local pattern contrast without interpreting a composition as wettability.
    """
    from scipy.spatial import cKDTree

    s = _xyz(sites, 'sites')
    labels = np.asarray(positive_labels)
    origin, lengths, p = _box(bounds, periodic)
    if labels.shape != (len(s),) or labels.dtype.kind != 'b' or not len(s):
        raise ValueError("One boolean label is required for every nonempty site")
    if not np.isfinite(lateral_radius_A) or lateral_radius_A <= 0:
        raise ValueError("A positive finite radius is required")
    xy = _wrap(s, origin, lengths, p)[:, :2]
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("Duplicate projected sites would bias neighborhood weights")
    neighbors = cKDTree(xy, boxsize=np.where(p[:2], lengths[:2], 0.)).query_ball_point(
        xy, lateral_radius_A)
    counts = np.asarray([len(indices) for indices in neighbors])
    fractions = np.asarray([labels[indices].mean() for indices in neighbors])
    return fractions, counts
