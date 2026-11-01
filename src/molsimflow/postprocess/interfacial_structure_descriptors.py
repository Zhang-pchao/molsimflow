"""Directional spatial descriptors for marked periodic material sites.

Squared mark differences describe arrangement at a declared distance scale.
They are not forces, barriers, bonds, or a causal ordering of dynamical events.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def periodic_site_edges(xy, lengths, cutoff):
    """Return unique unordered edges and minimum-image vectors in a planar box.

    Coordinates may be translated by any periodic image. The cutoff must be
    less than half either box length to exclude ambiguous displacement images.
    Coincident periodic sites are rejected. Input site ordering is preserved.
    """
    x = np.asarray(xy, dtype=float)
    box = np.asarray(lengths, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all()
            or box.shape != (2,) or not np.isfinite(box).all()
            or np.any(box <= 0) or not np.isfinite(cutoff)
            or not 0 < cutoff < box.min() / 2):
        raise ValueError("Finite planar sites, positive box and unambiguous cutoff required")
    wrapped = x % box
    if len(np.unique(wrapped, axis=0)) != len(wrapped):
        raise ValueError("Coincident periodic sites")
    pairs = np.asarray(sorted(cKDTree(wrapped, boxsize=box).query_pairs(cutoff)),
                       dtype=int).reshape(-1, 2)
    vectors = wrapped[pairs[:, 1]] - wrapped[pairs[:, 0]]
    vectors = (vectors + box / 2) % box - box / 2
    return pairs, vectors


def directional_mark_contrast(marks, edges, vectors, *, axis=(1., 0.), weights=None):
    """Measure squared mark differences parallel and transverse to an axis.

    marks has shape (..., sites); edges is (pairs, 2); vectors is (pairs, 2).
    Optional nonnegative edge weights affect every leading mark channel.
    For each edge, a=(unit_vector dot unit_axis)**2. Parallel and transverse
    means use weights*a and weights*(1-a), normalized separately. Contrast
    is parallel minus transverse. Isotropic is the ordinary weighted mean.
    For binary marks, isotropic is the weighted unlike-edge fraction.

    Empty or zero directional support returns NaN, never a fabricated zero.
    Uniform marks with nonzero support return zero. Axis reversal leaves
    descriptors unchanged; exchanging perpendicular axes reverses contrast.
    """
    z = np.asarray(marks, dtype=float)
    e = np.asarray(edges)
    v = np.asarray(vectors, dtype=float)
    u = np.asarray(axis, dtype=float)
    if (z.ndim < 1 or not np.isfinite(z).all()
            or e.ndim != 2 or e.shape[1] != 2 or e.dtype.kind not in "iu"
            or v.shape != e.shape or not np.isfinite(v).all()
            or np.any(e < 0) or np.any(e >= z.shape[-1])
            or np.any(e[:, 0] >= e[:, 1])
            or len(np.unique(e, axis=0)) != len(e)
            or u.shape != (2,) or not np.isfinite(u).all() or np.linalg.norm(u) == 0):
        raise ValueError("Invalid marks, unique ordered edges, vectors or axis")
    norm2 = np.sum(v * v, axis=1)
    if np.any(norm2 <= 0):
        raise ValueError("Edge vectors must have nonzero length")
    w = np.ones(len(e)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(e),) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Finite nonnegative edge weights required")
    u = u / np.linalg.norm(u)
    a = np.clip((v @ u)**2 / norm2, 0., 1.)
    d = (z[..., e[:, 0]] - z[..., e[:, 1]])**2

    def mean(weight):
        total = float(weight.sum())
        value = (np.sum(d * weight, axis=-1) / total if total > 1e-12
                 else np.full(z.shape[:-1], np.nan))
        return value, total

    iso, total = mean(w)
    parallel, wp = mean(w * a)
    transverse, wt = mean(w * (1 - a))
    return dict(isotropic=iso, parallel=parallel, transverse=transverse,
                contrast=parallel-transverse, weight=total,
                parallel_weight=wp, transverse_weight=wt,
                positive_edges=int(np.count_nonzero(w)))
