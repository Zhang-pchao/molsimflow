"""Scale-defined sampled contact fields on fixed periodic surface charts.

These observables are not thermodynamic contact lines or molecular lifetimes.
The caller supplies immutable site ordering and independently qualified epochs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

from .interfacial_contact_geometry import SurfaceGrid


def sampled_window_means(values, steps, cadence, segments, half_window, *,
                         centers=None, stride=1, offset=0):
    """Average complete native windows, retaining every decimation phase.

    Gaps and segment boundaries censor entire windows, including native samples
    not selected by decimation. Offsets refer to the input row index modulo
    stride. Returned indices, means, counts and sample-time offsets are aligned.
    No interpolation or extension across missing support is performed.
    """
    v = np.asarray(values, dtype=float)
    t, s = np.asarray(steps), np.asarray(segments)
    integers = (cadence, half_window, stride, offset)
    if (any(not isinstance(x, (int, np.integer)) for x in integers)
            or cadence <= 0 or half_window < 0 or stride < 1
            or not 0 <= offset < stride or t.ndim != 1
            or t.dtype.kind not in "iu" or s.shape != t.shape
            or v.ndim < 1 or len(v) != len(t) or not np.isfinite(v).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Invalid samples, segments, cadence or window")
    c = np.arange(len(t)) if centers is None else np.asarray(centers)
    if c.ndim != 1 or c.dtype.kind not in "iu" or np.any((c < 0) | (c >= len(t))):
        raise ValueError("Centers must be valid integer row indices")
    edge = (np.diff(t) != cadence) | (s[1:] != s[:-1])
    breaks = np.r_[0, np.cumsum(edge)]
    c = c[(c >= half_window) & (c + half_window < len(t))]
    c = c[breaks[c + half_window] == breaks[c - half_window]]
    indices, means, counts, offsets = [], [], [], []
    for i in c:
        rows = np.arange(i - half_window, i + half_window + 1)
        rows = rows[rows % stride == offset]
        if not len(rows):
            continue
        indices.append(i)
        means.append(v[rows].mean(axis=0))
        counts.append(len(rows))
        offsets.append(float(np.mean(t[rows] - t[i])))
    shape = (len(indices),) + v.shape[1:]
    return (np.asarray(indices, dtype=int),
            np.asarray(means, dtype=float).reshape(shape),
            np.asarray(counts, dtype=int), np.asarray(offsets, dtype=float))


def periodic_contact_field(grid: SurfaceGrid, site_values, sigma: float):
    """Rasterize occupancy then convolve periodically in XY.

    Occupancies must lie in [0, 1]. Gaussian sigma uses the chart length unit.
    Leading dimensions are preserved and never smoothed. Thresholding is an
    explicit downstream choice. A constant field and its cell mean are retained.
    """
    v = np.asarray(site_values, dtype=float)
    if (v.ndim < 1 or v.shape[-1] != grid.site_count
            or not np.isfinite(v).all() or np.any((v < 0) | (v > 1))
            or not np.isfinite(sigma) or sigma < 0):
        raise ValueError("Finite site occupancies and nonnegative sigma required")
    field = v[..., grid.owner]
    if sigma == 0:
        return field.copy()
    widths = [0.] * (field.ndim - 2) + list(sigma / grid.spacing)
    return gaussian_filter(field, widths, mode="wrap")


def periodic_mask_labels(mask):
    """Label all four-neighbor components with periodic XY adjacency.

    Zero denotes background. No component, hole or satellite is removed.
    Sizes are indexed by label minus one.
    """
    a = np.asarray(mask)
    if a.ndim != 2 or a.dtype != bool or min(a.shape) < 3:
        raise ValueError("Boolean XY grid with at least three cells per axis required")
    labels, n = label(a, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    parent = np.arange(n + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for axis in (0, 1):
        for i, j in zip(np.take(labels, 0, axis=axis),
                        np.take(labels, -1, axis=axis)):
            if i and j:
                parent[root(j)] = root(i)
    remap, mapping = np.zeros(n + 1, dtype=int), {}
    for i in range(1, n + 1):
        r = root(i)
        mapping.setdefault(r, len(mapping) + 1)
        remap[i] = mapping[r]
    out = remap[labels]
    return out, np.bincount(out.ravel(), minlength=len(mapping) + 1)[1:]
