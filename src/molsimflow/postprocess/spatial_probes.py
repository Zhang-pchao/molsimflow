"""Explicit axis-aligned cuboid occupancy probes."""

from __future__ import annotations
import numpy as np


def cuboid_counts(
    positions,
    centers,
    widths,
    *,
    lengths,
    periodic,
    lower_closed=(True, True, True),
    upper_closed=(False, False, False),
):
    """Count selected points in each probe with explicit boundary inclusion.

    Arrays use shape (points, 3), (probes, 3) and (3,) widths. Callers select
    atoms or molecule centers and supply probe centers in the intended registered
    frame. Widths are full extents. The default intervals are [lower, upper).
    """
    xyz, centers = np.asarray(positions, float), np.asarray(centers, float)
    widths, box = np.asarray(widths, float), np.asarray(lengths, float)
    flags = np.asarray(periodic)
    low, high = np.asarray(lower_closed), np.asarray(upper_closed)
    if any(x.shape != (3,) or x.dtype.kind != "b" for x in (flags, low, high)):
        raise ValueError("Periodic and endpoint policies must have three boolean flags")
    if any(a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() for a in (xyz, centers)):
        raise ValueError("Finite point/probe arrays with three coordinates are required")
    if (
        widths.shape != (3,)
        or box.shape != (3,)
        or not np.isfinite(widths).all()
        or np.any(widths <= 0)
    ):
        raise ValueError("Three positive full probe widths are required")
    if not np.isfinite(box).all() or np.any(box[flags] <= 0) or np.any(widths[flags] >= box[flags]):
        raise ValueError("Periodic probes must be narrower than their positive box lengths")
    counts = np.zeros(len(centers), dtype=int)
    for index, center in enumerate(centers):
        delta = xyz - center
        delta[:, flags] -= np.rint(delta[:, flags] / box[flags]) * box[flags]
        above = np.where(low, delta >= -widths / 2, delta > -widths / 2)
        below = np.where(high, delta <= widths / 2, delta < widths / 2)
        counts[index] = np.count_nonzero(np.all(above & below, axis=1))
    return counts
