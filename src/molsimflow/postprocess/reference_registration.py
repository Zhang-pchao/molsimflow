"""Translation registration of explicitly matched anchors in orthorhombic boxes."""

from __future__ import annotations
import numpy as np


def anchor_translation(reference_ids, reference_xyz, frame_ids, frame_xyz, *, lengths, periodic):
    """Return the componentwise median anchor shift and residuals in reference-ID order.

    Callers select chemically stable anchors. Each displacement uses its nearest
    image on explicitly periodic axes before taking the median. A near-half-box
    displacement is ambiguous and rejected. The strict Si/O scaffold observer
    retains its own anchor-unwrapping, element and deformation contracts.
    """
    reference_ids, frame_ids = np.asarray(reference_ids), np.asarray(frame_ids)
    ref, xyz = np.asarray(reference_xyz, float), np.asarray(frame_xyz, float)
    box, flags = np.asarray(lengths, float), np.asarray(periodic)
    if flags.shape != (3,) or flags.dtype.kind != "b" or box.shape != (3,):
        raise ValueError("Three box lengths and boolean periodic flags are required")
    if not np.isfinite(box).all() or np.any(box[flags] <= 0):
        raise ValueError("Periodic lengths must be finite and positive")
    for ids, positions in ((reference_ids, ref), (frame_ids, xyz)):
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or not len(ids)
            or len(np.unique(ids)) != len(ids)
            or positions.shape != (len(ids), 3)
            or not np.isfinite(positions).all()
        ):
            raise ValueError("Unique integer atom IDs and finite N-by-3 positions are required")
    lookup = {int(atom): index for index, atom in enumerate(frame_ids)}
    if not set(reference_ids) <= set(frame_ids):
        raise ValueError("Reference anchor IDs are missing from the frame")
    delta = xyz[[lookup[int(atom)] for atom in reference_ids]] - ref
    delta[:, flags] -= np.rint(delta[:, flags] / box[flags]) * box[flags]
    if np.any(np.isclose(np.abs(delta[:, flags]), box[flags] / 2, rtol=0, atol=1e-10)):
        raise ValueError("Anchor displacement is ambiguous at half a box")
    shift = np.median(delta, axis=0)
    residuals = delta - shift
    residuals[:, flags] -= np.rint(residuals[:, flags] / box[flags]) * box[flags]
    return shift, residuals
