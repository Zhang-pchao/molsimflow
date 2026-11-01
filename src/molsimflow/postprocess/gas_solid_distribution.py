"""Whole-gas nearest-solid distances with explicit PPP or PPF geometry."""
import numpy as np
from molsimflow.postprocess.nanobubble_attachment import molecule_centers, largest_cluster
from molsimflow.postprocess.constant_force_species_timeseries import _periodic_query


def gas_solid_distances(nitrogen, solid, bounds, *, periodic_z=True, cutoff=5.5):
    """Return per-molecule distances and largest-cluster membership.

    Nonperiodic z uses a padded computational box only for existing cluster
    helpers. Padding exceeds the coordinate span and cluster cutoff, so no
    artificial z edge or bond image can occur. Physical distances use the
    established partial-periodic nearest-neighbor implementation.
    """
    nitrogen = np.asarray(nitrogen, float)
    solid = np.asarray(solid, float)
    bounds = np.asarray(bounds, float)
    if not len(nitrogen) or len(nitrogen) % 2 or not len(solid):
        raise ValueError("Complete gas pairs and a nonempty solid are required")
    if not np.isfinite(nitrogen).all() or not np.isfinite(solid).all():
        raise ValueError("Nonfinite coordinates")
    effective = bounds.copy()
    if not periodic_z:
        z = np.concatenate((nitrogen[:, 2], solid[:, 2]))
        if z.min() < bounds[2, 0] or z.max() > bounds[2, 1]:
            raise ValueError("Nonperiodic coordinates outside declared box")
        span = np.ptp(z)
        pad = max(span + 2 * cutoff, 100.)
        effective[2] = [z.min() - pad, z.max() + pad]
    centers = molecule_centers(nitrogen, effective)
    indices = largest_cluster(centers, effective, cutoff)
    main = np.zeros(len(centers), dtype=bool)
    main[indices] = True
    _, distance = _periodic_query(nitrogen, solid, bounds, periodic_z=periodic_z)
    lengths = np.diff(effective, axis=1).ravel()
    bond = nitrogen[1::2] - nitrogen[::2]
    bond -= np.rint(bond / lengths) * lengths
    if np.linalg.norm(bond, axis=1).max() > 2.:
        raise ValueError("N2 bond/pairing qualification failed")
    return distance.reshape(-1, 2).min(axis=1), main


def time_mean_cdf(distance, main, grid):
    """Equal-frame empirical CDF; molecules normalized within each frame."""
    distance = np.asarray(distance)
    main = np.asarray(main, bool)
    grid = np.asarray(grid)
    if distance.shape != main.shape or distance.ndim != 2 or not len(distance):
        raise ValueError("Matching nonempty frame-by-molecule arrays required")
    if not np.isfinite(distance).all() or np.any(main.sum(axis=1) == 0):
        raise ValueError("Finite distances and nonempty main clusters required")
    result = np.zeros(len(grid), float)
    for d, m in zip(distance, main):
        result += np.searchsorted(np.sort(d[m]), grid, side="right") / m.sum()
    return result / len(distance)
