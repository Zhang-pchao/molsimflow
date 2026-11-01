"""Periodic nearest-solid identities without changing gas-distance statistics."""
import numpy as np
from scipy.spatial import cKDTree


def nearest_solid_identity(nitrogen, solid_xyz, solid_ids, solid_types, bounds, tie_atol=1e-12):
    """Return distance, winning solid ID/type, and tie flag per consecutive N2.

    Geometry is orthorhombic PPP. Ties within tie_atol Angstrom are resolved
    by minimum solid atom ID, then nitrogen order. Distance remains the exact
    numerical minimum, independent of the tie representative.
    """
    gas = np.asarray(nitrogen, float)
    solid = np.asarray(solid_xyz, float)
    ids = np.asarray(solid_ids, int)
    types = np.asarray(solid_types, int)
    bounds = np.asarray(bounds, float)
    if len(gas) % 2 or len(solid) < 2 or len(set(ids.tolist())) != len(ids):
        raise ValueError('Complete gas pairs and unique solid IDs required')
    lengths = np.diff(bounds, axis=1).ravel()
    g = (gas - bounds[:, 0]) % lengths
    s = (solid - bounds[:, 0]) % lengths
    tree = cKDTree(s, boxsize=lengths)
    dd, jj = tree.query(g, k=2)
    nearest = jj[:, 0].copy()
    atom_ties = np.abs(dd[:, 1] - dd[:, 0]) <= tie_atol
    for i in np.flatnonzero(atom_ties):
        candidates = np.asarray(tree.query_ball_point(g[i], dd[i, 0] + 2*tie_atol), int)
        delta = s[candidates] - g[i]
        delta -= np.rint(delta / lengths) * lengths
        distances = np.linalg.norm(delta, axis=1)
        candidates = candidates[np.abs(distances - dd[i, 0]) <= tie_atol]
        nearest[i] = candidates[np.argmin(ids[candidates])]
    d = dd[:, 0].reshape(-1, 2)
    pair_ties = np.abs(d[:, 0] - d[:, 1]) <= tie_atol
    side = np.argmin(d, axis=1)
    pair_ids = ids[nearest].reshape(-1, 2)
    side[pair_ties] = np.argmin(pair_ids[pair_ties], axis=1)
    atom = 2*np.arange(len(d)) + side
    winner = nearest[atom]
    return d.min(axis=1), ids[winner], types[winner], pair_ties | atom_ties[atom]
