"""Glycine protonation candidates from existing soft-Voronoi assignments.

The final three centers are N, carboxyl O, carboxyl O; preceding centers
are water oxygens. A complete path remains the statistical unit. Labels
are geometric descriptors, not dynamical states or committor assignments.
"""
from __future__ import annotations
import numpy as np

STATE_NAMES = ("N", "Z", "A", "C", "U")


def classify_glycine_states(soft, nearest, *, tolerance: float = 0.35):
    """Return hard candidates and conservative cores with an explicit U class.

Nearest-owner stoichiometry must match glycine and its solvent counterion.
Core labels additionally require N and total carboxyl occupancies, and any
solvent defect occupancies, to be within tolerance of that stoichiometry.
Ordinary-water soft fluctuations are not used to reject a whole configuration.
"""
    soft = np.asarray(soft, dtype=float)
    nearest = np.asarray(nearest)
    if soft.shape != nearest.shape or soft.ndim < 1 or soft.shape[-1] < 4:
        raise ValueError("Expected matching (..., water centers + N/O/O) arrays")
    if not np.isfinite(soft).all() or not np.isfinite(nearest).all():
        raise ValueError("Nonfinite occupancy")
    if not 0 < tolerance < 0.5 or np.any(nearest < 0) or np.any(nearest != np.rint(nearest)):
        raise ValueError("Invalid tolerance or nearest-owner counts")
    w, n, c = nearest[..., :-3], nearest[..., -3], nearest[..., -2:].sum(-1)
    hyd = (w == 3).sum(-1); hydrox = (w == 1).sum(-1)
    ordinary = ((w >= 1) & (w <= 3)).all(-1)
    codes = np.full(soft.shape[:-1], 4, dtype=np.int8)
    for code, nn, cc, hp, hm in [(0,2,1,0,0),(1,3,0,0,0),(2,2,0,1,0),(3,3,1,0,1)]:
        mask = ordinary & (n == nn) & (c == cc) & (hyd == hp) & (hydrox == hm)
        codes = np.where(mask, code, codes)
    defect_error = np.max(np.where(w != 2, abs(soft[..., :-3]-w), 0.), axis=-1)
    error = np.maximum.reduce([abs(soft[..., -3]-n), abs(soft[..., -2:].sum(-1)-c), defect_error])
    core = np.where(error <= tolerance, codes, 4).astype(np.int8)
    return {"hard_labels": codes.astype(np.int8), "core_labels": core,
            "core_error": error, "hydronium_count": hyd, "hydroxide_count": hydrox}


def state_path_fractions(labels, *, nstates: int = 5):
    """Average per-bead indicators, retaining empty inputs and unclassified mass."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] == 0:
        raise ValueError("Expected (frames, beads) labels")
    if np.any((labels < 0) | (labels >= nstates)):
        raise ValueError("Unknown state label")
    return np.stack([(labels == k).mean(axis=1) for k in range(nstates)], axis=1)
