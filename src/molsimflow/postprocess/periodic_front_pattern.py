"""Periodicity-correct observables for a migrating gas contact front.

This module supersedes the open-boundary treatment in
:mod:`molsimflow.postprocess.interfacial_front_pattern`. It exists
to make four analysis choices explicit for periodic-front applications:

1. The prepared surface is a periodic slab, but the connectivity
   analysis cut the box open in Y. A sublattice that is actually
   connected around the periodic direction appeared fragmented.
2. The tip-isolation test looked for neighbours only inside the
   destination half-face. A tip that is still continuous with the
   departure face through the seam was counted as isolated.
3. The tip-to-bulk comparison used the maximum tip over the whole
   history against the bulk position in the final frame, mixing two
   different times.
4. The advance hazard discarded every column the trajectory failed to
   leave, which removes exactly the failed attempts that carry the
   information about arrest.

Everything here takes the box lengths explicitly and applies the
minimum-image convention, and every quantity is reported with the
support that backs it. A quantity computed from a handful of
observations is labelled as such rather than being presented as a
surface-wide constant.

Design constraints:

* No quantity here is a rate, a free energy, a wetting rank, or a
  confidence interval. Frames are correlated samples from one
  trajectory.
* Affinity is reported as a contact-frequency enrichment with its
  denominator and support, never as an availability constraint.
  A site that the gas touches less often is still touchable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "minimum_image_delta",
    "periodic_neighbour_matrix",
    "connected_components",
    "sublattice_connectivity",
    "footprint_connectivity",
    "tip_isolation_full_surface",
    "synchronous_tip_bulk_gap",
    "complete_window_advance_fraction",
    "selectivity_with_support",
]


def minimum_image_delta(coords: np.ndarray, box: Sequence[float],
                        periodic: Sequence[bool] | None = None) -> np.ndarray:
    """Return the minimum-image displacement for every pair of points.

    ``coords`` holds an ``(n, 3)`` array of Cartesian coordinates and
    ``box`` the orthorhombic box lengths. ``periodic`` marks which axes
    are periodic; it defaults to every axis with a non-zero length.

    The default is only safe when every positive-length direction is
    periodic. Supply explicit flags from the simulation boundary contract;
    slab shape or a chemical seam does not establish those flags. An open
    observation chart must be declared separately from physical periodicity.
    Nonperiodic axes may have a zero length; periodic axes require a positive
    length. Coordinates and lengths must be finite.
    """
    pts = np.asarray(coords, dtype=float)
    lengths = np.asarray(box, dtype=float)
    if (lengths.ndim != 1 or not lengths.size or pts.ndim != 2
            or pts.shape[1] != lengths.size):
        raise ValueError("coords must be (n, len(box)) with a one-dimensional box")
    if (not np.isfinite(pts).all() or not np.isfinite(lengths).all()
            or np.any(lengths < 0)):
        raise ValueError("Finite coordinates and nonnegative box lengths required")
    if periodic is None:
        periodic = lengths > 0
    periodic = np.asarray(periodic, dtype=bool)
    if periodic.shape != lengths.shape:
        raise ValueError("periodic must match the box")
    if np.any(periodic & (lengths <= 0)):
        raise ValueError("Periodic axes require positive box lengths")
    delta = pts[:, None, :] - pts[None, :, :]
    for axis, length in enumerate(lengths):
        if periodic[axis] and length > 0:
            delta[:, :, axis] = (delta[:, :, axis] + length / 2) % length - length / 2
    return delta


def periodic_neighbour_matrix(coords: np.ndarray, box: Sequence[float],
                              cutoff: float,
                              periodic: Sequence[bool] | None = None) -> np.ndarray:
    """Return the boolean pairwise neighbour matrix under minimum image.

    The diagonal is set to ``False``: a site is not its own neighbour.
    """
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("A finite positive neighbor cutoff is required")
    delta = minimum_image_delta(coords, box, periodic)
    dist = np.linalg.norm(delta, axis=-1)
    matrix = dist <= cutoff
    np.fill_diagonal(matrix, False)
    return matrix


def connected_components(adjacency: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Label the connected components of a symmetric boolean adjacency.

    Returns the component count, the label of each node, and the size of
    each component indexed by label.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components as _cc

    adjacency = np.asarray(adjacency)
    if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
            or adjacency.dtype != bool or not np.array_equal(adjacency, adjacency.T)):
        raise ValueError("A square symmetric boolean adjacency is required")
    n_components, labels = _cc(sp.csr_matrix(adjacency), directed=False)
    sizes = np.bincount(labels)
    return int(n_components), labels, sizes


def sublattice_connectivity(coords: np.ndarray, box: Sequence[float],
                            selection: np.ndarray, cutoff: float,
                            periodic: Sequence[bool] | None = None) -> dict:
    """Measure the connectivity of a selected subset of surface sites.

    ``selection`` is a boolean mask over the sites. The returned
    dictionary reports the number of selected sites, the number of
    connected fragments, the size of the largest fragment, its
    fraction of the selection, and the X span and Y span of that
    largest fragment.

    A connected fraction below one does not by itself mean that no
    spanning path exists. The X span of the largest fragment is
    reported for exactly this reason: a fragment that spans the whole
    face in the transport direction is a percolating path even if it
    covers only a fraction of the sites.
    """
    mask = np.asarray(selection, dtype=bool)
    pts = np.asarray(coords, dtype=float)
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        return {"n_sites": 0, "n_fragments": 0, "largest_fragment": 0,
                "connected_fraction": float("nan"),
                "largest_x_span_A": float("nan"),
                "largest_y_span_A": float("nan")}

    full = periodic_neighbour_matrix(pts, box, cutoff, periodic)
    sub = full[np.ix_(ids, ids)]
    n_comp, labels, sizes = connected_components(sub)
    largest_label = int(np.argmax(sizes))
    largest_ids = ids[labels == largest_label]
    coords = pts[largest_ids]
    return {
        "n_sites": int(ids.size),
        "n_fragments": n_comp,
        "largest_fragment": int(sizes.max()),
        "connected_fraction": float(sizes.max() / ids.size),
        "largest_x_span_A": float(coords[:, 0].max() - coords[:, 0].min()),
        "largest_y_span_A": float(coords[:, 1].max() - coords[:, 1].min()),
    }


def footprint_connectivity(contacted_parents: Sequence[np.ndarray],
                            parent_xyz: np.ndarray, box: Sequence[float],
                            half_mask: np.ndarray, cutoff: float,
                            periodic: Sequence[bool] | None = None) -> dict:
    """Report the footprint geometry on the whole surface and on one face.

    The full-surface footprint is what physically holds the bubble down.
    The destination-side footprint is what the bubble must build in order
    to continue into new territory. Reporting only the latter made an
    early version of this analysis read a healthy bubble as a collapsed
    one, so both are returned side by side.
    """
    mask = np.asarray(half_mask, dtype=bool)
    pts = np.asarray(parent_xyz, dtype=float)
    full = periodic_neighbour_matrix(pts, box, cutoff, periodic)

    sizes_full: list[int] = []
    sizes_half: list[int] = []
    for contacted in contacted_parents:
        nodes = np.unique(np.asarray(contacted, dtype=int))
        if nodes.size == 0:
            continue
        sub = full[np.ix_(nodes, nodes)]
        _, labels, sizes = connected_components(sub)
        sizes_full.append(int(sizes.max()))
        on_face = nodes[mask[nodes]]
        if on_face.size:
            _, _, hsizes = connected_components(full[np.ix_(on_face, on_face)])
            sizes_half.append(int(hsizes.max()))
    if not sizes_full:
        return {"median_full_largest": float("nan"),
                "median_half_largest": float("nan"),
                "n_frames": 0}
    return {
        "median_full_largest": float(np.median(sizes_full)),
        "median_half_largest": float(np.median(sizes_half)) if sizes_half else float("nan"),
        "n_frames": len(sizes_full),
    }


def tip_isolation_full_surface(contacted_parents: Sequence[np.ndarray],
                               parent_xyz: np.ndarray, box: Sequence[float],
                               half_mask: np.ndarray, cutoff: float,
                               periodic: Sequence[bool] | None = None) -> dict:
    """Test whether the leading contact is isolated on the whole surface.

    The leading contact is the contacted parent with the largest X on
    the destination face. It is called isolated when no other contacted
    parent lies within ``cutoff`` under the minimum-image convention.
    Crucially the neighbourhood search covers every contacted parent,
    not only the ones on the destination face: a tip that is still
    connected to the departure face through the seam is not isolated.
    """
    mask = np.asarray(half_mask, dtype=bool)
    pts = np.asarray(parent_xyz, dtype=float)
    full = periodic_neighbour_matrix(pts, box, cutoff, periodic)

    isolated = 0
    frames = 0
    degrees: list[int] = []
    for contacted in contacted_parents:
        nodes = np.unique(np.asarray(contacted, dtype=int))
        if nodes.size == 0:
            continue
        on_face = nodes[mask[nodes]]
        if on_face.size == 0:
            continue
        # Break ties in x by y, so that the leading contact is a single
        # well-defined site. The two conventions differ only on the few
        # frames where two sites share the same x.
        tip = int(on_face[np.lexsort((pts[on_face, 1], pts[on_face, 0]))[-1]])
        degree = int(full[tip, nodes].sum())
        degrees.append(degree)
        isolated += 1 if degree == 0 else 0
        frames += 1
    if frames == 0:
        return {"n_frames": 0, "isolated_fraction": float("nan"),
                "median_degree": float("nan")}
    return {
        "n_frames": frames,
        "isolated_fraction": float(isolated / frames),
        "median_degree": float(np.median(degrees)),
    }


def synchronous_tip_bulk_gap(tip_x: np.ndarray, bulk_x: np.ndarray) -> dict:
    """Compare the tip and the bulk front at the same instants.

    Taking the maximum tip over a whole history against the final bulk
    position mixes two different times and produces a gap that no
    single frame ever showed. Every gap here is formed from one frame.
    """
    tip = np.asarray(tip_x, dtype=float)
    bulk = np.asarray(bulk_x, dtype=float)
    if tip.ndim != 1 or tip.shape != bulk.shape:
        raise ValueError("tip and bulk must be paired one-dimensional arrays")
    ok = np.isfinite(tip) & np.isfinite(bulk)
    if not ok.any():
        return {"n_frames": 0, "median_gap_A": float("nan"),
                "max_gap_A": float("nan"), "final_gap_A": float("nan")}
    gap = tip[ok] - bulk[ok]
    return {
        "n_frames": int(ok.sum()),
        "median_gap_A": float(np.median(gap)),
        "max_gap_A": float(np.max(gap)),
        "final_gap_A": float(gap[-1]),
    }


def complete_window_advance_fraction(column_index: np.ndarray,
                                     lookahead: int) -> dict[int, dict]:
    """Count every complete forward window, including failed ones.

    A column the trajectory occupied but never left is the interesting
    case, not a missing one. Dropping it removes the failed attempts
    that carry the information about arrest. Here every occupied column
    contributes all of its complete windows, and the fraction that
    advanced is reported together with the number of windows behind it.

    The result is a descriptive forward-window fraction. It is not a
    hazard, because the windows overlap and are not independent. Rows must
    belong to one uninterrupted, uniformly sampled segment; callers split
    native gaps and restarts before invoking this function. A -1 denotes an
    observed absence of a front, not a missing or unread trajectory frame.
    """
    idx = np.asarray(column_index)
    if idx.ndim != 1 or idx.dtype.kind not in "iu" or np.any(idx < -1):
        raise ValueError("One-dimensional integer columns with optional -1 absence required")
    if (not isinstance(lookahead, (int, np.integer))
            or isinstance(lookahead, (bool, np.bool_)) or lookahead < 1):
        raise ValueError("lookahead must be a positive integer")
    out: dict[int, dict] = {}
    for column in sorted(set(idx[idx >= 0].tolist())):
        starts = np.flatnonzero(idx == column)
        starts = starts[starts + lookahead < idx.size]
        advanced = 0
        for f in starts:
            if np.any(idx[f + 1: f + 1 + lookahead] > column):
                advanced += 1
        out[int(column)] = {
            "n_occupied": int(np.count_nonzero(idx == column)),
            "n_complete_windows": int(starts.size),
            "n_advanced": int(advanced),
            "forward_fraction": (advanced / starts.size) if starts.size else None,
        }
    return out


def selectivity_with_support(selectivity_rows: Sequence[dict],
                              support_key: str = "contacted_observations") -> dict:
    """Summarise a per-column affinity ratio together with its support.

    A median over columns that carry very different numbers of
    observations is not a surface property. This helper reports the
    observation-weighted mean of the supplied ratios, their unweighted
    median, and the smallest set of columns carrying 80 percent of support.
    It does not reconstruct a pooled ratio from numerators and denominators.
    """
    rows = [r for r in selectivity_rows
            if r.get("selectivity") is not None
            and np.isfinite(r.get("selectivity", np.nan))]
    if not rows:
        return {"n_columns": 0}
    values = np.array([float(r["selectivity"]) for r in rows])
    support = np.array([float(r.get(support_key, 0.0)) for r in rows])
    if not np.isfinite(support).all() or np.any(support < 0):
        raise ValueError("Support must be finite and nonnegative")
    total = support.sum()
    if total <= 0:
        return {"n_columns": len(rows), "median": float(np.median(values))}
    order = np.argsort(-support)
    cumulative = np.cumsum(support[order]) / total
    # the smallest number of leading columns that carries 80 % of support
    k = int(np.searchsorted(cumulative, 0.8) + 1)
    dominant = order[:k]
    return {
        "n_columns": len(rows),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "n_columns_carrying_80pct_support": k,
        "share_of_support_in_dominant_columns": float(support[dominant].sum() / total),
        "median_over_dominant_columns": float(np.median(values[dominant])),
        "observation_weighted_mean": float(np.average(values, weights=support)),
        "total_support": float(total),
    }
