"""Molecule-supported surface footprints and source-aware sampled episodes.

Projected footprints are geometric observables, not electron-density contact.
Episode endpoints are sampled bounds, not continuous-time event times.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from scipy.spatial import cKDTree


def capsule_membership(
    points_xy: np.ndarray, bonds_xy: np.ndarray, box_xy: np.ndarray, radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distinct (point index, molecule index) hits for periodic capsules.

    Each bond is a two-endpoint short segment. Coordinates may use any origin.
    Only XY is periodic; no assumption about Z is made by this projected helper.
    """
    points = np.asarray(points_xy, float)
    bonds = np.asarray(bonds_xy, float)
    box = np.asarray(box_xy, float)
    if points.ndim != 2 or points.shape[1] != 2 or bonds.ndim != 3 or bonds.shape[1:] != (2, 2):
        raise ValueError("Expected points (P,2) and bonds (M,2,2)")
    if box.shape != (2,) or np.any(box <= 0) or radius <= 0:
        raise ValueError("Positive XY box lengths and footprint radius required")
    if not np.isfinite(points).all() or not np.isfinite(bonds).all():
        raise ValueError("Missing coordinates cannot imply absent support")
    if not len(bonds) or not len(points):
        return np.array([], int), np.array([], int)
    vec = bonds[:, 1] - bonds[:, 0]
    vec -= np.rint(vec / box) * box
    length = np.linalg.norm(vec, axis=1)
    if np.any(length + 2 * radius >= box.min()):
        raise ValueError("Capsules must be local relative to the periodic box")
    mid = (bonds[:, 0] + vec / 2) % box
    tree = cKDTree(mid, boxsize=box)
    neighbors = tree.query_ball_point(points % box, radius + length.max() / 2 + 1e-10)
    count = np.fromiter((len(x) for x in neighbors), int, count=len(points))
    pi = np.repeat(np.arange(len(points)), count)
    if not len(pi):
        return np.array([], int), np.array([], int)
    mi = np.concatenate([np.asarray(x, int) for x in neighbors if len(x)])
    delta = points[pi] - mid[mi]
    delta -= np.rint(delta / box) * box
    vv = (vec[mi] ** 2).sum(axis=1)
    alpha = np.divide((delta * vec[mi]).sum(axis=1), vv, out=np.zeros(len(pi)), where=vv > 0)
    alpha = np.clip(alpha, -.5, .5)
    closest = delta - alpha[:, None] * vec[mi]
    keep = (closest ** 2).sum(axis=1) <= radius ** 2 + 1e-12
    return pi[keep], mi[keep]


def supported_cell_counts(
    point_indices: np.ndarray, molecule_indices: np.ndarray, point_cells: np.ndarray,
    n_cells: int, n_molecules: int,
) -> dict[str, np.ndarray]:
    """Separate union samples, unique molecules and conserved fractional counts."""
    pi = np.asarray(point_indices, int)
    mi = np.asarray(molecule_indices, int)
    cells = np.asarray(point_cells, int)
    if pi.shape != mi.shape or pi.ndim != 1 or n_cells <= 0 or n_molecules < 0:
        raise ValueError("Incompatible membership inputs")
    if np.any(pi < 0) or np.any(pi >= len(cells)) or np.any(mi < 0) or np.any(mi >= n_molecules):
        raise ValueError("Membership index out of range")
    if np.any(cells < 0) or np.any(cells >= n_cells):
        raise ValueError("Cell index out of range")
    pairs = np.unique(np.stack([pi, mi], axis=1), axis=0)
    pi, mi = pairs[:, 0], pairs[:, 1]
    union = np.bincount(cells[np.unique(pi)], minlength=n_cells)
    per_cell = np.unique(np.stack([cells[pi], mi], axis=1), axis=0)
    distinct = np.bincount(per_cell[:, 0], minlength=n_cells)
    total = np.bincount(mi, minlength=n_molecules)
    weights = np.bincount(cells[pi], weights=1.0 / total[mi], minlength=n_cells)
    assert np.isclose(weights.sum(), np.count_nonzero(total))
    return {"union_samples": union, "distinct_molecules": distinct,
            "fractional_molecules": weights, "molecule_sample_hits": total}


def sampled_episodes(
    states: Sequence[int], times: Sequence[float], segments: Sequence[str], expected_dt: float,
) -> list[dict[str, object]]:
    """Runs of state 1, with explicit unknown (-1), source and end censoring."""
    s, t, seg = np.asarray(states), np.asarray(times, float), np.asarray(segments)
    if s.ndim != 1 or t.shape != s.shape or seg.shape != s.shape or expected_dt <= 0:
        raise ValueError("Expected aligned one-dimensional states, times and segments")
    if np.any(~np.isin(s, [-1, 0, 1])) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Invalid state or time sequence")
    joined = (seg[1:] == seg[:-1]) & np.isclose(np.diff(t), expected_dt, atol=1e-7, rtol=0)
    results = []
    i = 0
    while i < len(s):
        if s[i] != 1:
            i += 1
            continue
        start = i
        while i + 1 < len(s) and joined[i] and s[i + 1] == 1:
            i += 1
        end = i
        left_known = start > 0 and joined[start - 1] and s[start - 1] == 0
        right_known = end + 1 < len(s) and joined[end] and s[end + 1] == 0
        results.append({
            "first_index": start, "last_index": end,
            "sampled_span": float(t[end] - t[start]), "samples": end - start + 1,
            "left_censored": not left_known, "right_censored": not right_known,
            "onset_lower": float(t[start - 1]) if left_known else None,
            "onset_upper": float(t[start]),
            "end_lower": float(t[end]), "end_upper": float(t[end + 1]) if right_known else None,
            "end_reason": "observed_absence" if right_known else "window_end" if end + 1 == len(s)
                else "source_or_gap" if not joined[end] else "unknown",
        })
        i += 1
    return results


def establishment_outcome(
    contact: Sequence[int], times: Sequence[float], segments: Sequence[str], origin_index: int,
    lag: float, dwell: float, expected_dt: float,
) -> dict[str, object]:
    """Observed establishment following an independent supply origin.

    contact is 1 (supported), 0 (observed non-contact), or -1 (unknown).
    The endpoint is confirmation by lag, without backdating. Unknowns before a
    confirmation preserve that confirmation but make its first-passage time unknown.
    """
    s, t, seg = np.asarray(contact), np.asarray(times, float), np.asarray(segments)
    if s.shape != t.shape or seg.shape != t.shape or origin_index < 0 or origin_index >= len(t):
        raise ValueError("Invalid aligned contact trace or origin")
    if lag < dwell or dwell < 0 or expected_dt <= 0:
        raise ValueError("Invalid lag, dwell or cadence")
    i = origin_index
    if s[i] != 0:
        return {"status": "NOT_AT_RISK", "outcome": None}
    end = int(np.searchsorted(t, t[i] + lag, side="right"))
    j = i + 1
    while j < end and seg[j] == seg[i] and np.isclose(t[j] - t[j-1], expected_dt):
        j += 1
    end = j
    runs = sampled_episodes(s[i:end], t[i:end], seg[i:end], expected_dt)
    qualified = [r for r in runs if r["sampled_span"] + 1e-8 >= dwell]
    if qualified:
        r = qualified[0]
        first = i + int(r["first_index"])
        confirm = int(np.searchsorted(t, t[first] + dwell))
        return {"status": "CONFIRMED_SAMPLED_CONTACT", "outcome": 1, "first_index": first,
                "confirmed_index": confirm, "first_passage_resolved": bool(np.all(s[i:first] == 0))}
    full = end > i and np.isclose(t[end - 1], t[i] + lag)
    if full and np.all(s[i:end] >= 0):
        return {"status": "NO_CONFIRMATION_WITHIN_LAG", "outcome": 0}
    return {"status": "CENSORED_UNKNOWN_OR_WINDOW", "outcome": None}
