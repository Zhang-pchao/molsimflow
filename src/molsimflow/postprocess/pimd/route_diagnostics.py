"""Frame-grouped diagnostics for bead-resolved PIMD reaction coordinates.

These functions describe sampled configurations and declared frame weights.
They do not identify transition states, independent events, or FES convergence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import minimum_image_vectors
from .weights import normalized_log_weights


@dataclass(frozen=True)
class DistanceFrameMetrics:
    """Per-path moments; the bead axis is a correlated within-frame dimension."""

    thresholds: np.ndarray
    quantile_levels: np.ndarray
    quantiles: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    fraction_ge: np.ndarray
    straddles: np.ndarray
    deep_split: np.ndarray


def distance_frame_metrics(
    bead_distance: np.ndarray,
    *,
    thresholds: Sequence[float],
    deep_bounds: tuple[float, float],
    quantile_levels: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> DistanceFrameMetrics:
    """Measure bead heterogeneity without treating beads as independent draws."""
    values = np.asarray(bead_distance, dtype=float)
    cuts = np.asarray(thresholds, dtype=float)
    levels = np.asarray(quantile_levels, dtype=float)
    bounds = np.asarray(deep_bounds, dtype=float)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("bead_distance must be a nonempty finite frame-by-bead array")
    if cuts.ndim != 1 or cuts.size == 0 or not np.isfinite(cuts).all() or np.any(np.diff(cuts) <= 0):
        raise ValueError("thresholds must be finite, nonempty, and increasing")
    if (levels.ndim != 1 or levels.size == 0 or not np.isfinite(levels).all()
            or np.any((levels < 0) | (levels > 1)) or np.any(np.diff(levels) <= 0)):
        raise ValueError("quantile_levels must increase within [0,1]")
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("deep_bounds must be two increasing finite values")
    fraction = np.mean(values[:, :, None] >= cuts[None, None, :], axis=1)
    return DistanceFrameMetrics(
        cuts, levels, np.quantile(values, levels, axis=1).T,
        np.mean(values, axis=1), np.var(values, axis=1), fraction,
        (fraction > 0) & (fraction < 1),
        (np.min(values, axis=1) < bounds[0]) & (np.max(values, axis=1) >= bounds[1]),
    )


def contiguous_episodes(
    steps: Sequence[int], mask: Sequence[bool], *, stride: int,
) -> list[dict[str, int]]:
    """Group adjacent saved frames; durations are grid spans, not reaction lifetimes."""
    times = np.asarray(steps)
    active = np.asarray(mask)
    if (times.ndim != 1 or active.shape != times.shape or times.size == 0
            or not np.issubdtype(times.dtype, np.integer)
            or not np.issubdtype(active.dtype, np.bool_)
            or not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0
            or np.any(np.diff(times) <= 0)):
        raise ValueError("steps/mask must be aligned, increasing integer/bool vectors")
    episodes: list[dict[str, int]] = []
    start = last = None
    count = 0
    for step, selected in zip(times.tolist(), active.tolist()):
        if selected:
            if start is None or step - last != stride:
                if start is not None:
                    episodes.append({"first_step": start, "last_step": last, "frames": count})
                start, count = step, 0
            last = step
            count += 1
        elif start is not None:
            episodes.append({"first_step": start, "last_step": last, "frames": count})
            start = last = None
            count = 0
    if start is not None:
        episodes.append({"first_step": start, "last_step": last, "frames": count})
    return episodes


def regional_weight_support(
    frame_fraction: Sequence[float], log_weights: Sequence[float], block_ids: Sequence[int],
) -> dict[str, object]:
    """Summarize a region using whole-frame weights and within-frame bead fractions."""
    fraction = np.asarray(frame_fraction, dtype=float)
    raw = np.asarray(log_weights, dtype=float)
    blocks = np.asarray(block_ids)
    if (fraction.ndim != 1 or raw.shape != fraction.shape or blocks.shape != fraction.shape
            or fraction.size == 0 or not np.isfinite(fraction).all()
            or np.any((fraction < 0) | (fraction > 1))
            or not np.issubdtype(blocks.dtype, np.integer)):
        raise ValueError("region fractions, log weights, and block IDs must align")
    weights = np.exp(normalized_log_weights(raw))
    contributions = weights * fraction
    mass = float(np.sum(contributions))
    labels = np.unique(blocks)
    block_mass = [float(np.sum(contributions[blocks == label])) for label in labels]
    if mass == 0:
        ess = max_share = top_five_share = 0.0
    else:
        shares = contributions / mass
        ess = float(1.0 / np.sum(shares**2))
        max_share = float(np.max(shares))
        top_five_share = float(np.sum(np.sort(shares)[-5:]))
    return {
        "frames_with_bead_support": int(np.count_nonzero(fraction)),
        "weighted_probability_mass": mass,
        "contribution_kish_ess": ess,
        "maximum_contribution_share": max_share,
        "top_five_contribution_share": top_five_share,
        "block_labels": labels.astype(int).tolist(),
        "block_probability_masses": block_mass,
    }



def read_plumed_selected(
    path: Path, times_fs: Sequence[int], columns: Sequence[str], *, time_scale_to_fs: float = 1.0,
) -> dict[str, np.ndarray]:
    """Stream exact PLUMED rows for declared times without loading dense COLVAR."""
    if not np.isfinite(time_scale_to_fs) or time_scale_to_fs <= 0:
        raise ValueError("time_scale_to_fs must be finite and positive")
    wanted = np.asarray(times_fs)
    if (wanted.ndim != 1 or wanted.size == 0
            or not np.issubdtype(wanted.dtype, np.integer)
            or np.any(np.diff(wanted) <= 0)):
        raise ValueError("times_fs must be a nonempty increasing integer vector")
    names = tuple(columns)
    if not names or len(set(names)) != len(names):
        raise ValueError("columns must be nonempty and unique")
    selected: list[list[float]] = []
    first, last = int(wanted[0]), int(wanted[-1])
    with Path(path).open(encoding="utf-8") as handle:
        header = handle.readline().split()
        if header[:2] != ["#!", "FIELDS"] or len(set(header[2:])) != len(header[2:]):
            raise ValueError("invalid PLUMED FIELDS header")
        fields = header[2:]
        if any(name not in fields for name in names) or "time" not in fields:
            raise ValueError("required PLUMED column missing")
        time_index = fields.index("time")
        indices = [fields.index(name) for name in names]
        last_time = -np.inf
        for line in handle:
            if line.startswith("#! FIELDS") and line.split()[2:] != fields:
                raise ValueError("PLUMED FIELDS changed")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != len(fields):
                raise ValueError("PLUMED row has incorrect column count")
            time = float(parts[time_index]) * time_scale_to_fs
            if not np.isfinite(time):
                raise ValueError("nonfinite PLUMED time")
            if time <= last_time:
                raise ValueError("PLUMED times must be strictly increasing")
            last_time = time
            if time > last:
                break
            if time < first:
                continue
            next_index = len(selected)
            if next_index >= len(wanted):
                raise ValueError("extra selected PLUMED row")
            if time < wanted[next_index]:
                continue
            if time != wanted[next_index]:
                raise ValueError("missing or misaligned PLUMED row")
            values = [float(parts[index]) for index in indices]
            if not np.isfinite(values).all():
                raise ValueError("nonfinite selected PLUMED value")
            selected.append(values)
    if len(selected) != len(wanted):
        raise ValueError("selected PLUMED times are incomplete")
    data = np.asarray(selected, dtype=float)
    return {name: data[:, index] for index, name in enumerate(names)}


def nearest_oxygen_proxy(
    positions: np.ndarray,
    atom_types: np.ndarray,
    box: np.ndarray,
    *,
    oxygen_type: int = 2,
    hydrogen_type: int = 1,
    ambiguity_gap: float = 0.15,
) -> dict[str, object]:
    """Return PBC O-H assignment proxies, without declaring chemical species."""
    xyz = np.asarray(positions, dtype=float)
    types = np.asarray(atom_types)
    lengths = np.asarray(box, dtype=float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or types.shape != (xyz.shape[0],)
            or lengths.shape != (3,) or not np.isfinite(xyz).all()
            or not np.isfinite(lengths).all() or np.any(lengths <= 0)
            or not np.isfinite(ambiguity_gap) or ambiguity_gap <= 0):
        raise ValueError("invalid positions, types, box, or ambiguity_gap")
    oxygen = xyz[types == oxygen_type]
    hydrogen = xyz[types == hydrogen_type]
    if len(oxygen) < 2 or len(hydrogen) == 0:
        raise ValueError("at least two oxygens and one hydrogen are required")
    delta = minimum_image_vectors(hydrogen[None, :, :] - oxygen[:, None, :], lengths)
    distances = np.linalg.norm(delta, axis=2)
    if not np.isfinite(distances).all() or np.any(distances <= np.finfo(float).eps):
        raise ValueError("O-H distances must be finite and positive")
    order = np.argsort(distances, axis=0, kind="stable")
    first, second = order[:2]
    columns = np.arange(len(hydrogen))
    nearest = distances[first, columns]
    next_nearest = distances[second, columns]
    gaps = next_nearest - nearest
    counts = np.bincount(first, minlength=len(oxygen))
    neutral = bool(np.all(counts == 2))
    pair = bool(np.count_nonzero(counts == 1) == 1
                and np.count_nonzero(counts == 3) == 1
                and np.count_nonzero(counts == 2) == len(oxygen) - 2)
    pair_oo = None
    pair_oxygen_indices = None
    if pair:
        anion = int(np.flatnonzero(counts == 1)[0])
        cation = int(np.flatnonzero(counts == 3)[0])
        pair_oxygen_indices = [anion, cation]
        pair_oo = float(np.linalg.norm(
            minimum_image_vectors(oxygen[anion] - oxygen[cation], lengths)
        ))
    h = int(np.argmin(gaps))
    return {
        "nearest_oxygen_counts": counts.astype(int).tolist(),
        "neutral_proxy": neutral,
        "single_pair_proxy": pair,
        "other_proxy": not (neutral or pair),
        "ambiguous_hydrogens": int(np.count_nonzero(gaps < ambiguity_gap)),
        "minimum_sharing_gap_A": float(gaps[h]),
        "most_shared_hydrogen_index": h,
        "two_nearest_oxygen_indices": [int(first[h]), int(second[h])],
        "two_nearest_oh_A": [float(nearest[h]), float(next_nearest[h])],
        "pair_oxygen_indices": pair_oxygen_indices,
        "pair_oo_A": pair_oo,
        "maximum_assigned_oh_A": float(np.max(nearest)),
    }
