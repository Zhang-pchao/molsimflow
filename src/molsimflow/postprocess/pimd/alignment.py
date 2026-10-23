"""Strict complete-path assembly, restart selection and time alignment."""
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


require = _require


def restart_unique_indices(
    frame_ids: Sequence[float] | np.ndarray,
    *,
    policy: str = "keep_first",
) -> Tuple[np.ndarray, int]:
    """Return indices after removing adjacent restart-seam duplicates."""

    ids = np.asarray(frame_ids, dtype=float)
    _require(ids.ndim == 1 and ids.size > 0, "frame ids must be a nonempty vector")
    _require(np.isfinite(ids).all(), "frame ids must be finite")
    _require(np.all(np.diff(ids) >= 0.0), "frame ids decrease across a restart seam")
    _require(policy in {"error", "keep_first", "keep_last"}, "invalid restart policy")
    duplicate = np.r_[False, ids[1:] == ids[:-1]]
    count = int(np.count_nonzero(duplicate))
    if count == 0:
        return np.arange(ids.size), 0
    if policy == "error":
        raise ValueError("duplicate restart-seam frame ids")
    if policy == "keep_first":
        keep = ~duplicate
    else:
        keep = ~np.r_[ids[:-1] == ids[1:], False]
    return np.flatnonzero(keep), count



def assemble_bead_frames(
    frame_ids: Sequence[int] | np.ndarray,
    bead_ids: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    expected_beads: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble a long bead table into complete frame-by-bead arrays."""

    frames = np.asarray(frame_ids)
    beads = np.asarray(bead_ids)
    data = np.asarray(values, dtype=float)
    _require(frames.ndim == beads.ndim == 1, "frame and bead ids must be vectors")
    _require(data.ndim in {1, 2}, "invalid bead value table")
    _require(frames.size == beads.size == data.shape[0], "long-table row counts differ")
    _require(frames.size > 0, "invalid bead value table")
    _require(np.isfinite(data).all(), "bead values must be finite")
    _require(
        isinstance(expected_beads, (int, np.integer))
        and not isinstance(expected_beads, (bool, np.bool_))
        and expected_beads > 0,
        "expected_beads must be a positive integer",
    )

    ordered_frames = []
    groups: Dict[object, list[int]] = {}
    for row, frame in enumerate(frames.tolist()):
        if frame not in groups:
            ordered_frames.append(frame)
            groups[frame] = []
        groups[frame].append(row)

    first_rows = groups[ordered_frames[0]]
    bead_labels = np.sort(beads[first_rows])
    _require(len(bead_labels) == int(expected_beads), "first frame has missing beads")
    _require(len(np.unique(bead_labels)) == len(bead_labels), "duplicate bead in first frame")
    assembled = []
    for frame in ordered_frames:
        rows = groups[frame]
        labels = beads[rows]
        _require(len(rows) == int(expected_beads), f"frame {frame} has missing beads")
        _require(len(np.unique(labels)) == len(labels), f"frame {frame} has duplicate beads")
        _require(np.array_equal(np.sort(labels), bead_labels), f"frame {frame} bead ids differ")
        order = np.argsort(labels)
        assembled.append(data[np.asarray(rows)[order]])
    return np.asarray(ordered_frames), bead_labels, np.asarray(assembled)



def time_window_mask(
    values: Sequence[float] | np.ndarray,
    first: float,
    last: float,
    *,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Select a finite closed time interval with an explicit tolerance."""
    times = np.asarray(values, dtype=float)
    require(times.ndim == 1 and times.size > 0, "invalid time axis")
    require(np.isfinite(times).all(), "non-finite time axis")
    require(float(first) <= float(last), "inverted time window")
    require(float(tolerance) >= 0.0, "negative time-window tolerance")
    return (times >= float(first) - float(tolerance)) & (
        times <= float(last) + float(tolerance)
    )



def time_scale_to_fs(value: object, label: str) -> float:
    """Validate one explicit conversion from an input time column to fs."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid {label}") from None
    require(np.isfinite(scale) and scale > 0.0, f"invalid {label}")
    return scale



def time_values_to_fs(values: Sequence[float] | np.ndarray, scale: float) -> np.ndarray:
    """Apply one validated raw-time conversion scale."""
    return np.asarray(values, dtype=float) * scale



def aligned_time_indices(
    source: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Match each target to one distinct source frame within a finite tolerance.

    Missing or ambiguous matches raise instead of interpolating or reusing data.
    """
    source_times = np.asarray(source, dtype=float)
    target_times = np.asarray(target, dtype=float)
    require(source_times.ndim == target_times.ndim == 1, "invalid time-grid rank")
    require(source_times.size > 0 and target_times.size > 0, "empty time grid")
    require(np.isfinite(source_times).all(), "non-finite source time grid")
    require(np.isfinite(target_times).all(), "non-finite target time grid")
    require(np.all(np.diff(source_times) > 0.0), "source time grid is not strictly increasing")
    require(np.all(np.diff(target_times) > 0.0), "target time grid is not strictly increasing")
    tolerance = float(tolerance)
    require(np.isfinite(tolerance) and tolerance >= 0.0, "invalid time-grid tolerance")
    # Find the entire closed tolerance interval, not just its right neighbour.
    indices = np.searchsorted(source_times, target_times - tolerance, side="left")
    stops = np.searchsorted(source_times, target_times + tolerance, side="right")
    matches = stops - indices
    require(np.all(matches > 0), "target time is absent from source grid")
    require(np.all(matches == 1), "ambiguous time-grid match within tolerance")
    require(np.all(np.diff(indices) > 0), "target times map to the same source frame")
    return indices
