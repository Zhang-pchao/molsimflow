"""Time-alignment helpers for post-processing tables."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np


def as_float(value: object, default: float = math.nan) -> float:
    """Return a finite float or `default`."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def nearest_row_index(
    rows: Sequence[Mapping[str, object]],
    time_value: float,
    tolerance: float,
    time_column: str = "time_ns",
) -> Optional[int]:
    """Return the nearest row index within a time tolerance."""

    if not rows:
        return None
    times = np.asarray([as_float(row.get(time_column)) for row in rows], dtype=float)
    index = int(np.searchsorted(times, float(time_value)))
    candidates = []
    if index < times.size:
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(float(times[item]) - float(time_value)))
    return best if abs(float(times[best]) - float(time_value)) <= float(tolerance) else None


def infer_timestep_time_scale(
    timesteps: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    tolerance: float,
    time_column: str = "time_ns",
    candidates: Sequence[float] = (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9),
) -> float:
    """Infer a timestep-to-time scale by maximizing table matches."""

    if not timesteps or not rows:
        return 1.0
    best_scale = 1.0
    best_matches = -1
    best_delta = math.inf
    for scale in candidates:
        matches = 0
        deltas = []
        for timestep in timesteps:
            frame_time = float(timestep) * float(scale)
            index = nearest_row_index(rows, frame_time, tolerance, time_column=time_column)
            if index is not None:
                matches += 1
                deltas.append(abs(as_float(rows[index].get(time_column)) - frame_time))
        mean_delta = float(np.mean(deltas)) if deltas else math.inf
        if matches > best_matches or (matches == best_matches and mean_delta < best_delta):
            best_scale = float(scale)
            best_matches = matches
            best_delta = mean_delta
    if best_matches <= 0:
        max_step = max(abs(float(item)) for item in timesteps)
        max_time = max(abs(as_float(row.get(time_column))) for row in rows)
        if max_step > 0 and max_time > 0:
            return max_time / max_step
    return best_scale
