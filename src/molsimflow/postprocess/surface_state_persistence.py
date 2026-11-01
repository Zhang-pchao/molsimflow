"""Causal confirmation of sampled states separated by an unresolved band.

A confirmed state is memory of supported observations, not current occupancy.
Callers must retain instantaneous coordinates/counts and explicit missingness.
"""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


def confirm_core_states(
    core: np.ndarray,
    times: Sequence[float],
    segments: Sequence[str],
    expected_dt: float,
    dwell: float,
) -> np.ndarray:
    """Confirm 0/1 after a continuous sampled span >= dwell, without backdating.

    Input has time first, optionally followed by an object axis. Values 0/1
    denote observed state cores, -1 is the unresolved band, and -2 is missing
    or physically ambiguous. Neutral observations retain an established state
    but interrupt pending confirmation. Missing observations, source changes,
    and cadence gaps reset memory. Output -1 means not yet confirmed.

    A dwell of five at unit cadence needs six saved observations. This does
    not establish uninterrupted continuous-time residence between frames.
    """
    c = np.asarray(core)
    t = np.asarray(times, dtype=float)
    seg = np.asarray(segments)
    if c.ndim not in (1, 2) or t.ndim != 1 or c.shape[0] != len(t) or seg.shape != t.shape:
        raise ValueError("Expected time or time-by-object cores with matching timestamps and segments")
    if not np.isin(c, [-2, -1, 0, 1]).all():
        raise ValueError("Core values must be -2, -1, 0 or 1")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if not np.isfinite(expected_dt) or expected_dt <= 0 or not np.isfinite(dwell) or dwell < 0:
        raise ValueError("Cadence must be positive and dwell must be nonnegative")
    scalar = c.ndim == 1
    a = c[:, None] if scalar else c
    result = np.full(a.shape, -1, dtype=np.int8)
    state = np.full(a.shape[1], -1, dtype=np.int8)
    pending = np.full(a.shape[1], -1, dtype=np.int8)
    since = np.full(a.shape[1], np.nan)
    for k, current in enumerate(a):
        if k and (seg[k] != seg[k - 1] or not np.isclose(
            t[k] - t[k - 1], expected_dt, rtol=0, atol=1e-7
        )):
            state[:] = -1
            pending[:] = -1
            since[:] = np.nan
        missing = current == -2
        state[missing] = -1
        active = (current >= 0) & (current != state)
        new = active & (current != pending)
        since[new] = t[k]
        since[~active] = np.nan
        pending[:] = np.where(active, current, -1)
        ready = active & (t[k] - since >= dwell - 1e-7)
        state[ready] = current[ready]
        pending[ready] = -1
        since[ready] = np.nan
        result[k] = state
    return result[:, 0] if scalar else result
