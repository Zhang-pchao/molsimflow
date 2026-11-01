"""Prospective count-loss candidates with explicit source and cadence censoring.

This observer knows counts, not molecular identities or chemical species.
Callers must establish chemistry, accessible geometry and molecular support.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CountLossCensus:
    """Arrays have shape (frames, sites), except available_followup_frames."""
    eligible: np.ndarray
    first_loss_index: np.ndarray
    available_followup_frames: np.ndarray


def _span_frames(span: float, dt: float) -> int:
    ratio = span / dt
    count = int(round(ratio))
    if span < 0 or not np.isclose(ratio, count, rtol=0, atol=1e-8):
        raise ValueError("Time spans must be nonnegative multiples of expected_dt")
    return count


def prospective_count_losses(
    counts: np.ndarray,
    times: np.ndarray,
    segments: np.ndarray,
    *,
    expected_dt: float,
    history_span: float,
    search_span: float,
    minimum_initial_count: int,
    minimum_loss: int,
    fractional_loss: float,
    history_absolute_tolerance: float,
    history_fractional_tolerance: float,
) -> CountLossCensus:
    """Register every past-eligible origin and its first observed net count loss.

    Eligibility uses only the initial count and the inclusive past window.
    A gap, source segment boundary or observation end limits follow-up. A loss
    can be observed in a shortened window; absence in that window is censored,
    not a full-window negative. No future outcome controls origin eligibility.
    Changes within one sampling interval have unresolved temporal order.

    The threshold is max(minimum_loss, ceil(fractional_loss * initial_count)).
    All count and time conventions are supplied by the caller.
    """
    w = np.asarray(counts)
    t = np.asarray(times, dtype=float)
    seg = np.asarray(segments)
    if w.ndim != 2 or t.ndim != 1 or seg.ndim != 1 or len(t) != len(w) or len(seg) != len(w):
        raise ValueError("Expected count matrix and aligned one-dimensional time/segment arrays")
    if not np.isfinite(w).all() or np.any(w < 0) or np.any(w != np.floor(w)):
        raise ValueError("Counts must be finite nonnegative integers")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Times must be finite and strictly increasing")
    if not np.isfinite(expected_dt) or expected_dt <= 0:
        raise ValueError("expected_dt must be positive")
    parameters = [minimum_initial_count, minimum_loss, fractional_loss,
                  history_absolute_tolerance, history_fractional_tolerance]
    if not np.isfinite(parameters).all() or min(parameters) < 0 or minimum_loss <= 0:
        raise ValueError("Invalid count or tolerance parameter")
    history = _span_frames(history_span, expected_dt)
    search = _span_frames(search_span, expected_dt)
    if search < 1:
        raise ValueError("search_span must contain at least one interval")
    eligible = np.zeros(w.shape, dtype=bool)
    loss = np.full(w.shape, -1, dtype=np.int64)
    available = np.zeros(len(t), dtype=np.int64)
    if not len(t):
        return CountLossCensus(eligible, loss, available)
    boundaries = np.flatnonzero(
        (seg[1:] != seg[:-1])
        | ~np.isclose(np.diff(t), expected_dt, rtol=0, atol=max(1e-10, expected_dt * 1e-8))) + 1
    edges = np.r_[0, boundaries, len(t)]
    for start, stop in zip(edges[:-1], edges[1:]):
        for i in range(int(start), int(stop)):
            available[i] = min(search, int(stop) - i - 1)
            if i - start < history:
                continue
            baseline = w[i].astype(float)
            tolerance = np.maximum(history_absolute_tolerance,
                                   history_fractional_tolerance * baseline)
            past = w[i - history:i + 1].astype(float)
            eligible[i] = ((baseline >= minimum_initial_count)
                           & np.all(np.abs(past - baseline) <= tolerance, axis=0))
            if not available[i] or not eligible[i].any():
                continue
            threshold = np.maximum(minimum_loss, np.ceil(fractional_loss * baseline))
            crossed = (baseline - w[i + 1:i + 1 + available[i]] >= threshold) & eligible[i]
            observed = crossed.any(axis=0)
            loss[i, observed] = i + 1 + crossed[:, observed].argmax(axis=0)
    return CountLossCensus(eligible, loss, available)
