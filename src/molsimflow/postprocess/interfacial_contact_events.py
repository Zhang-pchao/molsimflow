"""Gas-only sampled contact events and periodic exterior classification."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import label

def periodic_exterior_void(occupied: np.ndarray) -> np.ndarray:
    """Return four-neighbor void components with nonzero periodic winding.

    Label within-box components, then track integer image shifts through their
    periodic boundary connections. A nonzero cycle distinguishes winding from
    finite cavities crossing a boundary, including diagonal winding cycles.
    Supplied occupied cells are never modified or filled.
    """
    mask = np.asarray(occupied)
    if mask.ndim != 2 or mask.dtype != bool or min(mask.shape) < 3:
        raise ValueError("A boolean periodic two-dimensional grid is required")
    labels, count = label(~mask,
                         structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    neighbors = [[] for _ in range(count+1)]
    for axis in (0, 1):
        high = np.take(labels, -1, axis=axis)
        low = np.take(labels, 0, axis=axis)
        shift = np.eye(2, dtype=np.int64)[axis]
        for a, b in set(zip(high.tolist(), low.tolist())):
            if a and b:
                neighbors[a].append((b, shift))
                neighbors[b].append((a, -shift))
    visited = np.zeros(count+1, bool)
    images = np.zeros((count+1, 2), np.int64)
    is_exterior = np.zeros(count+1, bool)
    for root in range(1, count+1):
        if visited[root]:
            continue
        stack, members, winding = [root], [], False
        visited[root] = True
        while stack:
            i = stack.pop()
            members.append(i)
            for j, shift in neighbors[i]:
                target = images[i]+shift
                if visited[j]:
                    winding |= bool(np.any(target != images[j]))
                else:
                    visited[j] = True
                    images[j] = target
                    stack.append(j)
        if winding:
            is_exterior[members] = True
    return is_exterior[labels]

def complete_window_centers(timesteps: np.ndarray, cadence: int,
                            half_window: int) -> np.ndarray:
    """Require every native sample in a symmetric window, without gap joins."""
    steps = np.asarray(timesteps)
    if (steps.ndim != 1 or steps.dtype.kind not in "iu" or cadence <= 0
            or half_window < 0 or np.any(np.diff(steps) <= 0)):
        raise ValueError("Strictly increasing integer steps and valid window required")
    result = np.zeros(len(steps), bool)
    if len(steps) < 2*half_window+1:
        return result
    gaps = np.r_[0, np.cumsum(np.diff(steps) != cadence)]
    i = np.arange(half_window, len(steps)-half_window)
    result[i] = gaps[i+half_window] == gaps[i-half_window]
    return result

def saved_contact_states(contact: np.ndarray, timesteps: np.ndarray,
                         cadence: int, *, persistence: int = 5) -> np.ndarray:
    """Classify every center using only saved gas contact before/after it.

    Before: [t-k,t); after: [t,t+k), k=persistence.
    Codes: gain 1, loss -1, stable dry 2, stable wet 3, mixed 4, invalid 0.
    Five 5-ps samples span 20 ps; unsaved recrossings remain unresolved.
    These labels are not independent events or estimates of continuous rates.
    """
    c, steps = np.asarray(contact), np.asarray(timesteps)
    if (c.ndim != 2 or c.dtype != bool or c.shape[0] != len(steps)
            or not isinstance(persistence, (int, np.integer)) or persistence < 1):
        raise ValueError("Boolean time by identity contact and positive persistence required")
    valid = complete_window_centers(steps, cadence, persistence)
    out = np.zeros(c.shape, np.int8)
    ii = np.flatnonzero(valid)
    total = np.vstack([np.zeros(c.shape[1], np.int64), np.cumsum(c, axis=0)])
    b = total[ii]-total[ii-persistence]
    a = total[ii+persistence]-total[ii]
    labels = np.full(b.shape, 4, np.int8)
    labels[(b == 0) & (a == persistence)] = 1
    labels[(b == persistence) & (a == 0)] = -1
    labels[(b == 0) & (a == 0)] = 2
    labels[(b == persistence) & (a == persistence)] = 3
    out[ii] = labels
    return out


from dataclasses import dataclass


@dataclass(frozen=True)
class SavedStateRun:
    """A uniform known state at consecutive saved samples (inclusive indices)."""
    segment: int
    state: int
    first: int
    last: int
    confirmed: int


@dataclass(frozen=True)
class SavedStateTransition:
    """Transition bracket, not a precise time or continuous residence."""
    segment: int
    before: int
    after: int
    left: int
    right: int
    confirmed: int


def persistent_state_intervals(
    states: np.ndarray, timesteps: np.ndarray, cadence: int, *,
    persistence: int = 5, segments: np.ndarray | None = None,
) -> tuple[list[SavedStateRun], list[SavedStateTransition]]:
    """Bracket changes between persistent saved binary states.

    States are 0, 1, or unknown -1. Runs require persistence consecutive
    native samples; unknown values never extend a run. Native gaps and changes
    in explicit segment identity reset history. Consecutive qualified runs of
    the same state update the last known support. Opposite qualified runs
    bracket a transition between the previous run's last sample and the new
    run's first sample; its confirmation is a separate, later sample.

    Short opposite runs and unknown intervals remain inside the bracket.
    Unobserved recrossings are not excluded. All qualified runs are returned
    so callers can retain censored endpoints and nontransition histories.
    """
    s, t = np.asarray(states), np.asarray(timesteps)
    if (s.ndim != 1 or s.dtype.kind not in "iu" or
            t.ndim != 1 or t.dtype.kind not in "iu" or len(s) != len(t) or
            np.any(~np.isin(s, [-1, 0, 1])) or np.any(np.diff(t) <= 0) or
            not isinstance(cadence, (int, np.integer)) or cadence < 1 or
            not isinstance(persistence, (int, np.integer)) or persistence < 1):
        raise ValueError("Ordered integer samples, binary/unknown states and positive counts required")
    seg = np.zeros(len(s), dtype=int) if segments is None else np.asarray(segments)
    if seg.ndim != 1 or len(seg) != len(s):
        raise ValueError("One explicit segment identity per sample required")
    runs, transitions = [], []
    if not len(s):
        return runs, transitions
    reset = np.r_[True, (np.diff(t) != cadence) | (seg[1:] != seg[:-1])]
    block = np.cumsum(reset) - 1
    cuts = np.r_[0, np.flatnonzero(reset[1:] | (s[1:] != s[:-1])) + 1, len(s)]
    previous = None
    for first, stop in zip(cuts[:-1], cuts[1:]):
        if s[first] < 0 or stop-first < persistence:
            continue
        run = SavedStateRun(int(block[first]), int(s[first]), int(first),
                            int(stop-1), int(first+persistence-1))
        runs.append(run)
        if previous is not None and previous.segment == run.segment and previous.state != run.state:
            transitions.append(SavedStateTransition(
                run.segment, previous.state, run.state, previous.last,
                run.first, run.confirmed))
        previous = run
    return runs, transitions
