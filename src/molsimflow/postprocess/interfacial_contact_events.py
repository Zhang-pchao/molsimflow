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
