"""Identity-aware contact-water paths and sampled structural restoration.

These helpers operate on physical timestamps, explicit periodic XY lengths and
caller-supplied local surface heights. They do not assign hydrogen bonds or
infer continuous-time residence between saved frames.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np

PATH_LABELS = (
    "in_original_patch_layer",
    "lateral_in_layer",
    "outer_layer",
    "bulk_like",
    "below_or_ambiguous",
    "missing",
)


def cohort_position_states(
    xy: np.ndarray,
    local_height: np.ndarray,
    center_xy: np.ndarray,
    box_xy: np.ndarray,
    radius: float,
    layer_height: float = 3.0,
    outer_height: float = 6.0,
) -> np.ndarray:
    """Classify positions; local_height must follow the molecule's new XY."""
    xy = np.asarray(xy, dtype=float)
    h = np.asarray(local_height, dtype=float)
    box = np.asarray(box_xy, dtype=float)
    if xy.shape != h.shape + (2,) or box.shape != (2,):
        raise ValueError("Expected matching (..., 2) XY and (...) height arrays")
    if np.any(box <= 0) or not 0 < layer_height < outer_height or radius <= 0:
        raise ValueError("Invalid spatial bounds")
    delta = xy - np.asarray(center_xy, dtype=float)
    delta -= np.rint(delta / box) * box
    inside = np.sum(delta * delta, axis=-1) <= radius * radius
    finite = np.isfinite(xy).all(axis=-1) & np.isfinite(h)
    state = np.full(h.shape, 5, dtype=np.int8)
    state[finite & (h < 0)] = 4
    layer = finite & (h >= 0) & (h < layer_height)
    state[layer & inside] = 0
    state[layer & ~inside] = 1
    state[finite & (h >= layer_height) & (h < outer_height)] = 2
    state[finite & (h >= outer_height)] = 3
    return state


def observed_cohort_history(
    times_ps: Sequence[float],
    segments: Sequence[str],
    states: np.ndarray,
    expected_dt_ps: float,
) -> dict[str, object]:
    """Summarize sampled endpoints/returns, censoring gaps and source seams."""
    t = np.asarray(times_ps, dtype=float)
    seg = np.asarray(segments)
    s = np.asarray(states)
    if len(t) < 2 or s.ndim != 2 or s.shape[0] != len(t) or seg.shape != t.shape:
        raise ValueError("A trace needs >=2 timestamps and a time-by-cohort state matrix")
    if expected_dt_ps <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Time must increase with a positive expected cadence")
    if np.any((s < 0) | (s >= len(PATH_LABELS))):
        raise ValueError("Unknown state code")
    if np.any(s[0] != 0):
        raise ValueError("The initial cohort must be inside the original patch layer")
    if np.any(seg != seg[0]) or not np.allclose(np.diff(t), expected_dt_ps, rtol=0, atol=1e-7):
        return {"status": "CENSORED_TIME_OR_SOURCE", "cohort_size": s.shape[1]}
    counts = np.bincount(s[-1], minlength=len(PATH_LABELS))
    # A known departure precedes a known return; missingness is not an exit.
    away = (s > 0) & (s < 4)
    prior_away = np.vstack((np.zeros(s.shape[1], dtype=bool), np.maximum.accumulate(away, axis=0)[:-1]))
    returned = ((s == 0) & prior_away).any(axis=0)
    return {
        "status": "SAMPLED_PATH_PASS",
        "cohort_size": s.shape[1],
        **{label: int(counts[i]) for i, label in enumerate(PATH_LABELS)},
        "observed_return_anytime": int(returned.sum()),
        "returned_and_present_at_endpoint": int((returned & (s[-1] == 0)).sum()),
        "observed_layer_exit": int(((s == 2) | (s == 3)).any(axis=0).sum()),
        "any_missing": int((s == 5).any(axis=0).sum()),
        "any_ambiguous": int((s == 4).any(axis=0).sum()),
        "lag_ps": float(t[-1] - t[0]),
    }


def site_restoration(
    initial_edges: set[tuple[int, int]],
    current_edges: set[tuple[int, int]],
) -> dict[str, int]:
    """Compare water-O/site edges; preserve site identity independently of water ID."""
    old_by_site: dict[int, set[int]] = {}
    new_by_site: dict[int, set[int]] = {}
    for water, site in initial_edges:
        old_by_site.setdefault(int(site), set()).add(int(water))
    for water, site in current_edges:
        new_by_site.setdefault(int(site), set()).add(int(water))
    both_old_new = 0
    original = replacement = vacant = 0
    for site, old in old_by_site.items():
        now = new_by_site.get(site, set())
        if old & now:
            original += 1
            both_old_new += bool(now - old)
        elif now:
            replacement += 1
        else:
            vacant += 1
    return {
        "initial_sites": len(old_by_site),
        "same_water_same_site": original,
        "new_water_same_site_only": replacement,
        "site_unrestored": vacant,
        "same_site_old_and_new": both_old_new,
        "retained_identity_edges": len(initial_edges & current_edges),
        "current_edges": len(current_edges),
    }


def periodic_grid_components(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Label finite or winding 2D grid components on a periodic XY torus."""
    a = np.asarray(mask, dtype=bool)
    if a.ndim != 2 or connectivity not in (4, 8):
        raise ValueError("Expected a 2D mask and connectivity 4 or 8")
    labels = np.full(a.shape, -1, dtype=np.int32)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    label = 0
    for i, j in np.argwhere(a):
        if labels[i, j] >= 0:
            continue
        labels[i, j] = label
        stack = [(int(i), int(j))]
        while stack:
            u, v = stack.pop()
            for du, dv in offsets:
                x, y = (u + du) % a.shape[0], (v + dv) % a.shape[1]
                if a[x, y] and labels[x, y] < 0:
                    labels[x, y] = label
                    stack.append((x, y))
        label += 1
    return labels


def sampled_support_runs(
    mask: np.ndarray,
    times_ps: Sequence[float],
    segments: Sequence[str],
    expected_dt_ps: float,
) -> list[dict[str, object]]:
    """Return contiguous sampled support spans without bridging missing data."""
    m = np.asarray(mask, dtype=bool)
    t = np.asarray(times_ps, dtype=float)
    seg = np.asarray(segments)
    if m.ndim != 1 or m.shape != t.shape or seg.shape != t.shape:
        raise ValueError("Trace arrays must have the same one-dimensional shape")
    if expected_dt_ps <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Time must increase")
    result = []
    start = None
    for i in range(len(t) + 1):
        continuous = i > 0 and i < len(t) and seg[i] == seg[i - 1] and np.isclose(
            t[i] - t[i - 1], expected_dt_ps, rtol=0, atol=1e-7
        )
        if start is not None and (i == len(t) or not m[i] or not continuous):
            end = i - 1
            result.append({
                "first_index": start, "last_index": end,
                "sampled_span_ps": float(t[end] - t[start]),
                "frames": end - start + 1,
                "left_censored": start == 0 or seg[start] != seg[start - 1] or
                    not np.isclose(t[start] - t[start - 1], expected_dt_ps, rtol=0, atol=1e-7),
                "right_censored": i == len(t) or not continuous,
            })
            start = None
        if i < len(t) and m[i] and start is None:
            start = i
    return result
