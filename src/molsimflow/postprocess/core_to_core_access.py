"""Discrete, identity-preserving core-to-core access observers."""
from __future__ import annotations
import numpy as np

def core_labels(relative_xyz: np.ndarray, source: dict, target: dict) -> np.ndarray:
    """Label disjoint cylindrical cores: buffer=0, source=1, target=2."""
    x = np.asarray(relative_xyz)
    r = np.linalg.norm(x[..., :2], axis=-1)
    h = x[..., 2]
    b = (r >= source["r_lo_A"]) & (r <= source["r_hi_A"]) & (h >= source["h_lo_A"]) & (h <= source["h_hi_A"])
    t = (r <= target["radius_A"]) & (h >= target["h_lo_A"]) & (h <= target["h_hi_A"])
    if np.any(b & t):
        raise ValueError("Source and target cores overlap")
    return b.astype(np.int8) + 2 * t.astype(np.int8)

def inventory(labels: np.ndarray, target: int = 2) -> dict:
    """Sampled entries minus exits reproduce occupancy, not continuous flux."""
    occupied = np.asarray(labels) == target
    count = occupied.sum(axis=1)
    entries = np.r_[0, (occupied[1:] & ~occupied[:-1]).sum(axis=1)]
    exits = np.r_[0, (~occupied[1:] & occupied[:-1]).sum(axis=1)]
    residual = count - count[0] - np.cumsum(entries - exits)
    return dict(count=count, entries=entries, exits=exits, residual=residual)

def first_passages(labels: np.ndarray, origin_ok: np.ndarray | None = None) -> list[dict]:
    """One cohort origin per molecule; returns never reset its clock."""
    a = np.asarray(labels)
    ok = np.ones_like(a, dtype=bool) if origin_ok is None else np.asarray(origin_ok, dtype=bool)
    rows = []
    for m in range(a.shape[1]):
        starts = np.flatnonzero((a[:, m] == 1) & ok[:, m])
        if not len(starts):
            continue
        start = int(starts[0])
        hit = np.flatnonzero(a[start:, m] == 2)
        end = start + int(hit[0]) if len(hit) else len(a) - 1
        rows.append(dict(molecule=m, start=start, end=end, arrived=bool(len(hit)), initial=start == 0))
    return rows

def excursions(labels: np.ndarray, origin: int = 1, target: int = 2) -> list[dict]:
    """Each departure ends at first return, opposite core, or window censor."""
    a = np.asarray(labels)
    rows = []
    for m in range(a.shape[1]):
        start = None
        for k in range(1, len(a)):
            if start is None and a[k-1, m] == origin and a[k, m] != origin:
                start = k - 1
            if start is not None and a[k, m] in (origin, target):
                rows.append(dict(molecule=m, start=start, end=k,
                                 outcome="return" if a[k, m] == origin else "hit"))
                start = None
        if start is not None:
            rows.append(dict(molecule=m, start=start, end=len(a)-1, outcome="censored"))
    return rows
