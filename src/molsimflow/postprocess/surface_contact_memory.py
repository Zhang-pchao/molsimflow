"""Material-region endpoint recontact from complete, source-aware trajectories.

Regions may overlap: their counts are neither additive areas nor independent samples.
Endpoint recontact includes leaving and returning; it is not continuous survival.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors
from molsimflow.postprocess.nanobubble_attachment import largest_cluster


@dataclass(frozen=True)
class RegionContact:
    """Main-gas membership, contact molecules, and supported material locations."""
    main: np.ndarray
    supported: np.ndarray
    molecule_members: np.ndarray
    region_counts: np.ndarray
    near_members: np.ndarray | None = None
    near_counts: np.ndarray | None = None
    main_near_counts: np.ndarray | None = None
    shell_counts: np.ndarray | None = None
    main_shell_counts: np.ndarray | None = None


def _component_sizes(xyz: np.ndarray, bounds: np.ndarray, cutoff: float) -> np.ndarray:
    """Sizes of periodic components, including singleton molecules."""
    from scipy.spatial import cKDTree
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=int)
    lengths = box_lengths(bounds)
    parent = np.arange(n)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    tree = cKDTree((xyz - bounds[:, 0]) % lengths, boxsize=lengths)
    for a, b in tree.query_pairs(cutoff):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    labels = np.array([root(i) for i in range(n)])
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return counts[inverse]


def supported_material_regions(
    molecule_xyz: np.ndarray,
    solid_distance: np.ndarray,
    local_height: np.ndarray,
    region_xy: np.ndarray,
    bounds: np.ndarray,
    *,
    shells: Sequence[float] = (3.5, 4.0, 4.5),
    radius: float = 6.0,
    near_height: float = 8.5,
    cluster_cutoff: float = 5.5,
    minimum_support: int = 3,
) -> RegionContact:
    """Find locally connected contact support in a fixed material reference.

    All three dimensions must be periodic and orthorhombic; the caller enforces
    the actual trajectory boundary flags. Distances and heights must already
    refer to the declared current solid and surface-height reference.
    """
    xyz = np.asarray(molecule_xyz, dtype=float)
    distance = np.asarray(solid_distance, dtype=float)
    height = np.asarray(local_height, dtype=float)
    regions = np.asarray(region_xy, dtype=float)
    box = np.asarray(bounds, dtype=float)
    cutoffs = np.asarray(shells, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) == 0:
        raise ValueError("Expected nonempty molecule coordinates (N,3)")
    if distance.shape != (len(xyz),) or height.shape != (len(xyz),):
        raise ValueError("Distance and height must match molecule identities")
    if regions.ndim != 2 or regions.shape[1:] != (2,) or not len(regions):
        raise ValueError("Expected nonempty region centers (P,2)")
    if box.shape != (3, 2) or np.any(box[:, 1] <= box[:, 0]):
        raise ValueError("Expected positive orthorhombic bounds")
    if not all(np.isfinite(a).all() for a in [xyz, distance, height, regions, box, cutoffs]):
        raise ValueError("Missing geometry cannot be treated as no contact")
    if np.any(distance < 0) or cutoffs.ndim != 1 or not len(cutoffs) or np.any(cutoffs <= 0):
        raise ValueError("Invalid distances or shells")
    if min(radius, near_height, cluster_cutoff) <= 0 or minimum_support < 1:
        raise ValueError("Positive radii and minimum support required")
    lengths = box_lengths(box)
    main = np.zeros(len(xyz), dtype=bool)
    main[largest_cluster(xyz, box, cluster_cutoff)] = True
    xy = xyz[None, :, :2] - regions[:, None, :]
    xy -= np.rint(xy / lengths[:2]) * lengths[:2]
    near = (xy * xy).sum(axis=-1) <= radius**2
    near &= ((height >= 0) & (height < near_height))[None, :]
    near_all = near.copy()
    shell_members = near_all[None, :, :] & (distance[None, None, :] <= cutoffs[:, None, None])
    shell_counts = shell_members.sum(axis=-1)
    main_shell_counts = (shell_members & main[None, None, :]).sum(axis=-1)
    near &= main[None, :]
    counts = np.zeros((len(cutoffs), len(regions)), dtype=np.int32)
    members = np.zeros((len(cutoffs), len(xyz)), dtype=bool)
    for k, shell in enumerate(cutoffs):
        for j in range(len(regions)):
            use = np.flatnonzero(near[j] & (distance <= shell))
            sizes = _component_sizes(xyz[use], box, cluster_cutoff)
            accepted = use[sizes >= minimum_support]
            counts[k, j] = len(accepted)
            members[k, accepted] = True
    return RegionContact(main, counts >= minimum_support, members, counts,
                         near_all, near_all.sum(axis=-1), near.sum(axis=-1),
                         shell_counts, main_shell_counts)


def region_recontact_curve(
    supported: np.ndarray,
    times: Sequence[float],
    epochs: Sequence[object],
    lag_steps: Sequence[int],
    *,
    frame_dt: float,
    valid: Sequence[bool] | None = None,
) -> list[dict[str, float | int | None]]:
    """Average original-location endpoint overlap over every eligible origin.

    C(t) is the set of supported region identities. Each nonempty time origin
    contributes |C(t) & C(t+lag)| / |C(t)| with equal time-origin weight.
    Empty endpoints contribute zero; empty starts have no denominator and are
    omitted. No pair may span an epoch boundary, a cadence gap or an invalid
    frame. A repeated epoch label after a seam does not reconnect histories.
    """
    s = np.asarray(supported)
    t = np.asarray(times, dtype=float)
    e = np.asarray(epochs)
    lags = np.asarray(lag_steps)
    if s.ndim != 2 or s.dtype != np.bool_ or not s.shape[1]:
        raise ValueError("Expected a boolean (frames,regions) support matrix")
    if t.shape != (len(s),) or e.shape != (len(s),) or not len(s):
        raise ValueError("Frame metadata must align with support")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or frame_dt <= 0:
        raise ValueError("Times must be finite and strictly increasing")
    if lags.ndim != 1 or not np.issubdtype(lags.dtype, np.integer) or np.any(lags < 0):
        raise ValueError("Lag steps must be nonnegative integers")
    ok = np.ones(len(s), bool) if valid is None else np.asarray(valid, dtype=bool)
    if ok.shape != (len(s),):
        raise ValueError("Validity flags must align with frames")
    joined = (e[1:] == e[:-1]) & np.isclose(np.diff(t), frame_dt, rtol=0, atol=1e-8)
    joined &= ok[1:] & ok[:-1]
    run = np.r_[0, np.cumsum(~joined)]
    denominator = s.sum(axis=1)
    result = []
    for value in lags:
        lag = int(value)
        if lag >= len(s):
            use = np.zeros(0, dtype=bool)
            overlap = np.zeros(0)
        else:
            start = np.arange(len(s) - lag)
            end = start + lag
            use = ok[start] & ok[end] & (run[start] == run[end]) & (denominator[start] > 0)
            a, b = start[use], end[use]
            overlap = np.count_nonzero(s[a] & s[b], axis=1) / denominator[a]
        result.append({
            "lag_steps": lag,
            "lag_time": lag * frame_dt,
            "n_origins": int(use.sum()),
            "mean_original_region_fraction": float(overlap.mean()) if len(overlap) else None,
        })
    return result
