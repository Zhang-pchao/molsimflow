"""Periodic site-owned contact geometry and endpoint occupation bookkeeping.

This is a discrete material chart, not a solvent-accessible surface or a
continuous flux estimator. Caller-supplied ordered sites and particles must
retain their identities between frames. Length units are caller-consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree


def _points(value: np.ndarray, dimension: int) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[1] != dimension or not np.isfinite(a).all():
        raise ValueError("Expected finite N by dimension coordinates")
    return a


@dataclass(frozen=True)
class SurfaceGrid:
    """Fixed periodic XY chart; array axes are X then Y, cell-centered."""

    bounds: np.ndarray
    spacing: np.ndarray
    centers: np.ndarray
    owner: np.ndarray
    site_count: int

    @property
    def cell_area(self) -> float:
        return float(np.prod(self.spacing))


def surface_grid(site_xy: np.ndarray, bounds: np.ndarray, spacing: float) -> SurfaceGrid:
    """Assign each cell to its nearest prepared site under periodic XY distance.

    Grid spacing never exceeds the requested value. Ownership is fixed at
    construction: mobile site XY fluctuations do not relabel material cells.
    Coincident periodic sites are rejected because ownership would be arbitrary.
    """
    sites = _points(site_xy, 2)
    box = np.asarray(bounds, dtype=float)
    if (box.shape != (3, 2) or not np.isfinite(box).all()
            or np.any(box[:, 1] <= box[:, 0]) or not len(sites)
            or not np.isfinite(spacing) or spacing <= 0):
        raise ValueError("Invalid box, sites or spacing")
    lengths = np.diff(box, axis=1)[:, 0]
    wrapped = (sites - box[:2, 0]) % lengths[:2]
    if len(np.unique(wrapped, axis=0)) != len(sites):
        raise ValueError("Coincident periodic sites")
    ratio = lengths[:2] / spacing
    shape = np.maximum(3, np.ceil(ratio - 16*np.finfo(float).eps*ratio).astype(int))
    dx = lengths[:2] / shape
    centers = np.stack(np.meshgrid(*[(np.arange(n) + .5) * h + low
                                    for n, h, low in zip(shape, dx, box[:2, 0])],
                                   indexing="ij"), axis=-1)
    owner = cKDTree(wrapped, boxsize=lengths[:2]).query(
        ((centers - box[:2, 0]) % lengths[:2]).reshape(-1, 2))[1].reshape(shape)
    return SurfaceGrid(box.copy(), dx, centers, owner, len(sites))


@dataclass(frozen=True)
class ContactGeometry:
    grid: SurfaceGrid
    footprint: np.ndarray
    signed_distance: np.ndarray
    normal: np.ndarray
    height: np.ndarray
    status: str


def geometry_from_mask(grid: SurfaceGrid, mask: np.ndarray, site_z: np.ndarray,
                       *, normal_smoothing: float = 2.) -> ContactGeometry:
    """Preserve every cell of a footprint; smooth only its orientation field.

    Signed distance is negative inside and positive outside. Its magnitude is
    the periodic distance to the nearest opposite-label cell center, not exact
    distance to a subcell contour (uncertainty of order grid spacing). A 3 by 3
    tiling resolves the orthorhombic periodic minimum. No hole filling, convex
    hull, component removal or morphological closing is performed.
    """
    mask = np.asarray(mask)
    z = np.asarray(site_z, dtype=float)
    if mask.dtype != bool or mask.shape != grid.owner.shape:
        raise ValueError("Footprint must be a boolean array with the grid shape")
    if z.shape != (grid.site_count,) or not np.isfinite(z).all():
        raise ValueError("One finite height per ordered surface site required")
    if not np.isfinite(normal_smoothing) or normal_smoothing < 0:
        raise ValueError("Invalid normal smoothing length")
    status = "DEFINED" if mask.any() and not mask.all() else (
        "EMPTY_CONTACT" if not mask.any() else "FULL_CONTACT")
    signed = np.full(mask.shape, np.nan)
    normal = np.zeros(mask.shape + (2,))
    if status == "DEFINED":
        tiled = np.tile(mask, (3, 3))
        signed_big = (distance_transform_edt(~tiled, sampling=grid.spacing)
                      - distance_transform_edt(tiled, sampling=grid.spacing))
        nx, ny = mask.shape
        signed = signed_big[nx:2*nx, ny:2*ny]
        smooth = gaussian_filter(signed, normal_smoothing / grid.spacing, mode="wrap")
        gradient = np.stack([(np.roll(smooth, -1, axis=i)
                              - np.roll(smooth, 1, axis=i)) / (2*h)
                             for i, h in enumerate(grid.spacing)], axis=-1)
        norm = np.linalg.norm(gradient, axis=-1)
        np.divide(gradient, norm[..., None], out=normal, where=norm[..., None] > 1e-12)
    return ContactGeometry(grid, mask.copy(), signed, normal, z[grid.owner], status)


def contact_geometry(grid: SurfaceGrid, site_xyz: np.ndarray, gas_xyz: np.ndarray,
                     cutoff: float, *, normal_smoothing: float = 2.) -> ContactGeometry:
    """Mark sites within cutoff of supplied gas atoms, then rasterize ownership.

    The caller selects the main gas cluster independently of water occupation.
    Distances are full 3D periodic distances. Only the selected slab face should
    be supplied as sites; the chart must not contain both faces together.
    """
    sites, gas = _points(site_xyz, 3), _points(gas_xyz, 3)
    if len(sites) != grid.site_count or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Invalid site count or contact cutoff")
    box = grid.bounds
    lengths = np.diff(box, axis=1)[:, 0]
    contact = np.zeros(len(sites), dtype=bool)
    if len(gas):
        distances = cKDTree((gas - box[:, 0]) % lengths, boxsize=lengths).query(
            (sites - box[:, 0]) % lengths)[0]
        contact = distances <= cutoff
    return geometry_from_mask(grid, contact[grid.owner], sites[:, 2],
                              normal_smoothing=normal_smoothing)


def orientation_weights(normal: np.ndarray) -> np.ndarray:
    """Return front(+X), rear(-X), flank, unresolved nonnegative weights.

    Weights partition unity. Degenerate normals enter unresolved, not an
    invented direction. Orientation weights are geometric conventions.
    """
    n = np.asarray(normal, dtype=float)
    if n.shape[-1:] != (2,) or not np.isfinite(n).all():
        raise ValueError("Finite planar normals required")
    raw = np.stack([np.maximum(n[..., 0], 0), np.maximum(-n[..., 0], 0),
                    abs(n[..., 1]), np.zeros(n.shape[:-1])], axis=-1)
    total = raw.sum(axis=-1)
    raw[..., 3] = total <= 1e-12
    return raw / raw.sum(axis=-1)[..., None]


def region_weights(geometry: ContactGeometry, points: np.ndarray, *,
                   distance_range: tuple[float, float] = (0., 5.),
                   height_range: tuple[float, float] = (0., 6.)) -> np.ndarray:
    """Return N by 4 conformal-band membership weights for ordered particles.

    Both ranges are lower-inclusive, upper-exclusive. Heights use periodic Z
    displacement above the cell owner's local terminal site. This is an
    explicit geometric band, not an asserted first hydration layer. Undefined
    contact raises rather than silently returning zero occupation.
    """
    p = _points(points, 3)
    for interval in (distance_range, height_range):
        if len(interval) != 2 or not np.isfinite(interval).all() or interval[1] <= interval[0]:
            raise ValueError("Finite increasing intervals required")
    if geometry.status != "DEFINED":
        raise ValueError("Contact boundary is undefined: " + geometry.status)
    grid = geometry.grid
    lengths = np.diff(grid.bounds, axis=1)[:, 0]
    scaled = ((p[:, :2] - grid.bounds[:2, 0]) % lengths[:2]) / grid.spacing
    # Keep exact cell-edge choices stable under common-frame roundoff.
    nearest = np.rint(scaled)
    scaled = np.where(abs(scaled-nearest) <= 32*np.finfo(float).eps*np.maximum(1, abs(scaled)),
                      nearest, scaled)
    index = np.floor(scaled).astype(int)
    index %= np.array(grid.owner.shape)
    ij = tuple(index.T)
    height = (p[:, 2] - geometry.height[ij] + lengths[2]/2) % lengths[2] - lengths[2]/2
    distance = geometry.signed_distance[ij]
    included = ((distance >= distance_range[0]) & (distance < distance_range[1])
                & (height >= height_range[0]) & (height < height_range[1]))
    return orientation_weights(geometry.normal[ij]) * included[:, None]


def endpoint_occupation_budget(m0x0: np.ndarray, m0x1: np.ndarray,
                               m1x0: np.ndarray, m1x1: np.ndarray) -> dict:
    """Exact symmetric endpoint split, summed over the same particle IDs.

    Inputs are N by K nonnegative memberships (boolean is allowed). Equal
    ordering/identity is a caller contract. Geometry includes normal weights,
    contact topology and height changes. This split is not a time integral of
    physical flux and does not resolve between-frame recrossings.
    """
    arrays = [np.asarray(a, dtype=float) for a in (m0x0, m0x1, m1x0, m1x1)]
    if any(a.ndim != 2 or a.shape != arrays[0].shape
           or not np.isfinite(a).all() or np.any(a < 0) for a in arrays):
        raise ValueError("Matching finite nonnegative N by K memberships required")
    n00, n01, n10, n11 = [a.sum(axis=0) for a in arrays]
    particle = .5 * ((n01 - n00) + (n11 - n10))
    geometry = .5 * ((n10 - n00) + (n11 - n01))
    delta = n11 - n00
    return dict(before=n00, after=n11, particle=particle, geometry=geometry,
                delta=delta, residual=delta-particle-geometry)


def prescribed_contact_shift(mask: np.ndarray, spacing: np.ndarray,
                             displacement: float) -> dict:
    """Conservatively translate a periodic XY footprint along X.

    This is a prospective geometric operation, not an atomic displacement or
    force calculation. Fractional shifts mix two adjacent integer rolls; holes
    and components are never filled. Gain and loss are nonnegative cell
    fractions, not future observed contacts.
    """
    m, dx = np.asarray(mask), np.asarray(spacing, dtype=float)
    if (m.ndim != 2 or not all(m.shape) or m.dtype != bool or dx.shape != (2,)
            or not np.isfinite(dx).all() or np.any(dx <= 0)
            or not np.isfinite(displacement)):
        raise ValueError("Boolean XY mask, positive spacing and finite shift required")
    shift = displacement / dx[0]
    k = int(np.floor(shift))
    fraction = shift - k
    moved = (1 - fraction) * np.roll(m, k, axis=0) + fraction * np.roll(m, k+1, axis=0)
    gain, loss = np.maximum(moved - m, 0), np.maximum(m.astype(float) - moved, 0)
    area = float(np.prod(dx))
    return dict(shifted=moved, gain=gain, loss=loss,
                gained_area=float(gain.sum()*area),
                lost_area=float(loss.sum()*area),
                area_residual=float((moved.sum()-m.sum())*area))


def site_weighted_load(cell_weights: np.ndarray, grid: SurfaceGrid,
                       site_counts: np.ndarray) -> dict:
    """Integrate site counts over fractional selected site ownership areas.

    Bounds allow arbitrary unresolved allocation inside each site's cell union.
    Equivalent counts are uniform-within-site estimates, not exact particles.
    """
    w, n = np.asarray(cell_weights, dtype=float), np.asarray(site_counts, dtype=float)
    if (w.shape != grid.owner.shape or n.shape != (grid.site_count,)
            or not np.isfinite(w).all() or not np.isfinite(n).all()
            or np.any(w < 0) or np.any(w > 1) or np.any(n < 0)):
        raise ValueError("Finite nonnegative counts and cell fractions in [0,1] required")
    area = np.bincount(grid.owner.ravel(), minlength=grid.site_count)*grid.cell_area
    selected = np.bincount(grid.owner.ravel(), weights=w.ravel(),
                           minlength=grid.site_count)*grid.cell_area
    fraction = np.divide(selected, area, out=np.zeros_like(area), where=area>0)
    return dict(site_area=area, selected_area=selected, fraction=fraction,
                equivalent=float(fraction@n),
                lower=float(n[(fraction >= 1-1e-12) & (area>0)].sum()),
                upper=float(n[fraction > 1e-12].sum()))
