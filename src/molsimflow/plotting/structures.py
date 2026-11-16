"""Parameterized structure projection and drawing primitives."""

from __future__ import annotations
import numpy as np


def project_coordinates(coordinates, basis, *, origin=(0.0, 0.0, 0.0)):
    """Return XY projection and signed cross-basis depth for a caller's view."""
    xyz, view, center = (
        np.asarray(coordinates, float),
        np.asarray(basis, float),
        np.asarray(origin, float),
    )
    if xyz.ndim != 2 or xyz.shape[1] != 3 or view.shape != (2, 3) or center.shape != (3,):
        raise ValueError("N-by-3 coordinates, two view vectors and a 3D origin are required")
    if (
        not all(np.isfinite(a).all() for a in (xyz, view, center))
        or np.linalg.norm(np.cross(*view)) == 0
    ):
        raise ValueError("Finite coordinates and nonparallel view vectors are required")
    shifted = xyz - center
    return shifted @ view.T, shifted @ np.cross(*view)


def find_bond_pairs(coordinates, elements, cutoffs, *, candidate_radius=None):
    """Return ordered index pairs under explicit unordered element-pair cutoffs.

    The candidate search defaults to the largest pair cutoff. A supplied radius
    explicitly caps that search and can reproduce a historical chart's scope.
    The final distance test is strict (< cutoff). Coordinates must already use
    the intended periodic images. This is a drawing neighbor rule.
    """
    from scipy.spatial import cKDTree

    xyz, labels = np.asarray(coordinates, float), np.asarray(elements)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or labels.shape != (len(xyz),)
        or not np.isfinite(xyz).all()
    ):
        raise ValueError("Finite N-by-3 coordinates and matching element labels are required")
    rules = {}
    for pair, distance in cutoffs.items():
        if len(pair) != 2 or not np.isfinite(distance) or distance <= 0:
            raise ValueError("Element pairs require finite positive distance cutoffs")
        key = tuple(sorted(pair))
        if key in rules and rules[key] != distance:
            raise ValueError("Conflicting unordered pair cutoffs")
        rules[key] = float(distance)
    if not rules or not len(xyz):
        return []
    radius = max(rules.values()) if candidate_radius is None else float(candidate_radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Candidate radius must be finite and positive")
    pairs = []
    for left, right in sorted(cKDTree(xyz).query_pairs(radius)):
        cutoff = rules.get(tuple(sorted((labels[left], labels[right]))))
        if cutoff is not None and np.linalg.norm(xyz[left] - xyz[right]) < cutoff:
            pairs.append((left, right))
    return pairs


def draw_atoms(
    ax,
    coordinates,
    elements,
    *,
    radii,
    colors,
    depth=None,
    edge_color="black",
    linewidth=0.15,
    alpha=1.0,
    zorder=3,
):
    """Draw caller-styled atoms with radii in projected coordinate units.

    Scatter diameters scale with the axes' physical width. Supply an equal
    aspect view for isotropic structure geometry. Styles and fonts belong to
    the caller. Returns the collection for legends or later manipulation.
    """
    xy, labels = np.asarray(coordinates, float), np.asarray(elements)
    if xy.ndim != 2 or xy.shape[1] != 2 or labels.shape != (len(xy),) or not np.isfinite(xy).all():
        raise ValueError("Finite N-by-2 projection and matching elements are required")
    try:
        radius = np.asarray([radii[str(label)] for label in labels], float)
        palette = [colors[str(label)] for label in labels]
    except KeyError as error:
        raise ValueError(f"Missing explicit atom style for {error.args[0]}") from error
    if not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError("Atom radii must be finite and positive")
    order = np.arange(len(xy))
    if depth is not None:
        depth = np.asarray(depth, float)
        if depth.shape != (len(xy),) or not np.isfinite(depth).all():
            raise ValueError("Finite depth must match atom count")
        order = np.argsort(depth, kind="stable")
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    if span <= 0:
        raise ValueError("An increasing projected x range is required")
    scale = ax.get_position().width * ax.get_figure().get_figwidth() * 72 / span
    return ax.scatter(
        xy[order, 0],
        xy[order, 1],
        s=(2 * radius[order] * scale) ** 2,
        c=[palette[index] for index in order],
        edgecolors=edge_color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


def draw_bonds(ax, coordinates, pairs, *, color, linewidth=0.6, alpha=1.0, zorder=2):
    """Draw caller-selected index pairs in an already projected structure."""
    from matplotlib.collections import LineCollection

    xy, indices = np.asarray(coordinates, float), np.asarray(pairs)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Finite N-by-2 projected coordinates are required")
    if not indices.size:
        return None
    if indices.ndim != 2 or indices.shape[1] != 2 or indices.dtype.kind not in "iu":
        raise ValueError("Bond pairs must be a two-column integer array")
    if np.any(indices < 0) or np.any(indices >= len(xy)):
        raise ValueError("Bond indices are outside the coordinate array")
    collection = LineCollection(
        xy[indices], colors=color, linewidths=linewidth, alpha=alpha, zorder=zorder
    )
    ax.add_collection(collection)
    return collection
