"""Identity-preserving contact renewal and moving near-surface water regions.

Proximity is not a liquid film; directed hydrogen bonds are not pair unions.
All geometry assumes a fixed orthorhombic fully periodic box.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from molsimflow.io.lammps_dump import minimum_image_vectors, periodic_center
from molsimflow.postprocess.interfacial_water_hbond import donor_points_to


def site_transition_metrics(gas_before, gas_after, water_before, water_after, weights):
    """Measure weighted endpoint transitions on the SAME ordered site IDs.

    Water replacement requires absent water before and present water after gas
    loss. Co-occupancy is permitted; this does not identify a continuous film.
    """
    gb, ga, wb, wa = [np.asarray(a, dtype=bool) for a in
                      (gas_before, gas_after, water_before, water_after)]
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or any(a.shape != weights.shape for a in (gb, ga, wb, wa)):
        raise ValueError("All site arrays must be matching one-dimensional arrays")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Site weights must be finite and nonnegative")
    lost, gained = gb & ~ga, ~gb & ga
    area = lambda mask: float(weights[mask].sum())
    return dict(gas_lost_area=area(lost), gas_gained_area=area(gained),
                gas_retained_area=area(gb & ga), gas_union_area=area(gb | ga),
                lost_then_wet_area=area(lost & wa),
                lost_and_water_gained_area=area(lost & ~wb & wa),
                gained_and_water_lost_area=area(gained & wb & ~wa))


def surface_region_masks(points, gas_centers, site_xyz, site_groups, bounds, *,
                         height_A=3., edge_halfwidth_A=5., transverse_padding_A=5.):
    """Return face-owned front/rear/interior masks and nearest-site indices.

    Edges use gas q05/q95. Positive height points out of the nearest slab face.
    Regions can overlap for very small bubbles; broad periodic clusters fail.
    """
    points, gas, sites, bounds = [np.asarray(a, dtype=float) for a in
                                 (points, gas_centers, site_xyz, bounds)]
    groups = np.asarray(site_groups).astype(str)
    if bounds.shape != (3, 2) or any(a.ndim != 2 or a.shape[1] != 3 for a in
                                    (points, gas, sites)):
        raise ValueError("Coordinates require N by 3 and bounds 3 by 2")
    if len(gas) < 2 or not len(sites) or groups.shape != (len(sites),):
        raise ValueError("Nonempty sites and at least two gas centers required")
    if not all(np.isfinite(a).all() for a in (points, gas, sites, bounds)):
        raise ValueError("Nonfinite coordinates")
    lengths = bounds[:, 1] - bounds[:, 0]
    if np.any(lengths <= 0) or min(height_A, edge_halfwidth_A, transverse_padding_A) <= 0:
        raise ValueError("Invalid box or region width")
    allowed = {"top_left", "top_right", "bottom_left", "bottom_right"}
    if not set(groups) <= allowed or not all(any(np.char.startswith(groups, f + "_"))
                                           for f in ("top", "bottom")):
        raise ValueError("Both slab faces and known material groups required")
    center = periodic_center(gas, bounds)
    local_gas = minimum_image_vectors(gas - center, lengths)
    low, high = np.quantile(local_gas[:, :2], [.05, .95], axis=0)
    if np.any(high - low >= .8 * lengths[:2]):
        raise ValueError("Broad periodic cluster requires topology review")
    local = minimum_image_vectors(points - center, lengths)
    nearest = cKDTree((sites - bounds[:, 0]) % lengths, boxsize=lengths).query(
        (points - bounds[:, 0]) % lengths)[1]
    owner = groups[nearest]
    normal = minimum_image_vectors(points - sites[nearest], lengths)[:, 2]
    out = {}
    for face, sign in (("top", 1), ("bottom", -1)):
        shared = (np.char.startswith(owner, face + "_") & (normal * sign >= 0)
                  & (normal * sign <= height_A)
                  & (local[:, 1] >= low[1] - transverse_padding_A)
                  & (local[:, 1] <= high[1] + transverse_padding_A))
        out[(face, "front")] = shared & (np.abs(local[:, 0] - high[0]) <= edge_halfwidth_A)
        out[(face, "rear")] = shared & (np.abs(local[:, 0] - low[0]) <= edge_halfwidth_A)
        out[(face, "interior")] = shared & (local[:, 0] > low[0] + edge_halfwidth_A) & (local[:, 0] < high[0] - edge_halfwidth_A)
    return out, nearest


def water_surface_hbond_degrees(water, water_oh, surface, surface_oh, bounds, *,
                               oo_cutoff=3.5, angle_cutoff_deg=30.):
    """Count each water's directed donor and acceptor edges to surface oxygen.

    Caller supplies current hydrogen ownership and chemically eligible sites.
    Each ordered donor/acceptor O pair counts once even if two H align.
    """
    water, surface, bounds = [np.asarray(a, dtype=float) for a in (water, surface, bounds)]
    if bounds.shape != (3, 2) or any(a.ndim != 2 or a.shape[1] != 3 for a in (water, surface)):
        raise ValueError("Invalid coordinates or box")
    if len(water_oh) != len(water) or len(surface_oh) != len(surface):
        raise ValueError("One OH-vector array per oxygen required")
    lengths = bounds[:, 1] - bounds[:, 0]
    if np.any(lengths <= 0) or oo_cutoff <= 0 or not 0 < angle_cutoff_deg < 90:
        raise ValueError("Invalid geometry cutoff")
    if not all(np.isfinite(a).all() for a in (water, surface, bounds)):
        raise ValueError("Nonfinite coordinates")
    donor = np.zeros(len(water), dtype=int)
    acceptor = donor.copy()
    if not len(water) or not len(surface):
        return donor, acceptor
    tree = cKDTree((surface - bounds[:, 0]) % lengths, boxsize=lengths)
    for i, neighbors in enumerate(tree.query_ball_point((water - bounds[:, 0]) % lengths, oo_cutoff)):
        for j in neighbors:
            delta = minimum_image_vectors(surface[j] - water[i], lengths)
            donor[i] += donor_points_to(water_oh[i], delta, angle_cutoff_deg)
            acceptor[i] += donor_points_to(surface_oh[j], -delta, angle_cutoff_deg)
    return donor, acceptor


def spatial_contact_budget(before, after, weights, coordinates):
    """Exact area and centroid-change budget on fixed, ordered reference sites.

    Coordinates must share ONE unwrapped reference for both endpoint masks.
    Caller handles periodic images. Site movement is deliberately excluded.
    Centroids and their decomposition are undefined when either area is zero.
    No persistence, uncertainty, free energy or transport causality is implied.
    """
    b, a = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    w, x = np.asarray(weights, dtype=float), np.asarray(coordinates, dtype=float)
    q = site_transition_metrics(b, a, np.zeros_like(b), np.zeros_like(a), w)
    if x.ndim != 2 or x.shape[0] != len(w) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Coordinates must be finite N by D reference positions")
    ab, aa = float(w[b].sum()), float(w[a].sum())
    gain, loss = a & ~b, b & ~a
    dmoment = (w[gain, None] * x[gain]).sum(0) - (w[loss, None] * x[loss]).sum(0)
    moment_residual = (w[a, None] * x[a]).sum(0) - (w[b, None] * x[b]).sum(0) - dmoment
    valid = ab > 0 and aa > 0
    cb = (w[b, None] * x[b]).sum(0) / ab if ab > 0 else np.full(x.shape[1], np.nan)
    ca = (w[a, None] * x[a]).sum(0) / aa if aa > 0 else np.full(x.shape[1], np.nan)
    gs = (w[gain, None] * (x[gain] - cb)).sum(0) / aa if valid else np.full(x.shape[1], np.nan)
    ls = -(w[loss, None] * (x[loss] - cb)).sum(0) / aa if valid else np.full(x.shape[1], np.nan)
    return dict(area_before=ab, area_after=aa, gained_area=q['gas_gained_area'],
                lost_area=q['gas_lost_area'], retained_area=q['gas_retained_area'],
                area_change=aa-ab,
                area_residual=aa-ab-q['gas_gained_area']+q['gas_lost_area'],
                first_moment_change=dmoment, first_moment_residual=moment_residual,
                centroid_before=cb, centroid_after=ca, centroid_change=ca-cb,
                gain_centroid_change=gs, loss_centroid_change=ls,
                centroid_residual=ca-cb-gs-ls, centroid_valid=valid)
