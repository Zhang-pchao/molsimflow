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


def endpoint_identity_exchange(before: np.ndarray, after: np.ndarray) -> dict:
    """Count saved-endpoint identity overlap in caller-defined regions.

    Boolean arrays are N particles by K regions. Both must have identical
    ordered particle identities. Freeze the same region at both endpoints to
    attribute membership change to particles; a moving region mixes geometry
    with motion. This counts endpoint identity overlap, not continuous
    residence, first exits, crossing multiplicity or a resolved physical flux.
    Empty denominators return NaN and must remain missing in summaries.
    """
    b, a = np.asarray(before), np.asarray(after)
    if (b.dtype != bool or a.dtype != bool or b.ndim != 2
            or a.shape != b.shape or not b.shape[1]):
        raise ValueError("Matching boolean N by K memberships required")
    n0, n1 = b.sum(axis=0), a.sum(axis=0)
    retained = (b & a).sum(axis=0)
    gained, lost = (~b & a).sum(axis=0), (b & ~a).sum(axis=0)
    gross, net = gained + lost, gained - lost

    def ratio(numerator, denominator):
        out = np.full(denominator.shape, np.nan, dtype=float)
        np.divide(numerator, denominator, out=out, where=denominator > 0)
        return out

    return dict(before=n0, after=n1, retained=retained, gained=gained, lost=lost,
                net=net, gross=gross,
                retained_fraction=ratio(retained, n0),
                replacement_fraction=ratio(gained, n1),
                cancellation_fraction=1-ratio(abs(net), gross))


def water_surface_hbond_edges(water, water_oh, surface, surface_oh, bounds, *,
                             oo_cutoff=3.5, angle_cutoff_deg=30.):
    """Return identity-resolved directed rows (water index, site index, donor).

    Donor is 0 for water and 1 for surface. Both directions may occur for one
    oxygen pair; project unique pairs before counting bridging waters.
    Input oxygen/particle ordering is the caller's persistent identity map.
    Geometric hydrogen bonds neither quantify bond strength nor causal pinning.
    """
    water, surface, bounds = [np.asarray(a, dtype=float) for a in (water, surface, bounds)]
    if (bounds.shape != (3, 2) or any(a.ndim != 2 or a.shape[1] != 3 for a in
                                    (water, surface))
            or len(water_oh) != len(water) or len(surface_oh) != len(surface)):
        raise ValueError("Matching oxygen coordinates and OH-vector lists required")
    lengths = bounds[:, 1] - bounds[:, 0]
    if (not all(np.isfinite(a).all() for a in (water, surface, bounds))
            or np.any(lengths <= 0) or not np.isfinite(oo_cutoff) or oo_cutoff <= 0
            or not np.isfinite(angle_cutoff_deg) or not 0 < angle_cutoff_deg < 90):
        raise ValueError("Invalid box, coordinates or hydrogen-bond cutoffs")
    for vectors in list(water_oh) + list(surface_oh):
        a = np.asarray(vectors, dtype=float)
        if (a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all()
                or np.any(np.linalg.norm(a, axis=1) <= 0)):
            raise ValueError("OH vectors must be finite nonzero H by 3 arrays")
    rows = []
    if len(water) and len(surface):
        tree = cKDTree((surface - bounds[:, 0]) % lengths, boxsize=lengths)
        neighbors = tree.query_ball_point((water - bounds[:, 0]) % lengths, oo_cutoff)
        for i, sites in enumerate(neighbors):
            for j in sorted(sites):
                delta = minimum_image_vectors(surface[j] - water[i], lengths)
                if donor_points_to(water_oh[i], delta, angle_cutoff_deg):
                    rows.append((i, j, 0))
                if donor_points_to(surface_oh[j], -delta, angle_cutoff_deg):
                    rows.append((i, j, 1))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def shared_water_site_bridges(directed_edges):
    """Project water-site edges onto site pairs with one-water bridge counts.

    Duplicate rows and bidirectional H bonds do not multiply a bridging water.
    Returns sorted site-pair rows and the number of distinct waters per pair.
    This is a one-water path projection, not a free-energy or percolation test.
    """
    from collections import defaultdict
    from itertools import combinations
    e = np.asarray(directed_edges)
    if (e.ndim != 2 or e.shape[1] != 3 or e.dtype.kind not in "iu"
            or np.any(e < 0) or not np.isin(e[:, 2], [0, 1]).all()):
        raise ValueError("Nonnegative integer water/site/donor edge rows required")
    by_water = defaultdict(set)
    for water, site, _ in e:
        by_water[int(water)].add(int(site))
    count = defaultdict(int)
    for sites in by_water.values():
        for pair in combinations(sorted(sites), 2):
            count[pair] += 1
    pairs = sorted(count)
    return (np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
            np.asarray([count[p] for p in pairs], dtype=np.int64))


def origin_conditioned_transport(
    ids_before: np.ndarray,
    ids_after: np.ndarray,
    positions_before: np.ndarray,
    positions_after: np.ndarray,
    origin_labels: np.ndarray,
    destination_labels: np.ndarray,
    frame_displacement: np.ndarray,
    *,
    state_count: int,
) -> dict[str, np.ndarray]:
    """Accumulate endpoint transitions and displacement moments by origin.

    IDs must be identical ordered unique nonnegative integers. Positions are
    already unwrapped N by 3 coordinates in a common lab frame; this function
    never infers periodic images from a minimum-image endpoint displacement.
    Each particle has one integer state in [0, state_count), with 0 reserved
    by convention for outside. All initial particles remain in the accounting,
    including those that leave a region. Origin/destination geometry is owned
    by the caller; moving-state changes alone do not imply particle motion.

    Return counts[K,K], origin_support[K], destination_support[K] and lab/
    relative displacement sums and componentwise squared sums[K,K,3].
    Relative displacement is lab displacement minus frame_displacement.
    Summing destination axis 1 gives origin-conditioned moments without
    survivor selection. Empty supports have zero moments, not defined means.

    These endpoint moments do not count intervening crossings or prove
    continuous residence, a resolved flux, recirculation, or closed paths.
    """
    if (not isinstance(state_count, (int, np.integer))
            or isinstance(state_count, (bool, np.bool_)) or state_count < 1):
        raise ValueError("state_count must be a positive integer")
    before_ids, after_ids = np.asarray(ids_before), np.asarray(ids_after)
    if (before_ids.ndim != 1 or after_ids.shape != before_ids.shape
            or before_ids.dtype.kind not in "iu" or after_ids.dtype.kind not in "iu"
            or np.any(before_ids < 0) or np.any(after_ids < 0)
            or not np.array_equal(before_ids, after_ids)
            or len(np.unique(before_ids)) != len(before_ids)):
        raise ValueError("Identical ordered unique nonnegative integer IDs required")
    labels = [np.asarray(origin_labels), np.asarray(destination_labels)]
    if any(a.shape != before_ids.shape or a.dtype.kind not in "iu"
           or np.any(a < 0) or np.any(a >= state_count) for a in labels):
        raise ValueError("One integer state in [0, state_count) per particle required")
    raw_positions = [np.asarray(positions_before), np.asarray(positions_after),
                     np.asarray(frame_displacement)]
    if any(a.dtype.kind not in "iuf" for a in raw_positions):
        raise ValueError("Real coordinate and frame-displacement arrays required")
    x0, x1, shift = [np.asarray(a, dtype=float) for a in raw_positions]
    if (x0.shape != (len(before_ids), 3) or x1.shape != x0.shape
            or shift.shape != (3,)
            or not all(np.isfinite(a).all() for a in (x0, x1, shift))):
        raise ValueError("Finite N by 3 endpoints and 3-vector frame displacement required")

    origin, destination = [a.astype(np.int64, copy=False) for a in labels]
    shape = (int(state_count), int(state_count))
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(counts, (origin, destination), 1)
    result = dict(counts=counts, origin_support=counts.sum(axis=1),
                  destination_support=counts.sum(axis=0))
    try:
        with np.errstate(over="raise", invalid="raise"):
            lab = x1 - x0
            relative = lab - shift
            for name, displacement in (("lab", lab), ("relative", relative)):
                total = np.zeros(shape + (3,), dtype=float)
                squared = np.zeros_like(total)
                np.add.at(total, (origin, destination), displacement)
                np.add.at(squared, (origin, destination), displacement * displacement)
                result[name + "_displacement_sum"] = total
                result[name + "_displacement_squared_sum"] = squared
    except FloatingPointError as exc:
        raise ValueError("Displacement moment overflow") from exc
    return result


def sampled_run_support(states: np.ndarray, steps: np.ndarray, *,
                        span_steps: int, cadence_steps: int) -> np.ndarray:
    """Mark run END samples having a continuous saved-state span.

    States are time by site. Gaps reset all runs; false samples reset their site.
    A span requires elapsed time, so a span of four cadences needs five samples.
    This does not establish continuous residence between saved samples.
    """
    s, t = np.asarray(states), np.asarray(steps)
    if (s.ndim != 2 or s.dtype != bool or t.ndim != 1 or len(t) != len(s)
            or t.dtype.kind not in "iu" or np.any(np.diff(t) <= 0)
            or span_steps < 0 or cadence_steps <= 0):
        raise ValueError("Ordered integer times and boolean time-by-site states required")
    out = np.zeros_like(s)
    if not len(t):
        return out
    start = np.full(s.shape[1], t[0], dtype=np.int64)
    previous = np.zeros(s.shape[1], dtype=bool)
    for i, step in enumerate(t):
        if i and step-t[i-1] != cadence_steps:
            previous[:] = False
        begin = s[i] & ~previous
        start[begin] = step
        out[i] = s[i] & (step-start >= span_steps)
        previous = s[i].copy()
    return out
