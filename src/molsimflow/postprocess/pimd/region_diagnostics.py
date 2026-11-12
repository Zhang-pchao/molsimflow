"""Spatial activity and nuclear-resolution coverage diagnostics for complete paths.

These are observables, not confinement forces or adaptive nuclear dynamics.
Chemical occupancies must use the declared complete assignment/normalization
pool. These helpers do not change that pool or infer permanent molecules.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from molsimflow.io.lammps_dump import minimum_image_vectors


def region_weights(
    positions: np.ndarray,
    center: np.ndarray,
    *,
    core_radius: float,
    outer_radius: float,
    box_lengths: np.ndarray | None = None,
    geometry: str = "sphere",
) -> tuple[np.ndarray, np.ndarray]:
    """Return distances and C2 weights; coordinates must already match the CV mode.

    The final axes are (atoms, 3). Center may be fixed or have matching leading
    axes. Only orthorhombic minimum images are supported; no automatic path
    averaging, molecule reconstruction, or bead/centroid conversion is performed.
    """
    if geometry not in ("sphere", "slab_x", "slab_y", "slab_z"):
        raise ValueError("Geometry must be sphere, slab_x, slab_y or slab_z")
    axis = None if geometry == "sphere" else "xyz".index(geometry[-1])
    xyz, anchor = np.asarray(positions, dtype=float), np.asarray(center, dtype=float)
    if (
        xyz.ndim < 2
        or xyz.shape[-1] != 3
        or xyz.shape[-2] == 0
        or anchor.ndim < 1
        or anchor.shape[-1] != 3
        or not np.isfinite(xyz).all()
        or not np.isfinite(anchor).all()
        or not np.isfinite([core_radius, outer_radius]).all()
        or core_radius < 0
        or outer_radius <= core_radius
    ):
        raise ValueError("Require finite coordinates and 0 <= core_radius < outer_radius")
    delta = xyz - anchor[..., None, :]
    if delta.shape != xyz.shape:
        raise ValueError("Center axes must broadcast to the existing coordinate axes")
    if box_lengths is not None:
        lengths = np.asarray(box_lengths, dtype=float)
        if (
            lengths.shape != (3,)
            or not np.isfinite(lengths).all()
            or np.any(lengths <= 0)
            or outer_radius >= (np.min(lengths) if axis is None else lengths[axis]) / 2
        ):
            raise ValueError("Use a positive orthorhombic box with support below half its width")
        delta = minimum_image_vectors(delta, lengths)
    distances = np.linalg.norm(delta, axis=-1) if axis is None else np.abs(delta[..., axis])
    t = np.clip((distances - core_radius) / (outer_radius - core_radius), 0, 1)
    weights = (1 - t) ** 3 * (1 + 3 * t + 6 * t**2)
    return distances, weights


def sphere_weights(
    positions: np.ndarray,
    center: np.ndarray,
    *,
    core_radius: float,
    outer_radius: float,
    box_lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the spherical-region API with the explicit coordinate convention."""
    return region_weights(
        positions,
        center,
        core_radius=core_radius,
        outer_radius=outer_radius,
        box_lengths=box_lengths,
        geometry="sphere",
    )


def voronoi_dependency_support(
    centers: np.ndarray,
    assigned: np.ndarray,
    spatial_weights: np.ndarray,
    *,
    pair_switch: float,
    pair_cutoff: float,
    box_lengths: np.ndarray | None = None,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Find the complete compact-assignment support of a local coordination CV.

    Coordinates end in (atoms, 3); leading frame/bead axes must match exactly.
    For every active center, retain assigned atoms within pair_cutoff, then all
    competing centers within that cutoff of retained assigned atoms. The caller
    supplies complete candidate pools and the same coordinate view as the CV.
    This offline geometric audit is quadratic in candidate counts, with bounded
    temporary storage. It is not an online neighbor-list accelerator or a PES
    receptive-field analysis. Check anchor coverage separately.
    """
    c = np.asarray(centers, dtype=float)
    a = np.asarray(assigned, dtype=float)
    w = np.asarray(spatial_weights, dtype=float)
    if (
        c.ndim < 2
        or a.ndim != c.ndim
        or c.shape[-1] != 3
        or a.shape[-1] != 3
        or c.shape[:-2] != a.shape[:-2]
        or c.shape[-2] == 0
        or a.shape[-2] == 0
        or w.shape != c.shape[:-1]
        or not np.isfinite(c).all()
        or not np.isfinite(a).all()
        or not np.isfinite(w).all()
        or np.any((w < 0) | (w > 1))
        or not np.isfinite([pair_switch, pair_cutoff]).all()
        or not 0 < pair_switch < pair_cutoff
        or isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size <= 0
    ):
        raise ValueError(
            "Require aligned finite coordinates/weights, valid pair ranges and chunk size"
        )
    lengths = None
    if box_lengths is not None:
        lengths = np.asarray(box_lengths, dtype=float)
        if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("Box lengths must be three positive finite values")
        if pair_cutoff >= lengths.min() / 2:
            raise ValueError("Pair support must lie below the periodic half-box")
    active = w > 0
    competition = active.copy()
    support = np.zeros(a.shape[:-1], dtype=bool)
    missing = np.zeros(a.shape[:-1], dtype=bool)
    cf, af = c.reshape(-1, c.shape[-2], 3), a.reshape(-1, a.shape[-2], 3)
    wf = active.reshape(-1, c.shape[-2])
    nf = competition.reshape(-1, c.shape[-2])
    sf, mf = support.reshape(-1, a.shape[-2]), missing.reshape(-1, a.shape[-2])
    for frame, (cx, ax) in enumerate(zip(cf, af)):
        for start in range(0, len(ax), chunk_size):
            stop = min(len(ax), start + chunk_size)
            delta = cx[:, None, :] - ax[None, start:stop, :]
            if lengths is not None:
                delta = minimum_image_vectors(delta, lengths)
            distance2 = np.sum(delta * delta, axis=-1)
            pairs = distance2 < pair_cutoff**2
            retained = np.any(pairs & wf[frame, :, None], axis=0)
            sf[frame, start:stop] = retained
            mf[frame, start:stop] = retained & ~np.any(distance2 <= pair_switch**2, axis=0)
            nf[frame] |= np.any(pairs & retained[None, :], axis=1)
    return {
        "active_centers": active,
        "assigned_support": support,
        "normalization_centers": competition,
        "missing_unswitched_assignments": missing,
    }


def regional_activity(
    occupancies: np.ndarray,
    spatial_weights: np.ndarray,
    *,
    reference: float | Sequence[float],
) -> dict[str, np.ndarray]:
    """Partition squared positive/negative defects into local and outside parts.

    The last axis identifies centers. Leading frame/bead axes are retained;
    beads are never treated as independent samples. Outside is the complementary
    weight 1-chi using the SAME occupancy formula, not a second truncated pool.
    Activities are not integer species counts or electronic charges.
    """
    occupancy = np.asarray(occupancies, dtype=float)
    weights = np.asarray(spatial_weights, dtype=float)
    neutral = np.asarray(reference, dtype=float)
    if (
        occupancy.ndim < 1
        or occupancy.shape[-1] == 0
        or weights.shape != occupancy.shape
        or not np.isfinite(occupancy).all()
        or not np.isfinite(weights).all()
        or np.any((weights < 0) | (weights > 1))
        or not np.isfinite(neutral).all()
        or neutral.shape not in ((), (occupancy.shape[-1],))
    ):
        raise ValueError("Occupancies, weights in [0,1] and neutral references must align")
    defects = occupancy - neutral
    result = {}
    for sign, value in (("positive", np.maximum(defects, 0)), ("negative", np.minimum(defects, 0))):
        squared = value**2
        result[f"local_{sign}"] = (weights * squared).sum(axis=-1)
        result[f"outside_{sign}"] = ((1 - weights) * squared).sum(axis=-1)
        result[f"global_{sign}"] = squared.sum(axis=-1)
    return result


def fixed_bead_coverage(
    spatial_weights: np.ndarray,
    bead_counts: Sequence[int],
    *,
    required_beads: int,
) -> dict[str, np.ndarray]:
    """Report low-resolution support without assuming a chemical molecule map.

    Call on all relevant O/H or on each species separately. This measures spatial
    support only: normalization competitors or a PES receptive field may extend
    beyond it. Zero uncovered weight in an empty region is not evidence of coverage.
    Continuous adaptive amplitudes are a different contract, not bead counts.
    """
    weights, counts = np.asarray(spatial_weights, dtype=float), np.asarray(bead_counts)
    if (
        weights.ndim < 1
        or weights.shape[-1] == 0
        or counts.shape != (weights.shape[-1],)
        or not np.isfinite(weights).all()
        or np.any((weights < 0) | (weights > 1))
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts <= 0)
        or isinstance(required_beads, (bool, np.bool_))
        or not isinstance(required_beads, (int, np.integer))
        or required_beads <= 0
    ):
        raise ValueError("Require aligned weights and positive integer bead counts/requirement")
    low = counts < required_beads
    return {
        "support_weight": weights.sum(axis=-1),
        "support_present": np.any(weights > 0, axis=-1),
        "low_resolution_weight": (weights * low).sum(axis=-1),
        "low_resolution_atoms": ((weights > 0) & low).sum(axis=-1),
    }


def adaptive_amplitude_coverage(
    spatial_weights: np.ndarray,
    amplitudes: np.ndarray,
    *,
    amplitude_tolerance: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Report reduced path amplitude on spatial support, preserving frame/bead axes.

    Both inputs must have identical axes ending in atoms. Explicitly broadcast a
    centroid-defined amplitude to bead-resolved CV weights before calling. A zero
    support is reported separately; it is not evidence of quantum coverage. The
    deficit is a geometric diagnostic, not an effective bead count, a variance,
    or a bound on the error in a quantum observable. Check O and H separately when
    a chemical CV uses both. Matching support does not qualify the full assignment
    pool, PES neighborhood, quantum buffer or equilibrium ensemble.
    """
    weights = np.asarray(spatial_weights, dtype=float)
    amplitude = np.asarray(amplitudes, dtype=float)
    if (
        weights.ndim < 1
        or weights.shape[-1] == 0
        or amplitude.shape != weights.shape
        or not np.isfinite(weights).all()
        or not np.isfinite(amplitude).all()
        or np.any((weights < 0) | (weights > 1))
        or np.any((amplitude < 0) | (amplitude > 1))
        or not np.isfinite(amplitude_tolerance)
        or not 0 <= amplitude_tolerance < 1
    ):
        raise ValueError("Require matching weights/amplitudes in [0,1] and tolerance in [0,1)")
    reduced = amplitude < 1 - amplitude_tolerance
    return {
        "support_weight": weights.sum(axis=-1),
        "support_present": np.any(weights > 0, axis=-1),
        "reduced_amplitude_atoms": ((weights > 0) & reduced).sum(axis=-1),
        "reduced_amplitude_weight": (weights * reduced).sum(axis=-1),
        "amplitude_deficit_weight": (weights * (1 - amplitude)).sum(axis=-1),
    }


def membership_crossings(
    steps: Sequence[int],
    atom_ids: Sequence[int],
    distances: np.ndarray,
    *,
    radius: float,
    stride: int,
) -> list[dict[str, int | str]]:
    """Report atom-center crossings between adjacent saved frames.

    IDs identify atoms, not proton defects. Relay transport can change chemical
    host without an atom crossing. No first-passage time, real-time PIMD rate,
    within-stride crossing count, or return probability is inferred.
    """
    times, ids = np.asarray(steps), np.asarray(atom_ids)
    d = np.asarray(distances, dtype=float)
    if (
        times.ndim != 1
        or ids.ndim != 1
        or times.size == 0
        or ids.size == 0
        or not np.issubdtype(times.dtype, np.integer)
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids <= 0)
        or len(np.unique(ids)) != len(ids)
        or np.any(np.diff(times) <= 0)
        or d.shape != (len(times), len(ids))
        or not np.isfinite(d).all()
        or np.any(d < 0)
        or not np.isfinite(radius)
        or radius <= 0
        or isinstance(stride, (bool, np.bool_))
        or not isinstance(stride, (int, np.integer))
        or stride <= 0
    ):
        raise ValueError("Require aligned increasing steps, unique IDs and finite distances")
    inside = d < radius
    events = []
    for frame in range(1, len(times)):
        if times[frame] - times[frame - 1] != stride:
            continue
        for atom in np.flatnonzero(inside[frame] != inside[frame - 1]):
            events.append(
                {
                    "previous_step": int(times[frame - 1]),
                    "step": int(times[frame]),
                    "atom_id": int(ids[atom]),
                    "direction": "inward" if inside[frame, atom] else "outward",
                }
            )
    return events


def activity_change_terms(
    previous_occupancies: np.ndarray,
    current_occupancies: np.ndarray,
    previous_weights: np.ndarray,
    current_weights: np.ndarray,
    *,
    reference: float | Sequence[float],
) -> dict[str, np.ndarray]:
    """Exactly separate occupancy changes and spatial-weight changes over an interval.

    For activity a=q_positive^2 or q_negative^2, use the symmetric product rule
    delta(chi*a) = mean(chi)*delta(a) + mean(a)*delta(chi). A relay between fixed
    centers changes occupancy, so that term is NOT labelled reaction/creation.
    Atom identity/order, time interval and coordinate convention must be aligned
    by the caller. These are finite-interval balances, not continuous-time fluxes.
    """
    before = np.asarray(previous_occupancies, dtype=float)
    after = np.asarray(current_occupancies, dtype=float)
    w0 = np.asarray(previous_weights, dtype=float)
    w1 = np.asarray(current_weights, dtype=float)
    if before.shape != after.shape:
        raise ValueError("Previous and current center axes must match")
    regional_activity(before, w0, reference=reference)
    regional_activity(after, w1, reference=reference)
    q0, q1 = before - reference, after - reference
    output = {}
    for sign, choose in (("positive", np.maximum), ("negative", np.minimum)):
        a0, a1 = choose(q0, 0) ** 2, choose(q1, 0) ** 2
        output[f"{sign}_occupancy_change"] = (0.5 * (w1 + w0) * (a1 - a0)).sum(-1)
        output[f"{sign}_spatial_change"] = (0.5 * (a1 + a0) * (w1 - w0)).sum(-1)
        output[f"{sign}_local_change"] = (w1 * a1 - w0 * a0).sum(-1)
        output[f"{sign}_global_change"] = (a1 - a0).sum(-1)
    return output
