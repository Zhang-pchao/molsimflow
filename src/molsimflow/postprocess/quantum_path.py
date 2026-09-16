"""Offline Reactive Voronoi path moments for compact orthorhombic paths.

These are geometric descriptors, not free-energy estimates or physical dynamics.
Bead moments are invariant to bead permutations; arbitrary permutations are not
claimed to preserve the ring-polymer Hamiltonian.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from molsimflow.io.lammps_dump import minimum_image_vectors
from molsimflow.postprocess.pimd_reweight import soft_voronoi_occupancies


def unwrap_compact_path(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Return a (P, N, 3) path continuous about each atom in bead zero.

    ``box`` contains three orthorhombic lengths in the coordinate unit. Atom
    identity/order must be identical across beads. Every atom must occupy a
    compact arc whose per-axis span is strictly less than half a box length;
    ambiguous or extended paths fail instead of silently defining a centroid.
    Integer box shifts are immaterial to periodic descriptors. Returned atom
    images follow bead zero, so centroid coordinates themselves are not wrapped.
    """
    path = np.asarray(positions, dtype=float)
    lengths = np.asarray(box, dtype=float)
    if path.ndim != 3 or path.shape[2] != 3 or min(path.shape[:2]) == 0:
        raise ValueError("positions must have shape (P, N, 3), with P and N positive")
    if not np.all(np.isfinite(path)):
        raise ValueError("positions must be finite")
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise ValueError("box must contain three finite positive orthorhombic lengths")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            offsets = minimum_image_vectors(path - path[0], lengths)
            if np.any(np.ptp(offsets, axis=0) >= lengths / 2):
                raise ValueError("ambiguous path: per-atom span reaches half a box length")
            unwrapped = path[0] + offsets
        except FloatingPointError as exc:
            raise ValueError("path geometry exceeds floating-point range") from exc
    return unwrapped


def _frame_descriptors(
    positions: np.ndarray,
    types: np.ndarray,
    box: np.ndarray,
    center_type: int,
    assigned_type: int,
    kappa: float,
    distance_kappa: float,
    reference: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    centers = positions[types == center_type]
    assigned = positions[types == assigned_type]
    oh = np.linalg.norm(
        minimum_image_vectors(assigned[None, :, :] - centers[:, None, :], box), axis=-1
    )
    first, second = np.triu_indices(len(centers), k=1)
    oo = np.linalg.norm(minimum_image_vectors(centers[first] - centers[second], box), axis=-1)
    for distances in (oh, oo):
        if np.any(~np.isfinite(distances) | (distances <= np.finfo(float).eps)):
            raise ValueError("center-assigned and distinct center-center distances must exceed eps")
    occupancy, _ = soft_voronoi_occupancies(
        positions, types, box, center_type, assigned_type, kappa
    )
    defects = occupancy - reference
    if distance_kappa == kappa:
        distance_defects = defects
    else:
        occupancy, _ = soft_voronoi_occupancies(
            positions, types, box, center_type, assigned_type, distance_kappa
        )
        distance_defects = occupancy - reference
    distance = -np.sum(oo * distance_defects[first] * distance_defects[second])
    return defects, float(distance), oo


def describe_quantum_path(
    positions: np.ndarray,
    types: np.ndarray,
    box: np.ndarray,
    *,
    center_type: int,
    assigned_type: int,
    kappa: float,
    distance_kappa: float,
    reference: float,
    environment_r0: float,
) -> dict[str, object]:
    """Describe one complete P-bead frame, without time or statistical weighting.

    Coordinates, ``box``, and ``environment_r0`` share one length unit; ``kappa``
    and ``distance_kappa`` have its inverse unit. Type labels are positive
    integers. The groups must be distinct, with at least two centers and one
    assigned atom, matching GROUP1-only VORONOI_DISTANCE. Site arrays
    follow center atoms in input order. All variances are population moments
    over beads (ddof=0), not independent-sample uncertainty estimates.

    Returned arrays: ``centroid_positions`` (N, 3); ``defect_beads`` (P, C),
    ``defect_centroid`` and ``defect_bead_mean`` (C,); ``q_beads`` and
    ``distance_beads`` (P,); ``oo_coordination_centroid`` (C,).

    Returned scalars: ``q_centroid``, ``q_bead_mean``, ``q_bead_variance``,
    ``v_occ``, ``mean_defect_square``, ``variance_identity_residual``,
    ``distance_centroid``, ``distance_bead_mean``, ``distance_bead_variance``,
    and ``oo_coordination_centroid_mean``.

    Q = sum_i defect_i**2, with defect_i = occupancy_i - reference. In particular,
    mean_b(Q) = sum_i mean_b(defect_i)**2 + v_occ. The first term on the right
    is ``mean_defect_square``; it is generally NOT Q evaluated at the Cartesian
    centroid. Thus v_occ is not generally q_bead_mean - q_centroid.

    Distance uses its own Voronoi sharpness: D = -sum_(i<j) r_ij defect_i defect_j.
    Ordinary centroid O-O coordination sums 1 / (1 + (r_ij/r0)**6) over j != i.
    There is no cutoff, hard site selection, learned variable, or bias force.
    """
    path = unwrap_compact_path(positions, box)
    lengths = np.asarray(box, dtype=float)
    atom_types = np.asarray(types)
    if atom_types.shape != (path.shape[1],) or not np.issubdtype(atom_types.dtype, np.integer):
        raise ValueError("types must be an integer array with one entry per atom")
    if np.any(atom_types <= 0):
        raise ValueError("atom type labels must be positive integers")
    for label in (center_type, assigned_type):
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) or label <= 0:
            raise ValueError("center_type and assigned_type must be positive integers")
    if center_type == assigned_type:
        raise ValueError("center_type and assigned_type must be distinct")
    if np.count_nonzero(atom_types == center_type) < 2:
        raise ValueError("distance descriptor requires at least two center atoms")
    if not np.any(atom_types == assigned_type):
        raise ValueError("assigned group must be nonempty")
    for name, value in (
        ("kappa", kappa), ("distance_kappa", distance_kappa), ("environment_r0", environment_r0)
    ):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isscalar(reference) or not np.isfinite(reference):
        raise ValueError("reference must be finite")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            centroid = np.mean(path, axis=0)
            arguments = (atom_types, lengths, center_type, assigned_type, kappa,
                         distance_kappa, reference)
            bead_results = [_frame_descriptors(bead, *arguments) for bead in path]
            centroid_defects, centroid_distance, centroid_oo = _frame_descriptors(
                centroid, *arguments
            )
            defects = np.asarray([result[0] for result in bead_results])
            distances = np.asarray([result[1] for result in bead_results])
            mean_defects = np.mean(defects, axis=0)
            q_beads = np.sum(defects**2, axis=1)
            q_mean = float(np.mean(q_beads))
            mean_defect_square = float(np.sum(mean_defects**2))
            v_occ = float(np.sum(np.var(defects, axis=0)))
            first, second = np.triu_indices(len(centroid_defects), k=1)
            pair_coordination = 1 / (1 + (centroid_oo / environment_r0)**6)
            coordination = np.zeros(len(centroid_defects))
            np.add.at(coordination, first, pair_coordination)
            np.add.at(coordination, second, pair_coordination)
            return {
                "centroid_positions": centroid,
                "defect_beads": defects,
                "defect_centroid": centroid_defects,
                "defect_bead_mean": mean_defects,
                "q_beads": q_beads,
                "q_centroid": float(np.sum(centroid_defects**2)),
                "q_bead_mean": q_mean,
                "q_bead_variance": float(np.var(q_beads)),
                "v_occ": v_occ,
                "mean_defect_square": mean_defect_square,
                "variance_identity_residual": q_mean - mean_defect_square - v_occ,
                "distance_beads": distances,
                "distance_centroid": centroid_distance,
                "distance_bead_mean": float(np.mean(distances)),
                "distance_bead_variance": float(np.var(distances)),
                "oo_coordination_centroid": coordination,
                "oo_coordination_centroid_mean": float(np.mean(coordination)),
            }
        except FloatingPointError as exc:
            raise ValueError("descriptor calculation exceeds floating-point range") from exc
