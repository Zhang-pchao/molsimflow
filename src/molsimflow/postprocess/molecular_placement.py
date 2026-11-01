"""Bounded fixed-shape placement under explicit pair-distance conventions.

Only orthorhombic XY-periodic, Z-nonperiodic Cartesian geometry is supported.
The module proves geometry relative to the caller's declared complete atom roster;
it cannot establish that roster's provenance or the physical validity of a radius.
"""
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.io.lammps_dump import minimum_image_vectors

Vec3 = Tuple[float, float, float]


def _points(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(name + " must be a finite (N, 3) array")
    return array


def _ids(values, name):
    values = tuple(values)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in values):
        raise ValueError(name + " must contain integer atom or cell IDs")
    result = tuple(int(v) for v in values)
    if len(set(result)) != len(result):
        raise ValueError(name + " must be unique")
    return result


def _species(values, size):
    result = tuple(values)
    if len(result) != size or any(not isinstance(v, str) or not v for v in result):
        raise ValueError("species must contain one nonempty string per atom")
    return result


def _tuples(array):
    return tuple(tuple(float(x) for x in row) for row in array)


@dataclass(frozen=True)
class CompleteFrame:
    """All source atoms plus an explicit analytic obstacle selection.

    expected_atom_ids must come from an independently established complete roster.
    No halo subset or unverified completeness flag is accepted.
    """
    atom_ids: Tuple[int, ...]
    species: Tuple[str, ...]
    positions_A: Tuple[Vec3, ...]
    obstacle_atom_ids: Tuple[int, ...]
    bounds_A: Tuple[Tuple[float, float], ...]


def complete_frame(atom_ids, species, positions_A, *, expected_atom_ids,
                   obstacle_atom_ids, bounds_A) -> CompleteFrame:
    ids = _ids(atom_ids, "atom_ids")
    expected = _ids(expected_atom_ids, "expected_atom_ids")
    if not ids or set(ids) != set(expected):
        raise ValueError("UNKNOWN: full-frame atom roster is incomplete or mismatched")
    points = _points(positions_A, "positions_A")
    if len(points) != len(ids):
        raise ValueError("positions and atom IDs differ in length")
    labels = _species(species, len(ids))
    selected = _ids(obstacle_atom_ids, "obstacle_atom_ids")
    if not set(selected).issubset(ids):
        raise ValueError("obstacle selection contains atoms outside the complete frame")
    bounds = np.asarray(bounds_A, dtype=float)
    if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(
            bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds_A must be orthorhombic lower/upper bounds")
    return CompleteFrame(ids, labels, _tuples(points), selected, _tuples(bounds))


@dataclass(frozen=True)
class MolecularShape:
    species: Tuple[str, ...]
    masses: Tuple[float, ...]
    offsets_A: Tuple[Vec3, ...]


def molecular_shape(species, masses, offsets_A) -> MolecularShape:
    offsets = _points(offsets_A, "offsets_A")
    labels = _species(species, len(offsets))
    mass = np.asarray(masses, dtype=float)
    if not len(offsets) or mass.shape != (len(offsets),) or not np.all(
            np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("one positive finite mass is required per shape atom")
    if not np.allclose(np.average(offsets, axis=0, weights=mass), 0., atol=1e-10,
                       rtol=0.):
        raise ValueError("shape offsets must be centered at the mass-weighted COM")
    return MolecularShape(labels, tuple(float(x) for x in mass), _tuples(offsets))


@dataclass(frozen=True)
class PairConvention:
    """Explicit symmetric pair distances, never inferred from species labels."""
    mode: str
    distances_A: Tuple[Tuple[str, str, float], ...]

    def distance(self, first, second):
        key = tuple(sorted((first, second)))
        for a, b, value in self.distances_A:
            if (a, b) == key:
                return value
        raise ValueError("UNKNOWN: no pair convention for " + repr(key))


def pair_convention(distances_A: Mapping[Tuple[str, str], float], *,
                    mode: str) -> PairConvention:
    if mode not in ("center_distance", "exclusion_clearance"):
        raise ValueError("mode must be center_distance or exclusion_clearance")
    rows = {}
    for key, value in distances_A.items():
        if len(key) != 2 or any(not isinstance(v, str) or not v for v in key):
            raise ValueError("pair keys must contain two species labels")
        key = tuple(sorted(key))
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("pair distances must be finite and nonnegative")
        if (mode == "center_distance" and value != 0) or (
                mode == "exclusion_clearance" and value <= 0):
            raise ValueError("center_distance requires zero; exclusion requires positive values")
        if key in rows and rows[key] != value:
            raise ValueError("asymmetric pair distances")
        rows[key] = value
    if not rows:
        raise ValueError("pair table must not be empty")
    return PairConvention(mode, tuple((a, b, rows[(a, b)]) for a, b in sorted(rows)))


@dataclass(frozen=True)
class CellDomain:
    """Frozen union of half-open Cartesian boxes; periodic seam cells are split."""
    cell_ids: Tuple[int, ...]
    lower_A: Tuple[Vec3, ...]
    upper_A: Tuple[Vec3, ...]


def cell_domain(cell_ids, lower_A, upper_A) -> CellDomain:
    ids = _ids(cell_ids, "cell_ids")
    lower = _points(lower_A, "lower_A")
    upper = _points(upper_A, "upper_A")
    if not ids or len(lower) != len(ids) or upper.shape != lower.shape or np.any(
            upper <= lower):
        raise ValueError("one nonempty cell box is required per cell ID")
    return CellDomain(ids, _tuples(lower), _tuples(upper))


def _cell_memberships(points, domain, frame):
    bounds = np.asarray(frame.bounds_A)
    lower, upper = np.asarray(domain.lower_A), np.asarray(domain.upper_A)
    if np.any(lower[:, :2] < bounds[:2, 0]) or np.any(upper[:, :2] > bounds[:2, 1]):
        raise ValueError("domain XY cells must lie in the primary box; split seam cells")
    wrapped = np.array(points, dtype=float, copy=True)
    wrapped[:, :2] = ((wrapped[:, :2] - bounds[:2, 0]) %
                      (bounds[:2, 1] - bounds[:2, 0])) + bounds[:2, 0]
    return np.all((wrapped[:, None, :] >= lower) &
                  (wrapped[:, None, :] < upper), axis=2)


def _rotations(values):
    rotations = np.asarray(values, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or not len(rotations):
        raise ValueError("rotations must be a nonempty (R, 3, 3) array")
    if not np.all(np.isfinite(rotations)):
        raise ValueError("rotations must be finite")
    for rotation in rotations:
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.) or not (
                abs(np.linalg.det(rotation) - 1.) <= 1e-10):
            raise ValueError("only proper rotations are allowed")
    return rotations


@dataclass(frozen=True)
class Clearance:
    value_A: float
    # first: shape-local index; second: obstacle atom ID or second shape-local index.
    limiting_pair: Optional[Tuple[int, int]]


def _clearance(first, first_species, second, second_species, second_ids, frame, rule):
    if not len(second):
        return Clearance(float("inf"), None)
    delta = np.asarray(first)[:, None, :] - np.asarray(second)[None, :, :]
    lengths = np.diff(np.asarray(frame.bounds_A), axis=1).ravel()
    delta[:, :, :2] = minimum_image_vectors(delta[:, :, :2], lengths[:2])
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    thresholds = np.array([[rule.distance(a, b) for b in second_species]
                           for a in first_species])
    signed = distances - thresholds
    flat = int(np.argmin(signed))
    i, j = np.unravel_index(flat, signed.shape)
    return Clearance(float(signed[i, j]), (int(i), int(second_ids[j])))


class _ObstacleIndex:
    """Exact species-wise nearest neighbors using nine XY images, never Z images."""

    def __init__(self, positions, species, atom_ids, frame, rule):
        from scipy.spatial import cKDTree

        self.bounds = np.asarray(frame.bounds_A)
        self.lengths = np.diff(self.bounds, axis=1).ravel()
        self.rule = rule
        self.groups = []
        normalized = self.normalize(positions)
        species = np.asarray(species)
        atom_ids = np.asarray(atom_ids, dtype=np.int64)
        for label in sorted(set(species)):
            mask = species == label
            base = normalized[mask]
            images = np.concatenate([
                base + np.array([ix * self.lengths[0], iy * self.lengths[1], 0.])
                for ix in (-1, 0, 1) for iy in (-1, 0, 1)])
            self.groups.append((label, cKDTree(images), np.tile(atom_ids[mask], 9)))

    def normalize(self, points):
        result = np.array(points, dtype=float, copy=True).reshape(-1, 3)
        result[:, :2] = ((result[:, :2] - self.bounds[:2, 0]) %
                        self.lengths[:2]) + self.bounds[:2, 0]
        return result

    def clearances(self, positions, species):
        """Return one minimum and limiting pair per pose without pose-sized objects."""
        positions = np.asarray(positions)
        count, atoms = positions.shape[:2]
        values = np.full(count, np.inf)
        probe_indices = np.full(count, -1, dtype=np.int64)
        obstacle_ids = np.full(count, -1, dtype=np.int64)
        query = self.normalize(positions.reshape(-1, 3))
        for label, tree, ids in self.groups:
            distances, indices = tree.query(query, k=1, eps=0.)
            signed = distances.reshape(count, atoms) - np.array(
                [self.rule.distance(first, label) for first in species])
            local_indices = np.argmin(signed, axis=1)
            local_values = signed[np.arange(count), local_indices]
            update = local_values < values
            values[update] = local_values[update]
            probe_indices[update] = local_indices[update]
            nearest = indices.reshape(count, atoms)[np.arange(count), local_indices]
            obstacle_ids[update] = ids[nearest[update]]
        return values, probe_indices, obstacle_ids

    def clearance(self, positions, species):
        query = self.normalize(positions)
        best = Clearance(float("inf"), None)
        for label, tree, ids in self.groups:
            distances, indices = tree.query(query, k=1, eps=0.)
            for atom_i, first in enumerate(species):
                signed = float(distances[atom_i]) - self.rule.distance(first, label)
                if signed < best.value_A:
                    best = Clearance(signed, (atom_i, int(ids[indices[atom_i]])))
        return best


def _obstacles(frame, omitted_ids=()):
    omitted = set(omitted_ids)
    selected = set(frame.obstacle_atom_ids) - omitted
    indices = [i for i, atom in enumerate(frame.atom_ids) if atom in selected]
    return (np.asarray(frame.positions_A)[indices],
            tuple(frame.species[i] for i in indices),
            tuple(frame.atom_ids[i] for i in indices))


@dataclass(frozen=True)
class PoseResult:
    sample_index: int
    rotation_index: int
    cell_ids: Tuple[int, ...]
    com_A: Vec3
    positions_A: Tuple[Vec3, ...]
    all_probe_nuclei_in_domain: bool
    obstacle_clearance: Clearance


@dataclass(frozen=True)
class TwoPoseResult:
    first: PoseResult
    second: PoseResult
    intermolecular_clearance: Clearance
    joint_clearance_A: float
    limiting_component: str


@dataclass(frozen=True)
class SearchResult:
    status: str
    mode: str
    poses_evaluated: int
    poses_planned: int
    pairs_evaluated: int
    pairs_planned: int
    pairs_pruned: int
    best_two_upper_bound_A: Optional[float]
    best_two_gap_A: Optional[float]
    obstacle_queries: int
    obstacle_query_budget: int
    finite_grid_optimum: bool
    best_single: Optional[PoseResult]
    best_two: Optional[TwoPoseResult]
    two_pose_threshold_witness: Optional[bool]


def _budget(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def search_placements(shape: MolecularShape, frame: CompleteFrame, rule: PairConvention,
                      domain: CellDomain, com_samples_A, rotations, *,
                      max_pose_evaluations: int, max_pair_evaluations: int,
                      max_obstacle_queries: Optional[int] = None) -> SearchResult:
    """Maximize sampled single and joint two-pose clearance within one frozen domain.

    The pair score is min(first-obstacle, second-obstacle, all intermolecular pairs).
    Exhausting this finite grid does not establish continuous inaccessibility.
    No ordinary-probe self-exclusion option exists.
    """
    pose_budget = _budget(max_pose_evaluations, "max_pose_evaluations")
    pair_budget = _budget(max_pair_evaluations, "max_pair_evaluations")
    samples = _points(com_samples_A, "com_samples_A")
    if not len(samples):
        raise ValueError("at least one COM sample is required")
    turns = _rotations(rotations)
    memberships = _cell_memberships(samples, domain, frame)
    if not np.all(np.any(memberships, axis=1)):
        raise ValueError("all COM samples must belong to the same frozen cell union")
    obstacle_xyz, obstacle_species, obstacle_ids = _obstacles(frame)
    for first in set(shape.species):
        for second in set(obstacle_species) | set(shape.species):
            rule.distance(first, second)
    obstacle_index = _ObstacleIndex(obstacle_xyz, obstacle_species, obstacle_ids, frame, rule)
    planned = len(samples) * len(turns)
    queries_per_pose = len(shape.species) * len(set(obstacle_species))
    query_budget = (pose_budget * queries_per_pose if max_obstacle_queries is None else
                    _budget(max_obstacle_queries, "max_obstacle_queries"))
    if queries_per_pose:
        pose_budget = min(pose_budget, query_budget // queries_per_pose)
    evaluated = min(planned, pose_budget)
    flat_indices = np.arange(evaluated)
    sample_indices, rotation_indices = flat_indices // len(turns), flat_indices % len(turns)
    rotated = np.einsum("aij,nj->ani", turns, np.asarray(shape.offsets_A))
    positions = rotated[rotation_indices] + samples[sample_indices, None, :]
    singles = np.empty(evaluated)
    single_probe = np.empty(evaluated, dtype=np.int64)
    single_obstacle = np.empty(evaluated, dtype=np.int64)
    # The KD query buffer is bounded; the compact pose arrays scale linearly.
    query_chunk = max(1, 32768 // len(shape.species))
    for start in range(0, evaluated, query_chunk):
        stop = min(start + query_chunk, evaluated)
        singles[start:stop], single_probe[start:stop], single_obstacle[start:stop] = (
            obstacle_index.clearances(positions[start:stop], shape.species))

    def pose(index):
        sample_i, rotation_i = int(sample_indices[index]), int(rotation_indices[index])
        limiting_pair = (None if single_probe[index] < 0 else
                         (int(single_probe[index]), int(single_obstacle[index])))
        return PoseResult(
            sample_i, rotation_i,
            tuple(domain.cell_ids[i] for i in np.flatnonzero(memberships[sample_i])),
            tuple(float(x) for x in samples[sample_i]), _tuples(positions[index]),
            bool(np.all(np.any(_cell_memberships(positions[index], domain, frame), axis=1))),
            Clearance(float(singles[index]), limiting_pair))

    order = np.argsort(-singles, kind="stable")
    best_single = pose(int(order[0])) if evaluated else None
    best_indices = None
    best_joint = float("-inf")
    best_inter = None
    best_component = None
    checked = 0
    pruned = 0
    pairs_planned = planned * (planned - 1) // 2
    available_pairs = evaluated * (evaluated - 1) // 2
    remaining_upper = float("-inf")
    budget_stop = False
    chunk_size = max(1, 262144 // (len(shape.species) ** 2 * 3))
    pair_thresholds = np.array([[rule.distance(a, b) for b in shape.species]
                                for a in shape.species])
    lengths = np.diff(np.asarray(frame.bounds_A), axis=1).ravel()
    for i, first in enumerate(order[:-1]):
        if best_indices is not None and singles[order[i + 1]] <= best_joint:
            remaining = evaluated - i
            pruned += remaining * (remaining - 1) // 2
            break
        for start in range(i + 1, evaluated, chunk_size):
            if best_indices is not None and singles[order[start]] <= best_joint:
                pruned += evaluated - start
                break
            if checked == pair_budget:
                future = singles[order[i + 2]] if i + 2 < evaluated else float("-inf")
                remaining_upper = max(float(singles[order[start]]), float(future))
                budget_stop = True
                break
            count = min(chunk_size, evaluated - start, pair_budget - checked)
            seconds = order[start:start + count]
            delta = (positions[first][None, :, None, :] -
                     positions[seconds][:, None, :, :])
            delta[..., :2] = minimum_image_vectors(delta[..., :2], lengths[:2])
            signed = np.sqrt(np.sum(delta * delta, axis=-1)) - pair_thresholds
            flattened = signed.reshape(count, -1)
            minimum_indices = np.argmin(flattened, axis=1)
            inter_scores = flattened[np.arange(count), minimum_indices]
            scores = np.column_stack((
                np.full(count, singles[first]), singles[seconds], inter_scores))
            joint_scores = np.min(scores, axis=1)
            best_i = int(np.argmax(joint_scores))
            joint = float(joint_scores[best_i])
            if best_indices is None or joint > best_joint:
                atom_pair = np.unravel_index(minimum_indices[best_i], pair_thresholds.shape)
                best_inter = Clearance(float(inter_scores[best_i]),
                                       tuple(int(x) for x in atom_pair))
                best_indices = (int(first), int(seconds[best_i]))
                best_joint = joint
                best_component = ("first_obstacles", "second_obstacles",
                                  "intermolecular")[int(np.argmin(scores[best_i]))]
            checked += count
            next_start = start + count
            if count < min(chunk_size, evaluated - start):
                future = singles[order[i + 2]] if i + 2 < evaluated else float("-inf")
                remaining_upper = max(float(singles[order[next_start]]), float(future))
                budget_stop = True
                break
        if budget_stop:
            break
    certified_pairs = checked + pruned == available_pairs
    if best_indices is not None and remaining_upper <= best_joint:
        certified_pairs = True
        pruned = available_pairs - checked
    complete = evaluated == planned and certified_pairs
    best_two = None
    upper = None
    gap = None
    if best_indices is not None:
        best_two = TwoPoseResult(pose(best_indices[0]), pose(best_indices[1]),
                                 best_inter, best_joint, best_component)
        upper = max(best_joint, remaining_upper) if evaluated == planned else float("inf")
        gap = upper - best_joint
    witness = None
    if rule.mode == "exclusion_clearance":
        if best_two is not None and best_joint >= 0.:
            witness = True
        elif complete:
            witness = False
    status = ("CERTIFIED_DISCRETE_OPTIMUM" if pruned else
              "EXHAUSTIVE_DISCRETE_SEARCH") if complete else "BUDGET_EXHAUSTED"
    return SearchResult(
        status, rule.mode, evaluated, planned, checked, pairs_planned,
        pruned, upper, gap, evaluated * queries_per_pose, query_budget, complete,
        best_single, best_two, witness)


@dataclass(frozen=True)
class PosePairEvaluation:
    first_obstacle_clearance: Clearance
    second_obstacle_clearance: Clearance
    intermolecular_clearance: Clearance
    joint_clearance_A: float
    limiting_component: str


def evaluate_pose_positions(frame: CompleteFrame, species, positions_A,
                            rule: PairConvention) -> Clearance:
    """Evaluate supplied atom positions without search or self-exclusion.

    The caller owns molecule membership and coordinate provenance. This does not
    fit or alter the pose. An explicit background mask remains part of the frame.
    """
    positions = _points(positions_A, "positions_A")
    if not len(positions):
        raise ValueError("a pose must contain at least one atom")
    labels = _species(species, len(positions))
    obstacles, obstacle_species, obstacle_ids = _obstacles(frame)
    for first in set(labels):
        for second in set(obstacle_species):
            rule.distance(first, second)
    return _ObstacleIndex(obstacles, obstacle_species, obstacle_ids, frame, rule).clearance(
        positions, labels)


def evaluate_pose_pair_positions(frame: CompleteFrame, first_species, first_positions_A,
                                 second_species, second_positions_A,
                                 rule: PairConvention) -> PosePairEvaluation:
    """Evaluate two fixed poses, including every intermolecular atom pair."""
    first = _points(first_positions_A, "first_positions_A")
    second = _points(second_positions_A, "second_positions_A")
    if not len(first) or not len(second):
        raise ValueError("both poses must contain atoms")
    first_labels = _species(first_species, len(first))
    second_labels = _species(second_species, len(second))
    obstacles, obstacle_species, obstacle_ids = _obstacles(frame)
    for a in set(first_labels) | set(second_labels):
        for b in set(obstacle_species):
            rule.distance(a, b)
    index = _ObstacleIndex(obstacles, obstacle_species, obstacle_ids, frame, rule)
    first_score = index.clearance(first, first_labels)
    second_score = index.clearance(second, second_labels)
    inter = _clearance(first, first_labels, second, second_labels,
                       tuple(range(len(second))), frame, rule)
    scores = (first_score.value_A, second_score.value_A, inter.value_A)
    component = ("first_obstacles", "second_obstacles", "intermolecular")[int(np.argmin(scores))]
    return PosePairEvaluation(first_score, second_score, inter, min(scores), component)


@dataclass(frozen=True)
class OccupiedControl:
    atom_ids: Tuple[int, ...]
    positions_A: Tuple[Vec3, ...]
    excluded_self_ids: Tuple[int, ...]
    clearance: Clearance
    mode: str


def occupied_pose_control(frame: CompleteFrame, occupied_atom_ids: Sequence[int],
                          rule: PairConvention) -> OccupiedControl:
    """Evaluate only actual source coordinates, excluding only those exact atom IDs.

    The caller must supply complete molecule membership from an admitted external
    molecular roster; atom existence alone does not prove bonded membership.
    A negative exclusion clearance diagnoses the chosen probe convention at an
    observed pose. It does not establish an invalid trajectory or inaccessible site.
    """
    ids = _ids(occupied_atom_ids, "occupied_atom_ids")
    if not ids or not set(ids).issubset(frame.atom_ids):
        raise ValueError("occupied atom IDs must name an actual complete-frame molecule")
    index = {atom: i for i, atom in enumerate(frame.atom_ids)}
    indices = [index[atom] for atom in ids]
    xyz = np.asarray(frame.positions_A)[indices]
    labels = tuple(frame.species[i] for i in indices)
    obstacles, species, obstacle_ids = _obstacles(frame, ids)
    clearance = _ObstacleIndex(obstacles, species, obstacle_ids, frame, rule).clearance(xyz, labels)
    excluded = tuple(atom for atom in ids if atom in frame.obstacle_atom_ids)
    return OccupiedControl(ids, _tuples(xyz), excluded, clearance, rule.mode)
