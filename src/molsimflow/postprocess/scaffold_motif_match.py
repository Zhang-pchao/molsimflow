"""Pure-array matching of two anchor vectors at a fixed center.

The caller owns periodic unwrapping, chemical eligibility and source provenance.
A small mismatch is not a steric, chemical, energetic or candidate-admission test.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Matrix3 = Tuple[Tuple[float, float, float], ...]


def _tolerance(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")
    return value


def _anchors(values):
    vectors = np.asarray(values, dtype=float)
    if vectors.shape != (2, 3) or not np.all(np.isfinite(vectors)):
        raise ValueError("anchor vectors must be a finite (2, 3) array")
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("anchor vectors must have positive finite lengths")
    return vectors, lengths


def _matrix(values):
    return tuple(tuple(float(x) for x in row) for row in values)


def _basis(vectors, lengths, collinearity_tolerance):
    unit = vectors / lengths[:, None]
    if np.linalg.norm(np.cross(unit[0], unit[1])) <= collinearity_tolerance:
        raise ValueError("collinear or numerically singular anchor directions")
    x = unit[0] - unit[1]
    z = -(unit[0] + unit[1])
    x /= np.linalg.norm(x)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    # Roundoff-safe orthogonalization retains the normalized sum/difference chart.
    x = np.cross(y, z)
    return np.column_stack((x, y, z))


def bridge_basis(anchor_vectors_A, *, collinearity_tolerance: float = 1e-12) -> Matrix3:
    """Return a proper basis from the two normalized anchor directions.

    x follows their difference, z opposes their sum, and y = z cross x.
    collinearity_tolerance is dimensionless and is a numerical-singularity guard.
    """
    tolerance = _tolerance(collinearity_tolerance, "collinearity_tolerance")
    if tolerance >= 1:
        raise ValueError("collinearity_tolerance must be smaller than one")
    vectors, lengths = _anchors(anchor_vectors_A)
    return _matrix(_basis(vectors, lengths, tolerance))


@dataclass(frozen=True)
class AnchorPairMatch:
    target_orders: Tuple[Tuple[int, int], ...]
    scores_A: Tuple[float, float]
    rotations: Tuple[Matrix3, Matrix3]
    best_score_A: float
    best_flip_indices: Tuple[int, ...]
    ambiguous: bool
    tie_tolerance_A: float


def match_anchor_pair(source_anchor_vectors_A, target_anchor_vectors_A, *,
                      tie_tolerance_A: float = 1e-10,
                      collinearity_tolerance: float = 1e-12) -> AnchorPairMatch:
    """Compare the source with both target-anchor correspondences.

    Column-vector rotation is R = B_target @ B_source.T. For row-vector inputs,
    transformed source anchors are source @ R.T. The score is the square root
    of the mean squared 3D residual over the two corresponding anchor vectors,
    in the same distance unit as the inputs. No angle/length weighting is used.
    """
    tie = _tolerance(tie_tolerance_A, "tie_tolerance_A")
    source, source_lengths = _anchors(source_anchor_vectors_A)
    target, target_lengths = _anchors(target_anchor_vectors_A)
    tolerance = _tolerance(collinearity_tolerance, "collinearity_tolerance")
    if tolerance >= 1:
        raise ValueError("collinearity_tolerance must be smaller than one")
    source_basis = _basis(source, source_lengths, tolerance)
    orders = ((0, 1), (1, 0))
    scores = []
    rotations = []
    for order in orders:
        target_ordered = target[list(order)]
        target_basis = _basis(target_ordered, target_lengths[list(order)], tolerance)
        rotation = target_basis @ source_basis.T
        residual = source @ rotation.T - target_ordered
        score = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        if not np.isfinite(score):
            raise ValueError("nonfinite residual score")
        scores.append(score)
        rotations.append(_matrix(rotation))
    best = min(scores)
    flips = tuple(i for i, value in enumerate(scores) if value - best <= tie)
    return AnchorPairMatch(orders, tuple(scores), tuple(rotations), best,
                           flips, len(flips) > 1, tie)


def _id(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


@dataclass(frozen=True)
class RankedSource:
    source_id: int
    tie_group: int
    match: AnchorPairMatch


@dataclass(frozen=True)
class SourceRanking:
    selected_source_id: int
    selected_score_A: float
    global_minimum_score_A: float
    first_tie_group_ids: Tuple[int, ...]
    ranked_sources: Tuple[RankedSource, ...]
    preferred_source_id: Optional[int]
    preferred_source_eligible: bool
    tie_tolerance_A: float


def rank_scaffold_motifs(source_ids, source_anchor_vectors_A, target_anchor_vectors_A, *,
                         preferred_source_id: Optional[int] = None,
                         tie_tolerance_A: float = 1e-10,
                         collinearity_tolerance: float = 1e-12) -> SourceRanking:
    """Rank an already chemically eligible source set and select exactly one ID.

    For each rank group, compare all remaining scores with their global minimum,
    rather than using a nontransitive pairwise epsilon comparator. Within that
    minimum-plus-tolerance group, prefer the requested ID if present, then the
    lowest ID. No fallback to a different source after an external clash check
    is implied. Both anchor-order rotations remain in every source record.
    """
    ids = tuple(_id(value, "source ID") for value in source_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("source IDs must be nonempty and unique")
    vectors = np.asarray(source_anchor_vectors_A, dtype=float)
    if vectors.shape != (len(ids), 2, 3):
        raise ValueError("one (2, 3) anchor array is required per source ID")
    preferred = None if preferred_source_id is None else _id(
        preferred_source_id, "preferred_source_id")
    tie = _tolerance(tie_tolerance_A, "tie_tolerance_A")
    matches = {
        source_id: match_anchor_pair(
            anchors, target_anchor_vectors_A, tie_tolerance_A=tie,
            collinearity_tolerance=collinearity_tolerance)
        for source_id, anchors in zip(ids, vectors)}
    remaining = set(ids)
    ranked = []
    first_group = ()
    group_index = 0
    global_minimum = min(match.best_score_A for match in matches.values())
    while remaining:
        minimum = min(matches[source_id].best_score_A for source_id in remaining)
        tied = [source_id for source_id in remaining
                if matches[source_id].best_score_A - minimum <= tie]
        tied.sort(key=lambda source_id: (source_id != preferred, source_id))
        if group_index == 0:
            first_group = tuple(tied)
        ranked.extend(RankedSource(source_id, group_index, matches[source_id])
                      for source_id in tied)
        remaining.difference_update(tied)
        group_index += 1
    selected = ranked[0]
    return SourceRanking(selected.source_id, selected.match.best_score_A, global_minimum,
                         first_group, tuple(ranked), preferred, preferred in matches, tie)
