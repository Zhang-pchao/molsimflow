"""Small numerical helpers for validating atomistic model predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceErrorMetrics:
    """Per-atom vector-error statistics in the input force unit."""

    mean: float
    rms: float
    maximum: float
    atom_count: int


@dataclass(frozen=True)
class NearestReference:
    """Nearest coordinate row under direct atom-index comparison."""

    index: int
    rmsd: float


def _atom_indices(atom_count: int, atom_indices: Sequence[int] | None) -> np.ndarray:
    indices = (
        np.arange(atom_count, dtype=int)
        if atom_indices is None
        else np.asarray(atom_indices, dtype=int)
    )
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("atom_indices must be a non-empty one-dimensional sequence")
    if np.any(indices < 0) or np.any(indices >= atom_count):
        raise IndexError("atom_indices contains an out-of-range zero-based index")
    return indices


def force_error_metrics(
    reference_forces: np.ndarray,
    predicted_forces: np.ndarray,
    atom_indices: Sequence[int] | None = None,
) -> ForceErrorMetrics:
    """Calculate mean, RMS, and maximum atom-wise force-vector errors.

    ``atom_indices`` uses normal zero-based Python indexing. The function does
    not assume any atom ordering, species, reaction site, or force unit.
    """

    reference = np.asarray(reference_forces, dtype=float)
    predicted = np.asarray(predicted_forces, dtype=float)
    if reference.shape != predicted.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Force arrays must have matching shape (n_atoms, 3)")
    indices = _atom_indices(reference.shape[0], atom_indices)
    magnitudes = np.linalg.norm(predicted[indices] - reference[indices], axis=1)
    return ForceErrorMetrics(
        mean=float(np.mean(magnitudes)),
        rms=float(np.sqrt(np.mean(magnitudes**2))),
        maximum=float(np.max(magnitudes)),
        atom_count=int(magnitudes.size),
    )


def relative_energy_errors(
    reference_energies: Sequence[float],
    predicted_energies: Sequence[float],
    groups: Sequence[object] | None = None,
) -> np.ndarray:
    """Return prediction errors after independent per-group minimum shifts."""

    reference = np.asarray(reference_energies, dtype=float)
    predicted = np.asarray(predicted_energies, dtype=float)
    if reference.ndim != 1 or predicted.shape != reference.shape or reference.size == 0:
        raise ValueError("Energy arrays must be matching, non-empty one-dimensional sequences")
    labels = (
        np.zeros(reference.size, dtype=int) if groups is None else np.asarray(groups, dtype=object)
    )
    if labels.shape != reference.shape:
        raise ValueError("groups must contain one label per energy")
    errors = np.empty_like(reference)
    seen = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    for label in seen:
        mask = labels == label
        errors[mask] = (predicted[mask] - np.min(predicted[mask])) - (
            reference[mask] - np.min(reference[mask])
        )
    return errors


def nearest_coordinate_rmsd(
    query_coordinates: np.ndarray,
    candidate_coordinates: np.ndarray,
    *,
    box_lengths: np.ndarray | None = None,
    atom_indices: Sequence[int] | None = None,
    chunk_size: int = 256,
) -> NearestReference:
    """Find the nearest coordinate row by direct, optionally periodic RMSD.

    This intentionally does not rotate, translate, permute, or align atoms. It
    is suitable for checking indexed simulation frames against datasets with
    the same atom order. ``box_lengths`` may be shape ``(3,)`` or
    ``(n_candidates, 3)`` and therefore supports orthorhombic periodic boxes.
    """

    query = np.asarray(query_coordinates, dtype=float)
    candidates = np.asarray(candidate_coordinates, dtype=float)
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("query_coordinates must have shape (n_atoms, 3)")
    if candidates.ndim != 3 or candidates.shape[1:] != query.shape or candidates.shape[0] == 0:
        raise ValueError("candidate_coordinates must have shape (n_frames, n_atoms, 3)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    indices = _atom_indices(query.shape[0], atom_indices)

    lengths = None
    if box_lengths is not None:
        lengths = np.asarray(box_lengths, dtype=float)
        if lengths.shape == (3,):
            lengths = np.broadcast_to(lengths, (candidates.shape[0], 3))
        if lengths.shape != (candidates.shape[0], 3) or np.any(lengths <= 0.0):
            raise ValueError("box_lengths must be positive with shape (3,) or (n_frames, 3)")

    best = NearestReference(index=-1, rmsd=math.inf)
    for start in range(0, candidates.shape[0], chunk_size):
        stop = min(start + chunk_size, candidates.shape[0])
        delta = query[None, indices, :] - candidates[start:stop, indices, :]
        if lengths is not None:
            current_lengths = lengths[start:stop, None, :]
            delta -= current_lengths * np.rint(delta / current_lengths)
        rmsd = np.sqrt(np.mean(delta**2, axis=(1, 2)))
        local = int(np.argmin(rmsd))
        if float(rmsd[local]) < best.rmsd:
            best = NearestReference(index=start + local, rmsd=float(rmsd[local]))
    return best


def coordinate_coverage_status(
    rmsd: float,
    *,
    exact_tolerance: float = 1.0e-8,
    near_tolerance: float = 0.05,
) -> str:
    """Classify a nearest-coordinate RMSD without claiming dataset provenance."""

    if not math.isfinite(rmsd) or rmsd < 0.0:
        raise ValueError("rmsd must be finite and non-negative")
    if exact_tolerance < 0.0 or near_tolerance < exact_tolerance:
        raise ValueError("Require 0 <= exact_tolerance <= near_tolerance")
    if rmsd <= exact_tolerance:
        return "exact_coordinate_match"
    if rmsd <= near_tolerance:
        return "near_coordinate_neighbor"
    return "nonexact_coordinate_neighbor"
