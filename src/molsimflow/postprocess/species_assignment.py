"""Periodic oxygen-hydrogen assignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors


@dataclass(frozen=True)
class OxygenHydrogenAssignment:
    """Result of assigning hydrogens to nearest oxygen atoms."""

    h_count_per_oxygen: np.ndarray
    hydrogen_to_oxygen_index: np.ndarray
    hydrogen_distance: np.ndarray

    @property
    def hydrogen_indices_by_oxygen(self) -> Dict[int, List[int]]:
        """Return local hydrogen indices grouped by local oxygen index."""

        grouped: Dict[int, List[int]] = {}
        for h_index, oxygen_index in enumerate(self.hydrogen_to_oxygen_index):
            if int(oxygen_index) < 0:
                continue
            grouped.setdefault(int(oxygen_index), []).append(int(h_index))
        return grouped


def bounds_from_box_dims(box_dims: Sequence[float]) -> np.ndarray:
    """Convert box lengths to origin-based LAMMPS-style bounds."""

    lengths = np.asarray(box_dims, dtype=float)
    if lengths.shape != (3,):
        raise ValueError("box_dims must have shape (3,)")
    if np.any(lengths <= 0.0):
        raise ValueError("box_dims must be positive")
    return np.column_stack([np.zeros(3, dtype=float), lengths])


def normalize_bounds(bounds_or_lengths: Union[Sequence[Sequence[float]], Sequence[float]]) -> np.ndarray:
    """Return `(3, 2)` bounds from either LAMMPS bounds or box lengths."""

    arr = np.asarray(bounds_or_lengths, dtype=float)
    if arr.shape == (3, 2):
        lengths = box_lengths(arr)
        if np.any(lengths <= 0.0):
            raise ValueError("bounds lengths must be positive")
        return arr
    if arr.shape == (3,):
        return bounds_from_box_dims(arr)
    raise ValueError(f"bounds_or_lengths must have shape (3,2) or (3,), got {arr.shape}")


def wrap_positions_to_box(coords: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Wrap Cartesian positions into an orthorhombic box with origin at zero."""

    bounds = normalize_bounds(bounds)
    coords = np.asarray(coords, dtype=float)
    lengths = box_lengths(bounds)
    lo = bounds[:, 0]
    return np.mod(coords - lo, lengths)


def assign_hydrogen_to_nearest_oxygen(
    oxygen_coords: np.ndarray,
    hydrogen_coords: np.ndarray,
    bounds_or_lengths: Union[Sequence[Sequence[float]], Sequence[float]],
    oh_cutoff: float,
    chunk_size: int = 4096,
) -> OxygenHydrogenAssignment:
    """Assign each hydrogen to its nearest oxygen under periodic boundaries.

    This implementation uses NumPy chunking and does not require SciPy.  The
    oxygen and hydrogen arrays are local arrays with shape `(N, 3)`.  The box
    can be passed either as LAMMPS bounds with shape `(3, 2)` or as box lengths
    with shape `(3,)`.
    """

    oxygen = np.asarray(oxygen_coords, dtype=float)
    hydrogen = np.asarray(hydrogen_coords, dtype=float)
    bounds = normalize_bounds(bounds_or_lengths)

    if oxygen.ndim != 2 or oxygen.shape[1] != 3:
        raise ValueError("oxygen_coords must have shape (N, 3)")
    if hydrogen.ndim != 2 or hydrogen.shape[1] != 3:
        raise ValueError("hydrogen_coords must have shape (N, 3)")
    if oxygen.shape[0] == 0:
        raise ValueError("No oxygen coordinates provided for O-H assignment")
    if oh_cutoff <= 0.0:
        raise ValueError("oh_cutoff must be positive")

    n_oxygen = oxygen.shape[0]
    n_hydrogen = hydrogen.shape[0]
    if n_hydrogen == 0:
        return OxygenHydrogenAssignment(
            h_count_per_oxygen=np.zeros(n_oxygen, dtype=int),
            hydrogen_to_oxygen_index=np.zeros(0, dtype=int),
            hydrogen_distance=np.zeros(0, dtype=float),
        )

    lengths = box_lengths(bounds)
    oxygen_wrapped = wrap_positions_to_box(oxygen, bounds)
    hydrogen_wrapped = wrap_positions_to_box(hydrogen, bounds)

    nearest_indices = np.full(n_hydrogen, -1, dtype=int)
    nearest_distances = np.full(n_hydrogen, np.inf, dtype=float)
    chunk = max(1, int(chunk_size))
    cutoff_sq = float(oh_cutoff) ** 2
    for start in range(0, n_hydrogen, chunk):
        stop = min(start + chunk, n_hydrogen)
        deltas = hydrogen_wrapped[start:stop, None, :] - oxygen_wrapped[None, :, :]
        deltas = minimum_image_vectors(deltas, lengths)
        dist_sq = np.einsum("hox,hox->ho", deltas, deltas)
        local_nearest = np.argmin(dist_sq, axis=1)
        local_dist_sq = dist_sq[np.arange(stop - start), local_nearest]
        valid = local_dist_sq <= cutoff_sq
        chunk_indices = nearest_indices[start:stop]
        chunk_distances = nearest_distances[start:stop]
        chunk_indices[valid] = local_nearest[valid]
        chunk_distances[valid] = np.sqrt(local_dist_sq[valid])

    assigned = nearest_indices >= 0
    h_count_per_oxygen = np.bincount(nearest_indices[assigned], minlength=n_oxygen).astype(int)
    return OxygenHydrogenAssignment(
        h_count_per_oxygen=h_count_per_oxygen,
        hydrogen_to_oxygen_index=nearest_indices,
        hydrogen_distance=nearest_distances,
    )


def classify_oxygen_species_indices(h_count_per_oxygen: np.ndarray) -> Dict[str, np.ndarray]:
    """Return local oxygen-index arrays grouped by assigned hydrogen count."""

    counts = np.asarray(h_count_per_oxygen, dtype=int)
    return {
        "oh": np.where(counts == 1)[0],
        "h2o": np.where(counts == 2)[0],
        "h3o": np.where(counts == 3)[0],
        "other": np.where(counts >= 4)[0],
    }


def count_oxygen_species(h_count_per_oxygen: np.ndarray) -> Dict[str, int]:
    """Return oxygen species counts from assigned hydrogen counts."""

    grouped = classify_oxygen_species_indices(h_count_per_oxygen)
    return {name: int(indices.size) for name, indices in grouped.items()}
