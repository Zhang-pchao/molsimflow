"""Dynamic surface-plane reference from a translated periodic slab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.extxyz import read_extxyz_positions
from molsimflow.io.lammps_dump import box_lengths, minimum_image_vectors, periodic_center


@dataclass(frozen=True)
class SurfaceReference:
    nominal_plane_z_A: float
    center_z_A: float
    cell_lengths_A: np.ndarray

    def plane_z(self, surface_coordinates: np.ndarray, bounds: np.ndarray) -> float:
        lengths = box_lengths(bounds)
        if not np.allclose(lengths, self.cell_lengths_A):
            raise ValueError("Reference structure and trajectory cell lengths differ")
        current_center_z = periodic_center(surface_coordinates, bounds)[2]
        shift = minimum_image_vectors(
            np.array([[0.0, 0.0, current_center_z - self.center_z_A]]), lengths
        )[0, 2]
        return float(self.nominal_plane_z_A + shift)


def load_surface_reference(
    structure: Path, surface_range: tuple[int, int], nominal_plane_z_A: float
) -> SurfaceReference:
    _, coordinates, lengths = read_extxyz_positions(structure)
    start, end = surface_range[0] - 1, surface_range[1]
    if end > len(coordinates):
        raise ValueError("Surface range exceeds the reference structure atom count")
    bounds = np.column_stack((np.zeros(3), lengths))
    center_z = periodic_center(coordinates[start:end], bounds)[2]
    return SurfaceReference(nominal_plane_z_A, float(center_z), lengths)
