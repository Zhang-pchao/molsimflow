"""Fixed-axis two-water virtual maps and exact generalized-force projections.

Six positions: [O, H, H, O, H, H]. Coordinates: water rotation (rad), water
translation, neighbor translation. This is not an integrator or free energy.
"""
from __future__ import annotations
import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError('Expected a finite three-vector.')
    if not np.isclose(np.linalg.norm(vector), 1., atol=1e-12, rtol=0):
        raise ValueError('Axis must already be a unit vector.')
    return vector


def virtual_map(reference: np.ndarray, lengths: np.ndarray, rotation_axis: np.ndarray,
                translation_axis: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and exact Jacobian [coordinate, atom, Cartesian].

    Axes stay fixed to the reference. Periodic representatives and internal
    molecular distances are retained through continuous small displacements.
    """
    r = np.asarray(reference, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    z = np.asarray(coordinates, dtype=float)
    axis, direction = _unit(rotation_axis), _unit(translation_axis)
    if r.shape != (6, 3) or lengths.shape != (3,) or z.shape != (3,):
        raise ValueError('Expected six positions, three cell lengths and three coordinates.')
    if not np.isfinite(r).all() or not np.isfinite(z).all() or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Invalid position, displacement or cell.')
    h = r[1:3] - r[0]
    h -= np.rint(h / lengths) * lengths
    theta, water, neighbor = z
    rotated = h * np.cos(theta) + np.cross(axis, h) * np.sin(theta) + np.outer(h @ axis, axis) * (1 - np.cos(theta))
    moved = r.copy()
    moved[:3] += water * direction
    moved[1:3] += rotated - h
    moved[3:] += neighbor * direction
    jacobian = np.zeros((3, 6, 3))
    jacobian[0, 1:3] = np.cross(axis, rotated)
    jacobian[1, :3] = direction
    jacobian[2, 3:] = direction
    return moved, jacobian


def generalized_forces(forces: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    """Project total Cartesian forces onto exact coordinate-map tangents."""
    forces = np.asarray(forces, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    if forces.shape != (6, 3) or jacobian.shape != (3, 6, 3):
        raise ValueError('Wrong force or Jacobian shape.')
    if not np.isfinite(forces).all() or not np.isfinite(jacobian).all():
        raise ValueError('Nonfinite projection input.')
    return np.einsum('jai,ai->j', jacobian, forces)
