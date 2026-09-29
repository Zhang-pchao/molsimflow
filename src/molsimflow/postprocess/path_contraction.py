"""Reference coordinate map and bias-force pullback for PIMD contraction.

Arrays are complete paths with beads on axis zero. Callers must supply a
consistent Cartesian periodic lift and identical atom ordering on every bead.
These helpers do not unwrap coordinates, evaluate a physical PES, apply the
engine's force normalization, or infer equilibrium from an adaptive bias.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def validate_contraction(value: float) -> float:
    """Return a fixed finite contraction in [0, 1]; reject boolean parameters."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("path contraction must be a finite scalar in [0, 1]")
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("path contraction must be a finite scalar in [0, 1]") from exc
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("path contraction must be a finite scalar in [0, 1]")
    return fraction


def _path_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim < 2 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must contain complete nonempty bead arrays")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def contract_coordinates(coordinates: np.ndarray, fraction: float) -> np.ndarray:
    """Map lifted real coordinates to virtual coordinates without mutation.

    Mean coordinates are unchanged; coordinate covariance scales by fraction^2.
    This is a bias-input map, not a contraction of the physical simulation.
    """
    fraction = validate_contraction(fraction)
    real = _path_array(coordinates, "coordinates")
    centroid = np.mean(real, axis=0, keepdims=True)
    return centroid + fraction * (real - centroid)


def pullback_bias_forces(virtual_forces: np.ndarray, fraction: float) -> np.ndarray:
    """Pull back only the virtual bias increment to the original coordinates.

    The input must already be the derivative of one scalar complete-path bias
    with respect to the virtual coordinates. No extra bead factor belongs here.
    This map also preserves the summed force for a general differentiable
    virtual-coordinate potential, though the intended CV graph is bead-mean.
    """
    fraction = validate_contraction(fraction)
    forces = _path_array(virtual_forces, "virtual forces")
    return fraction * forces + (1.0 - fraction) * np.mean(forces, axis=0, keepdims=True)


def validate_contraction_metadata(record: Mapping[str, object]) -> float:
    """Require the lift and real-observable identities for contracted records."""
    if not isinstance(record, Mapping) or set(record) != {
        "lambda", "coordinate_lift", "observable_coordinates"
    }:
        raise ValueError("path_contraction requires lambda, coordinate_lift and observable_coordinates")
    if record["coordinate_lift"] != "pimd_unwrapped":
        raise ValueError("path contraction requires the documented pimd_unwrapped lift")
    if record["observable_coordinates"] != "real_beads":
        raise ValueError("contracted virtual CVs cannot define the real bead observable")
    return validate_contraction(record["lambda"])
