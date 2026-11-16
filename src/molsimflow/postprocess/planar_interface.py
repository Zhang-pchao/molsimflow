"""Mechanical planar-interface observables with explicit units and geometry."""

from __future__ import annotations
import numpy as np


def mechanical_surface_tension(
    diagonal_pressure: np.ndarray,
    normal_length: np.ndarray,
    *,
    normal_axis: int,
    interfaces: int,
    pressure_unit: str,
    length_unit: str,
) -> np.ndarray:
    """Return L_normal * (P_normal - mean(P_tangent)) / interfaces in mN/m.

    Pressure has shape (..., 3), ordered x/y/z. normal_length broadcasts over
    the leading frame axes. The declared interface count belongs to the actual
    planar geometry. This mechanical observable accepts already defined total
    pressure, including the caller's kinetic/virial convention.
    """
    pressure = np.asarray(diagonal_pressure, dtype=float)
    length = np.asarray(normal_length, dtype=float)
    if pressure.ndim < 1 or pressure.shape[-1] != 3 or not np.isfinite(pressure).all():
        raise ValueError("Finite diagonal pressure with last axis x/y/z is required")
    if (
        not isinstance(normal_axis, (int, np.integer))
        or isinstance(normal_axis, bool)
        or normal_axis not in (0, 1, 2)
    ):
        raise ValueError("normal_axis must be 0, 1 or 2")
    if (
        isinstance(interfaces, bool)
        or not isinstance(interfaces, (int, np.integer))
        or interfaces < 1
    ):
        raise ValueError("Interface count must be a positive integer")
    factors_p = {"Pa": 1.0, "bar": 1e5, "atm": 101325.0}
    factors_l = {"m": 1.0, "nm": 1e-9, "angstrom": 1e-10}
    if pressure_unit not in factors_p or length_unit not in factors_l:
        raise ValueError("Unsupported pressure or length unit")
    if not np.isfinite(length).all() or np.any(length <= 0):
        raise ValueError("Finite positive normal lengths are required")
    try:
        length = np.broadcast_to(length, pressure.shape[:-1])
    except ValueError as error:
        raise ValueError("Normal lengths do not match pressure frame axes") from error
    tangential = [axis for axis in range(3) if axis != normal_axis]
    excess = pressure[..., normal_axis] - pressure[..., tangential].mean(axis=-1)
    return length * excess * (factors_p[pressure_unit] * factors_l[length_unit] * 1000 / interfaces)
