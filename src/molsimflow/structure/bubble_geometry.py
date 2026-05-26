"""Geometry helpers for bubble-containing molecular systems."""

from __future__ import annotations

import math
from collections.abc import Iterable


def sphere_volume(radius: float) -> float:
    """Return the volume of a sphere."""
    if radius <= 0:
        raise ValueError("Radius must be positive")
    return 4.0 * math.pi * radius**3 / 3.0


def cylinder_volume(radius: float, length: float) -> float:
    """Return the volume of a cylinder."""
    if radius <= 0:
        raise ValueError("Radius must be positive")
    if length <= 0:
        raise ValueError("Length must be positive")
    return math.pi * radius**2 * length


def two_sphere_intersection_volume(radius_a: float, radius_b: float, distance: float) -> float:
    """Return the overlap volume of two spheres."""
    if radius_a <= 0 or radius_b <= 0:
        raise ValueError("Radii must be positive")
    if distance < 0:
        raise ValueError("Distance cannot be negative")
    if distance >= radius_a + radius_b:
        return 0.0
    if distance <= abs(radius_a - radius_b):
        return sphere_volume(min(radius_a, radius_b))
    return (
        math.pi
        * (radius_a + radius_b - distance) ** 2
        * (distance**2 + 2.0 * distance * (radius_a + radius_b) - 3.0 * (radius_a - radius_b) ** 2)
        / (12.0 * distance)
    )


def equal_volume_radius(radii: Iterable[float]) -> float:
    """Return the radius giving the same mean sphere volume as the input radii."""
    values = [float(radius) for radius in radii]
    if not values:
        raise ValueError("At least one radius is required")
    if any(radius <= 0 for radius in values):
        raise ValueError("All radii must be positive")
    return (sum(radius**3 for radius in values) / len(values)) ** (1.0 / 3.0)


def molecule_count_from_density(
    volume_a3: float,
    density_kg_m3: float,
    molar_mass_g_mol: float,
    *,
    avogadro_number: float = 6.02214076e23,
) -> int:
    """Estimate molecule count from volume, density, and molar mass."""
    if volume_a3 <= 0:
        raise ValueError("Volume must be positive")
    if density_kg_m3 <= 0:
        raise ValueError("Density must be positive")
    if molar_mass_g_mol <= 0:
        raise ValueError("Molar mass must be positive")
    molecules_per_a3 = (density_kg_m3 * 1e-27) / (molar_mass_g_mol / avogadro_number)
    return max(1, int(volume_a3 * molecules_per_a3))
