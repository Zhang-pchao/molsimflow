"""Region primitives used by structure-preparation workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoxRegion:
    """Axis-aligned rectangular region in Angstrom units."""

    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    def volume(self) -> float:
        """Return the box volume."""
        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        dz = self.z_max - self.z_min
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError(f"Invalid box region with non-positive dimension: {self}")
        return dx * dy * dz

    def buffered_xy_volume(self, correction: float) -> float:
        """Return volume using a correction on x/y lengths for density estimates."""
        dx = self.x_max - self.x_min + correction
        dy = self.y_max - self.y_min + correction
        dz = self.z_max - self.z_min
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError(f"Invalid corrected box region with non-positive dimension: {self}")
        return dx * dy * dz

    def packmol_box(self) -> str:
        """Format a PACKMOL `inside box` region."""
        return (
            f"box {self.x_min:.3f} {self.y_min:.3f} {self.z_min:.3f} "
            f"{self.x_max:.3f} {self.y_max:.3f} {self.z_max:.3f}"
        )


@dataclass(frozen=True)
class SphereRegion:
    """Spherical PACKMOL region."""

    x: float
    y: float
    z: float
    radius: float

    def packmol_region(self) -> str:
        """Format a PACKMOL sphere region."""
        return f"sphere {self.x:.3f} {self.y:.3f} {self.z:.3f} {self.radius:.3f}"


@dataclass(frozen=True)
class CylinderRegion:
    """Cylinder PACKMOL region."""

    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    radius: float
    length: float

    def packmol_region(self) -> str:
        """Format a PACKMOL cylinder region."""
        return (
            f"cylinder {self.x:.3f} {self.y:.3f} {self.z:.3f} "
            f"{self.dx:.3f} {self.dy:.3f} {self.dz:.3f} "
            f"{self.radius:.3f} {self.length:.3f}"
        )
