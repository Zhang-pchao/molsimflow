"""Material ownership and seam diagnostics for a two-domain periodic surface."""
from __future__ import annotations
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import numpy as np
from molsimflow.io.lammps_dump import minimum_image_vectors

ZONES = ("left_interior", "internal_seam", "right_interior", "periodic_seam")


def zone_from_x(x: float, length: float, half_width: float = 6.0) -> str:
    if length <= 0 or half_width <= 0 or 4*half_width >= length:
        raise ValueError("Invalid stripe length or overlapping seam inspection bands")
    x = float(x) % length
    if min(x, length-x) <= half_width:
        return "periodic_seam"
    if abs(x-length/2) <= half_width:
        return "internal_seam"
    return "left_interior" if x < length/2 else "right_interior"


def periodic_translation(reference: np.ndarray, target: np.ndarray,
                         lengths: Sequence[float]) -> np.ndarray:
    if reference.shape != target.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Coordinate identity/shape mismatch")
    return np.median(minimum_image_vectors(target-reference, np.asarray(lengths)), axis=0)


def changed_edges(reference: Iterable[tuple[int, int]],
                  target: Iterable[tuple[int, int]]) -> dict:
    before, after = set(reference), set(target)
    return {"lost": sorted(before-after), "added": sorted(after-before),
            "retained": len(before & after), "reference_count": len(before),
            "target_count": len(after)}


def regional_edge_changes(changes: Mapping, si_zones: Mapping[int, str]) -> dict:
    result = {zone: {"lost": 0, "added": 0} for zone in ZONES}
    for kind in ("lost", "added"):
        for si_id, _ in changes[kind]:
            result[si_zones[si_id]][kind] += 1
    return result


def region_species_counts(species: Sequence[str], zones: Sequence[str]) -> dict:
    if len(species) != len(zones):
        raise ValueError("Species/zone lengths differ")
    result = {zone: Counter() for zone in ZONES}
    for name, zone in zip(species, zones):
        result[zone][name] += 1
    return {zone: dict(counts) for zone, counts in result.items()}
