"""Shared oxygen-trajectory primitives for constant-force morphology analyses."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records


@dataclass(frozen=True)
class OxygenFrame:
    """One identity-sorted oxygen frame with optional velocities."""

    timestep: int
    bounds: np.ndarray
    atom_ids: np.ndarray
    coordinates: np.ndarray
    velocities: np.ndarray | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def write_tsv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_output_hashes(output: Path) -> None:
    paths = [
        path
        for path in sorted(Path(output).iterdir())
        if path.is_file() and path.name != "OUTPUT-SHA256SUMS"
    ]
    with (Path(output) / "OUTPUT-SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256(path)}  ./{path.name}\n")


def _columns(frame) -> tuple[int, tuple[int, int, int], tuple[int, int, int] | None]:
    fields = frame.atom_fields
    required = ("id", "x", "y", "z")
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"Trajectory is missing {missing} at step {frame.timestep}")
    velocity = (
        tuple(fields.index(name) for name in ("vx", "vy", "vz"))
        if all(name in fields for name in ("vx", "vy", "vz"))
        else None
    )
    return fields.index("id"), tuple(fields.index(name) for name in ("x", "y", "z")), velocity


def iter_oxygen_frames(paths: Sequence[Path]) -> Iterator[OxygenFrame]:
    """Stream restart segments, remove a shared endpoint, and verify atom identity."""

    if not paths:
        raise ValueError("At least one oxygen trajectory is required")
    reference_ids: np.ndarray | None = None
    previous_step: int | None = None
    previous_bounds: np.ndarray | None = None
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        segment_frames = 0
        for frame in iter_lammps_dump_records(path):
            segment_frames += 1
            if previous_step is not None and frame.timestep == previous_step:
                continue
            if previous_step is not None and frame.timestep < previous_step:
                raise ValueError(f"Non-increasing timestep {frame.timestep} in {path}")
            id_column, xyz_columns, velocity_columns = _columns(frame)
            ids = np.asarray([int(row[id_column]) for row in frame.atom_rows], dtype=np.int64)
            order = np.argsort(ids)
            ids = ids[order]
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Duplicate oxygen IDs at step {frame.timestep}")
            if reference_ids is None:
                reference_ids = ids.copy()
            elif not np.array_equal(ids, reference_ids):
                raise ValueError(f"Oxygen identity changed at step {frame.timestep}")
            coordinates = np.asarray(
                [[float(row[index]) for index in xyz_columns] for row in frame.atom_rows],
                dtype=float,
            )[order]
            velocities = None
            if velocity_columns is not None:
                velocities = np.asarray(
                    [[float(row[index]) for index in velocity_columns] for row in frame.atom_rows],
                    dtype=float,
                )[order]
            if previous_bounds is not None and not np.allclose(frame.bounds, previous_bounds, atol=1e-8):
                raise ValueError(f"Box bounds changed at step {frame.timestep}")
            previous_step = frame.timestep
            previous_bounds = frame.bounds.copy()
            yield OxygenFrame(
                timestep=frame.timestep,
                bounds=frame.bounds,
                atom_ids=ids,
                coordinates=coordinates,
                velocities=velocities,
            )
        if segment_frames == 0:
            raise ValueError(f"No complete frames in {path}")


def xy_minimum_image(vectors: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    result = np.asarray(vectors, dtype=float).copy()
    lengths = np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]
    result[..., :2] -= lengths[:2] * np.round(result[..., :2] / lengths[:2])
    return result


def periodic_xy_center(coordinates: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Return a circular X/Y center and arithmetic nonperiodic Z center."""

    points = np.asarray(coordinates, dtype=float)
    if len(points) == 0:
        raise ValueError("Cannot center an empty component")
    lengths = bounds[:, 1] - bounds[:, 0]
    center = np.empty(3, dtype=float)
    for dim in (0, 1):
        lo = bounds[dim, 0]
        angles = 2.0 * np.pi * (points[:, dim] - lo) / lengths[dim]
        value = np.mean(np.exp(1j * angles))
        if abs(value) < 1e-10:
            reference = points[0, dim]
            delta = points[:, dim] - reference
            delta -= lengths[dim] * np.round(delta / lengths[dim])
            center[dim] = ((reference + float(np.mean(delta)) - lo) % lengths[dim]) + lo
        else:
            angle = float(np.angle(value)) % (2.0 * np.pi)
            center[dim] = lo + lengths[dim] * angle / (2.0 * np.pi)
    center[2] = float(np.mean(points[:, 2]))
    return center


def unwrap_xy_center(
    center: np.ndarray,
    previous_center: np.ndarray,
    previous_unwrapped: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    delta = xy_minimum_image(np.asarray(center) - np.asarray(previous_center), bounds)
    result = np.asarray(previous_unwrapped, dtype=float) + delta
    result[2] = center[2]
    return result


def connected_components(
    coordinates: np.ndarray,
    bounds: np.ndarray,
    cutoff_A: float,
) -> list[np.ndarray]:
    """Return O--O components with periodic X/Y and nonperiodic Z."""

    from scipy.spatial import cKDTree

    points = np.asarray(coordinates, dtype=float)
    if cutoff_A <= 0.0:
        raise ValueError("cutoff_A must be positive")
    lengths = bounds[:, 1] - bounds[:, 0]
    shifted = np.empty_like(points)
    shifted[:, 0] = (points[:, 0] - bounds[0, 0]) % lengths[0]
    shifted[:, 1] = (points[:, 1] - bounds[1, 0]) % lengths[1]
    pseudo_z = max(1.0e5, 10.0 * lengths[2])
    shifted[:, 2] = points[:, 2] - bounds[2, 0] + 0.25 * pseudo_z
    parent = np.arange(len(points), dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    tree = cKDTree(shifted, boxsize=np.asarray([lengths[0], lengths[1], pseudo_z]))
    for left, right in tree.query_pairs(float(cutoff_A)):
        root_left, root_right = find(int(left)), find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    components = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    return sorted(components, key=lambda values: (-len(values), int(values[0])))
