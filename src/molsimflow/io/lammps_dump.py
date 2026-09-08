"""Small LAMMPS dump readers and periodic geometry helpers."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np


@dataclass(frozen=True)
class LammpsFrame:
    """Selected atom positions from one LAMMPS dump frame."""

    frame_index: int
    timestep: int
    bounds: np.ndarray
    selected_positions: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class LammpsDumpFrame:
    """One complete orthorhombic LAMMPS custom-dump frame."""

    frame_index: int
    timestep: int
    bounds: np.ndarray
    box_header: str
    atom_fields: tuple[str, ...]
    atom_rows: tuple[tuple[str, ...], ...]

    @property
    def atom_count(self) -> int:
        return len(self.atom_rows)


@contextmanager
def open_lammps_dump_text(dump_path: Path) -> Iterator[TextIO]:
    """Open a plain or zstd-compressed LAMMPS dump as a streaming text handle."""

    path = Path(dump_path)
    if path.suffix != ".zst":
        with path.open(encoding="utf-8") as handle:
            yield handle
        return
    try:
        process = subprocess.Popen(
            ["zstd", "-q", "-dc", "--", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Reading .zst dumps requires the zstd executable") from exc
    assert process.stdout is not None
    try:
        yield process.stdout
    except BaseException:
        # A consumer may stop early, in which case zstd can exit on SIGPIPE.
        process.stdout.close()
        process.wait()
        raise
    else:
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else ""
        if process.wait():
            raise ValueError(f"zstd decode failed for {path}: {stderr.strip()}")


def iter_lammps_dump_records(dump_path: Path) -> Iterator[LammpsDumpFrame]:
    """Iterate complete dump rows while preserving every atom column."""

    with open_lammps_dump_text(dump_path) as handle:
        frame_index = 0
        while True:
            line = handle.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError("Unexpected dump format: expected ITEM: TIMESTEP")
            timestep = int(handle.readline().strip())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Unexpected dump format: missing ITEM: NUMBER OF ATOMS")
            atom_count = int(handle.readline().strip())
            if atom_count <= 0:
                raise ValueError(f"Invalid atom count at timestep {timestep}: {atom_count}")
            box_header = handle.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected dump format: missing ITEM: BOX BOUNDS")
            bounds = np.zeros((3, 2), dtype=float)
            for dim in range(3):
                parts = handle.readline().split()
                if len(parts) != 2:
                    raise ValueError("Only orthorhombic LAMMPS dump boxes are supported")
                bounds[dim] = [float(parts[0]), float(parts[1])]
            atom_header = handle.readline().strip()
            if not atom_header.startswith("ITEM: ATOMS"):
                raise ValueError("Unexpected dump format: missing ITEM: ATOMS")
            atom_fields = tuple(atom_header.split()[2:])
            if not atom_fields:
                raise ValueError(f"LAMMPS dump has no atom columns at timestep {timestep}")
            rows = []
            for _ in range(atom_count):
                parts = tuple(handle.readline().split())
                if len(parts) != len(atom_fields):
                    raise ValueError(
                        f"Atom column count mismatch at timestep {timestep}: "
                        f"expected {len(atom_fields)}, got {len(parts)}"
                    )
                rows.append(parts)
            yield LammpsDumpFrame(
                frame_index=frame_index,
                timestep=timestep,
                bounds=bounds,
                box_header=box_header,
                atom_fields=atom_fields,
                atom_rows=tuple(rows),
            )
            frame_index += 1


def _validate_dump_identity(
    dump_path: Path,
    expected_fields: Sequence[str],
    expected_identity: tuple[tuple[int, int], ...] | None = None,
) -> tuple[dict[str, object], tuple[int, ...], tuple[tuple[int, int], ...]]:
    steps = []
    identity_reference = expected_identity
    atom_count = 0
    for frame in iter_lammps_dump_records(dump_path):
        if frame.atom_fields != tuple(expected_fields):
            raise ValueError(
                f"{dump_path}: expected fields {tuple(expected_fields)}, got {frame.atom_fields}"
            )
        if steps and frame.timestep <= steps[-1]:
            raise ValueError(f"{dump_path}: non-increasing timestep {frame.timestep}")
        identity = tuple((int(row[0]), int(row[1])) for row in frame.atom_rows)
        if len({atom_id for atom_id, _ in identity}) != frame.atom_count:
            raise ValueError(f"{dump_path}: duplicate atom id at timestep {frame.timestep}")
        if identity_reference is None:
            identity_reference = identity
            atom_count = frame.atom_count
        elif identity != identity_reference:
            raise ValueError(f"{dump_path}: atom id/type mismatch at timestep {frame.timestep}")
        steps.append(frame.timestep)
    if not steps or identity_reference is None:
        raise ValueError(f"{dump_path}: no complete frames")
    return (
        {
            "path": str(dump_path),
            "fields": list(expected_fields),
            "frames": len(steps),
            "atom_count": atom_count or len(identity_reference),
            "first_step": steps[0],
            "last_step": steps[-1],
        },
        tuple(steps),
        identity_reference,
    )


def validate_lammps_dump_bundle(
    coordinate_path: Path,
    velocity_path: Path,
    force_path: Path,
    start_step: int,
    expected_final_step: int,
    coordinate_stride: int,
    vector_stride: int,
) -> dict[str, object]:
    """Validate aligned coordinate, velocity, and force custom dumps."""

    if coordinate_stride <= 0 or vector_stride <= 0 or expected_final_step <= start_step:
        raise ValueError("Invalid timestep range or dump stride")
    expected_coordinate_steps = tuple(
        range(start_step + coordinate_stride, expected_final_step + 1, coordinate_stride)
    )
    expected_vector_steps = tuple(
        range(start_step + vector_stride, expected_final_step + 1, vector_stride)
    )
    if not expected_coordinate_steps or expected_coordinate_steps[-1] != expected_final_step:
        raise ValueError("Coordinate stride does not land on the expected final step")
    if not expected_vector_steps or expected_vector_steps[-1] != expected_final_step:
        raise ValueError("Vector stride does not land on the expected final step")
    coordinate, coordinate_steps, identity = _validate_dump_identity(
        Path(coordinate_path), ("id", "type", "x", "y", "z")
    )
    velocity, velocity_steps, _ = _validate_dump_identity(
        Path(velocity_path), ("id", "type", "vx", "vy", "vz"), identity
    )
    force, force_steps, _ = _validate_dump_identity(
        Path(force_path), ("id", "type", "fx", "fy", "fz"), identity
    )
    if velocity_steps != force_steps:
        raise ValueError("Velocity and force timesteps differ")
    if not set(velocity_steps).issubset(coordinate_steps):
        raise ValueError("Velocity/force timesteps are not a subset of coordinate timesteps")
    if coordinate_steps != expected_coordinate_steps:
        raise ValueError("Coordinate timesteps do not match the requested output cadence")
    if velocity_steps != expected_vector_steps:
        raise ValueError("Velocity/force timesteps do not match the requested output cadence")
    return {
        "status": "PASS",
        "start_step": int(start_step),
        "expected_final_step": int(expected_final_step),
        "coordinate_stride": int(coordinate_stride),
        "vector_stride": int(vector_stride),
        "coordinate": coordinate,
        "velocity": velocity,
        "force": force,
    }


def write_lammps_dump_frame(
    handle: TextIO,
    frame: LammpsDumpFrame,
    atom_rows: Sequence[Sequence[object]] | None = None,
) -> None:
    """Write one parsed frame, optionally replacing its atom rows."""

    rows = frame.atom_rows if atom_rows is None else atom_rows
    if len(rows) != frame.atom_count:
        raise ValueError("Replacement atom rows must preserve the atom count")
    handle.write(f"ITEM: TIMESTEP\n{frame.timestep}\n")
    handle.write(f"ITEM: NUMBER OF ATOMS\n{frame.atom_count}\n")
    handle.write(frame.box_header + "\n")
    handle.writelines(f"{low:.16g} {high:.16g}\n" for low, high in frame.bounds)
    handle.write("ITEM: ATOMS " + " ".join(frame.atom_fields) + "\n")
    for row in rows:
        if len(row) != len(frame.atom_fields):
            raise ValueError("Replacement atom row has the wrong number of columns")
        handle.write(" ".join(str(value) for value in row) + "\n")


def _choose_coord_field(fields: Sequence[str], dim: str) -> tuple[int, bool]:
    for name in (dim, dim + "u", dim + "s"):
        if name in fields:
            return fields.index(name), name.endswith("s")
    raise ValueError(f"LAMMPS dump is missing {dim}/{dim}u/{dim}s coordinate column")


def iter_lammps_dump_frames(
    dump_path: Path,
    needed_atom_ids: Iterable[int] | None = None,
    max_frames: int | None = None,
) -> Iterator[LammpsFrame]:
    """Iterate LAMMPS dump frames while retaining selected atom positions.

    Coordinates may be stored as `x/y/z`, `xu/yu/zu`, or scaled `xs/ys/zs`.
    Scaled coordinates are converted to box coordinates.  The reader assumes an
    orthorhombic dump box, which matches the migrated bridge workflows.
    """

    needed = set(needed_atom_ids) if needed_atom_ids is not None else None
    for frame in iter_lammps_dump_records(dump_path):
        fields = frame.atom_fields
        if "id" not in fields:
            raise ValueError("LAMMPS dump ATOMS line must contain id")
        id_index = fields.index("id")
        x_index, x_scaled = _choose_coord_field(fields, "x")
        y_index, y_scaled = _choose_coord_field(fields, "y")
        z_index, z_scaled = _choose_coord_field(fields, "z")
        lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
        selected: dict[int, np.ndarray] = {}
        for parts in frame.atom_rows:
            atom_id = int(parts[id_index])
            if needed is not None and atom_id not in needed:
                continue
            coords = np.asarray(
                [float(parts[x_index]), float(parts[y_index]), float(parts[z_index])],
                dtype=float,
            )
            for dim, scaled in enumerate((x_scaled, y_scaled, z_scaled)):
                if scaled:
                    coords[dim] = frame.bounds[dim, 0] + coords[dim] * lengths[dim]
            selected[atom_id] = coords
        yield LammpsFrame(
            frame_index=frame.frame_index,
            timestep=frame.timestep,
            bounds=frame.bounds,
            selected_positions=selected,
        )
        if max_frames is not None and frame.frame_index + 1 >= int(max_frames):
            break


def box_lengths(bounds: np.ndarray) -> np.ndarray:
    """Return orthorhombic box lengths from a `(3, 2)` bounds array."""

    return np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]


def minimum_image_vectors(vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Apply an orthorhombic minimum-image transform to displacement vectors."""

    values = np.asarray(vectors, dtype=float)
    return values - np.asarray(lengths, dtype=float) * np.round(values / np.asarray(lengths, dtype=float))


def wrap_point_to_box(point: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Wrap a point into an orthorhombic simulation box."""

    lengths = box_lengths(bounds)
    wrapped = np.empty(3, dtype=float)
    for dim in range(3):
        lo = float(bounds[dim, 0])
        wrapped[dim] = ((float(point[dim]) - lo) % float(lengths[dim])) + lo
    return wrapped


def periodic_center(coords: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Compute a periodic center robustly when coordinates straddle boundaries."""

    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        raise ValueError("Cannot compute periodic center for empty coordinate array")
    lengths = box_lengths(bounds)
    center = np.empty(3, dtype=float)
    for dim in range(3):
        lo = float(bounds[dim, 0])
        scaled = (coords[:, dim] - lo) / float(lengths[dim])
        angles = 2.0 * np.pi * scaled
        complex_mean = np.exp(1j * angles).mean()
        if np.isclose(abs(complex_mean), 0.0):
            center[dim] = float(np.mean(coords[:, dim]))
            continue
        angle = np.angle(complex_mean)
        if angle < 0:
            angle += 2.0 * np.pi
        center[dim] = lo + (angle / (2.0 * np.pi)) * float(lengths[dim])
    return center


def midpoint_minimum_image(center_a: np.ndarray, center_b: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Return the wrapped midpoint between two centers along the minimum image."""

    lengths = box_lengths(bounds)
    delta = minimum_image_vectors(np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float), lengths)
    return wrap_point_to_box(np.asarray(center_a, dtype=float) + 0.5 * delta, bounds)


def cylinder_membership(
    coords: np.ndarray,
    center: np.ndarray,
    bounds: np.ndarray,
    axis_index: int,
    radius_A: float,
    lower_A: float,
    upper_A: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cylinder mask, local axial coordinate, and radial distance."""

    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    lengths = box_lengths(bounds)
    deltas = minimum_image_vectors(coords - np.asarray(center, dtype=float), lengths)
    axial = deltas[:, axis_index]
    perp = [index for index in range(3) if index != axis_index]
    radial = np.sqrt(np.sum(deltas[:, perp] ** 2, axis=1))
    mask = (axial >= float(lower_A)) & (axial <= float(upper_A)) & (radial <= float(radius_A))
    return mask, axial, radial
