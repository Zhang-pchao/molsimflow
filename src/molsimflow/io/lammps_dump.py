"""Small LAMMPS dump readers and periodic geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, TextIO, Tuple

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
    atom_fields: Tuple[str, ...]
    atom_rows: Tuple[Tuple[str, ...], ...]

    @property
    def atom_count(self) -> int:
        return len(self.atom_rows)


def iter_lammps_dump_records(dump_path: Path) -> Iterator[LammpsDumpFrame]:
    """Iterate complete dump rows while preserving every atom column."""

    with Path(dump_path).open(encoding="utf-8") as handle:
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


def write_lammps_dump_frame(
    handle: TextIO,
    frame: LammpsDumpFrame,
    atom_rows: Optional[Sequence[Sequence[object]]] = None,
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


def _choose_coord_field(fields: Sequence[str], dim: str) -> Tuple[int, bool]:
    for name in (dim, dim + "u", dim + "s"):
        if name in fields:
            return fields.index(name), name.endswith("s")
    raise ValueError(f"LAMMPS dump is missing {dim}/{dim}u/{dim}s coordinate column")


def iter_lammps_dump_frames(
    dump_path: Path,
    needed_atom_ids: Optional[Iterable[int]] = None,
    max_frames: Optional[int] = None,
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
        selected: Dict[int, np.ndarray] = {}
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
