"""Relocate selected oxygen species with their PBC-assigned hydrogens."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.postprocess.species_assignment import (
    assign_hydrogen_to_nearest_oxygen,
)


@dataclass(frozen=True)
class MolecularRelocationResult:
    """Coordinates, identity mapping, and geometric gates for one relocation."""

    coordinates: np.ndarray
    image_flags: np.ndarray | None
    moved_atom_ids: tuple[int, ...]
    moved_oxygen_ids: tuple[int, ...]
    moved_hydrogen_ids: tuple[int, ...]
    parent_oxygen_by_atom_id: Mapping[int, int]
    oh_distance_by_hydrogen_id_A: Mapping[int, float]
    hydrogen_count_by_oxygen_id: Mapping[int, int]
    axis: int
    source_reference_A: float
    translation_A: float
    stationary_max_A: float
    moved_min_A: float
    moved_max_A: float
    stationary_buffer_A: float
    high_boundary_clearance_A: float
    maximum_oh_vector_change_A: float


@dataclass(frozen=True)
class _AtomicData:
    lines: tuple[str, ...]
    atom_line_by_id: Mapping[int, int]
    atom_fields_by_id: Mapping[int, tuple[str, ...]]
    atom_comments_by_id: Mapping[int, str]
    atom_ids: np.ndarray
    atom_types: np.ndarray
    coordinates: np.ndarray
    image_flags: np.ndarray | None
    bounds: np.ndarray


def _axis_index(axis: int | str) -> int:
    if isinstance(axis, str):
        try:
            return {"x": 0, "y": 1, "z": 2}[axis.lower()]
        except KeyError as exc:
            raise ValueError("axis must be x, y, z, 0, 1, or 2") from exc
    value = int(axis)
    if value not in (0, 1, 2):
        raise ValueError("axis must be x, y, z, 0, 1, or 2")
    return value


def _minimum_image_component(delta: np.ndarray, length: float) -> np.ndarray:
    return delta - length * np.rint(delta / length)


def relocate_selected_oxygen_species(
    atom_ids: Sequence[int],
    atom_types: Sequence[int],
    coordinates: np.ndarray,
    bounds: np.ndarray,
    selected_oxygen_ids: Sequence[int],
    *,
    oxygen_type: int = 2,
    hydrogen_type: int = 1,
    oh_cutoff_A: float = 1.3,
    allowed_hydrogen_counts: Sequence[int] = (1, 2, 3),
    axis: int | str = "z",
    source_reference_A: float | None = None,
    stationary_buffer_A: float = 4.0,
    high_boundary_buffer_A: float = 20.0,
    periodic: Sequence[bool] = (True, True, True),
    assignment_chunk_size: int = 256,
    image_flags: np.ndarray | None = None,
    clear_axis_image_flags: bool = True,
) -> MolecularRelocationResult:
    """Move selected O atoms and every H assigned to them as intact species.

    Hydrogens are assigned to the nearest oxygen under the configured periodic
    axes and accepted only within ``oh_cutoff_A``.  The selected species are
    first unwrapped around ``source_reference_A`` along ``axis``, preserving
    each minimum-image O-H vector, and are then translated above every
    stationary atom with the requested one-dimensional buffer.
    """

    ids = np.asarray(atom_ids, dtype=np.int64)
    types = np.asarray(atom_types, dtype=np.int64)
    xyz = np.asarray(coordinates, dtype=float)
    box_bounds = np.asarray(bounds, dtype=float)
    periodic_mask = np.asarray(periodic, dtype=bool)
    selected = tuple(int(value) for value in selected_oxygen_ids)
    allowed_counts = {int(value) for value in allowed_hydrogen_counts}
    axis_index = _axis_index(axis)

    if ids.ndim != 1 or types.shape != ids.shape or xyz.shape != (ids.size, 3):
        raise ValueError("atom_ids, atom_types, and coordinates have incompatible shapes")
    if box_bounds.shape != (3, 2):
        raise ValueError("bounds must have shape (3, 2)")
    lengths = box_bounds[:, 1] - box_bounds[:, 0]
    if np.any(~np.isfinite(box_bounds)) or np.any(lengths <= 0.0):
        raise ValueError("bounds must be finite with positive lengths")
    if np.any(~np.isfinite(xyz)):
        raise ValueError("coordinates must be finite")
    if ids.size == 0 or len(set(ids.tolist())) != ids.size or np.any(ids <= 0):
        raise ValueError("atom IDs must be unique positive integers")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected_oxygen_ids must be non-empty and unique")
    if not allowed_counts or any(value < 0 for value in allowed_counts):
        raise ValueError("allowed_hydrogen_counts must contain non-negative integers")
    if oh_cutoff_A <= 0.0:
        raise ValueError("oh_cutoff_A must be positive")
    if assignment_chunk_size < 1:
        raise ValueError("assignment_chunk_size must be positive")
    if stationary_buffer_A <= 0.0 or high_boundary_buffer_A <= 0.0:
        raise ValueError("stationary and high-boundary buffers must be positive")
    if periodic_mask.shape != (3,):
        raise ValueError("periodic must have shape (3,)")
    if not periodic_mask[axis_index]:
        raise ValueError("the relocation axis must be periodic in the source structure")

    copied_images: np.ndarray | None = None
    if image_flags is not None:
        copied_images = np.asarray(image_flags, dtype=np.int64).copy()
        if copied_images.shape != xyz.shape:
            raise ValueError("image_flags must have shape (N, 3)")

    id_to_index = {int(atom_id): index for index, atom_id in enumerate(ids)}
    missing = [atom_id for atom_id in selected if atom_id not in id_to_index]
    if missing:
        raise ValueError(f"selected oxygen IDs are absent: {missing}")
    wrong_type = [atom_id for atom_id in selected if types[id_to_index[atom_id]] != oxygen_type]
    if wrong_type:
        raise ValueError(f"selected atom IDs do not have oxygen_type={oxygen_type}: {wrong_type}")

    oxygen_indices = np.flatnonzero(types == oxygen_type)
    hydrogen_indices = np.flatnonzero(types == hydrogen_type)
    if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
        raise ValueError("the structure must contain both configured oxygen and hydrogen types")
    assignment = assign_hydrogen_to_nearest_oxygen(
        xyz[oxygen_indices],
        xyz[hydrogen_indices],
        box_bounds,
        oh_cutoff=oh_cutoff_A,
        chunk_size=assignment_chunk_size,
        periodic=periodic_mask,
    )
    oxygen_local_by_id = {
        int(ids[global_index]): local_index
        for local_index, global_index in enumerate(oxygen_indices)
    }
    selected_local = {oxygen_local_by_id[atom_id] for atom_id in selected}

    parent_by_hydrogen: dict[int, int] = {}
    distance_by_hydrogen: dict[int, float] = {}
    for hydrogen_local, oxygen_local in enumerate(assignment.hydrogen_to_oxygen_index):
        oxygen_local = int(oxygen_local)
        if oxygen_local not in selected_local:
            continue
        hydrogen_id = int(ids[hydrogen_indices[hydrogen_local]])
        oxygen_id = int(ids[oxygen_indices[oxygen_local]])
        parent_by_hydrogen[hydrogen_id] = oxygen_id
        distance_by_hydrogen[hydrogen_id] = float(
            assignment.hydrogen_distance[hydrogen_local]
        )

    hydrogen_count_by_oxygen = {oxygen_id: 0 for oxygen_id in selected}
    for oxygen_id in parent_by_hydrogen.values():
        hydrogen_count_by_oxygen[oxygen_id] += 1
    invalid_counts = {
        oxygen_id: count
        for oxygen_id, count in hydrogen_count_by_oxygen.items()
        if count not in allowed_counts
    }
    if invalid_counts:
        raise ValueError(
            "selected oxygen hydrogen counts are outside the allowed set "
            f"{sorted(allowed_counts)}: {invalid_counts}"
        )

    moved_hydrogen_ids = tuple(sorted(parent_by_hydrogen))
    moved_ids = tuple(sorted((*selected, *moved_hydrogen_ids)))
    moved_indices = np.asarray([id_to_index[atom_id] for atom_id in moved_ids], dtype=int)
    stationary_mask = np.ones(ids.size, dtype=bool)
    stationary_mask[moved_indices] = False
    if not np.any(stationary_mask):
        raise ValueError("relocation requires at least one stationary atom")

    source_reference = (
        float(box_bounds[axis_index, 0])
        if source_reference_A is None
        else float(source_reference_A)
    )
    if not math.isfinite(source_reference):
        raise ValueError("source_reference_A must be finite")
    axis_length = float(lengths[axis_index])
    unwrapped_axis: dict[int, float] = {}
    for oxygen_id in selected:
        value = float(xyz[id_to_index[oxygen_id], axis_index])
        unwrapped_axis[oxygen_id] = source_reference + float(
            _minimum_image_component(np.asarray(value - source_reference), axis_length)
        )
    for hydrogen_id, oxygen_id in parent_by_hydrogen.items():
        hydrogen_value = float(xyz[id_to_index[hydrogen_id], axis_index])
        oxygen_value = float(xyz[id_to_index[oxygen_id], axis_index])
        relative = float(
            _minimum_image_component(np.asarray(hydrogen_value - oxygen_value), axis_length)
        )
        unwrapped_axis[hydrogen_id] = unwrapped_axis[oxygen_id] + relative

    relocated = xyz.copy()
    stationary_max = float(np.max(relocated[stationary_mask, axis_index]))
    source_moved_min = min(unwrapped_axis.values())
    translation = stationary_max + float(stationary_buffer_A) - source_moved_min
    for atom_id, value in unwrapped_axis.items():
        relocated[id_to_index[atom_id], axis_index] = value + translation

    maximum_oh_vector_change = 0.0
    for hydrogen_id, oxygen_id in parent_by_hydrogen.items():
        source_vector = (
            xyz[id_to_index[hydrogen_id]] - xyz[id_to_index[oxygen_id]]
        ).copy()
        observed_vector = (
            relocated[id_to_index[hydrogen_id]] - relocated[id_to_index[oxygen_id]]
        ).copy()
        for dimension in range(3):
            if periodic_mask[dimension]:
                source_vector[dimension] = _minimum_image_component(
                    source_vector[dimension], float(lengths[dimension])
                )
                if dimension != axis_index:
                    observed_vector[dimension] = _minimum_image_component(
                        observed_vector[dimension], float(lengths[dimension])
                    )
        change = float(np.linalg.norm(observed_vector - source_vector))
        maximum_oh_vector_change = max(maximum_oh_vector_change, change)
    if maximum_oh_vector_change > 1.0e-10:
        raise RuntimeError(
            "internal error: relocation changed an assigned O-H vector by "
            f"{maximum_oh_vector_change:.8g} A"
        )

    moved_min = float(np.min(relocated[moved_indices, axis_index]))
    moved_max = float(np.max(relocated[moved_indices, axis_index]))
    observed_stationary_buffer = moved_min - stationary_max
    high_clearance = float(box_bounds[axis_index, 1] - np.max(relocated[:, axis_index]))
    tolerance = 1.0e-9
    if observed_stationary_buffer < stationary_buffer_A - tolerance:
        raise RuntimeError("internal error: stationary buffer was not preserved")
    if high_clearance < high_boundary_buffer_A - tolerance:
        raise ValueError(
            f"high-boundary clearance {high_clearance:.8g} A is below the required "
            f"{high_boundary_buffer_A:.8g} A"
        )

    if copied_images is not None and clear_axis_image_flags:
        copied_images[:, axis_index] = 0

    parent_by_atom = {oxygen_id: oxygen_id for oxygen_id in selected}
    parent_by_atom.update(parent_by_hydrogen)
    return MolecularRelocationResult(
        coordinates=relocated,
        image_flags=copied_images,
        moved_atom_ids=moved_ids,
        moved_oxygen_ids=tuple(sorted(selected)),
        moved_hydrogen_ids=moved_hydrogen_ids,
        parent_oxygen_by_atom_id=parent_by_atom,
        oh_distance_by_hydrogen_id_A=distance_by_hydrogen,
        hydrogen_count_by_oxygen_id=hydrogen_count_by_oxygen,
        axis=axis_index,
        source_reference_A=source_reference,
        translation_A=translation,
        stationary_max_A=stationary_max,
        moved_min_A=moved_min,
        moved_max_A=moved_max,
        stationary_buffer_A=observed_stationary_buffer,
        high_boundary_clearance_A=high_clearance,
        maximum_oh_vector_change_A=maximum_oh_vector_change,
    )


def _parse_lammps_atomic_data(path: Path) -> _AtomicData:
    lines = tuple(path.read_text(encoding="utf-8").splitlines())
    atom_count: int | None = None
    bounds: list[tuple[float, float]] = []
    bound_labels = ("xlo xhi", "ylo yhi", "zlo zhi")
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[1] == "atoms" and fields[0].isdigit():
            atom_count = int(fields[0])
        for label in bound_labels:
            if line.strip().endswith(label):
                bounds.append((float(fields[0]), float(fields[1])))
        if line.strip().endswith(("xy xz yz", "xz yz xy", "yz xy xz")):
            raise ValueError("triclinic LAMMPS data files are not supported")
    if atom_count is None or len(bounds) != 3:
        raise ValueError("LAMMPS data file is missing atom count or orthorhombic bounds")

    try:
        atoms_header = next(
            index for index, line in enumerate(lines) if line.strip().startswith("Atoms")
        )
    except StopIteration as exc:
        raise ValueError("LAMMPS data file is missing an Atoms section") from exc
    header = lines[atoms_header].lower()
    if "#" in header and "atomic" not in header:
        raise ValueError("only LAMMPS atom_style atomic data files are supported")

    line_by_id: dict[int, int] = {}
    fields_by_id: dict[int, tuple[str, ...]] = {}
    comments_by_id: dict[int, str] = {}
    rows: list[tuple[int, int, float, float, float, tuple[int, int, int] | None]] = []
    for line_index in range(atoms_header + 1, len(lines)):
        body, marker, comment = lines[line_index].partition("#")
        fields = body.split()
        if not fields or not fields[0].lstrip("+-").isdigit():
            if rows and len(rows) < atom_count and body.strip():
                raise ValueError("Atoms section ended before all atom rows were read")
            continue
        if len(rows) >= atom_count:
            break
        if len(fields) not in (5, 8):
            raise ValueError("atom_style atomic rows must contain 5 or 8 fields")
        atom_id, atom_type = int(fields[0]), int(fields[1])
        images = None if len(fields) == 5 else tuple(int(value) for value in fields[5:8])
        rows.append(
            (atom_id, atom_type, float(fields[2]), float(fields[3]), float(fields[4]), images)
        )
        line_by_id[atom_id] = line_index
        fields_by_id[atom_id] = tuple(fields)
        comments_by_id[atom_id] = f"#{comment}" if marker else ""
    if len(rows) != atom_count or len(line_by_id) != atom_count:
        raise ValueError(f"expected {atom_count} unique atom rows, found {len(line_by_id)}")
    rows.sort(key=lambda row: row[0])
    has_images = [row[5] is not None for row in rows]
    if any(has_images) and not all(has_images):
        raise ValueError("image flags must be present for either every atom or no atoms")
    return _AtomicData(
        lines=lines,
        atom_line_by_id=line_by_id,
        atom_fields_by_id=fields_by_id,
        atom_comments_by_id=comments_by_id,
        atom_ids=np.asarray([row[0] for row in rows], dtype=np.int64),
        atom_types=np.asarray([row[1] for row in rows], dtype=np.int64),
        coordinates=np.asarray([row[2:5] for row in rows], dtype=float),
        image_flags=(
            np.asarray([row[5] for row in rows], dtype=np.int64) if all(has_images) else None
        ),
        bounds=np.asarray(bounds, dtype=float),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _read_atom_ids(path: Path) -> tuple[int, ...]:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values.extend(int(token) for token in line.partition("#")[0].split())
    return tuple(values)


def relocate_lammps_atomic_data(
    input_path: Path,
    output_path: Path,
    selected_oxygen_ids_path: Path,
    *,
    mapping_path: Path | None = None,
    report_path: Path | None = None,
    oxygen_type: int = 2,
    hydrogen_type: int = 1,
    oh_cutoff_A: float = 1.3,
    allowed_hydrogen_counts: Sequence[int] = (1, 2, 3),
    axis: int | str = "z",
    source_anchor: str = "lower",
    stationary_buffer_A: float = 4.0,
    high_boundary_buffer_A: float = 20.0,
    periodic: Sequence[bool] = (True, True, True),
    assignment_chunk_size: int = 256,
) -> Mapping[str, object]:
    """Relocate selected oxygen species in an atomic-style LAMMPS data file."""

    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    id_path = Path(selected_oxygen_ids_path).expanduser().resolve()
    if not source.is_file() or not id_path.is_file():
        raise FileNotFoundError("input data and selected-oxygen ID files must exist")
    if source == output:
        raise ValueError("output_path must not overwrite input_path")
    mapping = (
        output.with_name(output.name + ".relocation.tsv")
        if mapping_path is None
        else Path(mapping_path).expanduser().resolve()
    )
    report = (
        output.with_name(output.name + ".relocation.json")
        if report_path is None
        else Path(report_path).expanduser().resolve()
    )
    targets = (output, mapping, report)
    if len(set(targets)) != len(targets):
        raise ValueError("output, mapping, and report paths must be distinct")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")

    data = _parse_lammps_atomic_data(source)
    axis_index = _axis_index(axis)
    if source_anchor not in ("lower", "upper"):
        raise ValueError("source_anchor must be 'lower' or 'upper'")
    source_reference = float(data.bounds[axis_index, 0 if source_anchor == "lower" else 1])
    selected_ids = _read_atom_ids(id_path)
    result = relocate_selected_oxygen_species(
        data.atom_ids,
        data.atom_types,
        data.coordinates,
        data.bounds,
        selected_ids,
        oxygen_type=oxygen_type,
        hydrogen_type=hydrogen_type,
        oh_cutoff_A=oh_cutoff_A,
        allowed_hydrogen_counts=allowed_hydrogen_counts,
        axis=axis_index,
        source_reference_A=source_reference,
        stationary_buffer_A=stationary_buffer_A,
        high_boundary_buffer_A=high_boundary_buffer_A,
        periodic=periodic,
        assignment_chunk_size=assignment_chunk_size,
        image_flags=data.image_flags,
    )

    id_to_sorted_index = {
        int(atom_id): index for index, atom_id in enumerate(data.atom_ids)
    }
    output_lines = list(data.lines)
    for atom_id in data.atom_ids:
        atom_id_int = int(atom_id)
        index = id_to_sorted_index[atom_id_int]
        fields = list(data.atom_fields_by_id[atom_id_int])
        fields[2:5] = [f"{value:.16g}" for value in result.coordinates[index]]
        if result.image_flags is not None:
            fields[5:8] = [str(int(value)) for value in result.image_flags[index]]
        comment = data.atom_comments_by_id[atom_id_int]
        output_lines[data.atom_line_by_id[atom_id_int]] = " ".join(fields) + (
            f" {comment}" if comment else ""
        )
    _write_new_text(output, "\n".join(output_lines) + "\n")

    mapping_header = (
        "atom_id\ttype\trole\tparent_oxygen_id\told_x\told_y\told_z\t"
        + "new_x\tnew_y\tnew_z\told_ix\told_iy\told_iz\tnew_ix\tnew_iy\tnew_iz\t"
        + "oh_distance_A"
    )
    mapping_rows = [mapping_header]
    moved_set = set(result.moved_atom_ids)
    for atom_id in result.moved_atom_ids:
        index = id_to_sorted_index[atom_id]
        role = "oxygen" if atom_id in result.moved_oxygen_ids else "assigned_hydrogen"
        distance = result.oh_distance_by_hydrogen_id_A.get(atom_id)
        old_images = (
            ("", "", "")
            if data.image_flags is None
            else tuple(str(int(value)) for value in data.image_flags[index])
        )
        new_images = (
            ("", "", "")
            if result.image_flags is None
            else tuple(str(int(value)) for value in result.image_flags[index])
        )
        mapping_rows.append(
            "\t".join(
                [
                    str(atom_id),
                    str(int(data.atom_types[index])),
                    role,
                    str(result.parent_oxygen_by_atom_id[atom_id]),
                    *(f"{value:.16g}" for value in data.coordinates[index]),
                    *(f"{value:.16g}" for value in result.coordinates[index]),
                    *old_images,
                    *new_images,
                    "" if distance is None else f"{distance:.16g}",
                ]
            )
        )
    _write_new_text(mapping, "\n".join(mapping_rows) + "\n")

    axis_name = "xyz"[result.axis]
    report_data: dict[str, object] = {
        "status": "PASS",
        "workflow": "relocate_lammps_atomic_data",
        "input": str(source),
        "output": str(output),
        "selected_oxygen_ids": str(id_path),
        "mapping": str(mapping),
        "report": str(report),
        "input_sha256": _sha256(source),
        "selected_oxygen_ids_sha256": _sha256(id_path),
        "output_sha256": _sha256(output),
        "mapping_sha256": _sha256(mapping),
        "atom_count": int(data.atom_ids.size),
        "atom_inventory_preserved": True,
        "non_coordinate_sections_preserved": True,
        "velocity_rows_preserved": True,
        "oxygen_type": oxygen_type,
        "hydrogen_type": hydrogen_type,
        "oh_cutoff_A": oh_cutoff_A,
        "assignment_chunk_size": assignment_chunk_size,
        "allowed_hydrogen_counts": sorted(int(value) for value in allowed_hydrogen_counts),
        "moved_oxygen_count": len(result.moved_oxygen_ids),
        "moved_hydrogen_count": len(result.moved_hydrogen_ids),
        "moved_atom_count": len(moved_set),
        "hydrogen_count_by_oxygen_id": {
            str(key): value for key, value in result.hydrogen_count_by_oxygen_id.items()
        },
        "periodic_source_axes": [
            name for name, enabled in zip("xyz", periodic) if enabled
        ],
        "relocation_axis": axis_name,
        "source_anchor": source_anchor,
        "source_reference_A": result.source_reference_A,
        "translation_A": result.translation_A,
        "stationary_max_A": result.stationary_max_A,
        "moved_range_A": [result.moved_min_A, result.moved_max_A],
        "stationary_buffer_A": result.stationary_buffer_A,
        "high_boundary_clearance_A": result.high_boundary_clearance_A,
        "maximum_oh_vector_change_A": result.maximum_oh_vector_change_A,
        "axis_image_flags_cleared_for_all_atoms": data.image_flags is not None,
    }
    _write_new_text(report, json.dumps(report_data, indent=2, sort_keys=True) + "\n")
    return report_data
