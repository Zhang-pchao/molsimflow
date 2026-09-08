"""Fail-closed validation for aligned V3 coordinate, ROI-velocity, and stress outputs."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from pathlib import Path

from molsimflow.io.lammps_dump import LammpsDumpFrame, iter_lammps_dump_records

DEFAULT_KINETIC_METAL_TO_BAR_A3 = 166.053882315


def parse_lammps_data_masses(data_path: Path) -> dict[int, float]:
    """Return LAMMPS atom-type masses from an atomic-data file."""

    masses: dict[int, float] = {}
    in_masses = False
    for raw in Path(data_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "Masses":
            in_masses = True
            continue
        if not in_masses:
            continue
        if line.startswith("Atoms"):
            break
        if line and not line.startswith("#"):
            fields = line.split()
            if len(fields) >= 2 and fields[0].isdigit():
                masses[int(fields[0])] = float(fields[1])
    if not masses:
        raise ValueError(f"could not read Masses from {data_path}")
    return masses


def parse_lammps_data_atoms(data_path: Path) -> dict[int, tuple[int, tuple[float, float, float]]]:
    """Return atom type and Cartesian coordinates from a LAMMPS atomic-data file."""

    atoms: dict[int, tuple[int, tuple[float, float, float]]] = {}
    in_atoms = False
    for raw in Path(data_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("Atoms"):
            in_atoms = True
            continue
        if not in_atoms or not line or line.startswith("#"):
            continue
        if not line[0].isdigit():
            break
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"{data_path}: malformed atom row")
        atom_id = int(fields[0])
        if atom_id in atoms:
            raise ValueError(f"{data_path}: duplicate atom id {atom_id}")
        atoms[atom_id] = (int(fields[1]), tuple(float(value) for value in fields[2:5]))
    if not atoms:
        raise ValueError(f"could not read Atoms from {data_path}")
    return atoms


def parse_ave_time(path: Path, expected_values: int) -> dict[int, list[float]]:
    """Read a LAMMPS ave/time table keyed by timestep."""

    values: dict[int, list[float]] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != expected_values + 1:
            raise ValueError(f"{path}: expected step plus {expected_values} values")
        step = int(fields[0])
        if step in values:
            raise ValueError(f"{path}: duplicate step {step}")
        values[step] = [float(value) for value in fields[1:]]
    if not values:
        raise ValueError(f"{path}: no numeric rows")
    return values


def expected_steps(start_step: int, end_step: int, interval: int) -> tuple[int, ...]:
    """Return the inclusive output schedule, rejecting incomplete cadence definitions."""

    if interval <= 0 or end_step < start_step or (end_step - start_step) % interval:
        raise ValueError("invalid output interval or timestep range")
    return tuple(range(start_step, end_step + 1, interval))


def _require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty required input: {path}")
    return path


def _frame_identity(frame: LammpsDumpFrame, fields: Sequence[str]) -> dict[int, int]:
    if frame.atom_fields != tuple(fields):
        raise ValueError(f"step {frame.timestep}: expected fields {tuple(fields)}, got {frame.atom_fields}")
    identity: dict[int, int] = {}
    for row in frame.atom_rows:
        atom_id, atom_type = int(row[0]), int(row[1])
        if atom_id in identity:
            raise ValueError(f"step {frame.timestep}: duplicate atom id {atom_id}")
        identity[atom_id] = atom_type
    if len(identity) != frame.atom_count:
        raise ValueError(f"step {frame.timestep}: non-unique atom IDs")
    return identity


def _kinetic_components(row: Sequence[str], mass: float, factor: float) -> tuple[float, ...]:
    vx, vy, vz = (float(value) for value in row[2:])
    return tuple(
        -factor * mass * value
        for value in (vx * vx, vy * vy, vz * vz, vx * vy, vx * vz, vy * vz)
    )


def _check_final_data(frame: LammpsDumpFrame, final_data: Path, tolerance_A: float) -> float:
    if tolerance_A <= 0:
        raise ValueError("final-data tolerance must be positive")
    final_atoms = parse_lammps_data_atoms(final_data)
    coordinate_identity = _frame_identity(frame, ("id", "type", "x", "y", "z"))
    if set(final_atoms) != set(coordinate_identity):
        raise ValueError("final.data atom IDs do not match the terminal coordinate frame")
    lengths = tuple(float(upper - lower) for lower, upper in frame.bounds)
    if min(lengths) <= 0:
        raise ValueError("terminal coordinate frame has an invalid box")
    maximum_displacement = 0.0
    for row in frame.atom_rows:
        atom_id = int(row[0])
        final_type, final_position = final_atoms[atom_id]
        if final_type != coordinate_identity[atom_id]:
            raise ValueError(f"final.data type differs for atom {atom_id}")
        for coordinate, reference, length in zip((float(value) for value in row[2:5]), final_position, lengths):
            displacement = coordinate - reference
            displacement -= round(displacement / length) * length
            maximum_displacement = max(maximum_displacement, abs(displacement))
    if maximum_displacement > tolerance_A:
        raise ValueError(
            "final.data does not match the terminal coordinate frame within "
            f"{tolerance_A:g} A (maximum minimum-image displacement {maximum_displacement:g} A)"
        )
    return maximum_displacement


def validate_v3_mechanism_io(
    *,
    coordinates: Path,
    velocity_roi: Path,
    roi_kinetic: Path,
    global_stress: Path,
    model_data: Path,
    final_data: Path,
    start_step: int,
    end_step: int,
    natoms: int,
    roi_types: Sequence[int],
    coordinate_every: int,
    velocity_every: int,
    thermo_every: int,
    kinetic_factor: float = DEFAULT_KINETIC_METAL_TO_BAR_A3,
    final_data_tolerance_A: float = 1.0e-6,
) -> dict[str, object]:
    """Validate V3 output alignment and ROI kinetic closure without requiring force dumps."""

    if natoms <= 0 or not roi_types or kinetic_factor <= 0:
        raise ValueError("natoms, roi_types, and kinetic_factor must be positive")
    if coordinate_every != velocity_every:
        raise ValueError("V3 coordinates and ROI velocities must have the same cadence")
    coordinate_path = _require_file(coordinates)
    velocity_path = _require_file(velocity_roi)
    online_roi = parse_ave_time(_require_file(roi_kinetic), 7)
    stress_rows = parse_ave_time(_require_file(global_stress), 6)
    masses = parse_lammps_data_masses(_require_file(model_data))
    terminal_data = _require_file(final_data)
    coordinate_steps = expected_steps(start_step, end_step, coordinate_every)
    thermo_steps = expected_steps(start_step, end_step, thermo_every)
    if tuple(sorted(online_roi)) != coordinate_steps:
        raise ValueError("ROI kinetic rows do not cover every coordinate timestep exactly")
    if tuple(sorted(stress_rows)) != thermo_steps:
        raise ValueError("global stress rows do not cover every thermo timestep exactly")

    selected_types = set(roi_types)
    max_absolute_error = 0.0
    max_relative_error = 0.0
    roi_min = math.inf
    roi_max = 0
    coordinate_frames = 0
    velocity_frames = 0
    terminal_coordinate: LammpsDumpFrame | None = None
    for coordinate, velocity in itertools.zip_longest(
        iter_lammps_dump_records(coordinate_path), iter_lammps_dump_records(velocity_path)
    ):
        if coordinate is None or velocity is None:
            raise ValueError("coordinate and ROI-velocity frame counts differ")
        coordinate_frames += 1
        velocity_frames += 1
        if coordinate.timestep != velocity.timestep:
            raise ValueError(
                f"coordinate/ROI-velocity timestep mismatch: {coordinate.timestep} != {velocity.timestep}"
            )
        if coordinate.atom_count != natoms:
            raise ValueError(f"step {coordinate.timestep}: {coordinate.atom_count} atoms, expected {natoms}")
        terminal_coordinate = coordinate
        coordinate_identity = _frame_identity(coordinate, ("id", "type", "x", "y", "z"))
        velocity_identity = _frame_identity(velocity, ("id", "type", "vx", "vy", "vz"))
        expected_roi = {
            atom_id for atom_id, atom_type in coordinate_identity.items() if atom_type in selected_types
        }
        if set(velocity_identity) != expected_roi:
            raise ValueError(f"step {velocity.timestep}: ROI membership does not match selected atom types")
        if any(coordinate_identity[atom_id] != atom_type for atom_id, atom_type in velocity_identity.items()):
            raise ValueError(f"step {velocity.timestep}: ROI atom id/type mismatch")
        components = [0.0] * 6
        for row in velocity.atom_rows:
            atom_type = int(row[1])
            if atom_type not in masses:
                raise ValueError(f"step {velocity.timestep}: missing mass for type {atom_type}")
            for index, value in enumerate(_kinetic_components(row, masses[atom_type], kinetic_factor)):
                components[index] += value
        online = online_roi[velocity.timestep]
        if round(online[0]) != len(velocity_identity):
            raise ValueError(f"step {velocity.timestep}: ROI count disagrees with roi_kinetic")
        roi_min = min(roi_min, len(velocity_identity))
        roi_max = max(roi_max, len(velocity_identity))
        for reconstructed, reported in zip(components, online[1:]):
            absolute_error = abs(reconstructed - reported)
            relative_error = absolute_error / max(1.0, abs(reported))
            max_absolute_error = max(max_absolute_error, absolute_error)
            max_relative_error = max(max_relative_error, relative_error)
            if relative_error > 5.0e-9 and absolute_error > 1.0e-2:
                raise ValueError(
                    f"step {velocity.timestep}: ROI kinetic closure failed "
                    f"(abs={absolute_error:.6g}, rel={relative_error:.6g})"
                )
    if coordinate_frames != len(coordinate_steps) or velocity_frames != len(coordinate_steps):
        raise ValueError("frame count does not match the requested coordinate cadence")
    if terminal_coordinate is None:
        raise ValueError("coordinate dump has no complete frames")
    terminal_displacement = _check_final_data(
        terminal_coordinate, terminal_data, final_data_tolerance_A
    )
    return {
        "status": "PASS",
        "coordinate_frames": coordinate_frames,
        "velocity_roi_frames": velocity_frames,
        "coordinate_step_first": coordinate_steps[0],
        "coordinate_step_last": coordinate_steps[-1],
        "roi_atom_count_min": int(roi_min),
        "roi_atom_count_max": roi_max,
        "roi_selector_types": sorted(selected_types),
        "global_stress_rows": len(stress_rows),
        "final_data_maximum_minimum_image_displacement_A": terminal_displacement,
        "kinetic_factor_bar_A3_per_g_mol_A2_ps2": kinetic_factor,
        "max_abs_kinetic_sum_error_bar_A3": max_absolute_error,
        "max_rel_kinetic_sum_error": max_relative_error,
    }
