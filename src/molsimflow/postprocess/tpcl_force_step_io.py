"""Fail-closed validation for high-frequency TPCL force-step outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import LammpsDumpFrame, iter_lammps_dump_records
from molsimflow.postprocess.constant_force_species_timeseries import (
    _frame_arrays,
    assign_hydrogen_parents,
    identify_fixed_carbon_hydrogen_ids,
    read_model_arrays,
    read_type_symbols,
    species_metrics,
)

COORDINATE_FIELDS = ("id", "type", "x", "y", "z", "ix", "iy", "iz")
DYNAMICS_FIELDS = (
    "id",
    "type",
    "vx",
    "vy",
    "vz",
    "fx",
    "fy",
    "fz",
    "f_RAW[1]",
    "f_RAW[2]",
    "f_RAW[3]",
)


def expected_regular_steps(start_step: int, total_steps: int, stride: int) -> tuple[int, ...]:
    """Return post-start output steps for a regular cadence."""

    if start_step < 0 or total_steps <= 0 or stride <= 0 or total_steps % stride:
        raise ValueError("invalid regular output schedule")
    return tuple(range(start_step + stride, start_step + total_steps + 1, stride))


def expected_multirate_steps(
    start_step: int,
    total_steps: int,
    fast_steps: int,
    fast_stride: int,
    slow_stride: int,
) -> tuple[int, ...]:
    """Return a two-rate coordinate schedule with one shared boundary."""

    if (
        start_step < 0
        or total_steps <= 0
        or fast_steps <= 0
        or fast_steps > total_steps
        or fast_stride <= 0
        or slow_stride <= 0
        or fast_steps % fast_stride
        or (total_steps - fast_steps) % slow_stride
    ):
        raise ValueError("invalid multirate output schedule")
    fast = range(start_step + fast_stride, start_step + fast_steps + 1, fast_stride)
    slow = range(
        start_step + fast_steps + slow_stride,
        start_step + total_steps + 1,
        slow_stride,
    )
    return tuple((*fast, *slow))


def _require_file(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise ValueError(f"missing or empty required file: {candidate}")
    return candidate


def _frame_identity(
    frame: LammpsDumpFrame,
    expected_fields: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    if frame.atom_fields != tuple(expected_fields):
        raise ValueError(
            f"step {frame.timestep}: expected fields {tuple(expected_fields)}, "
            f"got {frame.atom_fields}"
        )
    identity = tuple((int(row[0]), int(row[1])) for row in frame.atom_rows)
    if len({atom_id for atom_id, _ in identity}) != frame.atom_count:
        raise ValueError(f"step {frame.timestep}: duplicate atom IDs")
    values = np.asarray(
        [[float(value) for value in row[2:]] for row in frame.atom_rows],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"step {frame.timestep}: non-finite atom values")
    return identity


def _validate_dump(
    path: Path,
    expected_fields: Sequence[str],
    expected_steps: Sequence[int],
    expected_atom_count: int,
    expected_identity: tuple[tuple[int, int], ...] | None = None,
) -> tuple[dict[str, object], tuple[tuple[int, int], ...], LammpsDumpFrame]:
    steps: list[int] = []
    identity = expected_identity
    terminal: LammpsDumpFrame | None = None
    for frame in iter_lammps_dump_records(_require_file(path)):
        current = _frame_identity(frame, expected_fields)
        if frame.atom_count != expected_atom_count:
            raise ValueError(
                f"{path}: step {frame.timestep} has {frame.atom_count} atoms; "
                f"expected {expected_atom_count}"
            )
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(f"{path}: atom identity changed at step {frame.timestep}")
        steps.append(frame.timestep)
        terminal = frame
    if terminal is None or identity is None:
        raise ValueError(f"{path}: no complete frames")
    if tuple(steps) != tuple(expected_steps):
        raise ValueError(
            f"{path}: timestep schedule differs; got {steps[:3]}...{steps[-3:]}, "
            f"expected {tuple(expected_steps)[:3]}...{tuple(expected_steps)[-3:]}"
        )
    return (
        {
            "path": str(path),
            "frames": len(steps),
            "atom_count": expected_atom_count,
            "fields": list(expected_fields),
            "first_step": steps[0],
            "last_step": steps[-1],
            "size_bytes": path.stat().st_size,
        },
        identity,
        terminal,
    )


def _numeric_table_steps(path: Path) -> tuple[int, ...]:
    steps: list[int] = []
    for line_number, raw in enumerate(_require_file(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        try:
            step = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric row") from exc
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: empty or non-finite values")
        if steps and step <= steps[-1]:
            raise ValueError(f"{path}:{line_number}: non-increasing timestep")
        steps.append(step)
    if not steps:
        raise ValueError(f"{path}: no numeric rows")
    return tuple(steps)


def _validate_table(path: Path, expected_steps: Sequence[int]) -> dict[str, object]:
    steps = _numeric_table_steps(path)
    if steps != tuple(expected_steps):
        raise ValueError(f"{path}: timestep schedule differs")
    return {
        "path": str(path),
        "rows": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "size_bytes": path.stat().st_size,
    }


def _terminal_species_gate(
    terminal: LammpsDumpFrame,
    model_data: Path,
    substrate_atoms: int,
    expected_water_oxygen: int,
) -> dict[str, object]:
    type_symbols = read_type_symbols(model_data)
    model_ids, model_types, _, _ = read_model_arrays(model_data)
    solution_oxygen_ids = {
        int(atom_id)
        for atom_id, atom_type in zip(model_ids, model_types)
        if int(atom_id) > substrate_atoms and type_symbols[int(atom_type)] == "O"
    }
    if len(solution_oxygen_ids) != expected_water_oxygen:
        raise ValueError(
            f"model defines {len(solution_oxygen_ids)} solution O atoms; "
            f"expected {expected_water_oxygen}"
        )
    fixed_carbon_hydrogen_ids = identify_fixed_carbon_hydrogen_ids(
        model_data,
        type_symbols,
        1.25,
    )
    atom_ids, atom_types, coordinates = _frame_arrays(terminal)
    _, oxygen_ids, oxygen_parent_ids, assigned_any = assign_hydrogen_parents(
        atom_ids,
        atom_types,
        coordinates,
        terminal.bounds,
        type_symbols,
        1.35,
        fixed_carbon_hydrogen_ids,
    )
    metrics = species_metrics(
        oxygen_ids,
        oxygen_parent_ids,
        assigned_any,
        solution_oxygen_ids,
    )
    defect_fields = (
        "solution_O",
        "solution_OH",
        "solution_H3O",
        "solution_overcoordinated",
        "framework_overprotonated",
        "hydrogen_unassigned",
    )
    if metrics["solution_H2O"] != expected_water_oxygen or any(
        metrics[field] != 0 for field in defect_fields
    ):
        raise ValueError(f"terminal species gate failed: {metrics}")
    return {
        "status": "PASS",
        "step": terminal.timestep,
        "fixed_carbon_hydrogen": len(fixed_carbon_hydrogen_ids),
        **metrics,
    }


def _scaled_size(size: int, source_count: int, target_count: int) -> int:
    if source_count <= 0 or target_count <= 0:
        raise ValueError("size projection requires positive frame counts")
    return math.ceil(size * target_count / source_count)


def project_production_size(
    *,
    output_dir: Path,
    observed_counts: dict[str, int],
    start_step: int,
    projection_total_steps: int,
    projection_fast_steps: int,
    fast_coordinate_stride: int,
    slow_coordinate_stride: int,
    dynamics_stride: int,
    full_stride: int,
    table_stride: int,
    restart_stride: int,
) -> dict[str, object]:
    """Project compressed 100 ps output size from an observed smoke."""

    coordinate_target = len(
        expected_multirate_steps(
            start_step,
            projection_total_steps,
            projection_fast_steps,
            fast_coordinate_stride,
            slow_coordinate_stride,
        )
    )
    dynamics_target = len(
        expected_regular_steps(start_step, projection_total_steps, dynamics_stride)
    )
    full_target = len(expected_regular_steps(start_step, projection_total_steps, full_stride))
    table_target = len(expected_regular_steps(start_step, projection_total_steps, table_stride))
    targets = {
        "tpcl_coordinates.lammpstrj.zst": coordinate_target,
        "tpcl_dynamics.lammpstrj.zst": dynamics_target,
        "full_reference.lammpstrj.zst": full_target,
        "motion_energy_stress_0p01ps.dat": table_target,
        "force_sums_0p01ps.dat": table_target,
    }
    keys = {
        "tpcl_coordinates.lammpstrj.zst": "coordinates",
        "tpcl_dynamics.lammpstrj.zst": "dynamics",
        "full_reference.lammpstrj.zst": "full_reference",
        "motion_energy_stress_0p01ps.dat": "motion",
        "force_sums_0p01ps.dat": "force",
    }
    components: dict[str, int] = {}
    for filename, target_count in targets.items():
        path = output_dir / filename
        components[filename] = _scaled_size(
            _require_file(path).stat().st_size,
            observed_counts[keys[filename]],
            target_count,
        )
    restart_count = projection_total_steps // restart_stride
    components["restart_checkpoints_and_final"] = (
        restart_count + 1
    ) * _require_file(output_dir / "final.restart").stat().st_size
    components["final.data"] = _require_file(output_dir / "final.data").stat().st_size
    components["runtime_metadata_margin"] = 16 * 1024 * 1024
    return {
        "projection_total_steps": projection_total_steps,
        "projection_fast_steps": projection_fast_steps,
        "projected_coordinate_frames": coordinate_target,
        "projected_dynamics_frames": dynamics_target,
        "projected_full_reference_frames": full_target,
        "projected_table_rows": table_target,
        "projected_restart_checkpoints": restart_count,
        "components_bytes": components,
        "projected_total_bytes": sum(components.values()),
    }


def validate_tpcl_force_step_output(
    *,
    output_dir: Path,
    model_data: Path,
    start_step: int,
    total_steps: int,
    fast_steps: int,
    natoms: int,
    substrate_atoms: int,
    selected_atoms: int,
    expected_water_oxygen: int,
    fast_coordinate_stride: int = 20,
    slow_coordinate_stride: int = 100,
    dynamics_stride: int = 100,
    full_stride: int = 1000,
    table_stride: int = 20,
    restart_stride: int = 10000,
    projection_total_steps: int | None = None,
    projection_fast_steps: int = 40000,
    size_ceiling_bytes: int | None = None,
) -> dict[str, object]:
    """Validate cadence, identity, terminal species, and optional size projection."""

    root = Path(output_dir)
    coordinate_steps = expected_multirate_steps(
        start_step,
        total_steps,
        fast_steps,
        fast_coordinate_stride,
        slow_coordinate_stride,
    )
    dynamics_steps = expected_regular_steps(start_step, total_steps, dynamics_stride)
    full_steps = expected_regular_steps(start_step, total_steps, full_stride)
    table_steps = expected_regular_steps(start_step, total_steps, table_stride)
    coordinates, selected_identity, _ = _validate_dump(
        root / "tpcl_coordinates.lammpstrj.zst",
        COORDINATE_FIELDS,
        coordinate_steps,
        selected_atoms,
    )
    dynamics, _, _ = _validate_dump(
        root / "tpcl_dynamics.lammpstrj.zst",
        DYNAMICS_FIELDS,
        dynamics_steps,
        selected_atoms,
        selected_identity,
    )
    full_reference, _, terminal = _validate_dump(
        root / "full_reference.lammpstrj.zst",
        COORDINATE_FIELDS,
        full_steps,
        natoms,
    )
    motion = _validate_table(root / "motion_energy_stress_0p01ps.dat", table_steps)
    force = _validate_table(root / "force_sums_0p01ps.dat", table_steps)
    species = _terminal_species_gate(
        terminal,
        Path(model_data),
        substrate_atoms,
        expected_water_oxygen,
    )
    report: dict[str, object] = {
        "status": "PASS",
        "start_step": start_step,
        "end_step": start_step + total_steps,
        "total_steps": total_steps,
        "fast_steps": fast_steps,
        "coordinates": coordinates,
        "dynamics": dynamics,
        "full_reference": full_reference,
        "motion": motion,
        "force": force,
        "terminal_species": species,
    }
    if projection_total_steps is not None:
        projection = project_production_size(
            output_dir=root,
            observed_counts={
                "coordinates": int(coordinates["frames"]),
                "dynamics": int(dynamics["frames"]),
                "full_reference": int(full_reference["frames"]),
                "motion": int(motion["rows"]),
                "force": int(force["rows"]),
            },
            start_step=start_step,
            projection_total_steps=projection_total_steps,
            projection_fast_steps=projection_fast_steps,
            fast_coordinate_stride=fast_coordinate_stride,
            slow_coordinate_stride=slow_coordinate_stride,
            dynamics_stride=dynamics_stride,
            full_stride=full_stride,
            table_stride=table_stride,
            restart_stride=restart_stride,
        )
        if size_ceiling_bytes is None or size_ceiling_bytes <= 0:
            raise ValueError("projection requires a positive size ceiling")
        projection["size_ceiling_bytes"] = size_ceiling_bytes
        projection["ceiling_gate"] = (
            "PASS"
            if int(projection["projected_total_bytes"]) <= size_ceiling_bytes
            else "FAIL"
        )
        report["production_size_projection"] = projection
        if projection["ceiling_gate"] != "PASS":
            raise ValueError(
                "projected production output exceeds ceiling: "
                f"{projection['projected_total_bytes']} > {size_ceiling_bytes}"
            )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-data", type=Path, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--fast-steps", type=int, required=True)
    parser.add_argument("--natoms", type=int, required=True)
    parser.add_argument("--substrate-atoms", type=int, required=True)
    parser.add_argument("--selected-atoms", type=int, required=True)
    parser.add_argument("--expected-water-oxygen", type=int, required=True)
    parser.add_argument("--projection-total-steps", type=int)
    parser.add_argument("--projection-fast-steps", type=int, default=40000)
    parser.add_argument("--size-ceiling-bytes", type=int)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_tpcl_force_step_output(
        output_dir=args.output_dir,
        model_data=args.model_data,
        start_step=args.start_step,
        total_steps=args.total_steps,
        fast_steps=args.fast_steps,
        natoms=args.natoms,
        substrate_atoms=args.substrate_atoms,
        selected_atoms=args.selected_atoms,
        expected_water_oxygen=args.expected_water_oxygen,
        projection_total_steps=args.projection_total_steps,
        projection_fast_steps=args.projection_fast_steps,
        size_ceiling_bytes=args.size_ceiling_bytes,
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
