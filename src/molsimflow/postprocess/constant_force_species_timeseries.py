"""Audit geometric water/surface proton partition along constant-force trajectories."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import (
    LammpsDumpFrame,
    box_lengths,
    iter_lammps_dump_records_until,
)
from molsimflow.postprocess.constant_force_oxygen import (
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)


TIMESERIES_FIELDS = (
    "case_id",
    "branch_id",
    "direction",
    "step",
    "time_ps",
    "oh_cutoff_A",
    "solution_oxygen_total",
    "solution_O",
    "solution_OH",
    "solution_H2O",
    "solution_H3O",
    "solution_overcoordinated",
    "framework_OH",
    "framework_overprotonated",
    "proton_partition_pool",
    "hydrogen_unassigned",
)

SENSITIVITY_FIELDS = TIMESERIES_FIELDS

EVENT_FIELDS = (
    "case_id",
    "branch_id",
    "direction",
    "step",
    "time_ps",
    "hydrogen_id",
    "from_oxygen_id",
    "to_oxygen_id",
    "from_region",
    "to_region",
    "event_class",
    "new_assignment_duration_ps",
    "persistent",
)

SUMMARY_FIELDS = (
    "case_id",
    "branch_id",
    "direction",
    "first_step",
    "last_step",
    "frames",
    "duration_ps",
    "sampling_interval_ps",
    "solution_oxygen_total",
    "initial_H3O",
    "terminal_H3O",
    "H3O_delta",
    "H3O_min",
    "H3O_max",
    "H3O_last_50ps_range",
    "initial_framework_OH",
    "terminal_framework_OH",
    "initial_proton_pool",
    "terminal_proton_pool",
    "proton_pool_range",
    "terminal_solution_O",
    "terminal_solution_OH",
    "terminal_overcoordinated",
    "terminal_unassigned_H",
    "fixed_carbon_H",
    "partition_events",
    "persistent_partition_events",
    "cutoff_H3O_max_spread",
    "inventory_integrity_gate",
    "proton_partition_stationarity",
)


MASS_SYMBOLS = (
    ("H", 1.008),
    ("C", 12.011),
    ("N", 14.007),
    ("O", 15.999),
    ("Na", 22.989769),
    ("Si", 28.085),
    ("Cl", 35.45),
    ("Ti", 47.867),
)


def _infer_symbol(mass: float) -> str:
    symbol, reference = min(MASS_SYMBOLS, key=lambda item: abs(item[1] - mass))
    if abs(mass - reference) > 0.35:
        raise ValueError(f"Cannot infer an element from mass {mass}")
    return symbol


def read_type_symbols(model_data: Path) -> dict[int, str]:
    """Read LAMMPS Masses into an atom-type to element mapping."""

    lines = Path(model_data).read_text(encoding="utf-8").splitlines()
    mapping: dict[int, str] = {}
    in_masses = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("Masses"):
            in_masses = True
            continue
        if not in_masses or not line:
            continue
        if re.match(r"^[A-Za-z]", line):
            break
        match = re.match(
            r"^(\d+)\s+([0-9Ee+\-.]+)(?:\s*#\s*([A-Za-z][A-Za-z0-9]*))?",
            line,
        )
        if match:
            mapping[int(match.group(1))] = match.group(3) or _infer_symbol(float(match.group(2)))
    if not mapping:
        raise ValueError(f"No Masses mapping found in {model_data}")
    return mapping


def read_model_arrays(model_data: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read atom IDs, types, coordinates, and orthorhombic bounds from a data file."""

    lines = Path(model_data).read_text(encoding="utf-8").splitlines()
    atom_count = None
    bounds: list[tuple[float, float]] = []
    atoms_header = None
    for index, raw in enumerate(lines):
        fields = raw.split()
        if len(fields) == 2 and fields[1] == "atoms" and fields[0].isdigit():
            atom_count = int(fields[0])
        if len(fields) >= 4 and " ".join(fields[-2:]) in {"xlo xhi", "ylo yhi", "zlo zhi"}:
            bounds.append((float(fields[0]), float(fields[1])))
        if raw.strip().startswith("Atoms"):
            atoms_header = index
    if atom_count is None or len(bounds) != 3 or atoms_header is None:
        raise ValueError(f"Incomplete orthorhombic LAMMPS data file: {model_data}")
    records: list[tuple[int, int, float, float, float]] = []
    for raw in lines[atoms_header + 1 :]:
        fields = raw.partition("#")[0].split()
        if not fields:
            continue
        if not fields[0].lstrip("+-").isdigit():
            if records:
                break
            continue
        if len(fields) < 5:
            raise ValueError(f"Invalid atomic-style row in {model_data}: {raw!r}")
        records.append((int(fields[0]), int(fields[1]), *(float(value) for value in fields[2:5])))
        if len(records) == atom_count:
            break
    if len(records) != atom_count:
        raise ValueError(f"Expected {atom_count} atoms in {model_data}, found {len(records)}")
    array = np.asarray(records, dtype=float)
    return (
        array[:, 0].astype(np.int64),
        array[:, 1].astype(np.int64),
        array[:, 2:5],
        np.asarray(bounds, dtype=float),
    )


def _read_ids(path: Path) -> set[int]:
    values = {int(value) for value in Path(path).read_text(encoding="utf-8").split()}
    if not values:
        raise ValueError(f"No atom IDs found in {path}")
    return values


def _frame_arrays(
    frame: LammpsDumpFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = {name: index for index, name in enumerate(frame.atom_fields)}
    missing = {"id", "type", "x", "y", "z"}.difference(columns)
    if missing:
        raise ValueError(f"step {frame.timestep} is missing {sorted(missing)}")
    ids = np.fromiter(
        (int(row[columns["id"]]) for row in frame.atom_rows),
        dtype=np.int64,
        count=frame.atom_count,
    )
    types = np.fromiter(
        (int(row[columns["type"]]) for row in frame.atom_rows),
        dtype=np.int64,
        count=frame.atom_count,
    )
    coordinates = np.asarray(
        [[float(row[columns[name]]) for name in ("x", "y", "z")] for row in frame.atom_rows],
        dtype=float,
    )
    if len(set(ids.tolist())) != frame.atom_count:
        raise ValueError(f"step {frame.timestep} contains duplicate atom IDs")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"step {frame.timestep} contains non-finite coordinates")
    return ids, types, coordinates


def _validate_box_boundary(frame: LammpsDumpFrame, *, periodic_z: bool) -> None:
    """Reject a contract whose distance convention disagrees with dump boundaries."""

    flags = frame.box_header.split()[3:]
    if len(flags) < 3 or any(len(flag) != 2 or set(flag) - set("pfsm") for flag in flags[-3:]):
        raise ValueError(
            f"step {frame.timestep} has unsupported box boundary header: {frame.box_header}"
        )
    x_flag, y_flag, z_flag = flags[-3:]
    if x_flag != "pp" or y_flag != "pp" or (z_flag == "pp") != periodic_z:
        raise ValueError(
            f"step {frame.timestep} boundary {x_flag} {y_flag} {z_flag} "
            f"conflicts with periodic x/y and periodic_z={periodic_z}"
        )


def _periodic_query(
    sources: np.ndarray,
    targets: np.ndarray,
    bounds: np.ndarray,
    *,
    periodic_z: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest target index with periodic x/y and configurable z."""

    from scipy.spatial import cKDTree

    if len(targets) == 0:
        return np.full(len(sources), -1, dtype=int), np.full(len(sources), np.inf)
    lengths = box_lengths(bounds)
    pseudo_z = max(1.0e5, 10.0 * lengths[2])
    box = lengths if periodic_z else np.asarray([lengths[0], lengths[1], pseudo_z])

    def normalize(values: np.ndarray) -> np.ndarray:
        result = np.empty_like(values, dtype=float)
        result[:, 0] = (values[:, 0] - bounds[0, 0]) % lengths[0]
        result[:, 1] = (values[:, 1] - bounds[1, 0]) % lengths[1]
        if periodic_z:
            result[:, 2] = (values[:, 2] - bounds[2, 0]) % lengths[2]
        else:
            result[:, 2] = values[:, 2] - bounds[2, 0] + 0.25 * pseudo_z
        return result

    distance, index = cKDTree(normalize(targets), boxsize=box).query(
        normalize(sources),
        k=1,
    )
    return np.asarray(index, dtype=int), np.asarray(distance, dtype=float)


def identify_fixed_carbon_hydrogen_ids(
    model_data: Path,
    type_symbols: Mapping[int, str],
    ch_cutoff_A: float,
    *,
    periodic_z: bool = False,
) -> set[int]:
    """Identify model-defined methyl H atoms and keep them out of proton accounting."""

    ids, types, coordinates, bounds = read_model_arrays(model_data)
    symbols = np.asarray([type_symbols[int(atom_type)] for atom_type in types])
    h_indices = np.where(symbols == "H")[0]
    c_indices = np.where(symbols == "C")[0]
    if len(c_indices) == 0:
        return set()
    nearest_c, distance_c = _periodic_query(
        coordinates[h_indices], coordinates[c_indices], bounds, periodic_z=periodic_z
    )
    fixed: set[int] = set()
    for carbon_index in range(len(c_indices)):
        candidates = np.where((nearest_c == carbon_index) & (distance_c <= ch_cutoff_A))[0]
        if len(candidates) < 3:
            continue
        closest = candidates[np.argsort(distance_c[candidates])[:3]]
        fixed.update(int(ids[h_indices[index]]) for index in closest)
    return fixed


def assign_hydrogen_parents(
    ids: np.ndarray,
    types: np.ndarray,
    coordinates: np.ndarray,
    bounds: np.ndarray,
    type_symbols: Mapping[int, str],
    oh_cutoff_A: float,
    fixed_carbon_hydrogen_ids: set[int],
    *,
    periodic_z: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign non-methyl H to the nearest valid O and return oxygen parent IDs."""

    symbols = np.asarray([type_symbols[int(atom_type)] for atom_type in types])
    h_indices = np.where(symbols == "H")[0]
    o_indices = np.where(symbols == "O")[0]
    if len(h_indices) == 0 or len(o_indices) == 0:
        raise ValueError("Species assignment requires hydrogen and oxygen atoms")
    nearest_o, distance_o = _periodic_query(
        coordinates[h_indices], coordinates[o_indices], bounds, periodic_z=periodic_z
    )
    valid_o = distance_o <= oh_cutoff_A
    fixed_carbon = np.asarray(
        [int(atom_id) in fixed_carbon_hydrogen_ids for atom_id in ids[h_indices]],
        dtype=bool,
    )
    choose_o = valid_o & ~fixed_carbon
    oxygen_parent_ids = np.full(len(h_indices), -1, dtype=np.int64)
    oxygen_parent_ids[choose_o] = ids[o_indices[nearest_o[choose_o]]]
    assigned_any = choose_o | fixed_carbon
    return (
        ids[h_indices],
        ids[o_indices],
        oxygen_parent_ids,
        assigned_any,
    )


def species_metrics(
    oxygen_ids: np.ndarray,
    oxygen_parent_ids: np.ndarray,
    assigned_any: np.ndarray,
    solution_oxygen_ids: set[int],
) -> dict[str, int]:
    """Count geometric solution and framework oxygen protonation states."""

    index = {int(atom_id): position for position, atom_id in enumerate(oxygen_ids)}
    assigned = [index[int(parent)] for parent in oxygen_parent_ids if int(parent) >= 0]
    counts = np.bincount(assigned, minlength=len(oxygen_ids)).astype(int)
    solution_mask = np.asarray(
        [int(atom_id) in solution_oxygen_ids for atom_id in oxygen_ids],
        dtype=bool,
    )
    solution = counts[solution_mask]
    framework = counts[~solution_mask]
    return {
        "solution_oxygen_total": int(len(solution)),
        "solution_O": int(np.count_nonzero(solution == 0)),
        "solution_OH": int(np.count_nonzero(solution == 1)),
        "solution_H2O": int(np.count_nonzero(solution == 2)),
        "solution_H3O": int(np.count_nonzero(solution == 3)),
        "solution_overcoordinated": int(np.count_nonzero(solution >= 4)),
        "framework_OH": int(np.count_nonzero(framework == 1)),
        "framework_overprotonated": int(np.count_nonzero(framework >= 2)),
        "hydrogen_unassigned": int(np.count_nonzero(~assigned_any)),
    }


def _region(oxygen_id: int, solution_oxygen_ids: set[int]) -> str:
    if oxygen_id < 0:
        return "UNASSIGNED"
    return "SOLUTION" if oxygen_id in solution_oxygen_ids else "FRAMEWORK"


def _assignment_events(
    *,
    case_id: str,
    branch_id: str,
    direction: str,
    steps: Sequence[int],
    times: Sequence[float],
    hydrogen_ids: np.ndarray,
    assignments: Sequence[np.ndarray],
    solution_oxygen_ids: set[int],
    minimum_persistence_ps: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_index in range(1, len(assignments)):
        previous = assignments[frame_index - 1]
        current = assignments[frame_index]
        for local_index in np.where(previous != current)[0]:
            old_parent = int(previous[local_index])
            new_parent = int(current[local_index])
            old_region = _region(old_parent, solution_oxygen_ids)
            new_region = _region(new_parent, solution_oxygen_ids)
            if old_region == new_region:
                continue
            stop = frame_index
            while (
                stop + 1 < len(assignments)
                and int(assignments[stop + 1][local_index]) == new_parent
            ):
                stop += 1
            duration = float(times[stop] - times[frame_index])
            rows.append(
                {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "direction": direction,
                    "step": steps[frame_index],
                    "time_ps": times[frame_index],
                    "hydrogen_id": int(hydrogen_ids[local_index]),
                    "from_oxygen_id": old_parent if old_parent >= 0 else "",
                    "to_oxygen_id": new_parent if new_parent >= 0 else "",
                    "from_region": old_region,
                    "to_region": new_region,
                    "event_class": f"{old_region}_TO_{new_region}",
                    "new_assignment_duration_ps": duration,
                    "persistent": str(
                        old_region != "UNASSIGNED"
                        and new_region != "UNASSIGNED"
                        and duration >= minimum_persistence_ps
                    ).lower(),
                }
            )
    return rows


def _plot(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    keys = sorted({(str(row["case_id"]), str(row["branch_id"])) for row in rows})
    figure, axes = plt.subplots(len(keys), 1, figsize=(9.0, 2.7 * len(keys)), squeeze=False)
    for axis, key in zip(axes[:, 0], keys):
        selected = [row for row in rows if (row["case_id"], row["branch_id"]) == key]
        time = np.asarray([float(row["time_ps"]) for row in selected]) / 1000.0
        axis.plot(time, [row["solution_H3O"] for row in selected], label="H3O-like")
        axis.plot(time, [row["framework_OH"] for row in selected], label="framework-OH")
        axis.set(title=f"{key[0]} / {key[1]}", xlabel="Time (ns)", ylabel="Count")
        axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "species_partition_timeseries.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    """Run a contract-defined geometric proton-partition time-series audit."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    timestep_fs = float(raw["timestep_fs"])
    origin = int(raw["time_origin_step"])
    sampling_stride = int(raw.get("sampling_stride_steps", 20000))
    cutoffs = sorted({float(value) for value in raw.get("oh_cutoff_sensitivity_A", [1.35])})
    primary_cutoff = float(raw.get("oh_cutoff_A", 1.35))
    if primary_cutoff not in cutoffs:
        cutoffs.append(primary_cutoff)
        cutoffs.sort()
    ch_cutoff = float(raw.get("ch_cutoff_A", 1.25))
    periodic_z = raw.get("periodic_z", False)
    if not isinstance(periodic_z, bool):
        raise ValueError("periodic_z must be a boolean")
    minimum_persistence = float(raw.get("minimum_persistence_ps", 20.0))
    base = contract_path.parent
    rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = [
        {
            "path": str(contract_path),
            "size_bytes": contract_path.stat().st_size,
            "sha256": sha256(contract_path),
        }
    ]

    for entry in raw["cases"]:
        case_id = str(entry["case_id"])
        branch_id = str(entry["branch_id"])
        direction = str(entry["direction"])
        model_data = resolve_path(entry["model_data"], base)
        solution_ids_path = resolve_path(entry["solution_oxygen_ids"], base)
        paths = [resolve_path(path, base) for path in entry["trajectories"]]
        maximum_timestep = int(entry["maximum_timestep"])
        type_symbols = read_type_symbols(model_data)
        fixed_carbon_hydrogen_ids = identify_fixed_carbon_hydrogen_ids(
            model_data,
            type_symbols,
            ch_cutoff,
            periodic_z=periodic_z,
        )
        solution_ids = _read_ids(solution_ids_path)
        for path in [model_data, solution_ids_path, *paths]:
            inputs.append(
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            )

        branch_rows: list[dict[str, object]] = []
        branch_sensitivity: list[dict[str, object]] = []
        steps: list[int] = []
        times: list[float] = []
        assignments: list[np.ndarray] = []
        reference_hydrogen_ids: np.ndarray | None = None
        previous_step: int | None = None
        for path in paths:
            for frame in iter_lammps_dump_records_until(path, maximum_timestep):
                if previous_step is not None and frame.timestep == previous_step:
                    continue
                if previous_step is not None and frame.timestep < previous_step:
                    raise ValueError(f"Non-increasing timestep in {path}")
                previous_step = frame.timestep
                if (frame.timestep - origin) % sampling_stride != 0:
                    continue
                _validate_box_boundary(frame, periodic_z=periodic_z)
                ids, types, coordinates = _frame_arrays(frame)
                primary_assignment: np.ndarray | None = None
                hydrogen_ids: np.ndarray | None = None
                for cutoff in cutoffs:
                    h_ids, oxygen_ids, parent_ids, assigned_any = assign_hydrogen_parents(
                        ids,
                        types,
                        coordinates,
                        frame.bounds,
                        type_symbols,
                        cutoff,
                        fixed_carbon_hydrogen_ids,
                        periodic_z=periodic_z,
                    )
                    metrics = species_metrics(
                        oxygen_ids,
                        parent_ids,
                        assigned_any,
                        solution_ids,
                    )
                    item = {
                        "case_id": case_id,
                        "branch_id": branch_id,
                        "direction": direction,
                        "step": frame.timestep,
                        "time_ps": (frame.timestep - origin) * timestep_fs / 1000.0,
                        "oh_cutoff_A": cutoff,
                        **metrics,
                    }
                    item["proton_partition_pool"] = int(item["solution_H3O"]) + int(
                        item["framework_OH"]
                    )
                    branch_sensitivity.append(item)
                    if math.isclose(cutoff, primary_cutoff):
                        primary_assignment = parent_ids
                        hydrogen_ids = h_ids
                        branch_rows.append(item)
                assert primary_assignment is not None and hydrogen_ids is not None
                if reference_hydrogen_ids is None:
                    reference_hydrogen_ids = hydrogen_ids.copy()
                elif not np.array_equal(reference_hydrogen_ids, hydrogen_ids):
                    raise ValueError(f"Hydrogen identity changed in {case_id}/{branch_id}")
                steps.append(frame.timestep)
                times.append((frame.timestep - origin) * timestep_fs / 1000.0)
                assignments.append(primary_assignment.copy())
            if previous_step == maximum_timestep:
                break
        if not branch_rows or branch_rows[-1]["step"] != maximum_timestep:
            raise ValueError(
                f"{case_id}/{branch_id} did not reach accepted step {maximum_timestep}"
            )
        assert reference_hydrogen_ids is not None
        events = _assignment_events(
            case_id=case_id,
            branch_id=branch_id,
            direction=direction,
            steps=steps,
            times=times,
            hydrogen_ids=reference_hydrogen_ids,
            assignments=assignments,
            solution_oxygen_ids=solution_ids,
            minimum_persistence_ps=minimum_persistence,
        )
        rows.extend(branch_rows)
        sensitivity_rows.extend(branch_sensitivity)
        event_rows.extend(events)

        h3o = np.asarray([int(row["solution_H3O"]) for row in branch_rows])
        framework_oh = np.asarray([int(row["framework_OH"]) for row in branch_rows])
        proton_pool = h3o + framework_oh
        terminal = branch_rows[-1]
        last_50 = h3o[np.asarray(times) >= times[-1] - 50.0]
        spread_by_step: dict[int, list[int]] = defaultdict(list)
        for row in branch_sensitivity:
            spread_by_step[int(row["step"])].append(int(row["solution_H3O"]))
        cutoff_spread = max(max(values) - min(values) for values in spread_by_step.values())
        inventory_pass = (
            int(terminal["solution_oxygen_total"]) == len(solution_ids)
            and int(terminal["solution_O"]) == 0
            and int(terminal["solution_OH"]) == 0
            and int(terminal["solution_overcoordinated"]) == 0
            and int(terminal["hydrogen_unassigned"]) == 0
            and int(proton_pool[-1]) == int(proton_pool[0])
        )
        last_range = int(np.max(last_50) - np.min(last_50))
        summary_rows.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "first_step": steps[0],
                "last_step": steps[-1],
                "frames": len(steps),
                "duration_ps": times[-1] - times[0],
                "sampling_interval_ps": sampling_stride * timestep_fs / 1000.0,
                "solution_oxygen_total": len(solution_ids),
                "initial_H3O": int(h3o[0]),
                "terminal_H3O": int(h3o[-1]),
                "H3O_delta": int(h3o[-1] - h3o[0]),
                "H3O_min": int(np.min(h3o)),
                "H3O_max": int(np.max(h3o)),
                "H3O_last_50ps_range": last_range,
                "initial_framework_OH": int(framework_oh[0]),
                "terminal_framework_OH": int(framework_oh[-1]),
                "initial_proton_pool": int(proton_pool[0]),
                "terminal_proton_pool": int(proton_pool[-1]),
                "proton_pool_range": int(np.max(proton_pool) - np.min(proton_pool)),
                "terminal_solution_O": terminal["solution_O"],
                "terminal_solution_OH": terminal["solution_OH"],
                "terminal_overcoordinated": terminal["solution_overcoordinated"],
                "terminal_unassigned_H": terminal["hydrogen_unassigned"],
                "fixed_carbon_H": len(fixed_carbon_hydrogen_ids),
                "partition_events": len(events),
                "persistent_partition_events": sum(row["persistent"] == "true" for row in events),
                "cutoff_H3O_max_spread": cutoff_spread,
                "inventory_integrity_gate": "PASS" if inventory_pass else "FAIL",
                "proton_partition_stationarity": (
                    "NOT_ASSESSED_SHORT_WINDOW"
                    if times[-1] - times[0] < 50.0
                    else (
                        "STABLE_LAST_50PS_CANDIDATE"
                        if last_range <= 3
                        else "VARIABLE_LAST_50PS"
                    )
                ),
            }
        )

    output.mkdir(parents=True)
    write_tsv(output / "species_timeseries_10ps.tsv", rows, TIMESERIES_FIELDS)
    write_tsv(
        output / "species_cutoff_sensitivity.tsv",
        sensitivity_rows,
        SENSITIVITY_FIELDS,
    )
    write_tsv(output / "proton_partition_events.tsv", event_rows, EVENT_FIELDS)
    write_tsv(output / "branch_species_summary.tsv", summary_rows, SUMMARY_FIELDS)
    unique_inputs = {str(row["path"]): row for row in inputs}
    write_tsv(
        output / "input_manifest.tsv",
        list(unique_inputs.values()),
        ("path", "size_bytes", "sha256"),
    )
    if bool(raw.get("write_plots", True)):
        _plot(rows, output)
    summary = {
        "status": "PASS"
        if all(row["inventory_integrity_gate"] == "PASS" for row in summary_rows)
        else "FAIL",
        "case_branches": len(summary_rows),
        "timeseries_rows": len(rows),
        "cutoff_sensitivity_rows": len(sensitivity_rows),
        "partition_events": len(event_rows),
        "persistent_partition_events": sum(row["persistent"] == "true" for row in event_rows),
        "species_definition": "nearest-parent geometric proxy, not formal charge",
        "periodic_z": periodic_z,
        "scientific_limit": "descriptive time series; proton identity is cutoff sensitive",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        "# Constant-force species and proton-partition time series\n\n"
        "Species labels are nearest-parent geometric proxies, not formal charges. "
        "The inventory gate checks the accepted endpoint independently from the "
        "descriptive proton-partition stationarity label. "
        f"O-H assignment uses periodic x/y and {'periodic' if periodic_z else 'open'} z.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary
