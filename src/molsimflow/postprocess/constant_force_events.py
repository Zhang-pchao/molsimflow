"""Audit reactive-species and high-z events in constant-force interface trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from molsimflow.io.lammps_dump import (
    LammpsDumpFrame,
    box_lengths,
    iter_lammps_dump_records,
    minimum_image_vectors,
)
from molsimflow.postprocess.local_water_order import (
    local_structure_index,
    tetrahedral_order,
    water_hbond_edges,
)
from molsimflow.postprocess.species_assignment import (
    OxygenHydrogenAssignment,
    assign_hydrogen_to_nearest_oxygen,
)


@dataclass(frozen=True)
class EventEpisode:
    """One merged event episode and its extraction window."""

    event_id: str
    case_id: str
    branch_id: str
    anchor_time_ps: float
    anchor_step: int
    start_time_ps: float
    end_time_ps: float
    event_types: tuple[str, ...]
    sample_count: int
    tracked_oxygen_ids: tuple[int, ...]


EVENT_FIELDS = (
    "event_id",
    "case_id",
    "branch_id",
    "event_types",
    "anchor_time_ps",
    "anchor_step",
    "window_start_ps",
    "window_end_ps",
    "sample_count",
    "tracked_oxygen_ids",
    "state_frames",
    "terminal_O_solution",
    "terminal_OH_solution",
    "terminal_OH4plus_solution",
    "terminal_unassigned_H",
    "proton_pool_first",
    "proton_pool_last",
    "species_returned",
    "tracked_frames",
    "tracked_identity_complete",
    "tracked_h_counts",
    "tracked_shared_hydrogen_frames",
    "tracked_min_sharing_delta_A",
    "tracked_mean_q_tet",
    "tracked_mean_lsi_A2",
    "tracked_mean_water_hbond_degree",
    "tracked_mean_surface_hbond_count",
    "tracked_intact_water",
    "tracked_max_z_A",
    "tracked_returned_below_high_z",
    "nonzero_iz_max",
)

EVENT_SOURCE_FIELDS = (
    "event_id",
    "case_id",
    "branch_id",
    "event_type",
    "time_ps",
    "step",
    "oxygen_id",
    "severity",
    "O_solution",
    "OH_solution",
    "OH4plus_solution",
    "unassigned_H",
    "max_z_A",
    "image_z",
    "top_clearance_A",
)

FRAME_FIELDS = (
    "event_id",
    "case_id",
    "branch_id",
    "event_types",
    "anchor_time_ps",
    "anchor_step",
    "time_ps",
    "step",
    "relative_time_ps",
    "solution_O_total",
    "O_solution",
    "OH_solution",
    "H2O_solution",
    "H3O_solution",
    "OH4plus_solution",
    "framework_OH",
    "proton_pool",
    "carbon_owned_H",
    "unassigned_H",
    "max_z_oxygen_id",
    "max_z_A",
    "max_z_h_count",
    "nonzero_iz",
)

ATOM_FIELDS = (
    "event_id",
    "case_id",
    "branch_id",
    "event_types",
    "anchor_time_ps",
    "anchor_step",
    "time_ps",
    "step",
    "relative_time_ps",
    "oxygen_id",
    "is_solution",
    "h_count",
    "hydrogen_ids",
    "hydrogen_distances_A",
    "hydrogen_second_oxygen_ids",
    "hydrogen_second_distances_A",
    "hydrogen_sharing_deltas_A",
    "shared_hydrogen_count",
    "minimum_sharing_delta_A",
    "q_tet",
    "lsi_A2",
    "oo_coordination",
    "water_hbond_donor_count",
    "water_hbond_acceptor_count",
    "water_hbond_degree",
    "water_donor_surface_hbond_count",
    "surface_donor_water_hbond_count",
    "x_A",
    "y_A",
    "z_A",
    "distance_to_si_A",
    "image_z",
)

MOTION_FIELDS = (
    "event_id",
    "case_id",
    "branch_id",
    "event_types",
    "anchor_time_ps",
    "anchor_step",
    "samples",
    "minimum_top_clearance_A",
    "wall_samples",
    "dx_window_A",
    "dy_window_A",
    "vx_pre_mps",
    "vx_core_mps",
    "vx_post_mps",
    "vy_pre_mps",
    "vy_core_mps",
    "vy_post_mps",
)

SPECIES_COMPARISON_FIELDS = (
    "O_solution",
    "OH_solution",
    "H2O_solution",
    "H3O_solution",
    "OH4plus_solution",
    "framework_OH",
    "framework_OH2plus",
    "unassigned_H",
    "detached_C",
    "solution_charge_proxy",
)


def _float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: object, default: int = -1) -> int:
    number = _float(value)
    return int(round(number)) if math.isfinite(number) else default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_delimited(path: Path, delimiter: Optional[str] = None) -> list[dict[str, str]]:
    if delimiter is None:
        delimiter = "\t" if Path(path).suffix.lower() in {".tsv", ".tab"} else ","
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Table has no header: {path}")
        return [dict(row) for row in reader]


def _write_tsv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    preferred = [
        "event_id",
        "case_id",
        "branch_id",
        "event_types",
        "anchor_time_ps",
        "anchor_step",
        "time_ps",
        "step",
        "relative_time_ps",
        "oxygen_id",
    ]
    fields = list(fieldnames or ())
    fields.extend(key for key in preferred if key in keys and key not in fields)
    fields.extend(key for key in keys if key not in fields)
    if not fields:
        raise ValueError(f"Cannot write a headerless empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_fix_ave_table(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a LAMMPS fix-ave/time style table with a named second header line."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    headers = [line for line in lines if line.startswith("#")]
    if len(headers) < 2:
        raise ValueError(f"Expected two comment headers in {path}")
    columns = headers[1].lstrip("# ").split()
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(columns):
        raise ValueError(f"Column mismatch in {path}: {data.shape[1]} != {len(columns)}")
    return columns, data


def stitch_motion_tables(
    paths: Sequence[Path],
    *,
    step_column: str = "TimeStep",
    displacement_columns: Sequence[str] = ("v_dxrel", "v_dyrel"),
) -> tuple[list[str], np.ndarray]:
    """Join restart segments and offset displacement columns at shared endpoints."""

    if not paths:
        raise ValueError("At least one motion table is required")
    all_rows: list[np.ndarray] = []
    columns: Optional[list[str]] = None
    previous_last: Optional[np.ndarray] = None
    for path in paths:
        current_columns, data = read_fix_ave_table(path)
        if columns is None:
            columns = current_columns
        elif current_columns != columns:
            raise ValueError(f"Motion columns differ in {path}")
        index = {name: position for position, name in enumerate(columns)}
        if step_column not in index:
            raise ValueError(f"Missing {step_column} in {path}")
        if previous_last is not None:
            shared = int(round(data[0, index[step_column]])) == int(
                round(previous_last[index[step_column]])
            )
            for name in displacement_columns:
                if name not in index:
                    raise ValueError(f"Missing displacement column {name} in {path}")
                data[:, index[name]] += previous_last[index[name]] - data[0, index[name]]
            if shared:
                data = data[1:]
        if len(data):
            all_rows.append(data)
            previous_last = data[-1]
    assert columns is not None
    if not all_rows:
        raise ValueError("Motion tables contain no rows")
    combined = np.vstack(all_rows)
    step_index = columns.index(step_column)
    if np.any(np.diff(combined[:, step_index]) <= 0):
        raise ValueError("Stitched motion timesteps are not strictly increasing")
    return columns, combined


def _frame_arrays(
    frame: LammpsDumpFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    fields = frame.atom_fields
    required = ("id", "type", "x", "y", "z")
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"Trajectory is missing fields {missing} at step {frame.timestep}")
    column = {name: fields.index(name) for name in fields}
    ids = np.asarray([int(row[column["id"]]) for row in frame.atom_rows], dtype=np.int64)
    types = np.asarray([int(row[column["type"]]) for row in frame.atom_rows], dtype=np.int64)
    coordinates = np.asarray(
        [[float(row[column[name]]) for name in ("x", "y", "z")] for row in frame.atom_rows],
        dtype=float,
    )
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"Duplicate atom IDs at step {frame.timestep}")
    return ids, types, coordinates, column


def _minimum_distance(
    source: np.ndarray,
    target: np.ndarray,
    bounds: np.ndarray,
    *,
    periodic: Sequence[bool] = (True, True, False),
    chunk_size: int = 256,
) -> np.ndarray:
    if len(target) == 0:
        return np.full(len(source), np.inf)
    lengths = bounds[:, 1] - bounds[:, 0]
    mask = np.asarray(periodic, dtype=bool)
    result = np.full(len(source), np.inf)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        delta = source[start:stop, None, :] - target[None, :, :]
        delta[..., mask] -= lengths[mask] * np.round(delta[..., mask] / lengths[mask])
        result[start:stop] = np.sqrt(np.min(np.einsum("ijk,ijk->ij", delta, delta), axis=1))
    return result


def _xy_nonperiodic_z_tree_coordinates(
    coordinates: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed periodic X/Y and nonperiodic Z coordinates in a cKDTree box."""

    lengths = box_lengths(bounds)
    pseudo_z = max(1.0e5, 10.0 * lengths[2])
    shifted = np.asarray(coordinates, dtype=float).copy()
    shifted[:, 0] = (shifted[:, 0] - bounds[0, 0]) % lengths[0]
    shifted[:, 1] = (shifted[:, 1] - bounds[1, 0]) % lengths[1]
    shifted[:, 2] = shifted[:, 2] - bounds[2, 0] + 0.25 * pseudo_z
    return shifted, np.asarray([lengths[0], lengths[1], pseudo_z])


def nearest_two_oxygen(
    oxygen: np.ndarray,
    hydrogen: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two nearest oxygen indices and distances with periodic X/Y only."""

    from scipy.spatial import cKDTree

    if len(oxygen) < 2:
        raise ValueError("At least two oxygen atoms are required")
    oxygen_tree, tree_box = _xy_nonperiodic_z_tree_coordinates(oxygen, bounds)
    hydrogen_tree, _ = _xy_nonperiodic_z_tree_coordinates(hydrogen, bounds)
    distances, indices = cKDTree(oxygen_tree, boxsize=tree_box).query(hydrogen_tree, k=2)
    return np.asarray(indices, dtype=int), np.asarray(distances, dtype=float)


def _donates(oh_vectors: np.ndarray, donor_acceptor: np.ndarray, angle_deg: float) -> bool:
    distance = float(np.linalg.norm(donor_acceptor))
    if distance <= 0.0 or not len(oh_vectors):
        return False
    norms = np.linalg.norm(oh_vectors, axis=1)
    valid = norms > 0.0
    if not np.any(valid):
        return False
    cosine = (oh_vectors[valid] / norms[valid, None]) @ (donor_acceptor / distance)
    return bool(np.any(cosine >= math.cos(math.radians(angle_deg))))


def _water_order(
    oxygen: np.ndarray,
    bounds: np.ndarray,
    *,
    oo_cutoff_A: float,
    lsi_cutoff_A: float,
    lsi_neighbor_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q_tet, LSI, and O-O coordination for solution oxygen atoms."""

    from scipy.spatial import cKDTree

    shifted, tree_box = _xy_nonperiodic_z_tree_coordinates(oxygen, bounds)
    k = min(lsi_neighbor_cap + 2, len(oxygen))
    distances, neighbors = cKDTree(shifted, boxsize=tree_box).query(shifted, k=k)
    if len(oxygen) == 1:
        distances = distances.reshape((1, -1))
        neighbors = neighbors.reshape((1, -1))
    lengths = box_lengths(bounds)
    qtet = np.full(len(oxygen), np.nan)
    lsi = np.full(len(oxygen), np.nan)
    coordination = np.zeros(len(oxygen), dtype=int)
    for oxygen_index in range(len(oxygen)):
        ordered = sorted(
            (float(distance), int(neighbor))
            for distance, neighbor in zip(distances[oxygen_index], neighbors[oxygen_index])
            if int(neighbor) != oxygen_index and math.isfinite(float(distance))
        )
        ordered_distances = np.asarray([item[0] for item in ordered])
        coordination[oxygen_index] = int(np.count_nonzero(ordered_distances <= oo_cutoff_A))
        if coordination[oxygen_index] >= 4:
            nearest = np.asarray([item[1] for item in ordered[:4]], dtype=int)
            vectors = oxygen[nearest] - oxygen[oxygen_index]
            vectors[:, :2] = minimum_image_vectors(vectors[:, :2], lengths[:2])
            qtet[oxygen_index] = tetrahedral_order(vectors)
        lsi[oxygen_index] = local_structure_index(ordered_distances, lsi_cutoff_A)[0]
    return qtet, lsi, coordination


def _detect_species_samples(
    rows: Sequence[Mapping[str, object]],
    *,
    time_column: str,
    step_column: str,
) -> list[dict[str, object]]:
    columns = ("O_solution", "OH_solution", "OH4plus_solution", "unassigned_H")
    samples: list[dict[str, object]] = []
    for row in rows:
        time_ps = _float(row.get(time_column))
        step = _int(row.get(step_column))
        if not math.isfinite(time_ps) or step < 0:
            raise ValueError(f"Invalid species time/step: time={time_ps!r}, step={step!r}")
        values = {name: _int(row.get(name), 0) for name in columns}
        if any(value < 0 for value in values.values()):
            raise ValueError(f"Negative species count at step {step}")
        if max(values.values(), default=0) <= 0:
            continue
        severity = sum(values.values())
        samples.append(
            {
                "time_ps": time_ps,
                "step": step,
                "event_type": "species_geometry",
                "severity": float(severity),
                "oxygen_id": -1,
                **values,
            }
        )
    return samples


def _detect_high_z_samples(
    paths: Sequence[Path],
    *,
    time_origin_step: int,
    timestep_fs: float,
    oxygen_type: int,
    high_z_threshold_A: float,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    seen_steps: set[int] = set()
    for path in paths:
        for frame in iter_lammps_dump_records(path):
            if frame.timestep in seen_steps:
                continue
            seen_steps.add(frame.timestep)
            fields = frame.atom_fields
            for name in ("id", "type", "z"):
                if name not in fields:
                    raise ValueError(f"{path}: missing {name}")
            index = {name: fields.index(name) for name in fields}
            oxygen = [row for row in frame.atom_rows if int(row[index["type"]]) == oxygen_type]
            if not oxygen:
                raise ValueError(f"{path}: no oxygen atoms at step {frame.timestep}")
            time_ps = (frame.timestep - time_origin_step) * timestep_fs / 1000.0
            for row in oxygen:
                atom_id = int(row[index["id"]])
                z_A = float(row[index["z"]])
                image_z = int(row[index["iz"]]) if "iz" in index else 0
                if z_A >= high_z_threshold_A:
                    samples.append(
                        {
                            "time_ps": time_ps,
                            "step": frame.timestep,
                            "event_type": "high_z",
                            "severity": z_A / high_z_threshold_A,
                            "oxygen_id": atom_id,
                            "max_z_A": z_A,
                            "image_z": image_z,
                        }
                    )
                if image_z != 0:
                    samples.append(
                        {
                            "time_ps": time_ps,
                            "step": frame.timestep,
                            "event_type": "z_image",
                            "severity": float(abs(image_z)),
                            "oxygen_id": atom_id,
                            "max_z_A": z_A,
                            "image_z": image_z,
                        }
                    )
    return samples


def _detect_wall_samples(
    columns: Sequence[str],
    data: np.ndarray,
    *,
    time_origin_step: int,
    timestep_fs: float,
    step_column: str,
    clearance_column: str,
    wall_clearance_threshold_A: float,
) -> list[dict[str, object]]:
    index = {name: position for position, name in enumerate(columns)}
    for name in (step_column, clearance_column):
        if name not in index:
            raise ValueError(f"Motion table is missing {name}")
    result = []
    for row in data:
        clearance = float(row[index[clearance_column]])
        if clearance > wall_clearance_threshold_A:
            continue
        step = int(round(row[index[step_column]]))
        result.append(
            {
                "time_ps": (step - time_origin_step) * timestep_fs / 1000.0,
                "step": step,
                "event_type": "wall_approach",
                "severity": wall_clearance_threshold_A / max(clearance, 1.0e-12),
                "oxygen_id": -1,
                "top_clearance_A": clearance,
            }
        )
    return result


def merge_event_samples(
    samples: Sequence[Mapping[str, object]],
    *,
    case_id: str,
    branch_id: str,
    merge_gap_ps: float,
    window_ps: float,
) -> tuple[list[EventEpisode], list[dict[str, object]]]:
    """Merge nearby event samples while retaining every source observation."""

    ordered = sorted(
        samples,
        key=lambda row: (_float(row.get("time_ps")), str(row.get("event_type"))),
    )
    if any(not math.isfinite(_float(row.get("time_ps"))) for row in ordered):
        raise ValueError(f"Non-finite event time for {case_id}/{branch_id}")
    if any(_int(row.get("step")) < 0 for row in ordered):
        raise ValueError(f"Missing or negative event step for {case_id}/{branch_id}")
    groups: list[list[Mapping[str, object]]] = []
    for sample in ordered:
        if not groups:
            groups.append([sample])
            continue
        separation = _float(sample["time_ps"]) - _float(groups[-1][-1]["time_ps"])
        if separation > merge_gap_ps:
            groups.append([sample])
        else:
            groups[-1].append(sample)
    episodes: list[EventEpisode] = []
    source_rows: list[dict[str, object]] = []
    for number, group in enumerate(groups, start=1):
        anchor = max(group, key=lambda row: _float(row.get("severity"), 0.0))
        event_id = f"{case_id}__{branch_id}__e{number:04d}"
        times = [_float(row["time_ps"]) for row in group]
        tracked = sorted(
            {
                _int(row.get("oxygen_id"))
                for row in group
                if _int(row.get("oxygen_id")) > 0
            }
        )
        episode = EventEpisode(
            event_id=event_id,
            case_id=case_id,
            branch_id=branch_id,
            anchor_time_ps=_float(anchor["time_ps"]),
            anchor_step=_int(anchor.get("step")),
            start_time_ps=min(times) - window_ps,
            end_time_ps=max(times) + window_ps,
            event_types=tuple(sorted({str(row["event_type"]) for row in group})),
            sample_count=len(group),
            tracked_oxygen_ids=tuple(tracked),
        )
        episodes.append(episode)
        for sample in group:
            source_rows.append(
                {
                    "event_id": event_id,
                    "case_id": case_id,
                    "branch_id": branch_id,
                    **sample,
                }
            )
    return episodes, source_rows


def _event_rows(episodes: Sequence[EventEpisode]) -> list[dict[str, object]]:
    return [
        {
            "event_id": event.event_id,
            "case_id": event.case_id,
            "branch_id": event.branch_id,
            "event_types": ",".join(event.event_types),
            "anchor_time_ps": event.anchor_time_ps,
            "anchor_step": event.anchor_step,
            "window_start_ps": event.start_time_ps,
            "window_end_ps": event.end_time_ps,
            "sample_count": event.sample_count,
            "tracked_oxygen_ids": ",".join(map(str, event.tracked_oxygen_ids)),
        }
        for event in episodes
    ]


def _frame_species(
    frame: LammpsDumpFrame,
    *,
    hydrogen_type: int,
    oxygen_type: int,
    silicon_type: int,
    carbon_type: Optional[int],
    oh_cutoff_A: float,
    ch_cutoff_A: Optional[float],
    sio_cutoff_A: float,
    oo_cutoff_A: float,
    hbond_angle_deg: float,
    lsi_cutoff_A: float,
    lsi_neighbor_cap: int,
    sharing_delta_threshold_A: float,
) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    ids, types, coordinates, column = _frame_arrays(frame)
    oxygen_indices = np.flatnonzero(types == oxygen_type)
    hydrogen_indices = np.flatnonzero(types == hydrogen_type)
    silicon_indices = np.flatnonzero(types == silicon_type)
    carbon_indices = (
        np.flatnonzero(types == carbon_type)
        if carbon_type is not None
        else np.zeros(0, dtype=int)
    )
    if oxygen_indices.size == 0:
        raise ValueError(f"No oxygen atoms at step {frame.timestep}")
    if silicon_indices.size == 0:
        raise ValueError(f"No silicon atoms at step {frame.timestep}")
    assignment = assign_hydrogen_to_nearest_oxygen(
        coordinates[oxygen_indices],
        coordinates[hydrogen_indices],
        frame.bounds,
        oh_cutoff=oh_cutoff_A,
        periodic=(True, True, False),
    )
    carbon_owned = np.zeros(len(hydrogen_indices), dtype=bool)
    if carbon_type is not None:
        if ch_cutoff_A is None or not math.isfinite(ch_cutoff_A) or ch_cutoff_A <= 0.0:
            raise ValueError("ch_cutoff_A must be positive when carbon_type is configured")
        if carbon_indices.size:
            carbon_assignment = assign_hydrogen_to_nearest_oxygen(
                coordinates[carbon_indices],
                coordinates[hydrogen_indices],
                frame.bounds,
                oh_cutoff=ch_cutoff_A,
                periodic=(True, True, False),
            )
            oxygen_valid = assignment.hydrogen_to_oxygen_index >= 0
            carbon_valid = carbon_assignment.hydrogen_to_oxygen_index >= 0
            carbon_owned = carbon_valid & (
                ~oxygen_valid
                | (carbon_assignment.hydrogen_distance < assignment.hydrogen_distance)
            )
            oxygen_owner = assignment.hydrogen_to_oxygen_index.copy()
            oxygen_distance = assignment.hydrogen_distance.copy()
            oxygen_owner[carbon_owned] = -1
            oxygen_distance[carbon_owned] = np.inf
            assigned_to_oxygen = oxygen_owner >= 0
            assignment = OxygenHydrogenAssignment(
                h_count_per_oxygen=np.bincount(
                    oxygen_owner[assigned_to_oxygen],
                    minlength=len(oxygen_indices),
                ).astype(int),
                hydrogen_to_oxygen_index=oxygen_owner,
                hydrogen_distance=oxygen_distance,
            )
    distance_to_si = _minimum_distance(
        coordinates[oxygen_indices], coordinates[silicon_indices], frame.bounds
    )
    solution = distance_to_si > sio_cutoff_A
    if not np.any(solution):
        raise ValueError(f"No solution oxygen atoms at step {frame.timestep}")
    counts = assignment.h_count_per_oxygen
    solution_counts = counts[solution]
    framework_counts = counts[~solution]
    grouped_h = assignment.hydrogen_indices_by_oxygen
    oxygen_coordinates = coordinates[oxygen_indices]
    hydrogen_coordinates = coordinates[hydrogen_indices]
    nearest_two_indices, nearest_two_distances = nearest_two_oxygen(
        oxygen_coordinates, hydrogen_coordinates, frame.bounds
    )
    lengths = box_lengths(frame.bounds)
    oh_vectors_by_oxygen: list[np.ndarray] = []
    for local_o, global_o in enumerate(oxygen_indices):
        local_h = grouped_h.get(local_o, [])
        vectors = hydrogen_coordinates[local_h] - coordinates[global_o]
        if len(vectors):
            vectors[:, :2] = minimum_image_vectors(vectors[:, :2], lengths[:2])
        oh_vectors_by_oxygen.append(vectors)

    solution_local = np.flatnonzero(solution)
    framework_local = np.flatnonzero(~solution)
    water_coordinates = oxygen_coordinates[solution_local]
    water_oh = [oh_vectors_by_oxygen[int(index)] for index in solution_local]
    qtet, lsi, coordination = _water_order(
        water_coordinates,
        frame.bounds,
        oo_cutoff_A=oo_cutoff_A,
        lsi_cutoff_A=lsi_cutoff_A,
        lsi_neighbor_cap=lsi_neighbor_cap,
    )
    water_donor = np.zeros(len(solution_local), dtype=int)
    water_acceptor = np.zeros(len(solution_local), dtype=int)
    for donor, acceptor in water_hbond_edges(
        water_coordinates,
        water_oh,
        np.arange(len(solution_local), dtype=int),
        frame.bounds,
        oo_cutoff_A=oo_cutoff_A,
        angle_cutoff_deg=hbond_angle_deg,
    ):
        water_donor[donor] += 1
        water_acceptor[acceptor] += 1

    framework_coordinates = oxygen_coordinates[framework_local]
    framework_oh = [oh_vectors_by_oxygen[int(index)] for index in framework_local]
    water_surface_donor = np.zeros(len(solution_local), dtype=int)
    surface_water_donor = np.zeros(len(solution_local), dtype=int)
    if len(framework_coordinates):
        from scipy.spatial import cKDTree

        shifted_surface, tree_box = _xy_nonperiodic_z_tree_coordinates(
            framework_coordinates, frame.bounds
        )
        shifted_water, _ = _xy_nonperiodic_z_tree_coordinates(
            water_coordinates, frame.bounds
        )
        surface_tree = cKDTree(shifted_surface, boxsize=tree_box)
        for water_index, neighbors in enumerate(
            surface_tree.query_ball_point(shifted_water, oo_cutoff_A)
        ):
            for framework_index in neighbors:
                vector = framework_coordinates[framework_index] - water_coordinates[water_index]
                vector[:2] = minimum_image_vectors(vector[:2], lengths[:2])
                water_surface_donor[water_index] += _donates(
                    water_oh[water_index], vector, hbond_angle_deg
                )
                surface_water_donor[water_index] += _donates(
                    framework_oh[framework_index], -vector, hbond_angle_deg
                )
    water_index_by_oxygen = {
        int(oxygen_local): water_index
        for water_index, oxygen_local in enumerate(solution_local)
    }
    detail: dict[int, dict[str, object]] = {}
    for local_o, global_o in enumerate(oxygen_indices):
        hydrogen_local = grouped_h.get(local_o, [])
        hydrogen_global = [int(hydrogen_indices[index]) for index in hydrogen_local]
        second_oxygen_ids: list[int] = []
        second_distances: list[float] = []
        sharing_deltas: list[float] = []
        for hydrogen_index in hydrogen_local:
            nearest = nearest_two_indices[hydrogen_index]
            distances = nearest_two_distances[hydrogen_index]
            other_position = 1 if int(nearest[0]) == local_o else 0
            other_local = int(nearest[other_position])
            other_distance = float(distances[other_position])
            second_oxygen_ids.append(int(ids[oxygen_indices[other_local]]))
            second_distances.append(other_distance)
            sharing_deltas.append(
                other_distance - float(assignment.hydrogen_distance[hydrogen_index])
            )
        shared_count = sum(
            distance <= oh_cutoff_A and delta <= sharing_delta_threshold_A
            for distance, delta in zip(second_distances, sharing_deltas)
        )
        atom_id = int(ids[global_o])
        water_index = water_index_by_oxygen.get(local_o)
        detail[atom_id] = {
            "oxygen_id": atom_id,
            "is_solution": bool(solution[local_o]),
            "h_count": int(counts[local_o]),
            "hydrogen_ids": ",".join(str(int(ids[index])) for index in hydrogen_global),
            "hydrogen_distances_A": ",".join(
                f"{float(assignment.hydrogen_distance[index]):.8g}" for index in hydrogen_local
            ),
            "hydrogen_second_oxygen_ids": ",".join(map(str, second_oxygen_ids)),
            "hydrogen_second_distances_A": ",".join(
                f"{distance:.8g}" for distance in second_distances
            ),
            "hydrogen_sharing_deltas_A": ",".join(
                f"{delta:.8g}" for delta in sharing_deltas
            ),
            "shared_hydrogen_count": shared_count,
            "minimum_sharing_delta_A": min(sharing_deltas, default=math.nan),
            "q_tet": qtet[water_index] if water_index is not None else math.nan,
            "lsi_A2": lsi[water_index] if water_index is not None else math.nan,
            "oo_coordination": coordination[water_index] if water_index is not None else -1,
            "water_hbond_donor_count": (
                water_donor[water_index] if water_index is not None else 0
            ),
            "water_hbond_acceptor_count": (
                water_acceptor[water_index] if water_index is not None else 0
            ),
            "water_hbond_degree": (
                water_donor[water_index] + water_acceptor[water_index]
                if water_index is not None
                else 0
            ),
            "water_donor_surface_hbond_count": (
                water_surface_donor[water_index] if water_index is not None else 0
            ),
            "surface_donor_water_hbond_count": (
                surface_water_donor[water_index] if water_index is not None else 0
            ),
            "x_A": float(coordinates[global_o, 0]),
            "y_A": float(coordinates[global_o, 1]),
            "z_A": float(coordinates[global_o, 2]),
            "distance_to_si_A": float(distance_to_si[local_o]),
            "image_z": int(frame.atom_rows[global_o][column["iz"]]) if "iz" in column else 0,
        }
    unassigned = int(
        np.count_nonzero(
            (assignment.hydrogen_to_oxygen_index < 0) & ~carbon_owned
        )
    )
    max_solution_local = int(np.argmax(coordinates[oxygen_indices[solution], 2]))
    solution_global = oxygen_indices[solution]
    max_global = int(solution_global[max_solution_local])
    max_id = int(ids[max_global])
    summary = {
        "solution_O_total": int(solution.sum()),
        "O_solution": int(np.count_nonzero(solution_counts == 0)),
        "OH_solution": int(np.count_nonzero(solution_counts == 1)),
        "H2O_solution": int(np.count_nonzero(solution_counts == 2)),
        "H3O_solution": int(np.count_nonzero(solution_counts == 3)),
        "OH4plus_solution": int(np.count_nonzero(solution_counts >= 4)),
        "framework_OH": int(np.count_nonzero(framework_counts == 1)),
        "proton_pool": int(
            np.count_nonzero(solution_counts == 3)
            + np.count_nonzero(framework_counts == 1)
        ),
        "carbon_owned_H": int(np.count_nonzero(carbon_owned)),
        "unassigned_H": unassigned,
        "max_z_oxygen_id": max_id,
        "max_z_A": float(coordinates[max_global, 2]),
        "max_z_h_count": int(detail[max_id]["h_count"]),
        "nonzero_iz": int(
            sum(
                record["image_z"] != 0
                for record in detail.values()
                if record["is_solution"]
            )
        ),
    }
    return summary, detail


def extract_state_windows(
    paths: Sequence[Path],
    episodes: Sequence[EventEpisode],
    *,
    time_origin_step: int,
    timestep_fs: float,
    hydrogen_type: int,
    oxygen_type: int,
    silicon_type: int,
    carbon_type: Optional[int],
    oh_cutoff_A: float,
    ch_cutoff_A: Optional[float],
    sio_cutoff_A: float,
    oo_cutoff_A: float,
    hbond_angle_deg: float,
    lsi_cutoff_A: float,
    lsi_neighbor_cap: int,
    sharing_delta_threshold_A: float,
    high_z_threshold_A: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame_rows: list[dict[str, object]] = []
    atom_rows: list[dict[str, object]] = []
    seen_steps: set[int] = set()
    for path in paths:
        for frame in iter_lammps_dump_records(path):
            if frame.timestep in seen_steps:
                continue
            seen_steps.add(frame.timestep)
            time_ps = (frame.timestep - time_origin_step) * timestep_fs / 1000.0
            matched = [
                event
                for event in episodes
                if event.start_time_ps <= time_ps <= event.end_time_ps
            ]
            if not matched:
                continue
            summary, detail = _frame_species(
                frame,
                hydrogen_type=hydrogen_type,
                oxygen_type=oxygen_type,
                silicon_type=silicon_type,
                carbon_type=carbon_type,
                oh_cutoff_A=oh_cutoff_A,
                ch_cutoff_A=ch_cutoff_A,
                sio_cutoff_A=sio_cutoff_A,
                oo_cutoff_A=oo_cutoff_A,
                hbond_angle_deg=hbond_angle_deg,
                lsi_cutoff_A=lsi_cutoff_A,
                lsi_neighbor_cap=lsi_neighbor_cap,
                sharing_delta_threshold_A=sharing_delta_threshold_A,
            )
            abnormal = {
                atom_id
                for atom_id, record in detail.items()
                if record["is_solution"] and int(record["h_count"]) != 2
            }
            for event in matched:
                base = {
                    "event_id": event.event_id,
                    "case_id": event.case_id,
                    "branch_id": event.branch_id,
                    "event_types": ",".join(event.event_types),
                    "anchor_time_ps": event.anchor_time_ps,
                    "anchor_step": event.anchor_step,
                    "time_ps": time_ps,
                    "step": frame.timestep,
                    "relative_time_ps": time_ps - event.anchor_time_ps,
                }
                frame_rows.append({**base, **summary})
                selected = set(event.tracked_oxygen_ids) | abnormal
                if summary["max_z_A"] >= high_z_threshold_A:
                    selected.add(int(summary["max_z_oxygen_id"]))
                for atom_id in sorted(selected):
                    if atom_id in detail:
                        atom_rows.append({**base, **detail[atom_id]})
    return frame_rows, atom_rows


def _slope(time: np.ndarray, values: np.ndarray, low: float, high: float) -> float:
    mask = (time >= low) & (time <= high)
    if np.count_nonzero(mask) < 2:
        return math.nan
    return float(np.polyfit(time[mask], values[mask], 1)[0] * 100.0)


def summarize_motion_events(
    episodes: Sequence[EventEpisode],
    columns: Sequence[str],
    data: np.ndarray,
    *,
    time_origin_step: int,
    timestep_fs: float,
    step_column: str,
    x_column: str,
    y_column: str,
    clearance_column: str,
    wall_clearance_threshold_A: float,
) -> list[dict[str, object]]:
    index = {name: position for position, name in enumerate(columns)}
    for name in (step_column, x_column, y_column, clearance_column):
        if name not in index:
            raise ValueError(f"Motion data are missing {name}")
    time = (data[:, index[step_column]] - time_origin_step) * timestep_fs / 1000.0
    x = data[:, index[x_column]]
    y = data[:, index[y_column]]
    clearance = data[:, index[clearance_column]]
    rows = []
    for event in episodes:
        relative = time - event.anchor_time_ps
        window = (time >= event.start_time_ps) & (time <= event.end_time_ps)
        if not np.any(window):
            continue
        indices = np.flatnonzero(window)
        rows.append(
            {
                "event_id": event.event_id,
                "case_id": event.case_id,
                "branch_id": event.branch_id,
                "event_types": ",".join(event.event_types),
                "anchor_time_ps": event.anchor_time_ps,
                "anchor_step": event.anchor_step,
                "samples": len(indices),
                "minimum_top_clearance_A": float(np.min(clearance[window])),
                "wall_samples": int(
                    np.count_nonzero(clearance[window] <= wall_clearance_threshold_A)
                ),
                "dx_window_A": float(x[indices[-1]] - x[indices[0]]),
                "dy_window_A": float(y[indices[-1]] - y[indices[0]]),
                "vx_pre_mps": _slope(relative, x, -20.0, -5.0),
                "vx_core_mps": _slope(relative, x, -5.0, 5.0),
                "vx_post_mps": _slope(relative, x, 5.0, 20.0),
                "vy_pre_mps": _slope(relative, y, -20.0, -5.0),
                "vy_core_mps": _slope(relative, y, -5.0, 5.0),
                "vy_post_mps": _slope(relative, y, 5.0, 20.0),
            }
        )
    return rows


def summarize_event_outcomes(
    episodes: Sequence[EventEpisode],
    frame_rows: Sequence[Mapping[str, object]],
    atom_rows: Sequence[Mapping[str, object]],
    *,
    high_z_threshold_A: float,
) -> list[dict[str, object]]:
    results = []
    for event in episodes:
        frames = sorted(
            [row for row in frame_rows if row["event_id"] == event.event_id],
            key=lambda row: _float(row["time_ps"]),
        )
        atoms = [row for row in atom_rows if row["event_id"] == event.event_id]
        tracked = [row for row in atoms if _int(row["oxygen_id"]) in event.tracked_oxygen_ids]
        last = frames[-1] if frames else {}
        first = frames[0] if frames else {}
        tracked_by_id: dict[int, list[Mapping[str, object]]] = defaultdict(list)
        for row in tracked:
            tracked_by_id[_int(row["oxygen_id"])].append(row)
        unique_frame_steps = {_int(row["step"]) for row in frames}
        tracked_complete = bool(event.tracked_oxygen_ids) and all(
            {_int(row["step"]) for row in tracked_by_id.get(atom_id, [])} == unique_frame_steps
            for atom_id in event.tracked_oxygen_ids
        )
        tracked_returned = bool(tracked_by_id) and all(
            _float(max(rows, key=lambda row: _float(row["time_ps"]))["z_A"])
            < high_z_threshold_A
            for rows in tracked_by_id.values()
        )
        species_keys = (
            "O_solution",
            "OH_solution",
            "H3O_solution",
            "OH4plus_solution",
            "unassigned_H",
            "proton_pool",
        )
        species_returned = bool(frames) and all(
            _int(first.get(key)) == _int(last.get(key)) for key in species_keys
        )
        sharing_delta = [
            _float(row.get("minimum_sharing_delta_A"))
            for row in tracked
            if math.isfinite(_float(row.get("minimum_sharing_delta_A")))
        ]

        def tracked_mean(name: str) -> float:
            values = [
                _float(row.get(name))
                for row in tracked
                if math.isfinite(_float(row.get(name)))
            ]
            return float(np.mean(values)) if values else math.nan

        results.append(
            {
                **_event_rows([event])[0],
                "state_frames": len(frames),
                "terminal_O_solution": _int(last.get("O_solution")),
                "terminal_OH_solution": _int(last.get("OH_solution")),
                "terminal_OH4plus_solution": _int(last.get("OH4plus_solution")),
                "terminal_unassigned_H": _int(last.get("unassigned_H")),
                "proton_pool_first": _int(first.get("proton_pool")),
                "proton_pool_last": _int(last.get("proton_pool")),
                "species_returned": species_returned,
                "tracked_frames": len(tracked),
                "tracked_identity_complete": tracked_complete,
                "tracked_h_counts": ",".join(
                    map(str, sorted({_int(row["h_count"]) for row in tracked}))
                ),
                "tracked_shared_hydrogen_frames": sum(
                    _int(row.get("shared_hydrogen_count"), 0) > 0 for row in tracked
                ),
                "tracked_min_sharing_delta_A": min(sharing_delta, default=math.nan),
                "tracked_mean_q_tet": tracked_mean("q_tet"),
                "tracked_mean_lsi_A2": tracked_mean("lsi_A2"),
                "tracked_mean_water_hbond_degree": tracked_mean("water_hbond_degree"),
                "tracked_mean_surface_hbond_count": tracked_mean(
                    "water_donor_surface_hbond_count"
                )
                + tracked_mean("surface_donor_water_hbond_count"),
                "tracked_intact_water": bool(
                    tracked_complete and all(_int(row["h_count"]) == 2 for row in tracked)
                ),
                "tracked_max_z_A": max((_float(row["z_A"]) for row in tracked), default=math.nan),
                "tracked_returned_below_high_z": tracked_returned,
                "nonzero_iz_max": max(
                    (_int(row.get("nonzero_iz"), 0) for row in frames),
                    default=0,
                ),
            }
        )
    return results


def _validate_contract(raw: Mapping[str, object]) -> None:
    schema_version = _int(raw.get("schema_version"))
    if schema_version not in (1, 2):
        raise ValueError("schema_version must be 1 or 2")
    if schema_version >= 2:
        types = dict(raw.get("types", {}))
        cutoffs = dict(raw.get("cutoffs_A", {}))
        if "carbon" not in types or "ch" not in cutoffs:
            raise ValueError("schema_version 2 requires types.carbon and cutoffs_A.ch")
    if not isinstance(raw.get("cases"), list) or not raw["cases"]:
        raise ValueError("contract cases must be a non-empty list")
    if _float(raw.get("window_ps"), 20.0) <= 0.0:
        raise ValueError("window_ps must be positive")
    if _float(raw.get("merge_gap_ps"), 11.0) < 0.0:
        raise ValueError("merge_gap_ps must be non-negative")
    if _float(raw.get("timestep_fs"), 0.5) <= 0.0:
        raise ValueError("timestep_fs must be positive")
    identities = []
    for case in raw["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Each case entry must be an object")
        required = (
            "case_id",
            "branch_id",
            "state_trajectories",
            "oxygen_audit_trajectories",
            "motion_tables",
        )
        for key in required:
            if key not in case:
                raise ValueError(f"Case entry is missing {key}")
        for key in ("state_trajectories", "oxygen_audit_trajectories", "motion_tables"):
            if not isinstance(case[key], list) or not case[key]:
                raise ValueError(f"{key} must be a non-empty list")
        identities.append((str(case["case_id"]), str(case["branch_id"])))
    if len(identities) != len(set(identities)):
        raise ValueError("case_id/branch_id pairs must be unique")


def _deduplicate_species_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    step_column: str,
    time_column: str,
) -> list[dict[str, str]]:
    unique: dict[int, dict[str, str]] = {}

    def signature(row: Mapping[str, str]) -> tuple[tuple[str, float], ...]:
        values = [(time_column, _float(row.get(time_column)))]
        values.extend(
            (key, _float(row.get(key)))
            for key in SPECIES_COMPARISON_FIELDS
            if key in row
        )
        return tuple(values)

    for raw_row in rows:
        row = dict(raw_row)
        step = _int(row.get(step_column))
        time_ps = _float(row.get(time_column))
        if step < 0 or not math.isfinite(time_ps):
            raise ValueError(f"Invalid species row: step={step!r}, time={time_ps!r}")
        if step in unique and signature(unique[step]) != signature(row):
            raise ValueError(f"Conflicting duplicate species row at step {step}")
        unique[step] = row
    return [unique[step] for step in sorted(unique)]


def run_contract(contract_path: Path, output_dir: Path) -> dict[str, object]:
    """Run a contract-driven event audit and write inspectable TSV/JSON outputs."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(raw)
    base = contract_path.parent
    time_origin_step = _int(raw.get("time_origin_step"), 0)
    timestep_fs = _float(raw.get("timestep_fs"), 0.5)
    window_ps = _float(raw.get("window_ps"), 20.0)
    merge_gap_ps = _float(raw.get("merge_gap_ps"), 11.0)
    high_z_threshold_A = _float(raw.get("high_z_threshold_A"), 80.0)
    wall_clearance_threshold_A = _float(raw.get("wall_clearance_threshold_A"), 1.0)
    types = dict(raw.get("types", {}))
    cutoffs = dict(raw.get("cutoffs_A", {}))
    hydrogen_type = _int(types.get("hydrogen"), 1)
    oxygen_type = _int(types.get("oxygen"), 2)
    silicon_type = _int(types.get("silicon"), 8)
    carbon_type = _int(types.get("carbon")) if "carbon" in types else None
    oh_cutoff_A = _float(cutoffs.get("oh"), 1.35)
    ch_cutoff_A = _float(cutoffs.get("ch")) if carbon_type is not None else None
    sio_cutoff_A = _float(cutoffs.get("si_o"), 2.25)
    oo_cutoff_A = _float(cutoffs.get("oo"), 3.5)
    hbond_angle_deg = _float(cutoffs.get("hbond_angle_deg"), 30.0)
    lsi_cutoff_A = _float(cutoffs.get("lsi"), 3.7)
    sharing_delta_threshold_A = _float(cutoffs.get("proton_sharing_delta"), 0.2)
    lsi_neighbor_cap = _int(raw.get("lsi_neighbor_cap"), 24)
    if min(
        oh_cutoff_A,
        sio_cutoff_A,
        oo_cutoff_A,
        hbond_angle_deg,
        lsi_cutoff_A,
        sharing_delta_threshold_A,
        lsi_neighbor_cap,
    ) <= 0:
        raise ValueError("All geometry cutoffs and lsi_neighbor_cap must be positive")
    if carbon_type is not None and (
        ch_cutoff_A is None or not math.isfinite(ch_cutoff_A) or ch_cutoff_A <= 0.0
    ):
        raise ValueError("cutoffs_A.ch must be positive when types.carbon is configured")
    configured_types = [hydrogen_type, oxygen_type, silicon_type]
    if carbon_type is not None:
        configured_types.append(carbon_type)
    if len(configured_types) != len(set(configured_types)):
        raise ValueError("Configured atom types must be distinct")
    motion_columns = {
        "step": "TimeStep",
        "x": "v_dxrel",
        "y": "v_dyrel",
        "clearance": "v_topclear",
        **dict(raw.get("motion_columns", {})),
    }
    input_paths: set[Path] = {contract_path}
    all_events: list[EventEpisode] = []
    all_sources: list[dict[str, object]] = []
    all_frames: list[dict[str, object]] = []
    all_atoms: list[dict[str, object]] = []
    all_motion: list[dict[str, object]] = []
    for case in raw["cases"]:
        case_id = str(case["case_id"])
        branch_id = str(case["branch_id"])
        state_paths = [_resolve_path(value, base) for value in case["state_trajectories"]]
        oxygen_paths = [_resolve_path(value, base) for value in case["oxygen_audit_trajectories"]]
        motion_paths = [_resolve_path(value, base) for value in case["motion_tables"]]
        species_paths = [_resolve_path(value, base) for value in case.get("species_tables", [])]
        for path in (*state_paths, *oxygen_paths, *motion_paths, *species_paths):
            if not path.is_file():
                raise FileNotFoundError(path)
            input_paths.add(path)
        species_rows: list[dict[str, str]] = []
        for path in species_paths:
            species_rows.extend(_read_delimited(path))
        species_step_column = str(case.get("species_step_column", "step"))
        species_time_column = str(case.get("species_time_column", "time_ps"))
        unique_species = _deduplicate_species_rows(
            species_rows,
            step_column=species_step_column,
            time_column=species_time_column,
        )
        samples = _detect_species_samples(
            unique_species,
            time_column=species_time_column,
            step_column=species_step_column,
        )
        samples.extend(
            _detect_high_z_samples(
                oxygen_paths,
                time_origin_step=time_origin_step,
                timestep_fs=timestep_fs,
                oxygen_type=oxygen_type,
                high_z_threshold_A=high_z_threshold_A,
            )
        )
        columns, motion_data = stitch_motion_tables(
            motion_paths,
            step_column=motion_columns["step"],
            displacement_columns=(motion_columns["x"], motion_columns["y"]),
        )
        samples.extend(
            _detect_wall_samples(
                columns,
                motion_data,
                time_origin_step=time_origin_step,
                timestep_fs=timestep_fs,
                step_column=motion_columns["step"],
                clearance_column=motion_columns["clearance"],
                wall_clearance_threshold_A=wall_clearance_threshold_A,
            )
        )
        episodes, sources = merge_event_samples(
            samples,
            case_id=case_id,
            branch_id=branch_id,
            merge_gap_ps=merge_gap_ps,
            window_ps=window_ps,
        )
        frames, atoms = extract_state_windows(
            state_paths,
            episodes,
            time_origin_step=time_origin_step,
            timestep_fs=timestep_fs,
            hydrogen_type=hydrogen_type,
            oxygen_type=oxygen_type,
            silicon_type=silicon_type,
            carbon_type=carbon_type,
            oh_cutoff_A=oh_cutoff_A,
            ch_cutoff_A=ch_cutoff_A,
            sio_cutoff_A=sio_cutoff_A,
            oo_cutoff_A=oo_cutoff_A,
            hbond_angle_deg=hbond_angle_deg,
            lsi_cutoff_A=lsi_cutoff_A,
            lsi_neighbor_cap=lsi_neighbor_cap,
            sharing_delta_threshold_A=sharing_delta_threshold_A,
            high_z_threshold_A=high_z_threshold_A,
        )
        motion = summarize_motion_events(
            episodes,
            columns,
            motion_data,
            time_origin_step=time_origin_step,
            timestep_fs=timestep_fs,
            step_column=motion_columns["step"],
            x_column=motion_columns["x"],
            y_column=motion_columns["y"],
            clearance_column=motion_columns["clearance"],
            wall_clearance_threshold_A=wall_clearance_threshold_A,
        )
        all_events.extend(episodes)
        all_sources.extend(sources)
        all_frames.extend(frames)
        all_atoms.extend(atoms)
        all_motion.extend(motion)
    outcomes = summarize_event_outcomes(
        all_events,
        all_frames,
        all_atoms,
        high_z_threshold_A=high_z_threshold_A,
    )
    output.mkdir(parents=True)
    _write_tsv(output / "events.tsv", outcomes, fieldnames=EVENT_FIELDS)
    _write_tsv(output / "event_sources.tsv", all_sources, fieldnames=EVENT_SOURCE_FIELDS)
    _write_tsv(output / "frame_species.tsv", all_frames, fieldnames=FRAME_FIELDS)
    _write_tsv(output / "atom_identity.tsv", all_atoms, fieldnames=ATOM_FIELDS)
    _write_tsv(output / "motion_event_summary.tsv", all_motion, fieldnames=MOTION_FIELDS)
    with (output / "input_manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "bytes", "sha256"])
        for path in sorted(input_paths):
            writer.writerow([str(path), path.stat().st_size, _sha256(path)])
    summary = {
        "status": "PASS",
        "contract": str(contract_path),
        "case_branches": len(raw["cases"]),
        "events": len(all_events),
        "species_returned_events": sum(bool(row["species_returned"]) for row in outcomes),
        "tracked_high_z_events": sum(bool(row["tracked_frames"]) for row in outcomes),
        "tracked_intact_water_events": sum(bool(row["tracked_intact_water"]) for row in outcomes),
        "tracked_shared_hydrogen_events": sum(
            _int(row["tracked_shared_hydrogen_frames"], 0) > 0 for row in outcomes
        ),
        "z_image_crossing_events": sum(_int(row["nonzero_iz_max"], 0) > 0 for row in outcomes),
        "claim_limits": [
            "species labels use geometric O-H and Si-O cutoffs",
            "proton sharing is a nearest-two-oxygen distance diagnostic, not a formal charge state",
            "event association is descriptive and does not establish causality",
            "wall-window velocities are local regressions, not friction coefficients",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Constant-force event audit",
        "",
        f"Status: `{summary['status']}`.",
        "",
        (
            f"Audited `{summary['case_branches']}` case/branch entries and found "
            f"`{summary['events']}` merged episodes."
        ),
        "",
        (
            "The audit tracks atom identities, geometric species and proton sharing, local "
            "water order and H bonds, high-z oxygen return, Z image flags, and lateral motion "
            "around each event window."
        ),
        "",
        (
            "Species labels are geometric diagnostics. Event-aligned changes are associations "
            "and do not establish reaction kinetics, charge identity, wall causality, or friction."
        ),
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_contract(args.contract, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
