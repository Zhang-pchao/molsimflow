"""Local liquid supply and bubble geometry for driven interfacial nanobubbles.

The analysis joins full-system coordinate frames to per-atom fluid velocities,
tracks the largest non-relocated N2 component across periodic X/Y boundaries,
and evaluates liquid motion and flow-corrected kinetic temperature in a bubble-
attached frame.  Contact edges remain molecular screening proxies.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import (
    LammpsDumpFrame,
    iter_lammps_dump_records_selected,
    iter_synchronized_lammps_dump_records,
)
from molsimflow.postprocess.constant_force_oxygen import (
    connected_components,
    resolve_path,
    write_output_hashes,
    write_tsv,
    xy_minimum_image,
)
from molsimflow.postprocess.nanobubble_flow_response import (
    _molecule_centers,
    _tracked_periodic_center,
)

REGIONS = ("front", "rear", "interior_film", "above", "bulk")
COMMON_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps",
    "main_bubble_n2", "bubble_center_drive_A", "bubble_center_transverse_A",
    "bubble_center_z_A", "bubble_drive_unwrapped_A", "front_edge_unwrapped_A",
    "rear_edge_unwrapped_A", "front_edge_relative_A", "rear_edge_relative_A",
    "bubble_height_A", "bubble_drive_width_A", "bubble_transverse_width_A",
    "bubble_drive_span_fraction", "bubble_transverse_span_fraction",
    "pbc_spanning_drive", "pbc_spanning_transverse", "contact_n2",
    "contact_line_valid", "geometry_reference", "surface_top_z_A",
    "film_water_oxygen_count",
)
REGION_FIELDS = (
    "water_oxygen_count", "water_atom_count", "oxygen_density_A3",
    "water_drive_velocity_mps", "water_transverse_velocity_mps",
    "water_vertical_velocity_mps", "drive_velocity_gradient_ps_inv",
    "flow_corrected_temperature_K", "temperature_drive_K",
    "temperature_transverse_K", "temperature_vertical_K",
)
FRAME_FIELDS = COMMON_FIELDS + tuple(
    f"{region}_{field}" for region in REGIONS for field in REGION_FIELDS
)
INTERVAL_FIELDS = (
    "case_id", "branch_id", "direction", "start_step", "end_step",
    "start_time_ps", "end_time_ps", "mid_time_ps", "interval_ps",
    "bubble_velocity_mps", "front_edge_velocity_mps", "rear_edge_velocity_mps",
    "front_water_velocity_mps", "rear_water_velocity_mps",
    "bubble_minus_mean_edge_water_mps", "front_edge_minus_water_mps",
    "rear_edge_minus_water_mps", "center_mode_velocity_mps",
    "width_mode_velocity_mps", "bubble_height_change_A",
    "bubble_drive_width_change_A", "film_water_oxygen_change",
)
BLOCK_FIELDS = (
    "case_id", "branch_id", "direction", "block_index", "start_ps", "end_ps",
    "intervals", "bubble_velocity_mean_mps", "bubble_velocity_std_mps",
    "front_edge_velocity_mean_mps", "rear_edge_velocity_mean_mps",
    "front_water_velocity_mean_mps", "rear_water_velocity_mean_mps",
    "front_residual_velocity_mean_mps", "rear_residual_velocity_mean_mps",
    "center_mode_velocity_mean_mps", "width_mode_velocity_mean_mps",
    "front_temperature_mean_K", "rear_temperature_mean_K",
    "film_water_oxygen_mean", "bubble_water_velocity_correlation",
)

AMU_A2_PS2_TO_EV = 1.0364269656262175e-4
KB_EV_K = 8.617333262145e-5


def _range(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [start, end]")
    start, end = int(value[0]), int(value[1])
    if start < 1 or end < start:
        raise ValueError(f"{name} must satisfy 1 <= start <= end")
    return start, end


def _validate_contract(raw: Mapping[str, object]) -> None:
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    positive = (
        "timestep_fs", "sample_stride_steps", "cluster_cutoff_A",
        "contact_cutoff_A", "edge_window_halfwidth_A", "transverse_padding_A",
        "near_surface_height_A", "film_height_A", "bulk_z_min_A",
        "bulk_z_max_A", "block_ps", "minimum_region_water_oxygen",
    )
    for key in positive:
        if float(raw.get(key, 0.0)) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if float(raw["bulk_z_max_A"]) <= float(raw["bulk_z_min_A"]):
        raise ValueError("bulk_z_max_A must exceed bulk_z_min_A")
    if str(raw.get("no_contact_policy", "error")) not in {
        "error", "gas_projection_control",
    }:
        raise ValueError(
            "no_contact_policy must be error or gas_projection_control"
        )
    case = raw.get("case")
    if not isinstance(case, dict):
        raise TypeError("case must be an object")
    for key in (
        "case_id", "branch_id", "direction", "time_origin_step", "maximum_step",
        "surface_model_data", "surface_atom_range", "nitrogen_atom_range",
        "water_atom_range",
    ):
        if key not in case:
            raise ValueError(f"case is missing {key}")
    if str(case["direction"]).lower() not in {"x", "y"}:
        raise ValueError("direction must be x or y")
    for name in ("surface_atom_range", "nitrogen_atom_range", "water_atom_range"):
        _range(case[name], name)
    paired = "state_trajectories" in case or "velocity_trajectories" in case
    synchronized = "synchronized_state_trajectories" in case
    if paired == synchronized:
        raise ValueError(
            "case requires exactly one of paired state/velocity trajectories or "
            "synchronized_state_trajectories"
        )
    if paired:
        for name in ("state_trajectories", "velocity_trajectories"):
            if not isinstance(case.get(name), list) or not case[name]:
                raise ValueError(f"{name} must be a non-empty list")
    elif not isinstance(case["synchronized_state_trajectories"], list) or not case[
        "synchronized_state_trajectories"
    ]:
        raise ValueError("synchronized_state_trajectories must be a non-empty list")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_entries(values: Sequence[object], base: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for value in values:
        raw = value if isinstance(value, dict) else {"path": value}
        if "path" not in raw:
            raise ValueError("trajectory entries require path")
        entries.append({
            "path": resolve_path(raw["path"], base),
            "minimum_step": int(raw.get("minimum_step", -2**63)),
            "maximum_step": int(raw.get("maximum_step", 2**63 - 1)),
        })
    return entries


def _synchronized_trajectory_entries(
    values: Sequence[object], base: Path
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("paths"), list):
            raise ValueError("synchronized trajectory entries require a paths list")
        paths = [resolve_path(path, base) for path in value["paths"]]
        if not paths:
            raise ValueError("synchronized trajectory entries require at least one path")
        entries.append({
            "paths": paths,
            "minimum_step": int(value.get("minimum_step", -2**63)),
            "maximum_step": int(value.get("maximum_step", 2**63 - 1)),
        })
    return entries


def _iter_sampled_frames(
    entries: Sequence[Mapping[str, object]],
    *,
    time_origin_step: int,
    maximum_step: int,
    stride_steps: int,
) -> Iterator[LammpsDumpFrame]:
    previous_step: int | None = None
    previous_bounds: np.ndarray | None = None
    for entry in entries:
        path = Path(entry["path"])
        lower = max(int(entry["minimum_step"]), time_origin_step)
        upper = min(int(entry["maximum_step"]), maximum_step)

        def accept(step: int, lower: int = lower, upper: int = upper) -> bool:
            return lower <= step <= upper and (step - time_origin_step) % stride_steps == 0

        frames = 0
        for frame in iter_lammps_dump_records_selected(
            path, accept_timestep=accept, stop_after_timestep=upper
        ):
            frames += 1
            if previous_step is not None and frame.timestep == previous_step:
                continue
            if previous_step is not None and frame.timestep < previous_step:
                raise ValueError(f"Non-increasing sampled timestep {frame.timestep}")
            if previous_bounds is not None and not np.allclose(
                frame.bounds, previous_bounds, atol=1.0e-8
            ):
                raise ValueError(f"Box bounds changed at step {frame.timestep}")
            previous_step = frame.timestep
            previous_bounds = frame.bounds.copy()
            yield frame
        if frames == 0:
            raise ValueError(f"No sampled frames in {path}")


def _iter_sampled_synchronized_frames(
    entries: Sequence[Mapping[str, object]],
    *,
    time_origin_step: int,
    maximum_step: int,
    stride_steps: int,
) -> Iterator[LammpsDumpFrame]:
    previous_step: int | None = None
    previous_bounds: np.ndarray | None = None
    for entry in entries:
        paths = tuple(Path(path) for path in entry["paths"])
        lower = max(int(entry["minimum_step"]), time_origin_step)
        upper = min(int(entry["maximum_step"]), maximum_step)

        def accept(step: int, lower: int = lower, upper: int = upper) -> bool:
            return lower <= step <= upper and (step - time_origin_step) % stride_steps == 0

        frames = 0
        for frame in iter_synchronized_lammps_dump_records(
            paths, accept_timestep=accept, stop_after_timestep=upper
        ):
            frames += 1
            if previous_step is not None and frame.timestep == previous_step:
                continue
            if previous_step is not None and frame.timestep < previous_step:
                raise ValueError(f"Non-increasing sampled timestep {frame.timestep}")
            if previous_bounds is not None and not np.allclose(
                frame.bounds, previous_bounds, atol=1.0e-8
            ):
                raise ValueError(f"Box bounds changed at step {frame.timestep}")
            previous_step = frame.timestep
            previous_bounds = frame.bounds.copy()
            yield frame
        if frames == 0:
            raise ValueError(
                "No synchronized sampled frames in " + ", ".join(str(path) for path in paths)
            )


def _frame_arrays(
    frame: LammpsDumpFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("id", "type", "x", "y", "z")
    missing = [field for field in required if field not in frame.atom_fields]
    if missing:
        raise ValueError(f"step {frame.timestep}: coordinate frame missing {missing}")
    column = {name: frame.atom_fields.index(name) for name in required}
    atom_ids = np.asarray([int(row[column["id"]]) for row in frame.atom_rows], dtype=int)
    order = np.argsort(atom_ids)
    atom_ids = atom_ids[order]
    if len(np.unique(atom_ids)) != len(atom_ids):
        raise ValueError(f"step {frame.timestep}: duplicate coordinate atom IDs")
    atom_types = np.asarray(
        [int(row[column["type"]]) for row in frame.atom_rows], dtype=int
    )[order]
    coordinates = np.asarray(
        [[float(row[column[axis]]) for axis in "xyz"] for row in frame.atom_rows],
        dtype=float,
    )[order]
    return atom_ids, atom_types, coordinates


def _velocity_arrays(
    frame: LammpsDumpFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("id", "type", "vx", "vy", "vz")
    missing = [field for field in required if field not in frame.atom_fields]
    if missing:
        raise ValueError(f"step {frame.timestep}: velocity frame missing {missing}")
    column = {name: frame.atom_fields.index(name) for name in required}
    atom_ids = np.asarray([int(row[column["id"]]) for row in frame.atom_rows], dtype=int)
    order = np.argsort(atom_ids)
    atom_ids = atom_ids[order]
    if len(np.unique(atom_ids)) != len(atom_ids):
        raise ValueError(f"step {frame.timestep}: duplicate velocity atom IDs")
    atom_types = np.asarray(
        [int(row[column["type"]]) for row in frame.atom_rows], dtype=int
    )[order]
    velocities = np.asarray(
        [[float(row[column[f"v{axis}"]]) for axis in "xyz"] for row in frame.atom_rows],
        dtype=float,
    )[order]
    return atom_ids, atom_types, velocities


def _select_range(
    atom_ids: np.ndarray,
    atom_types: np.ndarray,
    values: np.ndarray,
    atom_range: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = (atom_ids >= atom_range[0]) & (atom_ids <= atom_range[1])
    expected = atom_range[1] - atom_range[0] + 1
    if int(np.count_nonzero(selected)) != expected:
        raise ValueError(f"Selected {np.count_nonzero(selected)} of {expected} declared atoms")
    return atom_ids[selected], atom_types[selected], values[selected]


def _select_present_range(
    atom_ids: np.ndarray,
    atom_types: np.ndarray,
    values: np.ndarray,
    atom_range: tuple[int, int],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a non-empty spatially filtered subset of a declared atom range."""

    selected = (atom_ids >= atom_range[0]) & (atom_ids <= atom_range[1])
    if not np.any(selected):
        raise ValueError(f"No present atoms in declared {name}")
    return atom_ids[selected], atom_types[selected], values[selected]


def _masses(path: Path) -> dict[int, float]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    output: dict[int, float] = {}
    in_masses = False
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if stripped == "Masses":
            in_masses = True
            continue
        if not in_masses:
            continue
        if not stripped:
            continue
        if stripped.startswith("Atoms"):
            break
        parts = stripped.split()
        if len(parts) >= 2 and parts[0].isdigit():
            output[int(parts[0])] = float(parts[1])
    if not output:
        raise ValueError(f"No Masses section parsed from {path}")
    return output


def _periodic_nearest_distance(
    query: np.ndarray, reference: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    from scipy.spatial import cKDTree

    lengths = bounds[:, 1] - bounds[:, 0]
    pseudo_z = max(1.0e5, 10.0 * lengths[2])

    def shifted(points: np.ndarray) -> np.ndarray:
        result = np.asarray(points, dtype=float).copy()
        result[:, 0] = (result[:, 0] - bounds[0, 0]) % lengths[0]
        result[:, 1] = (result[:, 1] - bounds[1, 0]) % lengths[1]
        result[:, 2] = result[:, 2] - bounds[2, 0] + 0.25 * pseudo_z
        return result

    tree = cKDTree(shifted(reference), boxsize=[lengths[0], lengths[1], pseudo_z])
    return np.asarray(tree.query(shifted(query), k=1)[0], dtype=float)


def _bubble_geometry(
    nitrogen_ids: np.ndarray,
    nitrogen: np.ndarray,
    surface: np.ndarray,
    bounds: np.ndarray,
    *,
    relocated_ids: set[int],
    cluster_cutoff_A: float,
    contact_cutoff_A: float,
    direction_index: int,
    transverse_index: int,
    reference_center: np.ndarray | None,
    no_contact_policy: str,
) -> dict[str, object]:
    if len(nitrogen_ids) % 2:
        raise ValueError("Nitrogen range must contain complete consecutive N2 pairs")
    pairs = nitrogen_ids.reshape((-1, 2))
    centers = _molecule_centers(nitrogen, bounds)
    relocated = np.asarray(
        [bool(relocated_ids.intersection(pair.tolist())) for pair in pairs], dtype=bool
    )
    candidates = np.flatnonzero(~relocated)
    components = connected_components(
        centers[candidates], bounds, float(cluster_cutoff_A)
    )
    if not components:
        raise ValueError("No non-relocated nitrogen component")
    members = candidates[components[0]]
    bubble = centers[members]
    center = _tracked_periodic_center(bubble, bounds, reference_center)
    vectors = bubble - center
    vectors[:, :2] = xy_minimum_image(vectors, bounds)[:, :2]
    distances = _periodic_nearest_distance(nitrogen, surface, bounds).reshape((-1, 2)).min(axis=1)
    contact = vectors[distances[members] <= contact_cutoff_A]
    if len(contact) < 3:
        if no_contact_policy != "gas_projection_control":
            raise ValueError("Fewer than three contact N2 molecules")
        contact_drive = vectors[:, direction_index]
        contact_line_valid = 0
        geometry_reference = "gas_projection_control"
    else:
        contact_drive = contact[:, direction_index]
        contact_line_valid = 1
        geometry_reference = "contact_n2"
    drive = vectors[:, direction_index]
    transverse = vectors[:, transverse_index]
    return {
        "center": center,
        "main_bubble_n2": len(members),
        "front_relative_A": float(np.quantile(contact_drive, 0.95)),
        "rear_relative_A": float(np.quantile(contact_drive, 0.05)),
        "drive_halfwidth_A": 0.5 * float(np.quantile(drive, 0.95) - np.quantile(drive, 0.05)),
        "transverse_halfwidth_A": float(np.quantile(np.abs(transverse), 0.95)),
        "height_A": float(np.quantile(vectors[:, 2], 0.95) - np.quantile(vectors[:, 2], 0.05)),
        "contact_n2": len(contact),
        "contact_line_valid": contact_line_valid,
        "geometry_reference": geometry_reference,
    }


def flow_corrected_temperature(
    atom_types: np.ndarray,
    velocities_A_ps: np.ndarray,
    masses_amu: Mapping[int, float],
    *,
    direction_index: int,
    transverse_index: int,
) -> tuple[float, float, float, float]:
    """Return total and directional temperatures after removing local mass COM flow."""

    if len(atom_types) < 2:
        return (math.nan,) * 4
    mass = np.asarray([masses_amu[int(atom_type)] for atom_type in atom_types], dtype=float)
    center = np.sum(mass[:, None] * velocities_A_ps, axis=0) / np.sum(mass)
    residual = velocities_A_ps - center
    directional = AMU_A2_PS2_TO_EV * np.sum(mass[:, None] * residual**2, axis=0)
    dof_axis = len(atom_types) - 1
    temperatures = directional / (dof_axis * KB_EV_K)
    total = float(np.sum(directional) / (3 * dof_axis * KB_EV_K))
    return (
        total,
        float(temperatures[direction_index]),
        float(temperatures[transverse_index]),
        float(temperatures[2]),
    )


def _region_stats(
    mask: np.ndarray,
    water_types: np.ndarray,
    water_velocities: np.ndarray,
    water_z_relative: np.ndarray,
    masses_amu: Mapping[int, float],
    *,
    oxygen_type: int,
    minimum_oxygen: int,
    volume_A3: float,
    direction_index: int,
    transverse_index: int,
) -> dict[str, object]:
    oxygen = mask & (water_types == oxygen_type)
    oxygen_count = int(np.count_nonzero(oxygen))
    atom_count = int(np.count_nonzero(mask))
    if oxygen_count < minimum_oxygen:
        return {field: math.nan for field in REGION_FIELDS} | {
            "water_oxygen_count": oxygen_count,
            "water_atom_count": atom_count,
            "oxygen_density_A3": oxygen_count / volume_A3 if volume_A3 > 0.0 else math.nan,
        }
    mean_velocity = np.mean(water_velocities[oxygen], axis=0)
    gradient = math.nan
    if oxygen_count >= max(3, minimum_oxygen):
        z = water_z_relative[oxygen]
        if float(np.ptp(z)) > 1.0e-8:
            gradient = float(np.polyfit(z, water_velocities[oxygen, direction_index], 1)[0])
    temperature = flow_corrected_temperature(
        water_types[mask], water_velocities[mask], masses_amu,
        direction_index=direction_index, transverse_index=transverse_index,
    )
    return {
        "water_oxygen_count": oxygen_count,
        "water_atom_count": atom_count,
        "oxygen_density_A3": oxygen_count / volume_A3 if volume_A3 > 0.0 else math.nan,
        "water_drive_velocity_mps": 100.0 * float(mean_velocity[direction_index]),
        "water_transverse_velocity_mps": 100.0 * float(mean_velocity[transverse_index]),
        "water_vertical_velocity_mps": 100.0 * float(mean_velocity[2]),
        "drive_velocity_gradient_ps_inv": gradient,
        "flow_corrected_temperature_K": temperature[0],
        "temperature_drive_K": temperature[1],
        "temperature_transverse_K": temperature[2],
        "temperature_vertical_K": temperature[3],
    }


def _interval_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for previous, current in zip(rows[:-1], rows[1:]):
        interval_ps = float(current["time_ps"]) - float(previous["time_ps"])
        if interval_ps <= 0.0:
            raise ValueError("Non-positive frame interval")

        bubble = 100.0 * (
            float(current["bubble_drive_unwrapped_A"])
            - float(previous["bubble_drive_unwrapped_A"])
        ) / interval_ps
        front = 100.0 * (
            float(current["front_edge_unwrapped_A"])
            - float(previous["front_edge_unwrapped_A"])
        ) / interval_ps
        rear = 100.0 * (
            float(current["rear_edge_unwrapped_A"])
            - float(previous["rear_edge_unwrapped_A"])
        ) / interval_ps
        front_water = _finite_mean([
            previous["front_water_drive_velocity_mps"],
            current["front_water_drive_velocity_mps"],
        ])
        rear_water = _finite_mean([
            previous["rear_water_drive_velocity_mps"],
            current["rear_water_drive_velocity_mps"],
        ])
        output.append({
            "case_id": current["case_id"], "branch_id": current["branch_id"],
            "direction": current["direction"], "start_step": previous["step"],
            "end_step": current["step"], "start_time_ps": previous["time_ps"],
            "end_time_ps": current["time_ps"],
            "mid_time_ps": 0.5 * (float(previous["time_ps"]) + float(current["time_ps"])),
            "interval_ps": interval_ps, "bubble_velocity_mps": bubble,
            "front_edge_velocity_mps": front, "rear_edge_velocity_mps": rear,
            "front_water_velocity_mps": front_water, "rear_water_velocity_mps": rear_water,
            "bubble_minus_mean_edge_water_mps": bubble - 0.5 * (front_water + rear_water),
            "front_edge_minus_water_mps": front - front_water,
            "rear_edge_minus_water_mps": rear - rear_water,
            "center_mode_velocity_mps": 0.5 * (front + rear),
            "width_mode_velocity_mps": 0.5 * (front - rear),
            "bubble_height_change_A": float(current["bubble_height_A"]) - float(previous["bubble_height_A"]),
            "bubble_drive_width_change_A": float(current["bubble_drive_width_A"]) - float(previous["bubble_drive_width_A"]),
            "film_water_oxygen_change": int(current["film_water_oxygen_count"]) - int(previous["film_water_oxygen_count"]),
        })
    return output


def _finite_mean(values: Sequence[object]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array[np.isfinite(array)])) if np.any(np.isfinite(array)) else math.nan


def _block_rows(
    frames: Sequence[Mapping[str, object]],
    intervals: Sequence[Mapping[str, object]],
    block_ps: float,
) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in intervals:
        grouped[math.floor(float(row["mid_time_ps"]) / block_ps)].append(row)
    output: list[dict[str, object]] = []
    for block, selected in sorted(grouped.items()):
        start, end = block * block_ps, (block + 1) * block_ps
        frame_rows = [row for row in frames if start <= float(row["time_ps"]) < end]
        bubble = np.asarray([row["bubble_velocity_mps"] for row in selected], dtype=float)
        water = np.asarray([
            0.5 * (float(row["front_water_velocity_mps"]) + float(row["rear_water_velocity_mps"]))
            for row in selected
        ])
        valid = np.isfinite(bubble) & np.isfinite(water)
        correlation = (
            float(np.corrcoef(bubble[valid], water[valid])[0, 1])
            if np.count_nonzero(valid) >= 3 and np.std(bubble[valid]) > 0 and np.std(water[valid]) > 0
            else math.nan
        )
        output.append({
            "case_id": selected[0]["case_id"], "branch_id": selected[0]["branch_id"],
            "direction": selected[0]["direction"], "block_index": block,
            "start_ps": start, "end_ps": end, "intervals": len(selected),
            "bubble_velocity_mean_mps": float(np.mean(bubble)),
            "bubble_velocity_std_mps": float(np.std(bubble, ddof=1)) if len(bubble) > 1 else math.nan,
            "front_edge_velocity_mean_mps": _finite_mean([row["front_edge_velocity_mps"] for row in selected]),
            "rear_edge_velocity_mean_mps": _finite_mean([row["rear_edge_velocity_mps"] for row in selected]),
            "front_water_velocity_mean_mps": _finite_mean([row["front_water_velocity_mps"] for row in selected]),
            "rear_water_velocity_mean_mps": _finite_mean([row["rear_water_velocity_mps"] for row in selected]),
            "front_residual_velocity_mean_mps": _finite_mean([row["front_edge_minus_water_mps"] for row in selected]),
            "rear_residual_velocity_mean_mps": _finite_mean([row["rear_edge_minus_water_mps"] for row in selected]),
            "center_mode_velocity_mean_mps": _finite_mean([row["center_mode_velocity_mps"] for row in selected]),
            "width_mode_velocity_mean_mps": _finite_mean([row["width_mode_velocity_mps"] for row in selected]),
            "front_temperature_mean_K": _finite_mean([row["front_flow_corrected_temperature_K"] for row in frame_rows]),
            "rear_temperature_mean_K": _finite_mean([row["rear_flow_corrected_temperature_K"] for row in frame_rows]),
            "film_water_oxygen_mean": _finite_mean([row["film_water_oxygen_count"] for row in frame_rows]),
            "bubble_water_velocity_correlation": correlation,
        })
    return output


def _plot(
    frames: Sequence[Mapping[str, object]],
    intervals: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    time = np.asarray([float(row["time_ps"]) for row in frames]) / 1000.0
    mid = np.asarray([float(row["mid_time_ps"]) for row in intervals]) / 1000.0
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex="col")
    axes[0, 0].plot(mid, [row["bubble_velocity_mps"] for row in intervals], label="bubble")
    axes[0, 0].plot(mid, [row["front_water_velocity_mps"] for row in intervals], label="front water")
    axes[0, 0].plot(mid, [row["rear_water_velocity_mps"] for row in intervals], label="rear water")
    axes[0, 0].set_ylabel("Velocity (m/s)"); axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].plot(mid, [row["front_edge_minus_water_mps"] for row in intervals], label="front")
    axes[1, 0].plot(mid, [row["rear_edge_minus_water_mps"] for row in intervals], label="rear")
    axes[1, 0].axhline(0.0, color="black", lw=0.7)
    axes[1, 0].set(xlabel="Time (ns)", ylabel="Edge - local water (m/s)")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].plot(time, [row["front_flow_corrected_temperature_K"] for row in frames], label="front")
    axes[0, 1].plot(time, [row["rear_flow_corrected_temperature_K"] for row in frames], label="rear")
    axes[0, 1].set_ylabel("Flow-corrected T (K)"); axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(time, [row["bubble_height_A"] for row in frames], label="height")
    axes[1, 1].plot(time, [row["bubble_drive_width_A"] for row in frames], label="drive width")
    axes[1, 1].plot(time, [row["bubble_transverse_width_A"] for row in frames], label="transverse width")
    axes[1, 1].set(xlabel="Time (ns)", ylabel="Bubble extent (A)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{frames[0]['case_id']} / {str(frames[0]['direction']).upper()}")
    figure.tight_layout(); figure.savefig(output / "local_flow_geometry.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(raw)
    case = raw["case"]
    base = contract_path.parent
    synchronized = "synchronized_state_trajectories" in case
    if synchronized:
        synchronized_entries = _synchronized_trajectory_entries(
            case["synchronized_state_trajectories"], base
        )
        state_entries: list[dict[str, object]] = []
        velocity_entries: list[dict[str, object]] = []
    else:
        synchronized_entries = []
        state_entries = _trajectory_entries(case["state_trajectories"], base)
        velocity_entries = _trajectory_entries(case["velocity_trajectories"], base)
    model_path = resolve_path(case["surface_model_data"], base)
    masses = _masses(model_path)
    surface_range = _range(case["surface_atom_range"], "surface_atom_range")
    nitrogen_range = _range(case["nitrogen_atom_range"], "nitrogen_atom_range")
    water_range = _range(case["water_atom_range"], "water_atom_range")
    origin = int(case["time_origin_step"])
    maximum = int(case["maximum_step"])
    stride = int(raw["sample_stride_steps"])
    if synchronized:
        states = _iter_sampled_synchronized_frames(
            synchronized_entries,
            time_origin_step=origin,
            maximum_step=maximum,
            stride_steps=stride,
        )
        joined_frames = ((frame, frame) for frame in states)
    else:
        states = _iter_sampled_frames(
            state_entries, time_origin_step=origin, maximum_step=maximum, stride_steps=stride
        )
        velocities = _iter_sampled_frames(
            velocity_entries,
            time_origin_step=origin,
            maximum_step=maximum,
            stride_steps=stride,
        )
        joined_frames = zip_longest(states, velocities)
    direction = str(case["direction"]).lower()
    drive_index, transverse_index = (0, 1) if direction == "x" else (1, 0)
    relocated_ids = {int(value) for value in case.get("relocated_nitrogen_ids", [])}
    rows: list[dict[str, object]] = []
    previous_center: np.ndarray | None = None
    previous_wrapped: np.ndarray | None = None
    unwrapped: np.ndarray | None = None
    origin_unwrapped: np.ndarray | None = None

    for state, velocity in joined_frames:
        if state is None or velocity is None:
            raise ValueError("Coordinate and velocity sampled-frame counts differ")
        if state.timestep != velocity.timestep:
            raise ValueError(
                f"Coordinate/velocity timestep mismatch: {state.timestep} != {velocity.timestep}"
            )
        atom_ids, atom_types, coordinates = _frame_arrays(state)
        velocity_ids, velocity_types, velocity_values = _velocity_arrays(velocity)
        selector = _select_present_range if synchronized else _select_range
        if synchronized:
            _, _, surface = selector(
                atom_ids, atom_types, coordinates, surface_range, name="surface_atom_range"
            )
        else:
            _, _, surface = selector(atom_ids, atom_types, coordinates, surface_range)
        nitrogen_ids, _, nitrogen = _select_range(
            atom_ids, atom_types, coordinates, nitrogen_range
        )
        if synchronized:
            water_ids, water_types, water = selector(
                atom_ids, atom_types, coordinates, water_range, name="water_atom_range"
            )
        else:
            water_ids, water_types, water = selector(
                atom_ids, atom_types, coordinates, water_range
            )
        positions = np.searchsorted(velocity_ids, water_ids)
        if np.any(positions >= len(velocity_ids)) or not np.array_equal(
            velocity_ids[positions], water_ids
        ):
            raise ValueError(f"step {state.timestep}: water velocity identities incomplete")
        if not np.array_equal(velocity_types[positions], water_types):
            raise ValueError(f"step {state.timestep}: coordinate/velocity types differ")
        water_velocity = velocity_values[positions]
        geometry = _bubble_geometry(
            nitrogen_ids, nitrogen, surface, state.bounds,
            relocated_ids=relocated_ids,
            cluster_cutoff_A=float(raw["cluster_cutoff_A"]),
            contact_cutoff_A=float(raw["contact_cutoff_A"]),
            direction_index=drive_index, transverse_index=transverse_index,
            reference_center=previous_center,
            no_contact_policy=str(raw.get("no_contact_policy", "error")),
        )
        center = np.asarray(geometry.pop("center"), dtype=float)
        if previous_wrapped is None:
            unwrapped = center.copy(); origin_unwrapped = center.copy()
        else:
            assert unwrapped is not None
            delta = xy_minimum_image(center - previous_wrapped, state.bounds)
            unwrapped = unwrapped + delta
            unwrapped[2] = center[2]
        assert unwrapped is not None and origin_unwrapped is not None
        relative = water - center
        relative[:, :2] = xy_minimum_image(relative, state.bounds)[:, :2]
        s = relative[:, drive_index]
        t = relative[:, transverse_index]
        surface_top = float(np.quantile(surface[:, 2], float(raw.get("surface_top_quantile", 0.99))))
        z = water[:, 2] - surface_top
        front = float(geometry["front_relative_A"])
        rear = float(geometry["rear_relative_A"])
        drive_half = float(geometry["drive_halfwidth_A"])
        lengths = state.bounds[:, 1] - state.bounds[:, 0]
        drive_span_fraction = 2.0 * drive_half / lengths[drive_index]
        transverse_span_fraction = (
            2.0 * float(geometry["transverse_halfwidth_A"]) / lengths[transverse_index]
        )
        transverse_half = min(
            float(geometry["transverse_halfwidth_A"]) + float(raw["transverse_padding_A"]),
            0.5 * lengths[transverse_index],
        )
        edge_half = float(raw["edge_window_halfwidth_A"])
        near_height = float(raw["near_surface_height_A"])
        film_height = float(raw["film_height_A"])
        lateral = np.abs(t) <= transverse_half
        masks = {
            "front": (np.abs(s - front) <= edge_half) & lateral & (z >= 0.0) & (z <= near_height),
            "rear": (np.abs(s - rear) <= edge_half) & lateral & (z >= 0.0) & (z <= near_height),
            "interior_film": (s > rear) & (s < front) & lateral & (z >= 0.0) & (z <= film_height),
            "above": (np.abs(s) <= drive_half) & lateral & (z > near_height) & (z <= near_height + float(geometry["height_A"])),
            "bulk": (
                (z >= float(raw["bulk_z_min_A"])) & (z <= float(raw["bulk_z_max_A"]))
                & ~((np.abs(s) <= drive_half + edge_half) & lateral)
            ),
        }
        volumes = {
            "front": 2.0 * edge_half * 2.0 * transverse_half * near_height,
            "rear": 2.0 * edge_half * 2.0 * transverse_half * near_height,
            "interior_film": max(front - rear, 0.0) * 2.0 * transverse_half * film_height,
            "above": 2.0 * drive_half * 2.0 * transverse_half * float(geometry["height_A"]),
            "bulk": math.nan,
        }
        row: dict[str, object] = {
            "case_id": case["case_id"], "branch_id": case["branch_id"],
            "direction": direction, "step": state.timestep,
            "time_ps": (state.timestep - origin) * float(raw["timestep_fs"]) / 1000.0,
            "main_bubble_n2": geometry["main_bubble_n2"],
            "bubble_center_drive_A": center[drive_index],
            "bubble_center_transverse_A": center[transverse_index],
            "bubble_center_z_A": center[2],
            "bubble_drive_unwrapped_A": unwrapped[drive_index] - origin_unwrapped[drive_index],
            "front_edge_unwrapped_A": unwrapped[drive_index] + front,
            "rear_edge_unwrapped_A": unwrapped[drive_index] + rear,
            "front_edge_relative_A": front, "rear_edge_relative_A": rear,
            "bubble_height_A": geometry["height_A"],
            "bubble_drive_width_A": 2.0 * drive_half,
            "bubble_transverse_width_A": 2.0 * float(geometry["transverse_halfwidth_A"]),
            "bubble_drive_span_fraction": drive_span_fraction,
            "bubble_transverse_span_fraction": transverse_span_fraction,
            "pbc_spanning_drive": int(drive_span_fraction >= float(raw.get("pbc_span_threshold", 0.8))),
            "pbc_spanning_transverse": int(
                transverse_span_fraction >= float(raw.get("pbc_span_threshold", 0.8))
            ),
            "contact_n2": geometry["contact_n2"],
            "contact_line_valid": geometry["contact_line_valid"],
            "geometry_reference": geometry["geometry_reference"],
            "surface_top_z_A": surface_top,
            "film_water_oxygen_count": int(np.count_nonzero(
                masks["interior_film"] & (water_types == int(raw.get("oxygen_type", 2)))
            )),
        }
        for region in REGIONS:
            stats = _region_stats(
                masks[region], water_types, water_velocity, z, masses,
                oxygen_type=int(raw.get("oxygen_type", 2)),
                minimum_oxygen=int(raw["minimum_region_water_oxygen"]),
                volume_A3=float(volumes[region]),
                direction_index=drive_index, transverse_index=transverse_index,
            )
            row.update({f"{region}_{key}": value for key, value in stats.items()})
        rows.append(row)
        previous_center = center
        previous_wrapped = center.copy()

    if len(rows) < 2:
        raise ValueError("Fewer than two joined sampled frames")
    intervals = _interval_rows(rows)
    blocks = _block_rows(rows, intervals, float(raw["block_ps"]))
    output.mkdir(parents=True)
    write_tsv(output / "local_flow_geometry.tsv", rows, FRAME_FIELDS)
    write_tsv(output / "edge_flow_intervals.tsv", intervals, INTERVAL_FIELDS)
    write_tsv(output / "block_summary.tsv", blocks, BLOCK_FIELDS)
    input_paths = [contract_path, model_path]
    if synchronized:
        input_paths.extend(
            Path(path) for entry in synchronized_entries for path in entry["paths"]
        )
    else:
        input_paths.extend(Path(entry["path"]) for entry in state_entries)
        input_paths.extend(Path(entry["path"]) for entry in velocity_entries)
    hash_inputs = bool(raw.get("hash_inputs", True))
    write_tsv(
        output / "input_manifest.tsv",
        [{
            "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": _sha256(path) if hash_inputs else "NOT_COMPUTED",
        } for path in input_paths],
        ("path", "size_bytes", "sha256"),
    )
    if bool(raw.get("write_plots", True)):
        _plot(rows, intervals, output)
    summary = {
        "status": "PASS", "case_id": case["case_id"], "branch_id": case["branch_id"],
        "direction": direction, "frames": len(rows), "intervals": len(intervals),
        "first_step": int(rows[0]["step"]), "last_step": int(rows[-1]["step"]),
        "sample_interval_ps": float(rows[1]["time_ps"]) - float(rows[0]["time_ps"]),
        "bubble_velocity_mean_mps": _finite_mean([row["bubble_velocity_mps"] for row in intervals]),
        "front_water_velocity_mean_mps": _finite_mean([row["front_water_velocity_mps"] for row in intervals]),
        "rear_water_velocity_mean_mps": _finite_mean([row["rear_water_velocity_mps"] for row in intervals]),
        "front_residual_velocity_mean_mps": _finite_mean([row["front_edge_minus_water_mps"] for row in intervals]),
        "rear_residual_velocity_mean_mps": _finite_mean([row["rear_edge_minus_water_mps"] for row in intervals]),
        "local_temperature_is_flow_corrected": True,
        "spatially_filtered_synchronized_streams": synchronized,
        "contact_edges_are_molecular_screening_proxies": True,
        "all_frames_have_contact_line": bool(
            all(int(row["contact_line_valid"]) == 1 for row in rows)
        ),
        "gas_projection_control_is_not_a_contact_line": True,
        "single_trajectory_blocks_are_not_replicate_uncertainty": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "# Nanobubble local-flow and geometry analysis\n\n"
        f"Status: `{summary['status']}`. Analyzed `{summary['frames']}` joined frames "
        f"at `{summary['sample_interval_ps']:.3f}` ps spacing.\n\n"
        "Local water velocities use oxygen atoms. Temperatures use all water atoms in each "
        "window after removing the local mass-weighted streaming velocity. Contact edges are "
        "molecular screening proxies and block variation is not replicate uncertainty.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary
