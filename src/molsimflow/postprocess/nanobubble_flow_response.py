"""Analyze a driven nanobubble relative to the surrounding water flow.

The workflow combines compact nitrogen coordinate/force dumps with LAMMPS
motion tables. A reference substrate from the initial LAMMPS data file is
translated by the instantaneous substrate COM before contact distances are
calculated. Contact-line coordinates are molecular screening proxies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import LammpsDumpFrame, iter_lammps_dump_records
from molsimflow.postprocess.constant_force_events import stitch_motion_tables
from molsimflow.postprocess.constant_force_kinematics import _linear_velocity_mps
from molsimflow.postprocess.constant_force_oxygen import (
    connected_components,
    periodic_xy_center,
    xy_minimum_image,
)
from molsimflow.postprocess.v3_mechanism_io import parse_lammps_data_atoms

FRAME_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps",
    "main_bubble_n2", "disconnected_n2", "relocated_n2",
    "bubble_center_x_A", "bubble_center_y_A", "bubble_center_z_A",
    "bubble_center_x_unwrapped_A", "bubble_center_y_unwrapped_A",
    "bubble_drive_A", "bubble_transverse_A", "water_drive_A", "water_transverse_A",
    "bubble_minus_water_drive_A", "all_n_motion_drive_A",
    "height_q05_q95_A", "drive_width_q05_q95_A", "transverse_width_q05_q95_A",
    "lateral_radius_p90_A", "relative_shape_anisotropy",
    "contact_n2", "footprint_area_A2", "contact_line_circularity",
    "advancing_position_A", "receding_position_A",
    "advancing_displacement_A", "receding_displacement_A",
    "min_surface_distance_A", "max_main_raw_force_eV_A", "max_all_raw_force_eV_A",
    "max_all_raw_force_atom_id",
)

BLOCK_FIELDS = (
    "case_id", "branch_id", "direction", "block_index", "start_ps", "end_ps", "frames",
    "bubble_velocity_mps", "water_velocity_mps", "relative_velocity_mps",
    "advancing_velocity_mps", "receding_velocity_mps", "response_pattern",
    "mean_height_A", "mean_drive_width_A", "mean_transverse_width_A",
    "mean_footprint_area_A2", "footprint_cv", "mean_shape_anisotropy",
    "minimum_main_bubble_n2", "maximum_raw_force_eV_A",
)


def _resolve(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), delimiter="\t", extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


def _validate_range(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-integer list")
    start, end = int(value[0]), int(value[1])
    if start < 1 or end < start:
        raise ValueError(f"{name} must satisfy 1 <= start <= end")
    return start, end


def _validate_contract(raw: Mapping[str, object]) -> None:
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    if str(raw.get("direction", "")).lower() not in {"x", "y"}:
        raise ValueError("direction must be x or y")
    required = (
        "case_id", "branch_id", "surface_model_data",
        "nitrogen_trajectories", "motion_tables",
    )
    for key in required:
        if key not in raw:
            raise ValueError(f"contract is missing {key}")
    _validate_range(raw.get("surface_atom_range"), "surface_atom_range")
    nitrogen_range = _validate_range(raw.get("nitrogen_atom_range"), "nitrogen_atom_range")
    if (nitrogen_range[1] - nitrogen_range[0] + 1) % 2:
        raise ValueError("nitrogen_atom_range must contain complete N2 pairs")
    if not isinstance(raw["nitrogen_trajectories"], list) or not raw["nitrogen_trajectories"]:
        raise ValueError("nitrogen_trajectories must be a non-empty list")
    if not isinstance(raw["motion_tables"], list) or not raw["motion_tables"]:
        raise ValueError("motion_tables must be a non-empty list")
    for key in ("timestep_fs", "cluster_cutoff_A", "contact_cutoff_A", "block_ps"):
        if float(raw.get(key, 0.0)) <= 0.0:
            raise ValueError(f"{key} must be positive")


def _surface_coordinates(path: Path, atom_range: tuple[int, int]) -> np.ndarray:
    atoms = parse_lammps_data_atoms(path)
    selected = [
        atoms[atom_id][1]
        for atom_id in range(atom_range[0], atom_range[1] + 1)
        if atom_id in atoms
    ]
    expected = atom_range[1] - atom_range[0] + 1
    if len(selected) != expected:
        raise ValueError(f"{path}: selected {len(selected)} of {expected} surface atoms")
    return np.asarray(selected, dtype=float)


def _nitrogen_arrays(
    frame: LammpsDumpFrame, atom_range: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    required = ("id", "x", "y", "z")
    missing = [name for name in required if name not in frame.atom_fields]
    if missing:
        raise ValueError(f"nitrogen trajectory is missing {missing} at step {frame.timestep}")
    index = {name: frame.atom_fields.index(name) for name in frame.atom_fields}
    selected = [
        row for row in frame.atom_rows
        if atom_range[0] <= int(row[index["id"]]) <= atom_range[1]
    ]
    selected.sort(key=lambda row: int(row[index["id"]]))
    ids = np.asarray([int(row[index["id"]]) for row in selected], dtype=np.int64)
    expected_ids = np.arange(atom_range[0], atom_range[1] + 1, dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"incomplete nitrogen selection at step {frame.timestep}")
    coordinates = np.asarray(
        [[float(row[index[name]]) for name in ("x", "y", "z")] for row in selected]
    )
    raw_force = None
    force_names = ("f_RAW[1]", "f_RAW[2]", "f_RAW[3]")
    if all(name in index for name in force_names):
        raw_force = np.asarray(
            [[float(row[index[name]]) for name in force_names] for row in selected]
        )
    return ids, coordinates, raw_force


def _molecule_centers(coordinates: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lengths = bounds[:, 1] - bounds[:, 0]
    delta = coordinates[1::2] - coordinates[0::2]
    delta[:, :2] -= lengths[:2] * np.round(delta[:, :2] / lengths[:2])
    centers = coordinates[0::2] + 0.5 * delta
    centers[:, :2] = (
        (centers[:, :2] - bounds[:2, 0]) % lengths[:2] + bounds[:2, 0]
    )
    return centers


def _tree_coordinates(points: np.ndarray, bounds: np.ndarray, pseudo_z: float) -> np.ndarray:
    lengths = bounds[:, 1] - bounds[:, 0]
    result = np.asarray(points, dtype=float).copy()
    result[:, 0] = (result[:, 0] - bounds[0, 0]) % lengths[0]
    result[:, 1] = (result[:, 1] - bounds[1, 0]) % lengths[1]
    result[:, 2] = result[:, 2] - bounds[2, 0] + 0.25 * pseudo_z
    return result


def _polygon_metrics(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 3:
        return math.nan, math.nan
    from scipy.spatial import ConvexHull, QhullError
    try:
        hull = ConvexHull(points)
    except QhullError:
        return math.nan, math.nan
    area, perimeter = float(hull.volume), float(hull.area)
    return area, 4.0 * math.pi * area / perimeter**2 if perimeter > 0.0 else math.nan


def _interp(step: int, steps: np.ndarray, values: np.ndarray) -> float:
    return float(np.interp(float(step), steps, values))


def _motion_series(raw: Mapping[str, object], base: Path) -> dict[str, object]:
    columns = {
        "step": "TimeStep", "water_x": "v_dxwaterO", "water_y": "v_dywaterO",
        "all_n_x": "v_dxrelN", "all_n_y": "v_dyrelN",
        "sub_x": "c_COMsub[1]", "sub_y": "c_COMsub[2]", "sub_z": "c_COMsub[3]",
        **dict(raw.get("motion_columns", {})),
    }
    paths = [_resolve(path, base) for path in raw["motion_tables"]]
    names, table = stitch_motion_tables(
        paths,
        step_column=columns["step"],
        displacement_columns=(
            columns["water_x"], columns["water_y"],
            columns["all_n_x"], columns["all_n_y"],
        ),
    )
    index = {name: position for position, name in enumerate(names)}
    missing = [name for name in columns.values() if name not in index]
    if missing:
        raise ValueError(f"motion tables are missing columns {missing}")
    result: dict[str, object] = {"paths": paths}
    for key, name in columns.items():
        result[key] = table[:, index[name]]
    return result


def _tracked_periodic_center(
    points: np.ndarray,
    bounds: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    """Return a PBC center on the branch nearest the previous center."""

    if reference is None:
        return periodic_xy_center(points, bounds)
    center = np.empty(3, dtype=float)
    delta = np.asarray(points, dtype=float) - np.asarray(reference, dtype=float)
    delta[:, :2] = xy_minimum_image(delta, bounds)[:, :2]
    candidate = np.asarray(reference, dtype=float) + np.mean(delta, axis=0)
    lengths = bounds[:, 1] - bounds[:, 0]
    center[:2] = (candidate[:2] - bounds[:2, 0]) % lengths[:2] + bounds[:2, 0]
    center[2] = float(np.mean(points[:, 2]))
    return center


def _analyze_frame(
    frame: LammpsDumpFrame,
    raw: Mapping[str, object],
    motion: Mapping[str, object],
    surface_tree: object,
    pseudo_z: float,
    origin_substrate: np.ndarray,
    reference_center: np.ndarray | None,
) -> dict[str, object]:
    atom_range = _validate_range(raw["nitrogen_atom_range"], "nitrogen_atom_range")
    ids, coordinates, raw_force = _nitrogen_arrays(frame, atom_range)
    centers = _molecule_centers(coordinates, frame.bounds)
    pairs = ids.reshape(-1, 2)
    relocated_ids = {int(value) for value in raw.get("relocated_nitrogen_ids", [])}
    relocated = np.asarray(
        [bool(relocated_ids.intersection(pair.tolist())) for pair in pairs], dtype=bool
    )
    candidates = np.flatnonzero(~relocated)
    components = connected_components(
        centers[candidates], frame.bounds, float(raw["cluster_cutoff_A"])
    )
    if not components:
        raise ValueError(f"no non-relocated N2 components at step {frame.timestep}")
    members = candidates[components[0]]
    bubble = centers[members]
    center = _tracked_periodic_center(bubble, frame.bounds, reference_center)
    vectors = bubble - center
    vectors[:, :2] = xy_minimum_image(vectors, frame.bounds)[:, :2]
    vectors[:, 2] = bubble[:, 2] - center[2]
    direction = str(raw["direction"]).lower()
    drive_index, transverse_index = (0, 1) if direction == "x" else (1, 0)

    steps = np.asarray(motion["step"], dtype=float)
    substrate = np.asarray([
        _interp(frame.timestep, steps, np.asarray(motion[key], dtype=float))
        for key in ("sub_x", "sub_y", "sub_z")
    ])
    shift = substrate - origin_substrate
    lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
    shift[:2] -= lengths[:2] * np.round(shift[:2] / lengths[:2])
    molecule_distance = surface_tree.query(
        _tree_coordinates(coordinates - shift, frame.bounds, pseudo_z), k=1
    )[0].reshape(-1, 2).min(axis=1)
    contact_vectors = vectors[
        molecule_distance[members] <= float(raw["contact_cutoff_A"])
    ]
    footprint, circularity = _polygon_metrics(contact_vectors[:, :2])

    z05, z95 = np.quantile(vectors[:, 2], [0.05, 0.95])
    d05, d95 = np.quantile(vectors[:, drive_index], [0.05, 0.95])
    t05, t95 = np.quantile(vectors[:, transverse_index], [0.05, 0.95])
    radii = np.linalg.norm(vectors[:, :2], axis=1)
    eigenvalues = np.linalg.eigvalsh(vectors.T @ vectors / len(vectors))
    total = float(np.sum(eigenvalues))
    anisotropy = (
        1.5 * float(np.sum(eigenvalues**2)) / total**2 - 0.5
        if total > 0.0 else math.nan
    )
    advancing = (
        float(np.quantile(contact_vectors[:, drive_index], 0.95))
        if len(contact_vectors) else math.nan
    )
    receding = (
        float(np.quantile(contact_vectors[:, drive_index], 0.05))
        if len(contact_vectors) else math.nan
    )
    max_main_force = max_all_force = math.nan
    max_force_id = -1
    if raw_force is not None:
        force_norm = np.linalg.norm(raw_force, axis=1)
        maximum = int(np.argmax(force_norm))
        max_all_force, max_force_id = float(force_norm[maximum]), int(ids[maximum])
        main_atoms = np.concatenate((2 * members, 2 * members + 1))
        max_main_force = float(np.max(force_norm[main_atoms]))
    return {
        "step": frame.timestep, "center": center,
        "main_bubble_n2": len(members),
        "disconnected_n2": len(candidates) - len(members),
        "relocated_n2": int(np.count_nonzero(relocated)),
        "height_q05_q95_A": float(z95 - z05),
        "drive_width_q05_q95_A": float(d95 - d05),
        "transverse_width_q05_q95_A": float(t95 - t05),
        "lateral_radius_p90_A": float(np.quantile(radii, 0.9)),
        "relative_shape_anisotropy": anisotropy,
        "contact_n2": len(contact_vectors),
        "footprint_area_A2": footprint,
        "contact_line_circularity": circularity,
        "advancing_relative_A": advancing, "receding_relative_A": receding,
        "min_surface_distance_A": float(np.min(molecule_distance[members])),
        "max_main_raw_force_eV_A": max_main_force,
        "max_all_raw_force_eV_A": max_all_force,
        "max_all_raw_force_atom_id": max_force_id,
    }


def _column_values(
    rows: Sequence[Mapping[str, object]], name: str
) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows])


def _block_rows(
    rows: Sequence[Mapping[str, object]], block_ps: float, minimum_motion_mps: float
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    start, final, block_index = 0.0, float(rows[-1]["time_ps"]), 0
    while start < final - 1.0e-9:
        end = min(start + block_ps, final)
        selected = [
            row for row in rows
            if start - 1.0e-9 <= float(row["time_ps"]) <= end + 1.0e-9
        ]
        if len(selected) >= 2:
            time = np.asarray([float(row["time_ps"]) for row in selected])
            values = lambda name, block=selected: _column_values(block, name)
            bubble_velocity = _linear_velocity_mps(time, values("bubble_drive_A"))
            water_velocity = _linear_velocity_mps(time, values("water_drive_A"))
            footprint = values("footprint_area_A2")
            finite = footprint[np.isfinite(footprint)]
            mean_footprint = float(np.mean(finite)) if len(finite) else math.nan
            pattern = (
                "FORWARD" if bubble_velocity > minimum_motion_mps else
                "REVERSE" if bubble_velocity < -minimum_motion_mps else
                "PINNED_OR_REORGANIZING"
            )
            output.append({
                "case_id": selected[0]["case_id"],
                "branch_id": selected[0]["branch_id"],
                "direction": selected[0]["direction"],
                "block_index": block_index, "start_ps": start, "end_ps": end,
                "frames": len(selected), "bubble_velocity_mps": bubble_velocity,
                "water_velocity_mps": water_velocity,
                "relative_velocity_mps": _linear_velocity_mps(
                    time, values("bubble_minus_water_drive_A")
                ),
                "advancing_velocity_mps": _linear_velocity_mps(
                    time, values("advancing_displacement_A")
                ),
                "receding_velocity_mps": _linear_velocity_mps(
                    time, values("receding_displacement_A")
                ),
                "response_pattern": pattern,
                "mean_height_A": float(np.nanmean(values("height_q05_q95_A"))),
                "mean_drive_width_A": float(np.nanmean(values("drive_width_q05_q95_A"))),
                "mean_transverse_width_A": float(np.nanmean(values("transverse_width_q05_q95_A"))),
                "mean_footprint_area_A2": mean_footprint,
                "footprint_cv": (
                    float(np.std(finite, ddof=1) / mean_footprint)
                    if len(finite) > 1 and mean_footprint > 0.0 else math.nan
                ),
                "mean_shape_anisotropy": float(np.nanmean(values("relative_shape_anisotropy"))),
                "minimum_main_bubble_n2": int(np.min(values("main_bubble_n2"))),
                "maximum_raw_force_eV_A": float(np.nanmax(values("max_all_raw_force_eV_A"))),
            })
        start += block_ps
        block_index += 1
    return output


def _write_plot(
    rows: Sequence[Mapping[str, object]],
    blocks: Sequence[Mapping[str, object]],
    output: Path,
    font_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    font_manager.findfont(properties, fallback_to_default=False)
    matplotlib.rcParams["font.family"] = properties.get_name()
    time = np.asarray([float(row["time_ps"]) for row in rows]) / 1000.0
    figure, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex="col")
    axes[0, 0].plot(time, [row["bubble_drive_A"] for row in rows], label="bubble")
    axes[0, 0].plot(time, [row["water_drive_A"] for row in rows], label="water O")
    axes[0, 0].set_ylabel("Driven displacement (A)"); axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(time, [row["advancing_displacement_A"] for row in rows], label="advancing")
    axes[0, 1].plot(time, [row["receding_displacement_A"] for row in rows], label="receding")
    axes[0, 1].set_ylabel("Contact-line displacement (A)"); axes[0, 1].legend(frameon=False)
    for key, label in (("height_q05_q95_A", "height"), ("drive_width_q05_q95_A", "drive width"), ("transverse_width_q05_q95_A", "transverse width")):
        axes[1, 0].plot(time, [row[key] for row in rows], label=label)
    axes[1, 0].set_ylabel("Bubble extent (A)"); axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(time, [row["footprint_area_A2"] for row in rows])
    axes[1, 1].set_ylabel("Footprint area (A2)")
    block_time = np.asarray([
        (float(row["start_ps"]) + float(row["end_ps"])) / 2000.0 for row in blocks
    ])
    axes[2, 0].plot(block_time, [row["bubble_velocity_mps"] for row in blocks], marker="o", label="bubble")
    axes[2, 0].plot(block_time, [row["water_velocity_mps"] for row in blocks], marker="o", label="water O")
    axes[2, 0].set(xlabel="Driven time (ns)", ylabel="Block velocity (m/s)"); axes[2, 0].legend(frameon=False)
    axes[2, 1].plot(time, [row["max_all_raw_force_eV_A"] for row in rows])
    axes[2, 1].set(xlabel="Driven time (ns)", ylabel="Max N raw force (eV/A)")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{rows[0]['case_id']} / {str(rows[0]['direction']).upper()}")
    figure.tight_layout(); figure.savefig(output, dpi=240); plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path, output = Path(contract_path).resolve(), Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(raw)
    base = contract_path.parent
    surface_path = _resolve(raw["surface_model_data"], base)
    surface = _surface_coordinates(
        surface_path, _validate_range(raw["surface_atom_range"], "surface_atom_range")
    )
    motion = _motion_series(raw, base)
    motion_steps = np.asarray(motion["step"], dtype=float)
    maximum_step = int(raw.get("analysis_max_step", int(motion_steps[-1])))
    trajectory_paths: list[Path] = []
    frames: dict[int, LammpsDumpFrame] = {}
    for entry in raw["nitrogen_trajectories"]:
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError("each nitrogen trajectory must be an object with path")
        path = _resolve(entry["path"], base)
        trajectory_paths.append(path)
        lower = int(entry.get("min_step", 0))
        upper = min(int(entry.get("max_step", maximum_step)), maximum_step)
        for frame in iter_lammps_dump_records(path):
            if frame.timestep < lower:
                continue
            if frame.timestep > upper:
                break
            frames[frame.timestep] = frame
    ordered = [frames[step] for step in sorted(frames) if step <= maximum_step]
    if len(ordered) < 2:
        raise ValueError("fewer than two admitted nitrogen frames")
    bounds = ordered[0].bounds
    if any(not np.allclose(frame.bounds, bounds, atol=1.0e-8) for frame in ordered[1:]):
        raise ValueError("trajectory box changed")
    from scipy.spatial import cKDTree
    lengths = bounds[:, 1] - bounds[:, 0]
    pseudo_z = max(1.0e5, 10.0 * lengths[2])
    surface_tree = cKDTree(
        _tree_coordinates(surface, bounds, pseudo_z),
        boxsize=np.asarray([lengths[0], lengths[1], pseudo_z]),
    )
    origin_substrate = np.asarray([
        float(np.asarray(motion[key])[0]) for key in ("sub_x", "sub_y", "sub_z")
    ])
    geometry = []
    tracking_center = None
    for frame in ordered:
        item = _analyze_frame(
            frame,
            raw,
            motion,
            surface_tree,
            pseudo_z,
            origin_substrate,
            tracking_center,
        )
        tracking_center = np.asarray(item["center"], dtype=float)
        geometry.append(item)
    direction = str(raw["direction"]).lower()
    drive_index, transverse_index = (0, 1) if direction == "x" else (1, 0)
    previous = unwrapped = origin = None
    advancing_origin = receding_origin = None
    rows: list[dict[str, object]] = []
    for item in geometry:
        center = np.asarray(item.pop("center"), dtype=float)
        if previous is None:
            unwrapped, origin = center.copy(), center.copy()
        else:
            delta = center - previous
            delta[:2] = xy_minimum_image(delta[None, :], bounds)[0, :2]
            unwrapped = np.asarray(unwrapped) + delta
            unwrapped[2] = center[2]
        step = int(item["step"])
        advance = float(unwrapped[drive_index]) + float(item.pop("advancing_relative_A"))
        recede = float(unwrapped[drive_index]) + float(item.pop("receding_relative_A"))
        if advancing_origin is None and math.isfinite(advance):
            advancing_origin = advance
        if receding_origin is None and math.isfinite(recede):
            receding_origin = recede
        water_drive_key = "water_x" if direction == "x" else "water_y"
        water_transverse_key = "water_y" if direction == "x" else "water_x"
        all_n_key = "all_n_x" if direction == "x" else "all_n_y"
        bubble_drive = float(unwrapped[drive_index] - origin[drive_index])
        water_drive = _interp(step, motion_steps, np.asarray(motion[water_drive_key]))
        rows.append({
            "case_id": raw["case_id"], "branch_id": raw["branch_id"], "direction": direction,
            "step": step,
            "time_ps": (step - int(raw.get("time_origin_step", 0))) * float(raw["timestep_fs"]) / 1000.0,
            **item,
            "bubble_center_x_A": float(center[0]), "bubble_center_y_A": float(center[1]),
            "bubble_center_z_A": float(center[2]),
            "bubble_center_x_unwrapped_A": float(unwrapped[0]),
            "bubble_center_y_unwrapped_A": float(unwrapped[1]),
            "bubble_drive_A": bubble_drive,
            "bubble_transverse_A": float(unwrapped[transverse_index] - origin[transverse_index]),
            "water_drive_A": water_drive,
            "water_transverse_A": _interp(step, motion_steps, np.asarray(motion[water_transverse_key])),
            "bubble_minus_water_drive_A": bubble_drive - water_drive,
            "all_n_motion_drive_A": _interp(step, motion_steps, np.asarray(motion[all_n_key])),
            "advancing_position_A": advance, "receding_position_A": recede,
            "advancing_displacement_A": advance - advancing_origin if advancing_origin is not None else math.nan,
            "receding_displacement_A": recede - receding_origin if receding_origin is not None else math.nan,
        })
        previous = center
    blocks = _block_rows(
        rows, float(raw["block_ps"]), float(raw.get("minimum_motion_mps", 0.1))
    )
    output.mkdir(parents=True, exist_ok=False)
    _write_tsv(output / "frame_metrics.tsv", rows, FRAME_FIELDS)
    _write_tsv(output / "block_metrics.tsv", blocks, BLOCK_FIELDS)
    input_paths = [
        contract_path, surface_path, *trajectory_paths,
        *[Path(path) for path in motion["paths"]],
    ]
    _write_tsv(
        output / "input_manifest.tsv",
        [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in input_paths
        ],
        ("path", "size_bytes", "sha256"),
    )
    summary = {
        "status": "PASS", "case_id": raw["case_id"], "branch_id": raw["branch_id"],
        "direction": direction, "frames": len(rows), "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "driven_time_ns": float(rows[-1]["time_ps"]) / 1000.0,
        "main_bubble_n2_min": min(int(row["main_bubble_n2"]) for row in rows),
        "main_bubble_n2_final": int(rows[-1]["main_bubble_n2"]),
        "relocated_n2": int(rows[-1]["relocated_n2"]),
        "valid_contact_fraction": sum(
            math.isfinite(float(row["footprint_area_A2"])) for row in rows
        ) / len(rows),
        "bubble_drive_final_A": float(rows[-1]["bubble_drive_A"]),
        "water_drive_final_A": float(rows[-1]["water_drive_A"]),
        "bubble_to_water_displacement_ratio": (
            float(rows[-1]["bubble_drive_A"]) / float(rows[-1]["water_drive_A"])
            if abs(float(rows[-1]["water_drive_A"])) > 1.0e-12 else math.nan
        ),
        "advancing_drive_final_A": float(rows[-1]["advancing_displacement_A"]),
        "receding_drive_final_A": float(rows[-1]["receding_displacement_A"]),
        "max_raw_force_eV_A": float(np.nanmax([
            float(row["max_all_raw_force_eV_A"]) for row in rows
        ])),
        "block_patterns": {
            label: sum(row["response_pattern"] == label for row in blocks)
            for label in sorted({str(row["response_pattern"]) for row in blocks})
        },
        "contact_line_is_static_surface_translation_proxy": True,
        "single_trajectory_block_statistics_are_not_replicate_uncertainty": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "# Driven nanobubble flow-response report\n\n"
        f"Status: `{summary['status']}`.\n\n"
        f"Case `{summary['case_id']}` / `{direction.upper()}` contains {len(rows)} admitted "
        f"10 ps frames through {summary['driven_time_ns']:.3f} ns.\n\n"
        f"The main bubble moved {summary['bubble_drive_final_A']:.3f} A while water O moved "
        f"{summary['water_drive_final_A']:.3f} A along the driving axis.\n\n"
        "Contact-line positions use molecular N2 contact points relative to a translated static "
        "substrate reference. They are screening proxies, not density-derived contact angles or "
        "proof of slip.\n",
        encoding="utf-8",
    )
    if bool(raw.get("write_plots", True)):
        font_path = _resolve(raw["font_path"], base)
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        _write_plot(rows, blocks, output / "flow_response.png", font_path)
    with (output / "OUTPUT-SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "OUTPUT-SHA256SUMS":
                handle.write(f"{_sha256(path)}  ./{path.name}\n")
    return summary
