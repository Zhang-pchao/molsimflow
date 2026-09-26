"""Build morphology-aware surface and transport anisotropy maps for Stage B."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, iter_lammps_dump_records_until
from molsimflow.postprocess.constant_force_oxygen import (
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)
from molsimflow.postprocess.surface_site_enrichment import identify_surface_sites
from molsimflow.postprocess.upper_water_admission import read_lammps_atomic_data


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def _read_table(path: Path, delimiter: str | None = None) -> list[dict[str, str]]:
    if delimiter is None:
        delimiter = "\t" if path.suffix in {".tsv", ".tab"} else ","
    with _open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _float(value: object) -> float:
    return float(str(value))


def _verify_hashes(root: Path) -> None:
    manifest = root / "OUTPUT-SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    marker = f"/{root.name}/"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, raw_path = line.split(maxsplit=1)
        raw_path = raw_path.strip().removeprefix("./")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        elif not path.exists() and marker in raw_path:
            path = root / raw_path.split(marker, 1)[1]
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Source result hash mismatch: {path}")


def _bin_indices(
    xy: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    fractional = ((xy - lower) % lengths) / lengths
    indices = np.floor(fractional * grid_size).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)
    return indices[:, 0], indices[:, 1]


def _deposit_path_segments(
    accumulator: np.ndarray,
    starts: np.ndarray,
    deltas: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    grid_size: int,
) -> None:
    """Deposit straight-line interval displacements at half-cell resolution."""

    cell_lengths = lengths / grid_size
    for start, delta in zip(starts, deltas):
        segments = max(1, int(math.ceil(float(np.max(np.abs(delta) / (0.5 * cell_lengths))))))
        fractions = (np.arange(segments, dtype=float) + 0.5) / segments
        midpoints = (start + fractions[:, None] * delta - lower) % lengths + lower
        ix, iy = _bin_indices(midpoints, lower, lengths, grid_size)
        np.add.at(accumulator[0], (ix, iy), delta[0] / segments)
        np.add.at(accumulator[1], (ix, iy), delta[1] / segments)


def _frame_water_xy(
    frame: object,
    water_range: tuple[int, int],
    oxygen_type: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fields = frame.atom_fields
    index = {name: position for position, name in enumerate(fields)}
    required = {"id", "type", "x", "y", "ix", "iy"}
    missing = required.difference(index)
    if missing:
        raise ValueError(f"timestep {frame.timestep} is missing {sorted(missing)}")
    selected = []
    start, end = water_range
    lengths = box_lengths(frame.bounds)
    for row in frame.atom_rows:
        atom_id = int(row[index["id"]])
        if start <= atom_id <= end and int(row[index["type"]]) == oxygen_type:
            x = _float(row[index["x"]])
            y = _float(row[index["y"]])
            selected.append(
                (
                    atom_id,
                    x,
                    y,
                    x + int(row[index["ix"]]) * lengths[0],
                    y + int(row[index["iy"]]) * lengths[1],
                )
            )
    selected.sort(key=lambda row: row[0])
    if not selected:
        raise ValueError(f"no water oxygen atoms at timestep {frame.timestep}")
    array = np.asarray(selected, dtype=float)
    return (
        array[:, 0].astype(np.int64),
        array[:, 1:3],
        array[:, 3:5],
        frame.bounds[:2].copy(),
    )


def analyze_branch(
    trajectories: Sequence[Path],
    *,
    water_range: tuple[int, int],
    oxygen_type: int,
    timestep_fs: float,
    grid_size: int,
    event_keys: set[tuple[int, int]] | None = None,
    maximum_timestep: int | None = None,
) -> dict[str, object]:
    """Accumulate occupancy and a substrate-grid number-current proxy."""

    occupancy = np.zeros((grid_size, grid_size), dtype=float)
    displacement = np.zeros((2, grid_size, grid_size), dtype=float)
    event_samples: dict[tuple[int, int], dict[str, float]] = {}
    previous: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    first_step: int | None = None
    last_step: int | None = None
    frame_count = 0
    interval_count = 0
    oxygen_count = 0
    maximum_interval_displacement_A = 0.0
    intervals_above_half_box = 0
    lower: np.ndarray | None = None
    lengths: np.ndarray | None = None
    for trajectory in trajectories:
        for frame in iter_lammps_dump_records_until(trajectory, maximum_timestep):
            if last_step is not None and frame.timestep <= last_step:
                if frame.timestep == last_step:
                    continue
                raise ValueError("trajectory timesteps are not strictly increasing")
            ids, wrapped, unwrapped, bounds = _frame_water_xy(
                frame,
                water_range,
                oxygen_type,
            )
            current_lower = bounds[:, 0]
            current_lengths = bounds[:, 1] - bounds[:, 0]
            if lower is None:
                lower, lengths = current_lower, current_lengths
            elif not np.allclose(current_lower, lower) or not np.allclose(current_lengths, lengths):
                raise ValueError("lateral box changed across the branch")
            assert lengths is not None and lower is not None
            ix, iy = _bin_indices(wrapped, lower, lengths, grid_size)
            np.add.at(occupancy, (ix, iy), 1.0)
            frame_count += 1
            oxygen_count = len(ids)
            if first_step is None:
                first_step = frame.timestep
            if previous is not None:
                previous_step, previous_ids, previous_wrapped, previous_unwrapped, _ = previous
                if not np.array_equal(ids, previous_ids):
                    raise ValueError(f"water oxygen identity changed at {frame.timestep}")
                delta = unwrapped - previous_unwrapped
                maximum_interval_displacement_A = max(
                    maximum_interval_displacement_A,
                    float(np.max(np.linalg.norm(delta, axis=1))),
                )
                intervals_above_half_box += int(
                    np.count_nonzero(np.any(np.abs(delta) > 0.5 * lengths, axis=1))
                )
                _deposit_path_segments(
                    displacement,
                    previous_wrapped,
                    delta,
                    lower,
                    lengths,
                    grid_size,
                )
                interval_count += 1
                if event_keys:
                    id_to_index = {int(atom_id): pos for pos, atom_id in enumerate(ids)}
                    for key in [key for key in event_keys if key[0] == frame.timestep]:
                        pos = id_to_index.get(key[1])
                        if pos is None:
                            raise ValueError(f"missing event oxygen {key[1]} at {frame.timestep}")
                        event_samples[key] = {
                            "x_A": float(wrapped[pos, 0]),
                            "y_A": float(wrapped[pos, 1]),
                            "dx_A": float(delta[pos, 0]),
                            "dy_A": float(delta[pos, 1]),
                            "previous_step": previous_step,
                        }
            previous = (frame.timestep, ids, wrapped, unwrapped, bounds)
            last_step = frame.timestep
        if maximum_timestep is not None and last_step == maximum_timestep:
            break
    if first_step is None or last_step is None or lower is None or lengths is None:
        raise ValueError("branch contains no frames")
    duration_ps = (last_step - first_step) * timestep_fs / 1000.0
    if duration_ps <= 0.0:
        raise ValueError("branch duration is not positive")
    total_delta = np.sum(displacement, axis=(1, 2))
    velocity_mps = total_delta / oxygen_count / duration_ps * 100.0
    area_A2 = float(np.prod(lengths))
    bin_area_A2 = area_A2 / grid_size**2
    return {
        "occupancy": occupancy,
        "current": displacement / duration_ps / bin_area_A2,
        "frame_count": frame_count,
        "interval_count": interval_count,
        "oxygen_count": oxygen_count,
        "duration_ps": duration_ps,
        "velocity_mps": velocity_mps,
        "lower": lower,
        "lengths": lengths,
        "bin_area_A2": bin_area_A2,
        "event_samples": event_samples,
        "maximum_interval_displacement_A": maximum_interval_displacement_A,
        "intervals_above_half_box": intervals_above_half_box,
    }


def response_matrix_rows(
    case_id: str,
    branches: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    baseline = np.asarray(branches["none"]["velocity_mps"], dtype=float)
    x_raw = np.asarray(branches["x"]["velocity_mps"], dtype=float)
    y_raw = np.asarray(branches["y"]["velocity_mps"], dtype=float)
    x_response = x_raw - baseline
    y_response = y_raw - baseline
    matrix = {
        "case_id": case_id,
        "Jx_Fx_mps": float(x_response[0]),
        "Jx_Fy_mps": float(y_response[0]),
        "Jy_Fx_mps": float(x_response[1]),
        "Jy_Fy_mps": float(y_response[1]),
        "x_longitudinal_mps": float(x_response[0]),
        "x_lateral_mps": float(x_response[1]),
        "y_lateral_mps": float(y_response[0]),
        "y_longitudinal_mps": float(y_response[1]),
        "x_response_angle_deg": float(math.degrees(math.atan2(x_response[1], x_response[0]))),
        "y_response_angle_deg": float(math.degrees(math.atan2(y_response[1], y_response[0]))),
        "evidence_limit": "single_trajectory_descriptive_response_not_independent_replicates",
    }
    detail = []
    for direction, raw, response in (("x", x_raw, x_response), ("y", y_raw, y_response)):
        detail.append(
            {
                "case_id": case_id,
                "drive_direction": direction,
                "raw_vx_mps": float(raw[0]),
                "raw_vy_mps": float(raw[1]),
                "baseline_vx_mps": float(baseline[0]),
                "baseline_vy_mps": float(baseline[1]),
                "excess_vx_mps": float(response[0]),
                "excess_vy_mps": float(response[1]),
                "longitudinal_mps": float(response[0 if direction == "x" else 1]),
                "lateral_mps": float(response[1 if direction == "x" else 0]),
            }
        )
    return matrix, detail


def _surface_sites(
    model_data: Path,
    slab_range: tuple[int, int],
    surface_z_A: float,
    surface_depth_A: float,
    bond_cutoff_A: float,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    ids, atom_types, coordinates, bounds = read_lammps_atomic_data(model_data)
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError(f"atom IDs are not contiguous in {model_data}")
    type_map = {1: "H", 2: "O", 7: "C", 8: "Si"}
    elements = np.asarray([type_map.get(int(atom_type), "X") for atom_type in atom_types])
    lengths = box_lengths(bounds)
    sites = identify_surface_sites(
        elements,
        coordinates,
        lengths,
        slab_range=slab_range,
        surface_z=surface_z_A,
        surface_depth=surface_depth_A,
        bond_cutoff=bond_cutoff_A,
    )
    return sites, bounds[:2, 0], lengths[:2]


def _surface_grid(
    sites: Sequence[Mapping[str, object]],
    lower: np.ndarray,
    lengths: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray([[_float(row["x_A"]), _float(row["y_A"])] for row in sites])
    ix, iy = _bin_indices(xy, lower, lengths, grid_size)
    total = np.zeros((grid_size, grid_size), dtype=float)
    oh = np.zeros((grid_size, grid_size), dtype=float)
    np.add.at(total, (ix, iy), 1.0)
    np.add.at(
        oh,
        (ix, iy),
        np.asarray([row["site_type"] == "SiOH" for row in sites], dtype=float),
    )
    return total, np.divide(oh, total, out=np.full_like(total, np.nan), where=total > 0)


def _grid_rows(
    case_id: str,
    branch_id: str,
    direction: str,
    result: Mapping[str, object],
    baseline: Mapping[str, object],
    grid_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lower = np.asarray(result["lower"])
    lengths = np.asarray(result["lengths"])
    occupancy = np.asarray(result["occupancy"])
    base_occupancy = np.asarray(baseline["occupancy"])
    current = np.asarray(result["current"])
    base_current = np.asarray(baseline["current"])
    residence_rows = []
    flux_rows = []
    for ix in range(grid_size):
        for iy in range(grid_size):
            common = {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "bin_x": ix,
                "bin_y": iy,
                "x_center_A": lower[0] + (ix + 0.5) * lengths[0] / grid_size,
                "y_center_A": lower[1] + (iy + 0.5) * lengths[1] / grid_size,
                "bin_area_A2": result["bin_area_A2"],
            }
            mean_count = occupancy[ix, iy] / _float(result["frame_count"])
            baseline_count = base_occupancy[ix, iy] / _float(baseline["frame_count"])
            residence_rows.append(
                {
                    **common,
                    "mean_oxygen_count_per_frame": mean_count,
                    "baseline_mean_oxygen_count_per_frame": baseline_count,
                    "excess_mean_oxygen_count_per_frame": mean_count - baseline_count,
                }
            )
            flux_rows.append(
                {
                    **common,
                    "raw_current_x_oxygen_A_per_ps_A-2": current[0, ix, iy],
                    "raw_current_y_oxygen_A_per_ps_A-2": current[1, ix, iy],
                    "baseline_current_x_oxygen_A_per_ps_A-2": base_current[0, ix, iy],
                    "baseline_current_y_oxygen_A_per_ps_A-2": base_current[1, ix, iy],
                    "excess_current_x_oxygen_A_per_ps_A-2": (
                        current[0, ix, iy] - base_current[0, ix, iy]
                    ),
                    "excess_current_y_oxygen_A_per_ps_A-2": (
                        current[1, ix, iy] - base_current[1, ix, iy]
                    ),
                    "map_definition": (
                        "linearized_interval_path_displacement_per_time_and_bin_area_"
                        "subsnapshot_path_unresolved"
                    ),
                }
            )
    return residence_rows, flux_rows


def _tpcl_dwell_rows(
    case_id: str,
    branch_id: str,
    direction: str,
    contour_path: Path,
    lower: np.ndarray,
    lengths: np.ndarray,
    grid_size: int,
) -> list[dict[str, object]]:
    rows = _read_table(contour_path)
    xy = np.asarray([[_float(row["contour_x_A"]), _float(row["contour_y_A"])] for row in rows])
    ix, iy = _bin_indices(xy, lower, lengths, grid_size)
    count = np.zeros((grid_size, grid_size), dtype=float)
    np.add.at(count, (ix, iy), 1.0)
    frame_count = len({int(row["step"]) for row in rows})
    output = []
    for x_index in range(grid_size):
        for y_index in range(grid_size):
            output.append(
                {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "direction": direction,
                    "bin_x": x_index,
                    "bin_y": y_index,
                    "x_center_A": lower[0] + (x_index + 0.5) * lengths[0] / grid_size,
                    "y_center_A": lower[1] + (y_index + 0.5) * lengths[1] / grid_size,
                    "contour_point_observations": int(count[x_index, y_index]),
                    "contour_points_per_frame": count[x_index, y_index] / frame_count,
                    "frame_count": frame_count,
                    "map_definition": "sampled_contour_point_dwell_not_unique_contact_line_probability",
                }
            )
    return output


def _pearson(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(left) & np.isfinite(right)
    count = int(np.count_nonzero(mask))
    if count < 3 or np.std(left[mask]) == 0.0 or np.std(right[mask]) == 0.0:
        return math.nan, count
    return float(np.corrcoef(left[mask], right[mask])[0, 1]), count


def _plot_maps(
    cases: Sequence[str],
    site_rows: Sequence[Mapping[str, object]],
    residence_rows: Sequence[Mapping[str, object]],
    flux_rows: Sequence[Mapping[str, object]],
    response_rows: Sequence[Mapping[str, object]],
    output: Path,
    grid_size: int,
    font_path: Path | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    if font_path is not None:
        font_manager.fontManager.addfont(font_path)
        properties = font_manager.FontProperties(fname=font_path)
        font_manager.findfont(properties, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = properties.get_name()
    figure, axes = plt.subplots(len(cases), 4, figsize=(14.0, 3.1 * len(cases)))
    for row_index, case_id in enumerate(cases):
        sites = [row for row in site_rows if row["case_id"] == case_id]
        colors = ["#2ca02c" if row["site_type"] == "CH3" else "#1f77b4" for row in sites]
        axes[row_index, 0].scatter(
            [row["x_A"] for row in sites],
            [row["y_A"] for row in sites],
            c=colors,
            s=6,
            alpha=0.85,
        )
        axes[row_index, 0].set_title(f"{case_id}: surface sites")
        for column, direction in ((1, "x"), (2, "y")):
            selected = [
                row
                for row in flux_rows
                if row["case_id"] == case_id and row["direction"] == direction
            ]
            magnitude = np.asarray(
                [
                    math.hypot(
                        _float(row["excess_current_x_oxygen_A_per_ps_A-2"]),
                        _float(row["excess_current_y_oxygen_A_per_ps_A-2"]),
                    )
                    for row in selected
                ]
            ).reshape(grid_size, grid_size)
            axes[row_index, column].imshow(
                magnitude.T,
                origin="lower",
                cmap="magma",
                aspect="auto",
            )
            axes[row_index, column].set_title(f"{direction.upper()} excess current magnitude")
        selected_residence = [
            row
            for row in residence_rows
            if row["case_id"] == case_id and row["direction"] == "none"
        ]
        residence = np.asarray(
            [row["mean_oxygen_count_per_frame"] for row in selected_residence]
        ).reshape(grid_size, grid_size)
        axes[row_index, 3].imshow(
            residence.T,
            origin="lower",
            cmap="Blues",
            aspect="auto",
        )
        axes[row_index, 3].set_title("F0 water occupancy")
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output / "stage_b4_surface_transport_maps.png", dpi=240)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(cases), figsize=(3.3 * len(cases), 3.2))
    by_case = {str(row["case_id"]): row for row in response_rows}
    for axis, case_id in zip(np.atleast_1d(axes), cases):
        row = by_case[case_id]
        matrix = np.asarray(
            [
                [row["Jx_Fx_mps"], row["Jx_Fy_mps"]],
                [row["Jy_Fx_mps"], row["Jy_Fy_mps"]],
            ],
            dtype=float,
        )
        limit = max(float(np.max(np.abs(matrix))), 1.0e-12)
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
        axis.set_xticks([0, 1], ["Fx", "Fy"])
        axis.set_yticks([0, 1], ["Jx", "Jy"])
        axis.set_title(case_id)
        figure.colorbar(image, ax=axis, shrink=0.75, label="m/s")
    figure.tight_layout()
    figure.savefig(output / "stage_b4_response_matrices.png", dpi=240)
    plt.close(figure)


def _report_text(
    case_ids: Sequence[str],
    morphology_by_case: Mapping[str, str],
    reference_result_count: int,
) -> str:
    finite_cases = sorted(
        case_id
        for case_id in case_ids
        if morphology_by_case[case_id] == "finite_droplet"
    )
    island_cases = sorted(
        case_id
        for case_id in case_ids
        if morphology_by_case[case_id] == "water_islands"
    )
    ch3_control_included = "ch3_only" in case_ids
    control_text = (
        "The ch3_only intrinsic X/Y response is included in this contract."
        if ch3_control_included
        else "No ch3_only intrinsic X/Y control is included in this contract."
    )
    return (
        "# Morphology-aware anisotropy maps\n\n"
        f"Analyzed {len(case_ids)} interface cases ({', '.join(case_ids)}) on the same "
        "substrate-fixed periodic grid within each case. Water occupancy and directed "
        "current are F0-subtracted, and each response matrix retains longitudinal and "
        "lateral components.\n\n"
        f"Finite-droplet TPCL applicability: {', '.join(finite_cases) or 'none'}. "
        f"Water-island transfer-channel applicability: {', '.join(island_cases) or 'none'}. "
        f"Verified external reference result sets: {reference_result_count}.\n\n"
        "Spatial correlations and X/Y differences are single-trajectory descriptive "
        f"associations. {control_text} Pattern causality requires an independently "
        "prepared or composition-preserving scrambled surface and remains untested.\n"
    )


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    """Run a contract-defined Stage-B4 anisotropy synthesis."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(contract.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    base = contract_path.parent
    timestep_fs = _float(contract["timestep_fs"])
    grid_size = int(contract.get("grid_size", 24))
    oxygen_type = int(contract.get("oxygen_type", 2))
    surface_depth_A = _float(contract.get("surface_depth_A", 3.0))
    bond_cutoff_A = _float(contract.get("bond_cutoff_A", 1.25))
    reference_results = [resolve_path(path, base) for path in contract.get("reference_results", [])]
    for root in reference_results:
        _verify_hashes(root)

    surface_rows: list[dict[str, object]] = []
    residence_rows: list[dict[str, object]] = []
    flux_rows: list[dict[str, object]] = []
    response_rows: list[dict[str, object]] = []
    response_detail: list[dict[str, object]] = []
    tpcl_rows: list[dict[str, object]] = []
    transfer_ledger: list[dict[str, object]] = []
    transfer_map_rows: list[dict[str, object]] = []
    applicability_rows: list[dict[str, object]] = []
    correspondence_rows: list[dict[str, object]] = []
    branch_quality_rows: list[dict[str, object]] = []
    input_paths: set[Path] = {contract_path}
    input_paths.update(root / "OUTPUT-SHA256SUMS" for root in reference_results)
    case_results: dict[str, dict[str, dict[str, object]]] = {}
    case_site_grids: dict[str, np.ndarray] = {}
    cases = contract.get("cases", [])
    if not cases:
        raise ValueError("At least one interface case is required")
    for entry in cases:
        case_id = str(entry["case_id"])
        morphology = str(entry["morphology_class"])
        model_data = resolve_path(entry["model_data"], base)
        slab_range = tuple(map(int, entry["slab_range"]))
        water_range = tuple(map(int, entry["water_range"]))
        sites, model_lower, model_lengths = _surface_sites(
            model_data,
            slab_range,
            _float(entry["surface_z_A"]),
            surface_depth_A,
            bond_cutoff_A,
        )
        for site in sites:
            surface_rows.append({"case_id": case_id, **site})
        _, site_oh_fraction = _surface_grid(
            sites,
            model_lower,
            model_lengths,
            grid_size,
        )
        case_site_grids[case_id] = site_oh_fraction
        input_paths.add(model_data)

        exchange_path = (
            resolve_path(entry["persistent_island_exchange"], base)
            if entry.get("persistent_island_exchange")
            else None
        )
        exchange_rows = _read_table(exchange_path) if exchange_path else []
        if exchange_path:
            input_paths.add(exchange_path)
        event_keys_by_branch: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for row in exchange_rows:
            if row["exchange_class"] == "PERSISTENT_ISLAND_TRANSFER":
                event_keys_by_branch[row["branch_id"]].add(
                    (int(row["step"]), int(row["oxygen_id"]))
                )

        branches: dict[str, dict[str, object]] = {}
        for branch in entry["branches"]:
            branch_id = str(branch["branch_id"])
            direction = str(branch["direction"]).lower()
            trajectories = [resolve_path(path, base) for path in branch["trajectories"]]
            input_paths.update(trajectories)
            result = analyze_branch(
                trajectories,
                water_range=water_range,
                oxygen_type=oxygen_type,
                timestep_fs=timestep_fs,
                grid_size=grid_size,
                event_keys=event_keys_by_branch.get(branch_id),
                maximum_timestep=(
                    int(branch["maximum_timestep"])
                    if branch.get("maximum_timestep")
                    else None
                ),
            )
            branches[direction] = result
            branch_quality_rows.append(
                {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "direction": direction,
                    "frame_count": result["frame_count"],
                    "interval_count": result["interval_count"],
                    "oxygen_count": result["oxygen_count"],
                    "duration_ps": result["duration_ps"],
                    "maximum_interval_displacement_A": result["maximum_interval_displacement_A"],
                    "intervals_above_half_box": result["intervals_above_half_box"],
                    "map_path_limit": (
                        "PASS_WITH_LINEARIZED_LONG_INTERVALS_SUBSNAPSHOT_PATH_UNRESOLVED"
                        if result["intervals_above_half_box"]
                        else "PASS"
                    ),
                }
            )
            if branch.get("tpcl_contours"):
                contour = resolve_path(branch["tpcl_contours"], base)
                input_paths.add(contour)
                tpcl_rows.extend(
                    _tpcl_dwell_rows(
                        case_id,
                        branch_id,
                        direction,
                        contour,
                        np.asarray(result["lower"]),
                        np.asarray(result["lengths"]),
                        grid_size,
                    )
                )
        if set(branches) != {"none", "x", "y"}:
            raise ValueError(f"{case_id}: branches must define none, x, and y")
        case_results[case_id] = branches
        matrix, detail = response_matrix_rows(case_id, branches)
        response_rows.append(matrix)
        response_detail.extend(detail)
        applicability_rows.append(
            {
                "case_id": case_id,
                "morphology_class": morphology,
                "surface_map": "APPLICABLE",
                "water_residence_map": "APPLICABLE",
                "directed_flux_map": "APPLICABLE",
                "tpcl_dwell_map": (
                    "APPLICABLE_FINITE_DROPLET"
                    if morphology == "finite_droplet"
                    else "NOT_APPLICABLE_NO_UNIQUE_TPCL"
                ),
                "island_transfer_channel_map": (
                    "APPLICABLE_WATER_ISLANDS"
                    if morphology == "water_islands"
                    else "NOT_APPLICABLE_OTHER_MORPHOLOGY"
                ),
            }
        )
        for direction in ("none", "x", "y"):
            branch_id = next(
                str(row["branch_id"])
                for row in entry["branches"]
                if str(row["direction"]).lower() == direction
            )
            residence, flux = _grid_rows(
                case_id,
                branch_id,
                direction,
                branches[direction],
                branches["none"],
                grid_size,
            )
            residence_rows.extend(residence)
            flux_rows.extend(flux)

        if exchange_rows:
            branch_by_id = {
                str(row["branch_id"]): str(row["direction"]).lower() for row in entry["branches"]
            }
            aggregate: dict[tuple[str, int, int], dict[str, float]] = defaultdict(
                lambda: {"count": 0.0, "dx": 0.0, "dy": 0.0}
            )
            for row in exchange_rows:
                if row["exchange_class"] != "PERSISTENT_ISLAND_TRANSFER":
                    continue
                branch_id = row["branch_id"]
                direction = branch_by_id[branch_id]
                result = branches[direction]
                key = (int(row["step"]), int(row["oxygen_id"]))
                sample = result["event_samples"].get(key)
                if sample is None:
                    raise ValueError(f"missing event sample for {branch_id}/{key}")
                lower = np.asarray(result["lower"])
                lengths = np.asarray(result["lengths"])
                indices = _bin_indices(
                    np.asarray([[sample["x_A"], sample["y_A"]]]),
                    lower,
                    lengths,
                    grid_size,
                )
                ix, iy = int(indices[0][0]), int(indices[1][0])
                transfer_ledger.append(
                    {
                        "case_id": case_id,
                        "branch_id": branch_id,
                        "direction": direction,
                        "step": row["step"],
                        "time_ps": row["time_ps"],
                        "oxygen_id": row["oxygen_id"],
                        "source_track_id": row["source_track_id"],
                        "target_track_id": row["target_track_id"],
                        "source_role": row["source_role"],
                        "target_role": row["target_role"],
                        "x_A": sample["x_A"],
                        "y_A": sample["y_A"],
                        "dx_A": sample["dx_A"],
                        "dy_A": sample["dy_A"],
                        "bin_x": ix,
                        "bin_y": iy,
                    }
                )
                cell = aggregate[(branch_id, ix, iy)]
                cell["count"] += 1
                cell["dx"] += sample["dx_A"]
                cell["dy"] += sample["dy_A"]
            for (branch_id, ix, iy), cell in sorted(aggregate.items()):
                direction = branch_by_id[branch_id]
                result = branches[direction]
                lower = np.asarray(result["lower"])
                lengths = np.asarray(result["lengths"])
                transfer_map_rows.append(
                    {
                        "case_id": case_id,
                        "branch_id": branch_id,
                        "direction": direction,
                        "bin_x": ix,
                        "bin_y": iy,
                        "x_center_A": lower[0] + (ix + 0.5) * lengths[0] / grid_size,
                        "y_center_A": lower[1] + (iy + 0.5) * lengths[1] / grid_size,
                        "persistent_transfer_count": int(cell["count"]),
                        "persistent_transfer_dx_A": cell["dx"],
                        "persistent_transfer_dy_A": cell["dy"],
                    }
                )

    for case_id, branches in case_results.items():
        oh_fraction = case_site_grids[case_id].ravel()
        for direction in ("x", "y"):
            residence = np.asarray(branches[direction]["occupancy"], dtype=float)
            residence /= _float(branches[direction]["frame_count"])
            baseline_residence = np.asarray(branches["none"]["occupancy"], dtype=float)
            baseline_residence /= _float(branches["none"]["frame_count"])
            excess_residence = (residence - baseline_residence).ravel()
            current = np.asarray(branches[direction]["current"], dtype=float)
            base_current = np.asarray(branches["none"]["current"], dtype=float)
            excess_current = np.hypot(*(current - base_current)).ravel()
            residence_corr, residence_bins = _pearson(oh_fraction, excess_residence)
            current_corr, current_bins = _pearson(oh_fraction, excess_current)
            correspondence_rows.append(
                {
                    "case_id": case_id,
                    "direction": direction,
                    "oh_fraction_vs_excess_residence_pearson_r": residence_corr,
                    "oh_fraction_vs_excess_residence_bins": residence_bins,
                    "oh_fraction_vs_excess_current_magnitude_pearson_r": current_corr,
                    "oh_fraction_vs_excess_current_bins": current_bins,
                    "evidence_limit": "spatial_association_not_pattern_causality",
                }
            )

    output.mkdir(parents=True)
    write_tsv(output / "surface_functional_group_map.tsv", surface_rows, tuple(surface_rows[0]))
    write_tsv(output / "water_residence_map.tsv", residence_rows, tuple(residence_rows[0]))
    write_tsv(output / "directed_flux_map.tsv", flux_rows, tuple(flux_rows[0]))
    write_tsv(output / "transport_response_matrix.tsv", response_rows, tuple(response_rows[0]))
    write_tsv(output / "transport_response_detail.tsv", response_detail, tuple(response_detail[0]))
    write_tsv(output / "map_applicability.tsv", applicability_rows, tuple(applicability_rows[0]))
    write_tsv(
        output / "spatial_correspondence.tsv",
        correspondence_rows,
        tuple(correspondence_rows[0]),
    )
    write_tsv(
        output / "branch_map_quality.tsv",
        branch_quality_rows,
        tuple(branch_quality_rows[0]),
    )
    if tpcl_rows:
        write_tsv(output / "tpcl_dwell_map.tsv", tpcl_rows, tuple(tpcl_rows[0]))
    if transfer_ledger:
        write_tsv(
            output / "persistent_island_transfer_spatial_ledger.tsv",
            transfer_ledger,
            tuple(transfer_ledger[0]),
        )
        write_tsv(
            output / "island_transfer_channel_map.tsv",
            transfer_map_rows,
            tuple(transfer_map_rows[0]),
        )
    input_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(input_paths)
    ]
    write_tsv(output / "input_manifest.tsv", input_rows, tuple(input_rows[0]))
    font = resolve_path(contract["font_path"], base) if contract.get("font_path") else None
    case_ids = [str(entry["case_id"]) for entry in cases]
    morphology_by_case = {
        str(entry["case_id"]): str(entry["morphology_class"]) for entry in cases
    }
    ch3_control_included = "ch3_only" in case_ids
    _plot_maps(
        case_ids,
        surface_rows,
        residence_rows,
        flux_rows,
        response_rows,
        output,
        grid_size,
        font,
    )
    summary = {
        "status": "PASS",
        "case_count": len(cases),
        "branch_count": len(branch_quality_rows),
        "grid_size": grid_size,
        "surface_site_rows": len(surface_rows),
        "residence_map_rows": len(residence_rows),
        "directed_flux_map_rows": len(flux_rows),
        "tpcl_dwell_map_rows": len(tpcl_rows),
        "persistent_island_transfer_rows": len(transfer_ledger),
        "response_matrix_rows": len(response_rows),
        "single_trajectory_descriptive_only": True,
        "pattern_causality_established": False,
        "case_ids": case_ids,
        "reference_result_count": len(reference_results),
        "ch3_intrinsic_xy_control_included": ch3_control_included,
        "new_md_submitted": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        _report_text(case_ids, morphology_by_case, len(reference_results)),
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_contract(args.contract, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
