"""Synthesize morphology-specific constant-force diagnostics from existing results.

The Stage A workflow consumes immutable post-processing tables plus an optional
three-atom identity trace. It does not reread full trajectories for transport,
network, contact-line, or morphology observables. Statistics describe one
trajectory per branch and are never promoted to replicate uncertainty.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import box_lengths, iter_lammps_dump_frames


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if delimiter is None:
        table_suffix = (
            Path(path.stem).suffix.lower() if path.suffix == ".gz" else path.suffix.lower()
        )
        delimiter = "\t" if table_suffix in {".tsv", ".tab", ".dat"} else ","
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, mode="rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Table has no header: {path}")
        return [dict(row) for row in reader]


def _read_motion(path: Path) -> list[dict[str, str]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                fields = stripped[1:].split()
                if fields and fields[0] == "TimeStep":
                    header = fields
                continue
            if header is None:
                raise ValueError(f"Missing '# TimeStep' header in {path}")
            values = stripped.split()
            if len(values) != len(header):
                raise ValueError(f"Column mismatch in {path}: {len(values)} != {len(header)}")
            rows.append(dict(zip(header, values)))
    if not rows:
        raise ValueError(f"No motion rows in {path}")
    return rows


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return math.fsum(finite) / len(finite) if finite else math.nan


def _weighted_mean(values: Iterable[tuple[float, float]]) -> float:
    finite = [
        (float(value), float(weight))
        for value, weight in values
        if math.isfinite(float(value)) and math.isfinite(float(weight)) and float(weight) > 0
    ]
    total = math.fsum(weight for _, weight in finite)
    return math.fsum(value * weight for value, weight in finite) / total if total else math.nan


def _slope_mps(times_ps: Sequence[float], positions_A: Sequence[float]) -> float:
    times = np.asarray(times_ps, dtype=float)
    positions = np.asarray(positions_A, dtype=float)
    mask = np.isfinite(times) & np.isfinite(positions)
    times = times[mask]
    positions = positions[mask]
    if len(times) < 2 or np.ptp(times) <= 0.0:
        return math.nan
    return 100.0 * float(np.polyfit(times, positions, 1)[0])


def _rank(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=float)
    start = 0
    while start < len(data):
        end = start + 1
        while end < len(data) and data[order[end]] == data[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, int]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    if np.count_nonzero(mask) < 4:
        return math.nan, int(np.count_nonzero(mask))
    x_rank = _rank(x_array[mask])
    y_rank = _rank(y_array[mask])
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return math.nan, len(x_rank)
    return float(np.corrcoef(x_rank, y_rank)[0, 1]), len(x_rank)


def _block_index(time_ps: float, block_ps: float, final_time_ps: float) -> int:
    index = math.floor(max(time_ps, 0.0) / block_ps + 1.0e-12)
    count = max(1, math.ceil(final_time_ps / block_ps - 1.0e-12))
    return min(index, count - 1)


def _group_blocks(
    rows: Sequence[Mapping[str, object]], block_ps: float, time_key: str = "time_ps"
) -> dict[int, list[Mapping[str, object]]]:
    final_time = max(_float(row[time_key]) for row in rows)
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_block_index(_float(row[time_key]), block_ps, final_time)].append(row)
    return dict(grouped)


def _add_input(manifest: list[dict[str, object]], path: Path, kind: str) -> None:
    manifest.append(
        {
            "kind": kind,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    )


def _distance_xy_periodic(a: np.ndarray, b: np.ndarray, lengths: np.ndarray) -> float:
    delta = np.asarray(a - b, dtype=float)
    delta[:2] -= lengths[:2] * np.round(delta[:2] / lengths[:2])
    return float(np.linalg.norm(delta))


def _extract_identity_track(
    config: Mapping[str, object], base: Path, manifest: list[dict[str, object]]
) -> list[dict[str, object]]:
    atom_ids = config.get("atom_ids")
    if not isinstance(atom_ids, dict):
        raise TypeError("mixed275.identity.atom_ids must be an object")
    hydrogen_id = int(atom_ids["hydrogen"])
    carbon_id = int(atom_ids["carbon"])
    oxygen_id = int(atom_ids["framework_oxygen"])
    needed = {hydrogen_id, carbon_id, oxygen_id}
    origin = int(config["time_origin_step"])
    timestep_fs = float(config["timestep_fs"])
    ch_cutoff = float(config.get("ch_cutoff_A", 1.35))
    oh_cutoff = float(config.get("oh_cutoff_A", 1.35))
    output: list[dict[str, object]] = []
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("mixed275.identity.cases must be a non-empty list")
    for entry in cases:
        branch_id = str(entry["branch_id"])
        by_step = {}
        for raw_path in entry["trajectories"]:
            path = _resolve(base, raw_path)
            _add_input(manifest, path, "identity_trajectory")
            for frame in iter_lammps_dump_frames(path, needed_atom_ids=needed):
                if set(frame.selected_positions) != needed:
                    missing = sorted(needed.difference(frame.selected_positions))
                    raise ValueError(f"{path}: missing selected atom ids {missing}")
                h = frame.selected_positions[hydrogen_id]
                c = frame.selected_positions[carbon_id]
                o = frame.selected_positions[oxygen_id]
                lengths = box_lengths(frame.bounds)

                d_ch = _distance_xy_periodic(c, h, lengths)
                d_oh = _distance_xy_periodic(o, h, lengths)
                owner = "carbon" if d_ch < d_oh else "framework_oxygen" if d_oh < d_ch else "tie"
                by_step[frame.timestep] = {
                    "case_id": "mixed275",
                    "branch_id": branch_id,
                    "step": frame.timestep,
                    "time_ps": (frame.timestep - origin) * timestep_fs / 1000.0,
                    "hydrogen_id": hydrogen_id,
                    "carbon_id": carbon_id,
                    "framework_oxygen_id": oxygen_id,
                    "distance_CH_A": d_ch,
                    "distance_OH_A": d_oh,
                    "distance_CO_A": _distance_xy_periodic(c, o, lengths),
                    "distance_CH_minus_OH_A": d_ch - d_oh,
                    "nearest_owner": owner,
                    "within_CH_cutoff": d_ch <= ch_cutoff,
                    "within_OH_cutoff": d_oh <= oh_cutoff,
                }
        output.extend(by_step[step] for step in sorted(by_step))
    return output


def _add_event_post_minus_pre_deltas(result: dict[str, object]) -> None:
    """Add consistently named event deltas to a mixed275 event row."""

    result["main_axis_velocity_post_minus_pre_mps"] = _float(
        result["main_axis_velocity_post_mps"]
    ) - _float(result["main_axis_velocity_pre_mps"])
    for field in (
        "local_surface_hbond",
        "local_water_hbond_degree",
        "local_q_tet",
        "local_lsi_A2",
        "identity_CH_A",
        "identity_OH_A",
    ):
        result[f"{field}_post_minus_pre"] = _float(result[f"{field}_post"]) - _float(
            result[f"{field}_pre"]
        )



def _mixed275_analysis(
    config: Mapping[str, object],
    base: Path,
    block_ps: float,
    event_half_window_ps: float,
    manifest: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    island_dir = _resolve(base, config["island_results"])
    event_dir = _resolve(base, config["event_results"])
    sources = {
        "islands": island_dir / "island_timeseries.tsv",
        "exchange": island_dir / "molecule_exchange.tsv",
        "lineage": island_dir / "lineage_events.tsv",
        "events": event_dir / "events.tsv",
        "identity": event_dir / "atom_identity.tsv",
    }
    for kind, path in sources.items():
        _add_input(manifest, path, f"mixed275_{kind}")
    island_rows = _read_rows(sources["islands"])
    exchange_rows = _read_rows(sources["exchange"])
    lineage_rows = _read_rows(sources["lineage"])
    event_rows = _read_rows(sources["events"])
    atom_rows = _read_rows(sources["identity"])

    identity_track = _extract_identity_track(config["identity"], base, manifest)
    identity_by_branch: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in identity_track:
        identity_by_branch[str(row["branch_id"])].append(row)

    by_branch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in island_rows:
        by_branch[row["branch_id"]].append(row)
    directions = {branch: rows[0]["direction"].lower() for branch, rows in by_branch.items()}
    if list(directions.values()).count("none") != 1:
        raise ValueError("mixed275 island results must contain one direction=none branch")
    baseline_branch = next(
        branch for branch, direction in directions.items() if direction == "none"
    )

    raw_blocks: dict[tuple[str, int], dict[str, float]] = {}
    for branch, rows in by_branch.items():
        for block, selected in _group_blocks(rows, block_ps).items():
            main = [row for row in selected if _int(row["component_rank"]) == 1]
            satellite = [row for row in selected if _int(row["component_rank"]) > 1]
            raw_blocks[(branch, block)] = {
                "main_vx_mps": _mean(_float(row["vx_mps"]) for row in main),
                "main_vy_mps": _mean(_float(row["vy_mps"]) for row in main),
                "satellite_vx_mps": _weighted_mean(
                    (_float(row["vx_mps"]), _float(row["size"])) for row in satellite
                ),
                "satellite_vy_mps": _weighted_mean(
                    (_float(row["vy_mps"]), _float(row["size"])) for row in satellite
                ),
                "main_fraction": _mean(_float(row["fraction"]) for row in main),
                "satellite_oxygen": _mean([sum(_float(item["size"]) for item in satellite)]),
            }
    block_rows: list[dict[str, object]] = []
    for (branch, block), values in sorted(raw_blocks.items()):
        direction = directions[branch]
        reference = raw_blocks.get((baseline_branch, block))
        if reference is None:
            raise ValueError(f"Missing mixed275 baseline block {block}")
        axis = "x" if direction in {"none", "x"} else "y"
        block_rows.append(
            {
                "case_id": "mixed275",
                "branch_id": branch,
                "direction": direction,
                "block_index": block,
                "start_ps": block * block_ps,
                "end_ps": (block + 1) * block_ps,
                **values,
                "main_axis_velocity_mps": values[f"main_v{axis}_mps"],
                "baseline_main_axis_velocity_mps": reference[f"main_v{axis}_mps"],
                "main_excess_axis_velocity_mps": (
                    0.0
                    if direction == "none"
                    else values[f"main_v{axis}_mps"] - reference[f"main_v{axis}_mps"]
                ),
                "satellite_axis_velocity_mps": values[f"satellite_v{axis}_mps"],
                "baseline_satellite_axis_velocity_mps": reference[f"satellite_v{axis}_mps"],
                "satellite_excess_axis_velocity_mps": (
                    0.0
                    if direction == "none"
                    else values[f"satellite_v{axis}_mps"] - reference[f"satellite_v{axis}_mps"]
                ),
            }
        )

    atom_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in atom_rows:
        atom_by_event[row["event_id"]].append(row)
    selected_events = [
        row
        for row in event_rows
        if _int(row["proton_pool_first"]) != _int(row["proton_pool_last"])
        or not _bool(row["species_returned"])
    ]
    proton_changes = sum(
        _int(row["proton_pool_first"]) != _int(row["proton_pool_last"]) for row in event_rows
    )
    nonreturning = sum(not _bool(row["species_returned"]) for row in event_rows)
    if "expected_proton_pool_change_events" in config and proton_changes != int(
        config["expected_proton_pool_change_events"]
    ):
        raise ValueError(f"Expected proton-pool event count mismatch: {proton_changes}")
    if "expected_nonreturning_events" in config and nonreturning != int(
        config["expected_nonreturning_events"]
    ):
        raise ValueError(f"Expected nonreturning event count mismatch: {nonreturning}")

    event_output: list[dict[str, object]] = []
    for event in selected_events:
        event_id = event["event_id"]
        branch = event["branch_id"]
        anchor = _float(event["anchor_time_ps"])
        axis = "x" if directions[branch] in {"none", "x"} else "y"
        main = [row for row in by_branch[branch] if _int(row["component_rank"]) == 1]
        pre_main = [
            row for row in main if -event_half_window_ps <= _float(row["time_ps"]) - anchor < 0
        ]
        post_main = [
            row for row in main if 0 < _float(row["time_ps"]) - anchor <= event_half_window_ps
        ]
        local = atom_by_event[event_id]
        pre_local = [row for row in local if _float(row["relative_time_ps"]) < 0]
        post_local = [row for row in local if _float(row["relative_time_ps"]) > 0]

        def local_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
            return _mean(_float(row[field]) for row in rows)

        def surface_hbond(rows: Sequence[Mapping[str, object]]) -> float:
            return _mean(
                _float(row["water_donor_surface_hbond_count"])
                + _float(row["surface_donor_water_hbond_count"])
                for row in rows
            )

        trace = identity_by_branch[branch]
        event_trace = [
            row for row in trace if abs(_float(row["time_ps"]) - anchor) <= event_half_window_ps
        ]
        pre_trace = [row for row in event_trace if _float(row["time_ps"]) < anchor]
        post_trace = [row for row in event_trace if _float(row["time_ps"]) > anchor]
        exchange = [
            row
            for row in exchange_rows
            if row["branch_id"] == branch
            and abs(_float(row["time_ps"]) - anchor) <= event_half_window_ps
        ]
        lineage = [
            row
            for row in lineage_rows
            if row["branch_id"] == branch
            and abs(_float(row["time_ps"]) - anchor) <= event_half_window_ps
        ]
        result: dict[str, object] = {
            "event_id": event_id,
            "branch_id": branch,
            "direction": directions[branch],
            "event_types": event["event_types"],
            "anchor_time_ps": anchor,
            "proton_pool_change": _int(event["proton_pool_last"])
            - _int(event["proton_pool_first"]),
            "species_returned": _bool(event["species_returned"]),
            "persistent_transfer_count_pm20ps": sum(
                row["exchange_class"] == "persistent_island_transfer" for row in exchange
            ),
            "lineage_reassignment_count_pm20ps": sum(
                row["exchange_class"] == "split_merge_lineage_reassignment" for row in exchange
            ),
            "split_event_count_pm20ps": sum(row["event_type"] == "split" for row in lineage),
            "merge_event_count_pm20ps": sum(row["event_type"] == "merge" for row in lineage),
            "main_axis_velocity_pre_mps": _mean(_float(row[f"v{axis}_mps"]) for row in pre_main),
            "main_axis_velocity_post_mps": _mean(_float(row[f"v{axis}_mps"]) for row in post_main),
            "local_surface_hbond_pre": surface_hbond(pre_local),
            "local_surface_hbond_post": surface_hbond(post_local),
            "local_water_hbond_degree_pre": local_mean(pre_local, "water_hbond_degree"),
            "local_water_hbond_degree_post": local_mean(post_local, "water_hbond_degree"),
            "local_q_tet_pre": local_mean(pre_local, "q_tet"),
            "local_q_tet_post": local_mean(post_local, "q_tet"),
            "local_lsi_A2_pre": local_mean(pre_local, "lsi_A2"),
            "local_lsi_A2_post": local_mean(post_local, "lsi_A2"),
            "identity_CH_A_pre": local_mean(pre_trace, "distance_CH_A"),
            "identity_CH_A_post": local_mean(post_trace, "distance_CH_A"),
            "identity_OH_A_pre": local_mean(pre_trace, "distance_OH_A"),
            "identity_OH_A_post": local_mean(post_trace, "distance_OH_A"),
            "identity_owner_changes_pm20ps": sum(
                left["nearest_owner"] != right["nearest_owner"]
                for left, right in zip(event_trace, event_trace[1:])
            ),
        }
        _add_event_post_minus_pre_deltas(result)
        event_output.append(result)
    return block_rows, event_output, identity_track


def _water_block_metrics(path: Path, block_ps: float) -> dict[tuple[str, int], dict[str, float]]:
    rows = [row for row in _read_rows(path) if row.get("region") == "all"]
    by_branch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_branch[row["branch_id"]].append(row)
    output: dict[tuple[str, int], dict[str, float]] = {}
    for branch, selected in by_branch.items():
        for block, block_rows in _group_blocks(selected, block_ps).items():
            output[(branch, block)] = {
                "water_q_tet": _mean(_float(row["mean_q_tet"]) for row in block_rows),
                "water_lsi_A2": _mean(_float(row["mean_lsi_A2"]) for row in block_rows),
                "water_hbond_degree": _mean(
                    2.0 * _float(row["water_water_hbond_edges"]) / _float(row["water_count"])
                    for row in block_rows
                    if _float(row["water_count"]) > 0
                ),
                "surface_hbond_per_water": _mean(
                    _float(row["water_surface_hbond_edges"]) / _float(row["water_count"])
                    for row in block_rows
                    if _float(row["water_count"]) > 0
                ),
                "water_edge_turnover": _mean(
                    _float(row["water_water_edge_turnover"]) for row in block_rows
                ),
                "surface_edge_turnover": _mean(
                    _float(row["water_surface_edge_turnover"]) for row in block_rows
                ),
            }
    return output


def _oh_analysis(
    config: Mapping[str, object],
    base: Path,
    block_ps: float,
    manifest: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    layers_dir = _resolve(base, config["layer_results"])
    paths = {
        "layers": layers_dir / "layer_timeseries.tsv",
        "modes": layers_dir / "density_modes.tsv",
        "exchange": layers_dir / "layer_exchange.tsv",
        "residence": layers_dir / "residence_summary.tsv",
        "water": _resolve(base, config["water_results"]) / "water_structure_by_frame.tsv",
    }
    for kind, path in paths.items():
        _add_input(manifest, path, f"oh_{kind}")
    layer_rows = _read_rows(paths["layers"])
    mode_rows = _read_rows(paths["modes"])
    exchange_rows = _read_rows(paths["exchange"])
    residence_rows = _read_rows(paths["residence"])
    water_blocks = _water_block_metrics(paths["water"], block_ps)
    directions = {row["branch_id"]: row["direction"].lower() for row in layer_rows}
    baseline_branch = next(
        branch for branch, direction in directions.items() if direction == "none"
    )
    core_layers = [int(value) for value in config.get("core_layer_indices", [0, 1, 2])]
    outer_layers = [int(value) for value in config.get("outer_layer_indices", [3, 4])]

    by_branch_layer: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in layer_rows:
        by_branch_layer[(row["branch_id"], _int(row["layer_index"]))].append(row)
    raw: dict[tuple[str, int, int], dict[str, float]] = {}
    for (branch, layer), rows in by_branch_layer.items():
        direction = directions[branch]
        axis = "x" if direction in {"none", "x"} else "y"
        for block, selected in _group_blocks(rows, block_ps).items():
            raw[(branch, layer, block)] = {
                "mean_count": _mean(_float(row["count"]) for row in selected),
                "occupied_fraction": _mean(float(_float(row["count"]) > 0) for row in selected),
                "axis_velocity_mps": _mean(_float(row[f"mean_v{axis}_mps"]) for row in selected),
                "axis_flux_molecules_per_A_ps": _mean(
                    _float(row[f"surface_flux_{axis}_molecules_per_A_ps"]) for row in selected
                ),
            }
    block_rows: list[dict[str, object]] = []
    for (branch, layer, block), values in sorted(raw.items()):
        reference = raw.get((baseline_branch, layer, block))
        if reference is None:
            raise ValueError(f"Missing OH baseline layer/block: {layer}/{block}")
        direction = directions[branch]
        row: dict[str, object] = {
            "case_id": "oh_only",
            "branch_id": branch,
            "direction": direction,
            "layer_index": layer,
            "human_layer": layer + 1,
            "block_index": block,
            "start_ps": block * block_ps,
            "end_ps": (block + 1) * block_ps,
            **values,
            "baseline_axis_velocity_mps": reference["axis_velocity_mps"],
            "excess_axis_velocity_mps": (
                0.0
                if direction == "none"
                else values["axis_velocity_mps"] - reference["axis_velocity_mps"]
            ),
            "baseline_axis_flux_molecules_per_A_ps": reference["axis_flux_molecules_per_A_ps"],
            "excess_axis_flux_molecules_per_A_ps": (
                0.0
                if direction == "none"
                else values["axis_flux_molecules_per_A_ps"]
                - reference["axis_flux_molecules_per_A_ps"]
            ),
            "layer_role": (
                "core" if layer in core_layers else "outer" if layer in outer_layers else "excluded"
            ),
        }
        row.update(water_blocks.get((branch, block), {}))
        block_rows.append(row)

    phase_rows: list[dict[str, object]] = []
    box_lengths_A = [float(value) for value in config["box_lengths_A"]]
    mode_by_group: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in mode_rows:
        direction = directions[row["branch_id"]]
        wanted = (1, 0) if direction in {"none", "x"} else (0, 1)
        if (_int(row["mode_x"]), _int(row["mode_y"])) == wanted:
            mode_by_group[(row["branch_id"], _int(row["layer_index"]))].append(row)
    phase_raw: dict[tuple[str, int, int], dict[str, float]] = {}
    for (branch, layer), rows in mode_by_group.items():
        rows = sorted(rows, key=lambda row: _float(row["time_ps"]))
        times = np.asarray([_float(row["time_ps"]) for row in rows], dtype=float)
        phases = np.unwrap(np.asarray([_float(row["phase_rad"]) for row in rows], dtype=float))
        amplitudes = np.asarray([_float(row["amplitude"]) for row in rows], dtype=float)
        direction = directions[branch]
        length = box_lengths_A[0 if direction in {"none", "x"} else 1]
        final_time = float(np.max(times))
        groups: dict[int, list[int]] = defaultdict(list)
        for index, time_ps in enumerate(times):
            groups[_block_index(float(time_ps), block_ps, final_time)].append(index)
        for block, indices in groups.items():
            phase_raw[(branch, layer, block)] = {
                "density_mode_velocity_mps": _slope_mps(
                    times[indices], phases[indices] * length / (2.0 * np.pi)
                ),
                "density_mode_mean_amplitude": float(np.nanmean(amplitudes[indices])),
            }
    block_lookup = {
        (str(row["branch_id"]), int(row["layer_index"]), int(row["block_index"])): row
        for row in block_rows
    }
    for key, values in sorted(phase_raw.items()):
        branch, layer, block = key
        reference = phase_raw.get((baseline_branch, layer, block))
        if reference is None:
            raise ValueError(f"Missing OH baseline density mode: {layer}/{block}")
        transport = block_lookup[key]
        phase_rows.append(
            {
                "case_id": "oh_only",
                "branch_id": branch,
                "direction": directions[branch],
                "layer_index": layer,
                "human_layer": layer + 1,
                "block_index": block,
                **values,
                "baseline_density_mode_velocity_mps": reference["density_mode_velocity_mps"],
                "excess_density_mode_velocity_mps": (
                    0.0
                    if directions[branch] == "none"
                    else values["density_mode_velocity_mps"]
                    - reference["density_mode_velocity_mps"]
                ),
                "excess_water_velocity_mps": transport["excess_axis_velocity_mps"],
            }
        )

    exchange_by_branch_block: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    final_exchange = max(_float(row["time_ps"]) for row in exchange_rows)
    for row in exchange_rows:
        block = _block_index(_float(row["time_ps"]), block_ps, final_exchange)
        exchange_by_branch_block[(row["branch_id"], block)].append(row)
    core_rows: list[dict[str, object]] = []
    for branch in sorted(directions):
        blocks = sorted({key[2] for key in raw if key[0] == branch})
        for block in blocks:
            selected = [
                block_lookup[(branch, layer, block)]
                for layer in core_layers
                if (branch, layer, block) in block_lookup
            ]
            outer = [
                block_lookup[(branch, layer, block)]
                for layer in outer_layers
                if (branch, layer, block) in block_lookup
            ]
            excess_values = [_float(row["excess_axis_velocity_mps"]) for row in selected]
            weights = [_float(row["mean_count"]) for row in selected]
            core_velocity = _weighted_mean(zip(excess_values, weights))
            mean_abs = _weighted_mean(
                (abs(value), weight) for value, weight in zip(excess_values, weights)
            )
            coherence = abs(core_velocity) / mean_abs if mean_abs > 0 else math.nan
            signs = [math.copysign(1.0, value) for value in excess_values if abs(value) > 1.0e-12]
            sign_agreement = (
                sum(sign == math.copysign(1.0, core_velocity) for sign in signs) / len(signs)
                if signs and abs(core_velocity) > 1.0e-12
                else math.nan
            )
            exchange = exchange_by_branch_block.get((branch, block), [])
            water = water_blocks.get((branch, block), {})
            core_rows.append(
                {
                    "case_id": "oh_only",
                    "branch_id": branch,
                    "direction": directions[branch],
                    "block_index": block,
                    "start_ps": block * block_ps,
                    "end_ps": (block + 1) * block_ps,
                    "core_excess_velocity_mps": core_velocity,
                    "core_velocity_coherence": coherence,
                    "core_layer_sign_agreement_fraction": sign_agreement,
                    "core_mean_count": _mean(weights),
                    "outer_mean_count": _mean(_float(row["mean_count"]) for row in outer),
                    "outer_mean_occupied_fraction": _mean(
                        _float(row["occupied_fraction"]) for row in outer
                    ),
                    "layer_exchange_molecules": sum(
                        _int(row["molecule_count"]) for row in exchange
                    ),
                    "core_outer_exchange_molecules": sum(
                        _int(row["molecule_count"])
                        for row in exchange
                        if ({_int(row["from_layer"]), _int(row["to_layer"])} & set(core_layers))
                        and ({_int(row["from_layer"]), _int(row["to_layer"])} & set(outer_layers))
                    ),
                    **water,
                }
            )

    residence_lookup = {(row["branch_id"], _int(row["layer_index"])): row for row in residence_rows}
    summary_rows: list[dict[str, object]] = []
    for branch in sorted(directions):
        for layer in sorted({key[1] for key in raw if key[0] == branch}):
            transport = [
                row
                for row in block_rows
                if row["branch_id"] == branch and row["layer_index"] == layer
            ]
            density = [
                row
                for row in phase_rows
                if row["branch_id"] == branch and row["layer_index"] == layer
            ]
            density_lookup = {int(row["block_index"]): row for row in density}
            paired = [
                (row, density_lookup[int(row["block_index"])])
                for row in transport
                if int(row["block_index"]) in density_lookup
            ]
            rho, n_blocks = _spearman(
                [_float(pair[0]["excess_axis_velocity_mps"]) for pair in paired],
                [_float(pair[1]["excess_density_mode_velocity_mps"]) for pair in paired],
            )
            residence = residence_lookup.get((branch, layer), {})
            summary_rows.append(
                {
                    "case_id": "oh_only",
                    "branch_id": branch,
                    "direction": directions[branch],
                    "layer_index": layer,
                    "human_layer": layer + 1,
                    "layer_role": transport[0]["layer_role"],
                    "mean_count": _mean(_float(row["mean_count"]) for row in transport),
                    "mean_occupied_fraction": _mean(
                        _float(row["occupied_fraction"]) for row in transport
                    ),
                    "mean_excess_velocity_mps": _mean(
                        _float(row["excess_axis_velocity_mps"]) for row in transport
                    ),
                    "mean_excess_flux_molecules_per_A_ps": _mean(
                        _float(row["excess_axis_flux_molecules_per_A_ps"]) for row in transport
                    ),
                    "mean_excess_density_mode_velocity_mps": _mean(
                        _float(row["excess_density_mode_velocity_mps"]) for row in density
                    ),
                    "velocity_density_mode_spearman_rho": rho,
                    "velocity_density_mode_blocks": n_blocks,
                    "mean_residence_ps": _float(residence.get("mean_residence_ps")),
                    "p95_residence_ps": _float(residence.get("p95_residence_ps")),
                    "right_censored_episodes": _int(residence.get("right_censored_episodes")),
                }
            )
    return block_rows, core_rows, summary_rows


def _motion_by_step(paths: Sequence[Path]) -> dict[int, dict[str, str]]:
    output: dict[int, dict[str, str]] = {}
    for path in paths:
        for row in _read_motion(path):
            output[_int(row["TimeStep"])] = row
    return output


def _finite_analysis(
    configs: Sequence[Mapping[str, object]],
    base: Path,
    block_ps: float,
    origin: int,
    timestep_fs: float,
    manifest: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_blocks: dict[tuple[str, str, int], dict[str, object]] = {}
    directions: dict[tuple[str, str], str] = {}
    for case in configs:
        case_id = str(case["case_id"])
        water_path = _resolve(base, case["water_results"]) / "water_structure_by_frame.tsv"
        _add_input(manifest, water_path, f"{case_id}_water")
        water = _water_block_metrics(water_path, block_ps)
        for branch in case["branches"]:
            branch_id = str(branch["branch_id"])
            direction = str(branch["direction"]).lower()
            directions[(case_id, branch_id)] = direction
            branch_dir = _resolve(base, branch["results"])
            paths = {
                "tpcl": branch_dir / "tpcl" / "frame_metrics.csv",
                "contour": branch_dir / "tpcl" / "contour_points.csv.gz",
                "hbond": branch_dir / "interfacial_hbond" / "hbond_by_frame.csv",
                "order": branch_dir / "local_water_order" / "water_order_by_frame.csv",
                "registry": branch_dir / "surface_site" / "surface_site_enrichment.csv",
                "flow": branch_dir / "flow_virial" / "flow_virial_by_region.tsv",
                "angle": _resolve(base, branch["contact_angle_blocks"]),
            }
            motion_paths = [_resolve(base, value) for value in branch["motion_tables"]]
            for kind, path in paths.items():
                _add_input(manifest, path, f"{case_id}_{kind}")
            for path in motion_paths:
                _add_input(manifest, path, f"{case_id}_motion")

            motion = _motion_by_step(motion_paths)
            axis = "x" if direction in {"none", "x"} else "y"
            displacement_key = "v_dxrel" if axis == "x" else "v_dyrel"
            motion_rows = [
                {
                    "step": step,
                    "time_ps": (step - origin) * timestep_fs / 1000.0,
                    "displacement_A": _float(row[displacement_key]),
                }
                for step, row in sorted(motion.items())
            ]
            tpcl = []
            contour_by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
            for contour_row in _read_rows(paths["contour"]):
                contour_by_step[_int(contour_row["step"])].append(contour_row)
            for row in _read_rows(paths["tpcl"]):
                step = _int(row["step"])
                center = _float(row[f"phase_center_{axis}_unwrapped_A"])
                contour = contour_by_step.get(step, [])
                contour_axis = [_float(item[f"contour_{axis}_unwrapped_A"]) for item in contour]
                if not contour_axis:
                    raise ValueError(f"Missing TPCL contour points at step {step}")
                tpcl.append(
                    {
                        "step": step,
                        "time_ps": (step - origin) * timestep_fs / 1000.0,
                        "center_A": center,
                        "front_A": max(contour_axis),
                        "rear_A": min(contour_axis),
                        "circularity": _float(row["contact_line_circularity"]),
                        "largest_cluster": _float(row["largest_cluster_size"]),
                    }
                )
            final_time = max(row["time_ps"] for row in tpcl)
            motion_blocks = _group_blocks(motion_rows, block_ps)
            tpcl_blocks = _group_blocks(tpcl, block_ps)
            angle_by_block = {_int(row["block_index"]): row for row in _read_rows(paths["angle"])}

            def mean_by_block(path: Path) -> dict[int, list[Mapping[str, object]]]:
                prepared = []
                for row in _read_rows(path):
                    item: dict[str, object] = dict(row)
                    item["time_ps"] = (_int(row["step"]) - origin) * timestep_fs / 1000.0
                    prepared.append(item)
                return _group_blocks(prepared, block_ps)

            hbond_blocks = mean_by_block(paths["hbond"])
            order_blocks = mean_by_block(paths["order"])
            registry_blocks = mean_by_block(paths["registry"])
            flow_blocks = _group_blocks(_read_rows(paths["flow"]), block_ps)
            first_center = tpcl[0]["center_A"]
            for block in sorted(tpcl_blocks):
                tp = tpcl_blocks[block]
                mo = motion_blocks.get(block, [])
                hb = hbond_blocks.get(block, [])
                order = order_blocks.get(block, [])
                registry = registry_blocks.get(block, [])
                flow = flow_blocks.get(block, [])
                angle = angle_by_block.get(block, {})
                front_flow = [
                    row
                    for row in flow
                    if row["layer"] == "all" and row["longitudinal_region"] == "front"
                ]
                rear_flow = [
                    row
                    for row in flow
                    if row["layer"] == "all" and row["longitudinal_region"] == "rear"
                ]
                all_flow = [
                    row
                    for row in flow
                    if row["layer"] == "all" and row["longitudinal_region"] == "all"
                ]
                front_flow_mean = _mean(
                    _float(row["mean_v_parallel_moving_frame_A_per_ps"]) for row in front_flow
                )
                rear_flow_mean = _mean(
                    _float(row["mean_v_parallel_moving_frame_A_per_ps"]) for row in rear_flow
                )
                raw_blocks[(case_id, branch_id, block)] = {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "direction": direction,
                    "block_index": block,
                    "start_ps": block * block_ps,
                    "end_ps": min((block + 1) * block_ps, final_time),
                    "com_velocity_mps": _slope_mps(
                        [_float(row["time_ps"]) for row in mo],
                        [_float(row["displacement_A"]) for row in mo],
                    ),
                    "tpcl_velocity_mps": _slope_mps(
                        [_float(row["time_ps"]) for row in tp],
                        [_float(row["center_A"]) for row in tp],
                    ),
                    "front_velocity_mps": _slope_mps(
                        [_float(row["time_ps"]) for row in tp],
                        [_float(row["front_A"]) for row in tp],
                    ),
                    "rear_velocity_mps": _slope_mps(
                        [_float(row["time_ps"]) for row in tp],
                        [_float(row["rear_A"]) for row in tp],
                    ),
                    "tpcl_minus_com_lag_A": (
                        _mean(
                            _float(row["center_A"])
                            - first_center
                            - _float(
                                min(
                                    mo,
                                    key=lambda item: abs(
                                        _float(item["time_ps"]) - _float(row["time_ps"])
                                    ),
                                )["displacement_A"]
                            )
                            for row in tp
                        )
                        if mo
                        else math.nan
                    ),
                    "contact_angle_deg": _float(angle.get("dense_phase_contact_angle_deg")),
                    "contact_angle_fit_rmse_A": _float(angle.get("fit_rmse_A")),
                    "circularity": _mean(_float(row["circularity"]) for row in tp),
                    "largest_cluster": _mean(_float(row["largest_cluster"]) for row in tp),
                    "tpcl_surface_hbond_per_water": _mean(
                        _float(row["tpcl_surface_water_hbond_per_h2o"]) for row in hb
                    ),
                    "tpcl_water_hbond_degree": _mean(
                        _float(row["tpcl_water_water_hbond_degree"]) for row in hb
                    ),
                    "tpcl_q_tet": _mean(_float(row["mean_q_tet"]) for row in order),
                    "tpcl_lsi_A2": _mean(_float(row["mean_lsi_A2"]) for row in order),
                    "tpcl_network_largest_component_fraction": _mean(
                        _float(row["hbond_largest_component_fraction"]) for row in order
                    ),
                    "tpcl_ch3_fraction": _mean(
                        _float(row["tpcl_ch3_fraction"]) for row in registry
                    ),
                    "registry_boundary_proxy_A": _mean(
                        _float(row["mean_contact_line_boundary_proxy_A"]) for row in registry
                    ),
                    "internal_flow_front_moving_A_per_ps": front_flow_mean,
                    "internal_flow_rear_moving_A_per_ps": rear_flow_mean,
                    "internal_flow_front_minus_rear_A_per_ps": front_flow_mean - rear_flow_mean,
                    "interfacial_stressvol_parallel_z_barA3": _mean(
                        _float(row["mean_signed_atom_stressvol_parallel_z_barA3"])
                        for row in all_flow
                    ),
                    **water.get((branch_id, block), {}),
                }

    output: list[dict[str, object]] = []
    for key, row in sorted(raw_blocks.items()):
        case_id, branch_id, block = key
        direction = directions[(case_id, branch_id)]
        baseline_branch = next(
            candidate
            for candidate, value in directions.items()
            if candidate[0] == case_id and value == "none"
        )[1]
        reference = raw_blocks.get((case_id, baseline_branch, block))
        if reference is None:
            raise ValueError(f"Missing finite-droplet baseline block: {case_id}/{block}")
        result = dict(row)
        for field in (
            "com_velocity_mps",
            "tpcl_velocity_mps",
            "front_velocity_mps",
            "rear_velocity_mps",
        ):
            result[f"excess_{field}"] = (
                0.0 if direction == "none" else _float(row[field]) - _float(reference[field])
            )
        result["front_minus_rear_velocity_mps"] = _float(row["front_velocity_mps"]) - _float(
            row["rear_velocity_mps"]
        )
        output.append(result)

    correlations: list[dict[str, object]] = []
    descriptors = (
        "contact_angle_deg",
        "tpcl_minus_com_lag_A",
        "internal_flow_front_minus_rear_A_per_ps",
        "tpcl_surface_hbond_per_water",
        "tpcl_water_hbond_degree",
        "tpcl_q_tet",
        "tpcl_lsi_A2",
        "tpcl_network_largest_component_fraction",
        "tpcl_ch3_fraction",
        "registry_boundary_proxy_A",
        "water_edge_turnover",
        "surface_edge_turnover",
        "interfacial_stressvol_parallel_z_barA3",
    )
    for case_id, branch_id in sorted(directions):
        if directions[(case_id, branch_id)] == "none":
            continue
        rows = [
            row for row in output if row["case_id"] == case_id and row["branch_id"] == branch_id
        ]
        for response in ("excess_com_velocity_mps", "front_minus_rear_velocity_mps"):
            for descriptor in descriptors:
                rho, samples = _spearman(
                    [_float(row[response]) for row in rows],
                    [_float(row.get(descriptor)) for row in rows],
                )
                correlations.append(
                    {
                        "case_id": case_id,
                        "branch_id": branch_id,
                        "direction": directions[(case_id, branch_id)],
                        "response": response,
                        "descriptor": descriptor,
                        "blocks": samples,
                        "spearman_rho": rho,
                        "evidence_limit": "single_trajectory_descriptive_association",
                    }
                )
    return output, correlations


def _cross_interface(
    aggregate_path: Path,
    mixed_blocks: Sequence[Mapping[str, object]],
    oh_blocks: Sequence[Mapping[str, object]],
    finite_blocks: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = _read_rows(aggregate_path)
    mixed_lookup: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in mixed_blocks:
        mixed_lookup[str(row["branch_id"])].append(row)
    oh_lookup: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in oh_blocks:
        oh_lookup[str(row["branch_id"])].append(row)
    finite_lookup: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in finite_blocks:
        finite_lookup[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    output: list[dict[str, object]] = []
    for row in rows:
        case_id, branch_id = row["case_id"], row["branch_id"]
        result: dict[str, object] = dict(row)
        if case_id == "mixed275":
            selected = mixed_lookup[branch_id]
            result.update(
                {
                    "stage_a_response_observable": "main_island_excess_velocity",
                    "stage_a_response_mean_mps": _mean(
                        _float(item["main_excess_axis_velocity_mps"]) for item in selected
                    ),
                    "stage_a_shape_integrity_min": min(
                        _float(item["main_fraction"]) for item in selected
                    ),
                }
            )
        elif case_id == "oh_only":
            selected = oh_lookup[branch_id]
            result.update(
                {
                    "stage_a_response_observable": "core_layer_excess_velocity",
                    "stage_a_response_mean_mps": _mean(
                        _float(item["core_excess_velocity_mps"]) for item in selected
                    ),
                    "stage_a_shape_integrity_min": _mean(
                        _float(item["core_velocity_coherence"]) for item in selected
                    ),
                }
            )
        else:
            selected = finite_lookup[(case_id, branch_id)]
            result.update(
                {
                    "stage_a_response_observable": "droplet_com_excess_velocity",
                    "stage_a_response_mean_mps": _mean(
                        _float(item["excess_com_velocity_mps"]) for item in selected
                    ),
                    "stage_a_shape_integrity_min": min(
                        _float(item["circularity"]) for item in selected
                    ),
                }
            )
        output.append(result)
    return output


def _plot(
    output: Path,
    mixed_blocks: Sequence[Mapping[str, object]],
    mixed_events: Sequence[Mapping[str, object]],
    oh_layers: Sequence[Mapping[str, object]],
    oh_core: Sequence[Mapping[str, object]],
    finite: Sequence[Mapping[str, object]],
    cross: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for branch in sorted({str(row["branch_id"]) for row in mixed_blocks}):
        rows = [row for row in mixed_blocks if row["branch_id"] == branch]
        axes[0, 0].plot(
            [0.001 * (float(row["start_ps"]) + float(row["end_ps"])) / 2 for row in rows],
            [_float(row["main_excess_axis_velocity_mps"]) for row in rows],
            label=branch,
        )
    axes[0, 0].axhline(0.0, color="black", lw=0.7)
    for event in mixed_events:
        axes[0, 0].axvline(_float(event["anchor_time_ps"]) / 1000.0, color="0.8", lw=0.4)
    axes[0, 0].set(
        title="mixed275: main-island response",
        xlabel="Time (ns)",
        ylabel="F - F0 velocity (m/s)",
    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    driven_layers = [row for row in oh_layers if row["direction"] != "none"]
    for branch in sorted({str(row["branch_id"]) for row in driven_layers}):
        means = []
        for layer in range(5):
            rows = [
                row
                for row in driven_layers
                if row["branch_id"] == branch and int(row["layer_index"]) == layer
            ]
            means.append(_mean(_float(row["excess_axis_velocity_mps"]) for row in rows))
        axes[0, 1].plot(range(1, 6), means, marker="o", label=branch)
    axes[0, 1].axhline(0.0, color="black", lw=0.7)
    axes[0, 1].set(
        title="oh_only: layer response",
        xlabel="Human layer",
        ylabel="F - F0 velocity (m/s)",
    )
    axes[0, 1].legend(frameon=False, fontsize=7)

    finite_keys = sorted(
        {
            (str(row["case_id"]), str(row["branch_id"]))
            for row in finite
            if row["direction"] != "none"
        }
    )
    for key in finite_keys:
        rows = [row for row in finite if (row["case_id"], row["branch_id"]) == key]
        axes[1, 0].plot(
            [0.001 * (float(row["start_ps"]) + float(row["end_ps"])) / 2 for row in rows],
            [_float(row["excess_com_velocity_mps"]) for row in rows],
            label="/".join(key),
        )
    axes[1, 0].axhline(0.0, color="black", lw=0.7)
    axes[1, 0].set(
        title="Finite droplets: 50 ps response",
        xlabel="Time (ns)",
        ylabel="COM F - F0 velocity (m/s)",
    )
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    driven = [row for row in cross if str(row["direction"]).lower() != "none"]
    colors = {"x": "#377eb8", "y": "#e41a1c"}
    for row in driven:
        direction = str(row["direction"]).lower()
        x_value = _float(row.get("water_mean_surface_hbond_per_water"))
        y_value = _float(row["stage_a_response_mean_mps"])
        axes[1, 1].scatter(x_value, y_value, color=colors.get(direction, "0.3"))
        axes[1, 1].annotate(f"{row['case_id']}/{direction}", (x_value, y_value), fontsize=7)
    axes[1, 1].axhline(0.0, color="black", lw=0.7)
    axes[1, 1].set(
        title="Cross-interface descriptive map",
        xlabel="Surface H bonds / water",
        ylabel="Morphology-specific response (m/s)",
    )
    figure.tight_layout()
    figure.savefig(output / "stage_a_overview.png", dpi=240)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    axes[0].scatter(
        [_float(row["persistent_transfer_count_pm20ps"]) for row in mixed_events],
        [_float(row["main_axis_velocity_post_minus_pre_mps"]) for row in mixed_events],
        c=[_float(row["proton_pool_change"]) for row in mixed_events],
        cmap="coolwarm",
    )
    axes[0].set(
        xlabel="Persistent transfers in +/-20 ps",
        ylabel="Post - pre island velocity (m/s)",
        title="mixed275 events",
    )
    for branch in sorted({str(row["branch_id"]) for row in oh_core if row["direction"] != "none"}):
        rows = [row for row in oh_core if row["branch_id"] == branch]
        axes[1].scatter(
            [_float(row["core_velocity_coherence"]) for row in rows],
            [_float(row["core_excess_velocity_mps"]) for row in rows],
            s=12,
            alpha=0.6,
            label=branch,
        )
    axes[1].set(xlabel="Core coherence", ylabel="Core response (m/s)", title="oh_only blocks")
    axes[1].legend(frameon=False, fontsize=7)
    driven_finite = [row for row in finite if row["direction"] != "none"]
    axes[2].scatter(
        [_float(row["front_minus_rear_velocity_mps"]) for row in driven_finite],
        [_float(row["tpcl_minus_com_lag_A"]) for row in driven_finite],
        s=10,
        alpha=0.5,
    )
    axes[2].set(
        xlabel="Front - rear velocity (m/s)",
        ylabel="TPCL - COM lag (A)",
        title="Finite droplets",
    )
    figure.tight_layout()
    figure.savefig(output / "stage_a_mechanism_panels.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    """Run the existing-trajectory Stage A synthesis contract."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    block_ps = float(raw.get("block_ps", 50.0))
    event_half_window_ps = float(raw.get("event_half_window_ps", 20.0))
    if block_ps <= 0.0 or event_half_window_ps <= 0.0:
        raise ValueError("block_ps and event_half_window_ps must be positive")
    base = contract_path.parent
    output.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    _add_input(manifest, contract_path, "contract")

    mixed_blocks, mixed_events, identity_track = _mixed275_analysis(
        raw["mixed275"], base, block_ps, event_half_window_ps, manifest
    )
    oh_layers, oh_core, oh_summary = _oh_analysis(raw["oh_only"], base, block_ps, manifest)
    finite_blocks, finite_correlations = _finite_analysis(
        raw["finite_droplets"],
        base,
        block_ps,
        int(raw["time_origin_step"]),
        float(raw["timestep_fs"]),
        manifest,
    )
    aggregate_path = _resolve(base, raw["aggregate_branch_summary"])
    _add_input(manifest, aggregate_path, "accepted_aggregate")
    cross_rows = _cross_interface(aggregate_path, mixed_blocks, oh_core, finite_blocks)

    _write_tsv(output / "mixed275_island_blocks_50ps.tsv", mixed_blocks)
    _write_tsv(output / "mixed275_event_alignment_pm20ps.tsv", mixed_events)
    _write_tsv(output / "mixed275_identity_track.tsv", identity_track)
    _write_tsv(output / "oh_layer_blocks_50ps.tsv", oh_layers)
    _write_tsv(output / "oh_core_blocks_50ps.tsv", oh_core)
    _write_tsv(output / "oh_layer_summary.tsv", oh_summary)
    _write_tsv(output / "finite_droplet_blocks_50ps.tsv", finite_blocks)
    _write_tsv(output / "finite_droplet_correlations.tsv", finite_correlations)
    _write_tsv(output / "cross_interface_stage_a.tsv", cross_rows)
    _write_tsv(output / "input_manifest.tsv", manifest)
    if bool(raw.get("write_plots", True)):
        _plot(
            output,
            mixed_blocks,
            mixed_events,
            oh_layers,
            oh_core,
            finite_blocks,
            cross_rows,
        )

    summary = {
        "status": "PASS",
        "block_ps": block_ps,
        "event_half_window_ps": event_half_window_ps,
        "mixed275_block_rows": len(mixed_blocks),
        "mixed275_selected_events": len(mixed_events),
        "mixed275_identity_frames": len(identity_track),
        "oh_layer_block_rows": len(oh_layers),
        "oh_core_block_rows": len(oh_core),
        "finite_droplet_block_rows": len(finite_blocks),
        "cross_interface_rows": len(cross_rows),
        "single_trajectory_descriptive_only": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Constant-force Stage A synthesis\n\n"
        "This package summarizes existing 4 ns trajectories in 50 ps blocks. "
        "It reports mixed275 island/event coupling, oh_only layer transport, finite-droplet "
        "contact response, and a cross-interface descriptive table.\n\n"
        "The H 1570, C 1640, and framework O 1826 trace preserves the identity ambiguity as "
        "C-H and O-H distances. Geometric species labels are not formal charges. Density-mode, "
        "hydrogen-bond, registry, and transport associations are descriptive because each branch "
        "contains one trajectory. Time blocks are not independent replicas.\n",
        encoding="utf-8",
    )
    hashes = []
    for path in sorted(candidate for candidate in output.iterdir() if candidate.is_file()):
        if path.name == "OUTPUT-SHA256SUMS":
            continue
        hashes.append(f"{_sha256(path)}  {path.name}\n")
    (output / "OUTPUT-SHA256SUMS").write_text("".join(hashes), encoding="utf-8")
    return summary
