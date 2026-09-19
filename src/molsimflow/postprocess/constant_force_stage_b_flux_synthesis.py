"""Synthesize Stage-B directed flux into mechanism-facing diagnostics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.constant_force_oxygen import (
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)
from molsimflow.postprocess.constant_force_stage_b_flux import (
    _as_int,
    _initial_owners,
    _read_tsv,
    _validate_owner_sizes,
    iter_unwrapped_oxygen_frames,
)


CATEGORIES = (
    "UNCHANGED_TRACK",
    "PERSISTENT_ISLAND_TRANSFER",
    "LINEAGE_REASSIGNMENT",
    "UNTRACKED_TRANSITION",
)
SIZE_CLASSES = (
    "MAIN_CONDENSED_ISLAND",
    "OTHER_MULTI_ISLAND",
    "SINGLETON_VAPOR",
    "UNTRACKED",
)


def _verify_output_hashes(results: Path) -> None:
    expected = {}
    for line in (results / "OUTPUT-SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative.removeprefix("./")] = digest
    for relative, digest in expected.items():
        path = results / relative
        if sha256(path) != digest:
            raise ValueError(f"Source result hash mismatch: {path}")


def _group_rows(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str], list[Mapping[str, str]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    return grouped


def _window_index(mid_time_ps: float, window_ps: float) -> int:
    return int(math.floor(mid_time_ps / window_ps + 1.0e-12))


def _aggregate_interval_windows(
    intervals: Sequence[Mapping[str, str]], window_ps: float
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in intervals:
        grouped[
            (
                str(row["case_id"]),
                str(row["branch_id"]),
                str(row["direction"]),
                _window_index(float(row["mid_time_ps"]), window_ps),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for (case_id, branch_id, direction, window), selected in sorted(grouped.items()):
        duration = sum(float(row["interval_ps"]) for row in selected)
        oxygen_count = int(selected[0]["oxygen_count"])
        item: dict[str, object] = {
            "case_id": case_id,
            "branch_id": branch_id,
            "direction": direction,
            "window_ps": window_ps,
            "window_index": window,
            "start_ps": window * window_ps,
            "end_ps": (window + 1) * window_ps,
            "duration_ps": duration,
            "oxygen_count": oxygen_count,
        }
        for axis in ("x", "y"):
            for name in ("total", *(category.lower() for category in CATEGORIES)):
                displacement = sum(float(row[f"{name}_{axis}_displacement_A"]) for row in selected)
                item[f"{name}_{axis}_displacement_A"] = displacement
                item[f"{name}_{axis}_velocity_mps"] = (
                    100.0 * displacement / (oxygen_count * duration)
                )
            item[f"net_{axis}_crossing_rate_per_ps"] = (
                sum(float(row[f"net_{axis}_crossings"]) for row in selected) / duration
            )
        output.append(item)
    return output


def _response_rows(
    windows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline = {
        (str(row["case_id"]), float(row["window_ps"]), int(row["window_index"])): row
        for row in windows
        if str(row["direction"]) == "none"
    }
    output: list[dict[str, object]] = []
    for row in windows:
        direction = str(row["direction"])
        if direction == "none":
            continue
        axis = direction
        reference = baseline[
            (str(row["case_id"]), float(row["window_ps"]), int(row["window_index"]))
        ]
        item: dict[str, object] = {
            "case_id": row["case_id"],
            "branch_id": row["branch_id"],
            "direction": direction,
            "axis": axis,
            "window_ps": row["window_ps"],
            "window_index": row["window_index"],
            "start_ps": row["start_ps"],
            "end_ps": row["end_ps"],
            "duration_ps": row["duration_ps"],
        }
        for name in ("total", *(category.lower() for category in CATEGORIES)):
            raw = float(row[f"{name}_{axis}_velocity_mps"])
            base = float(reference[f"{name}_{axis}_velocity_mps"])
            item[f"raw_{name}_velocity_mps"] = raw
            item[f"baseline_{name}_velocity_mps"] = base
            item[f"response_{name}_velocity_mps"] = raw - base
        raw_crossing = float(row[f"net_{axis}_crossing_rate_per_ps"])
        baseline_crossing = float(reference[f"net_{axis}_crossing_rate_per_ps"])
        item["raw_crossing_rate_per_ps"] = raw_crossing
        item["baseline_crossing_rate_per_ps"] = baseline_crossing
        item["response_crossing_rate_per_ps"] = raw_crossing - baseline_crossing
        output.append(item)
    return output


def _branch_summary(
    windows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    selected = [row for row in windows if float(row["window_ps"]) == 4000.0]
    baseline = {str(row["case_id"]): row for row in selected if str(row["direction"]) == "none"}
    output: list[dict[str, object]] = []
    for row in selected:
        for axis in ("x", "y"):
            reference = baseline[str(row["case_id"])]
            raw = float(row[f"total_{axis}_velocity_mps"])
            base = float(reference[f"total_{axis}_velocity_mps"])
            raw_crossing = float(row[f"net_{axis}_crossing_rate_per_ps"])
            base_crossing = float(reference[f"net_{axis}_crossing_rate_per_ps"])
            output.append(
                {
                    "case_id": row["case_id"],
                    "branch_id": row["branch_id"],
                    "direction": row["direction"],
                    "axis": axis,
                    "duration_ps": row["duration_ps"],
                    "raw_velocity_mps": raw,
                    "baseline_velocity_mps": base,
                    "response_velocity_mps": raw - base,
                    "raw_crossing_rate_per_ps": raw_crossing,
                    "baseline_crossing_rate_per_ps": base_crossing,
                    "response_crossing_rate_per_ps": raw_crossing - base_crossing,
                }
            )
    return output


def _category_summary(
    responses: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    full = [row for row in responses if float(row["window_ps"]) == 4000.0]
    output: list[dict[str, object]] = []
    for row in full:
        component_responses = {
            category: float(row[f"response_{category.lower()}_velocity_mps"])
            for category in CATEGORIES
        }
        total = float(row["response_total_velocity_mps"])
        absolute_denominator = sum(abs(value) for value in component_responses.values())
        for category, response in component_responses.items():
            output.append(
                {
                    "case_id": row["case_id"],
                    "branch_id": row["branch_id"],
                    "axis": row["axis"],
                    "category": category,
                    "raw_velocity_mps": row[f"raw_{category.lower()}_velocity_mps"],
                    "baseline_velocity_mps": row[f"baseline_{category.lower()}_velocity_mps"],
                    "response_velocity_mps": response,
                    "total_response_velocity_mps": total,
                    "signed_fraction_of_total_response": (
                        response / total if abs(total) > 1.0e-12 else float("nan")
                    ),
                    "absolute_component_l1_denominator_mps": absolute_denominator,
                    "absolute_fraction_of_component_l1": (
                        abs(response) / absolute_denominator
                        if absolute_denominator > 1.0e-12
                        else float("nan")
                    ),
                }
            )
    return output


def _size_class(previous_size: int | None, track: int | None, main_track: int | None) -> str:
    if previous_size is None or track is None:
        return "UNTRACKED"
    if track == main_track:
        return "MAIN_CONDENSED_ISLAND"
    if previous_size == 1:
        return "SINGLETON_VAPOR"
    return "OTHER_MULTI_ISLAND"


def _reconstruct_track_transport(
    raw: Mapping[str, object], base: Path, interval_rows: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
    interval_lookup = {
        (str(row["case_id"]), str(row["branch_id"]), _as_int(row["end_step"])): row
        for row in interval_rows
    }
    class_displacement: dict[tuple[str, str, str, str, str], float] = defaultdict(float)
    track_displacement: dict[tuple[str, str, int, str, str], float] = defaultdict(float)
    track_exposure: dict[tuple[str, str, int], float] = defaultdict(float)
    track_interval_count: dict[tuple[str, str, int], int] = defaultdict(int)
    track_size_min: dict[tuple[str, str, int], int] = {}
    track_size_max: dict[tuple[str, str, int], int] = {}
    track_time_min: dict[tuple[str, str, int], float] = {}
    track_time_max: dict[tuple[str, str, int], float] = {}
    branch_meta: dict[tuple[str, str], tuple[int, float, str]] = {}
    maximum_residual = 0.0
    timestep_fs = float(raw["timestep_fs"])
    cutoff_A = float(raw["cluster_cutoff_A"])
    origin = int(raw["time_origin_step"])
    for entry in raw["cases"]:
        case_id = str(entry["case_id"])
        branch_id = str(entry["branch_id"])
        direction = str(entry["direction"])
        paths = [resolve_path(path, base) for path in entry["trajectories"]]
        island_dir = resolve_path(entry["island_results"], base)
        exchange_rows = [
            row
            for row in _read_tsv(island_dir / "molecule_exchange.tsv")
            if row["case_id"] == case_id and row["branch_id"] == branch_id
        ]
        island_rows = [
            row
            for row in _read_tsv(island_dir / "island_timeseries.tsv")
            if row["case_id"] == case_id and row["branch_id"] == branch_id
        ]
        changes: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in exchange_rows:
            changes[_as_int(row["step"])].append(row)
        expected_sizes = {
            (_as_int(row["step"]), _as_int(row["track_id"])): _as_int(row["size"])
            for row in island_rows
        }
        frames = iter_unwrapped_oxygen_frames(paths)
        previous = next(frames)
        owners = _initial_owners(previous, island_rows, cutoff_A)
        _validate_owner_sizes(owners, expected_sizes, previous.step)
        start_step = previous.step
        last_step = previous.step
        for current in frames:
            dt_ps = (current.step - previous.step) * timestep_fs / 1000.0
            old_owners = dict(owners)
            old_sizes: dict[int, int] = defaultdict(int)
            for track in old_owners.values():
                old_sizes[int(track)] += 1
            main_track = min(old_sizes, key=lambda track: (-old_sizes[track], track))
            category_by_id: dict[int, str] = {}
            for change in changes.get(current.step, []):
                atom_id = _as_int(change["oxygen_id"])
                target = str(change["target_track_id"]).strip()
                if target:
                    owners[atom_id] = _as_int(target)
                else:
                    owners.pop(atom_id, None)
                exchange_class = str(change["exchange_class"])
                if exchange_class == "PERSISTENT_ISLAND_TRANSFER":
                    category = exchange_class
                elif "LINEAGE_REASSIGNMENT" in exchange_class:
                    category = "LINEAGE_REASSIGNMENT"
                else:
                    category = "UNTRACKED_TRANSITION"
                category_by_id[atom_id] = category
            _validate_owner_sizes(owners, expected_sizes, current.step)
            displacement = current.unwrapped - previous.unwrapped
            recomputed = [0.0, 0.0]
            for atom_index, atom_id_value in enumerate(current.atom_ids):
                atom_id = int(atom_id_value)
                track = old_owners.get(atom_id)
                previous_size = old_sizes.get(track) if track is not None else None
                size_class = _size_class(previous_size, track, main_track)
                category = category_by_id.get(atom_id, "UNCHANGED_TRACK")
                for axis_index, axis in enumerate(("x", "y")):
                    value = float(displacement[atom_index, axis_index])
                    class_displacement[(branch_id, direction, size_class, category, axis)] += value
                    recomputed[axis_index] += value
                    if track is not None:
                        track_displacement[(branch_id, direction, int(track), category, axis)] += (
                            value
                        )
                if track is not None:
                    key = (branch_id, direction, int(track))
                    track_exposure[key] += dt_ps
            for track, size in old_sizes.items():
                key = (branch_id, direction, track)
                track_interval_count[key] += 1
                track_size_min[key] = min(track_size_min.get(key, size), size)
                track_size_max[key] = max(track_size_max.get(key, size), size)
                start_ps = (previous.step - origin) * timestep_fs / 1000.0
                end_ps = (current.step - origin) * timestep_fs / 1000.0
                track_time_min[key] = min(track_time_min.get(key, start_ps), start_ps)
                track_time_max[key] = max(track_time_max.get(key, end_ps), end_ps)
            accepted = interval_lookup[(case_id, branch_id, current.step)]
            for axis_index, axis in enumerate(("x", "y")):
                residual = recomputed[axis_index] - float(accepted[f"total_{axis}_displacement_A"])
                maximum_residual = max(maximum_residual, abs(residual))
            previous = current
            last_step = current.step
        duration_ps = (last_step - start_step) * timestep_fs / 1000.0
        branch_meta[(branch_id, direction)] = (len(previous.atom_ids), duration_ps, case_id)

    class_rows: list[dict[str, object]] = []
    for branch_id, direction in sorted(branch_meta):
        oxygen_count, duration_ps, case_id = branch_meta[(branch_id, direction)]
        for size_class in SIZE_CLASSES:
            for axis in ("x", "y"):
                item: dict[str, object] = {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "direction": direction,
                    "size_class": size_class,
                    "axis": axis,
                    "duration_ps": duration_ps,
                    "oxygen_count": oxygen_count,
                }
                total = 0.0
                for category in CATEGORIES:
                    value = class_displacement[(branch_id, direction, size_class, category, axis)]
                    item[f"{category.lower()}_displacement_A"] = value
                    total += value
                item["total_displacement_A"] = total
                item["contribution_to_all_water_velocity_mps"] = (
                    100.0 * total / (oxygen_count * duration_ps)
                )
                class_rows.append(item)

    track_rows: list[dict[str, object]] = []
    for branch_id, direction, track in sorted(track_exposure):
        oxygen_count, duration_ps, case_id = branch_meta[(branch_id, direction)]
        for axis in ("x", "y"):
            item = {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "track_id": track,
                "axis": axis,
                "first_time_ps": track_time_min[(branch_id, direction, track)],
                "last_time_ps": track_time_max[(branch_id, direction, track)],
                "interval_count": track_interval_count[(branch_id, direction, track)],
                "minimum_previous_size": track_size_min[(branch_id, direction, track)],
                "maximum_previous_size": track_size_max[(branch_id, direction, track)],
                "oxygen_interval_exposure_ps": track_exposure[(branch_id, direction, track)],
            }
            total = 0.0
            for category in CATEGORIES:
                value = track_displacement[(branch_id, direction, track, category, axis)]
                item[f"{category.lower()}_displacement_A"] = value
                total += value
            item["total_displacement_A"] = total
            item["contribution_to_all_water_velocity_mps"] = (
                100.0 * total / (oxygen_count * duration_ps)
            )
            track_rows.append(item)
    return class_rows, track_rows, maximum_residual


def _size_class_responses(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (str(row["branch_id"]), str(row["size_class"]), str(row["axis"])): row for row in rows
    }
    baseline_branch = next(str(row["branch_id"]) for row in rows if str(row["direction"]) == "none")
    output: list[dict[str, object]] = []
    for row in rows:
        direction = str(row["direction"])
        axis = str(row["axis"])
        if direction == "none" or axis != direction:
            continue
        reference = lookup[(baseline_branch, str(row["size_class"]), axis)]
        raw_value = float(row["contribution_to_all_water_velocity_mps"])
        baseline_value = float(reference["contribution_to_all_water_velocity_mps"])
        output.append(
            {
                "case_id": row["case_id"],
                "branch_id": row["branch_id"],
                "axis": axis,
                "size_class": row["size_class"],
                "raw_contribution_mps": raw_value,
                "baseline_contribution_mps": baseline_value,
                "response_contribution_mps": raw_value - baseline_value,
            }
        )
    return output


def _center_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    main_by_interval: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (str(row["branch_id"]), _as_int(row["start_step"]))
        candidate = _as_int(row["track_id"])
        current = main_by_interval.get(key)
        if current is None:
            main_by_interval[key] = candidate
            continue
        current_row = next(
            item
            for item in rows
            if str(item["branch_id"]) == key[0]
            and _as_int(item["start_step"]) == key[1]
            and _as_int(item["track_id"]) == current
        )
        if (_as_int(row["previous_size"]), -candidate) > (
            _as_int(current_row["previous_size"]),
            -current,
        ):
            main_by_interval[key] = candidate
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        previous_size = _as_int(row["previous_size"])
        track = _as_int(row["track_id"])
        main = main_by_interval[(str(row["branch_id"]), _as_int(row["start_step"]))]
        size_class = _size_class(previous_size, track, main)
        for axis in ("x", "y"):
            grouped[(str(row["branch_id"]), str(row["direction"]), size_class, axis)].append(row)
    output: list[dict[str, object]] = []
    for (branch_id, direction, size_class, axis), selected in sorted(grouped.items()):
        suffix = "" if axis == "x" else "_y"
        exposure = sum(
            _as_int(row[f"previous_size{suffix}"]) * float(row["interval_ps"]) for row in selected
        )
        item: dict[str, object] = {
            "branch_id": branch_id,
            "direction": direction,
            "size_class": size_class,
            "axis": axis,
            "track_interval_rows": len(selected),
            "molecule_interval_exposure_ps": exposure,
        }
        total_weighted = 0.0
        for name in ("advective", "membership", "total_center"):
            field = f"{name}_displacement_A{suffix}"
            weighted = sum(
                _as_int(row[f"previous_size{suffix}"]) * float(row[field]) for row in selected
            )
            item[f"size_weighted_{name}_displacement_A"] = weighted
            item[f"size_weighted_{name}_velocity_mps"] = 100.0 * weighted / exposure
            if name == "total_center":
                total_weighted = weighted
        item["size_weighted_closure_residual_A"] = total_weighted - (
            float(item["size_weighted_advective_displacement_A"])
            + float(item["size_weighted_membership_displacement_A"])
        )
        output.append(item)
    return output


def _plot(
    branch_rows: Sequence[Mapping[str, object]],
    category_rows: Sequence[Mapping[str, object]],
    response_rows: Sequence[Mapping[str, object]],
    size_rows: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    driven = [
        row
        for row in branch_rows
        if str(row["direction"]) != "none" and str(row["axis"]) == str(row["direction"])
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    labels = [str(row["branch_id"]) for row in driven]
    x = np.arange(len(driven))
    width = 0.25
    axes[0, 0].bar(x - width, [row["raw_velocity_mps"] for row in driven], width, label="raw")
    axes[0, 0].bar(x, [row["baseline_velocity_mps"] for row in driven], width, label="F0")
    axes[0, 0].bar(x + width, [row["response_velocity_mps"] for row in driven], width, label="F-F0")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Velocity (m/s)")
    axes[0, 0].set_title("Four-nanosecond transport")
    axes[0, 0].legend(frameon=False)

    for index, branch in enumerate(labels):
        selected = [row for row in category_rows if row["branch_id"] == branch]
        axes[0, 1].bar(
            np.arange(len(selected)) + (index - 0.5) * 0.35,
            [row["response_velocity_mps"] for row in selected],
            0.35,
            label=branch,
        )
    axes[0, 1].set_xticks(
        np.arange(len(CATEGORIES)), [name.replace("_", "\n") for name in CATEGORIES], fontsize=8
    )
    axes[0, 1].set_ylabel("F-F0 contribution (m/s)")
    axes[0, 1].set_title("Category decomposition")
    axes[0, 1].legend(frameon=False)

    for branch in labels:
        selected = [
            row
            for row in response_rows
            if row["branch_id"] == branch and float(row["window_ps"]) == 1000.0
        ]
        time = [0.001 * (float(row["start_ps"]) + float(row["end_ps"])) / 2.0 for row in selected]
        axes[1, 0].plot(
            time,
            [row["response_total_velocity_mps"] for row in selected],
            marker="o",
            label=f"{branch} total",
        )
        axes[1, 0].plot(
            time,
            [row["response_persistent_island_transfer_velocity_mps"] for row in selected],
            marker="s",
            linestyle="--",
            label=f"{branch} transfer",
        )
    axes[1, 0].set_xlabel("Time (ns)")
    axes[1, 0].set_ylabel("F-F0 contribution (m/s)")
    axes[1, 0].set_title("One-nanosecond windows")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for index, branch in enumerate(labels):
        selected = [row for row in size_rows if row["branch_id"] == branch]
        axes[1, 1].bar(
            np.arange(len(selected)) + (index - 0.5) * 0.35,
            [row["response_contribution_mps"] for row in selected],
            0.35,
            label=branch,
        )
    axes[1, 1].set_xticks(
        np.arange(len(SIZE_CLASSES)), [name.replace("_", "\n") for name in SIZE_CLASSES], fontsize=8
    )
    axes[1, 1].set_ylabel("F-F0 contribution (m/s)")
    axes[1, 1].set_title("Previous-island size class")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.axhline(0.0, color="black", lw=0.7)
    figure.tight_layout()
    figure.savefig(output / "directed_flux_synthesis.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    """Create an immutable mechanism-facing synthesis of accepted Stage-B flux results."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(contract.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    base = contract_path.parent
    source_contract_path = resolve_path(contract["flux_contract"], base)
    results = resolve_path(contract["flux_results"], base)
    _verify_output_hashes(results)
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    intervals = _read_tsv(results / "directed_flux_intervals.tsv")
    centers = _read_tsv(results / "island_center_decomposition.tsv")
    windows = []
    for window_ps in (50.0, 1000.0, 4000.0):
        windows.extend(_aggregate_interval_windows(intervals, window_ps))
    response_rows = _response_rows(windows)
    branch_rows = _branch_summary(windows)
    category_rows = _category_summary(response_rows)
    class_rows, track_rows, recompute_residual = _reconstruct_track_transport(
        source_contract, source_contract_path.parent, intervals
    )
    size_response_rows = _size_class_responses(class_rows)
    center_rows = _center_summary(centers)
    requested_tracks = {
        (str(item["branch_id"]), int(item["track_id"]))
        for item in contract.get("highlight_tracks", [])
    }
    highlight_rows = [
        row
        for row in track_rows
        if (str(row["branch_id"]), int(row["track_id"])) in requested_tracks
    ]
    missing_tracks = sorted(
        requested_tracks - {(str(row["branch_id"]), int(row["track_id"])) for row in highlight_rows}
    )
    if missing_tracks:
        raise ValueError(f"Requested highlight tracks were not found: {missing_tracks}")
    output.mkdir(parents=True)
    write_tsv(output / "branch_transport_summary.tsv", branch_rows, tuple(branch_rows[0]))
    write_tsv(output / "category_response_summary.tsv", category_rows, tuple(category_rows[0]))
    write_tsv(output / "time_window_response.tsv", response_rows, tuple(response_rows[0]))
    write_tsv(output / "size_class_transport.tsv", class_rows, tuple(class_rows[0]))
    write_tsv(
        output / "size_class_response_summary.tsv",
        size_response_rows,
        tuple(size_response_rows[0]),
    )
    write_tsv(output / "track_transport_summary.tsv", track_rows, tuple(track_rows[0]))
    write_tsv(output / "highlight_track_summary.tsv", highlight_rows, tuple(highlight_rows[0]))
    write_tsv(output / "center_decomposition_summary.tsv", center_rows, tuple(center_rows[0]))
    input_rows = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in (
            contract_path,
            source_contract_path,
            results / "directed_flux_intervals.tsv",
            results / "transport_blocks_50ps.tsv",
            results / "island_center_decomposition.tsv",
            results / "summary.json",
            results / "OUTPUT-SHA256SUMS",
        )
    ]
    write_tsv(output / "input_manifest.tsv", input_rows, ("path", "size_bytes", "sha256"))
    _plot(branch_rows, category_rows, response_rows, size_response_rows, output)
    source_summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    maximum_category_closure = max(
        abs(
            float(row["response_total_velocity_mps"])
            - sum(
                float(row[f"response_{category.lower()}_velocity_mps"]) for category in CATEGORIES
            )
        )
        for row in response_rows
    )
    summary = {
        "status": "PASS",
        "source_status": source_summary["status"],
        "source_interval_rows": len(intervals),
        "branch_summary_rows": len(branch_rows),
        "time_window_rows": len(response_rows),
        "track_summary_rows": len(track_rows),
        "highlight_track_rows": len(highlight_rows),
        "maximum_recomputed_interval_residual_A": recompute_residual,
        "maximum_response_category_closure_mps": maximum_category_closure,
        "single_trajectory_descriptive_only": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Stage-B directed-flux synthesis\n\n"
        "Raw driven transport, the time-aligned F0 baseline, and F-F0 response are reported "
        "separately. Signed fractions use the total F-F0 response as denominator. Absolute "
        "fractions use the L1 sum of the four response components, so cancellation is explicit.\n\n"
        "Size classes are assigned from each oxygen's island at the start of every interval. "
        "The largest island is MAIN_CONDENSED_ISLAND, size-one islands are SINGLETON_VAPOR, "
        "and remaining tracked islands are OTHER_MULTI_ISLAND. Track and size-class velocities "
        "are contributions to the all-water center-of-mass velocity over the full branch.\n\n"
        "F0, X, and Y are different drive conditions, not independent replicas. Fifty-picosecond "
        "and one-nanosecond windows are single-trajectory descriptive diagnostics.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary


def main() -> int:
    """Command-line entry point for cluster packages."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_contract(args.contract, args.output)
    print(args.output.resolve())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
