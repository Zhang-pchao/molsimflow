"""Resolve directed water transport into persistent-island exchange classes."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records
from molsimflow.postprocess.constant_force_oxygen import (
    connected_components,
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)


@dataclass(frozen=True)
class UnwrappedOxygenFrame:
    """One identity-sorted water-oxygen frame with image-unwrapped coordinates."""

    step: int
    bounds: np.ndarray
    atom_ids: np.ndarray
    wrapped: np.ndarray
    unwrapped: np.ndarray


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _as_int(value: object) -> int:
    return int(float(str(value)))


def iter_unwrapped_oxygen_frames(paths: Sequence[Path]) -> Iterator[UnwrappedOxygenFrame]:
    """Stream restart segments and reconstruct coordinates from LAMMPS image flags."""

    reference_ids: np.ndarray | None = None
    previous_step: int | None = None
    previous_bounds: np.ndarray | None = None
    for path in paths:
        segment_frames = 0
        for frame in iter_lammps_dump_records(path):
            segment_frames += 1
            if previous_step is not None and frame.timestep == previous_step:
                continue
            if previous_step is not None and frame.timestep < previous_step:
                raise ValueError(f"Non-increasing timestep {frame.timestep} in {path}")
            fields = frame.atom_fields
            required = ("id", "x", "y", "z", "ix", "iy", "iz")
            missing = [name for name in required if name not in fields]
            if missing:
                raise ValueError(f"{path}: missing trajectory fields {missing}")
            column = {name: fields.index(name) for name in required}
            ids = np.asarray([int(row[column["id"]]) for row in frame.atom_rows])
            order = np.argsort(ids)
            ids = ids[order]
            if reference_ids is None:
                reference_ids = ids.copy()
            elif not np.array_equal(ids, reference_ids):
                raise ValueError(f"Oxygen identity changed at step {frame.timestep}")
            wrapped = np.asarray(
                [
                    [
                        float(row[column["x"]]),
                        float(row[column["y"]]),
                        float(row[column["z"]]),
                    ]
                    for row in frame.atom_rows
                ],
                dtype=float,
            )[order]
            images = np.asarray(
                [
                    [
                        int(row[column["ix"]]),
                        int(row[column["iy"]]),
                        int(row[column["iz"]]),
                    ]
                    for row in frame.atom_rows
                ],
                dtype=np.int64,
            )[order]
            if previous_bounds is not None and not np.allclose(
                frame.bounds, previous_bounds, atol=1.0e-8
            ):
                raise ValueError(f"Box bounds changed at step {frame.timestep}")
            lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
            unwrapped = wrapped + images * lengths
            yield UnwrappedOxygenFrame(
                step=frame.timestep,
                bounds=frame.bounds.copy(),
                atom_ids=ids,
                wrapped=wrapped,
                unwrapped=unwrapped,
            )
            previous_step = frame.timestep
            previous_bounds = frame.bounds.copy()
        if segment_frames == 0:
            raise ValueError(f"No complete frames in {path}")


def plane_crossing_counts(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    lower_bound: float,
    box_length: float,
    plane_count: int,
) -> tuple[float, float, float]:
    """Return plane-averaged positive, negative, and signed periodic crossings."""

    if plane_count <= 0 or box_length <= 0.0:
        raise ValueError("plane_count and box_length must be positive")
    positive = 0.0
    negative = 0.0
    for index in range(plane_count):
        offset = lower_bound + (index + 0.5) * box_length / plane_count
        delta = np.floor((current - offset) / box_length) - np.floor(
            (previous - offset) / box_length
        )
        positive += float(np.sum(delta[delta > 0.0]))
        negative += float(-np.sum(delta[delta < 0.0]))
    positive /= plane_count
    negative /= plane_count
    return positive, negative, positive - negative


def decompose_track_center(
    previous_ids: set[int],
    current_ids: set[int],
    atom_ids: np.ndarray,
    previous_coordinate: np.ndarray,
    current_coordinate: np.ndarray,
) -> dict[str, float | int]:
    """Exactly decompose a track-center change into motion and membership terms."""

    if not previous_ids or not current_ids:
        raise ValueError("Track-center decomposition requires non-empty memberships")
    index = {int(atom_id): position for position, atom_id in enumerate(atom_ids)}
    old = np.asarray([index[atom_id] for atom_id in sorted(previous_ids)])
    new = np.asarray([index[atom_id] for atom_id in sorted(current_ids)])
    center_previous = float(np.mean(previous_coordinate[old]))
    moved_old_center = float(np.mean(current_coordinate[old]))
    center_current = float(np.mean(current_coordinate[new]))
    advective = moved_old_center - center_previous
    membership = center_current - moved_old_center
    total = center_current - center_previous
    return {
        "previous_size": len(previous_ids),
        "current_size": len(current_ids),
        "common_size": len(previous_ids & current_ids),
        "advective_displacement_A": advective,
        "membership_displacement_A": membership,
        "total_center_displacement_A": total,
        "closure_residual_A": total - advective - membership,
    }


def _initial_owners(
    frame: UnwrappedOxygenFrame,
    island_rows: Sequence[Mapping[str, str]],
    cutoff_A: float,
) -> dict[int, int]:
    components = connected_components(frame.wrapped, frame.bounds, cutoff_A)
    rank_to_track = {
        _as_int(row["component_rank"]): _as_int(row["track_id"])
        for row in island_rows
        if _as_int(row["step"]) == frame.step
    }
    if len(rank_to_track) != len(components):
        raise ValueError("Initial component count differs from accepted island ledger")
    owners: dict[int, int] = {}
    for rank, indices in enumerate(components, start=1):
        track = rank_to_track[rank]
        owners.update({int(frame.atom_ids[index]): track for index in indices})
    return owners


def _validate_owner_sizes(
    owners: Mapping[int, int],
    expected: Mapping[tuple[int, int], int],
    step: int,
) -> None:
    observed: dict[int, int] = defaultdict(int)
    for track in owners.values():
        observed[int(track)] += 1
    expected_step = {
        track: size for (candidate_step, track), size in expected.items() if candidate_step == step
    }
    if observed != expected_step:
        raise ValueError(
            f"Reconstructed island membership differs at step {step}: "
            f"observed={observed}, expected={expected_step}"
        )


def _block_summary(
    interval_rows: Sequence[Mapping[str, object]], block_ps: float
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in interval_rows:
        block = int(math.floor(float(row["mid_time_ps"]) / block_ps))
        grouped[(str(row["case_id"]), str(row["branch_id"]), str(row["direction"]), block)].append(row)
    output: list[dict[str, object]] = []
    categories = (
        "UNCHANGED_TRACK",
        "PERSISTENT_ISLAND_TRANSFER",
        "LINEAGE_REASSIGNMENT",
        "UNTRACKED_TRANSITION",
    )
    for (case_id, branch_id, direction, block), rows in sorted(grouped.items()):
        duration = sum(float(row["interval_ps"]) for row in rows)
        count = int(rows[0]["oxygen_count"])
        row: dict[str, object] = {
            "case_id": case_id,
            "branch_id": branch_id,
            "direction": direction,
            "block_index": block,
            "start_ps": block * block_ps,
            "end_ps": (block + 1) * block_ps,
            "intervals": len(rows),
            "duration_ps": duration,
            "oxygen_count": count,
        }
        for axis in ("x", "y"):
            total_displacement = sum(float(item[f"total_{axis}_displacement_A"]) for item in rows)
            row[f"mean_{axis}_velocity_mps"] = 100.0 * total_displacement / (count * duration)
            row[f"net_{axis}_crossing_rate_per_ps"] = sum(
                float(item[f"net_{axis}_crossings"]) for item in rows
            ) / duration
            for category in categories:
                value = sum(
                    float(item[f"{category.lower()}_{axis}_displacement_A"])
                    for item in rows
                )
                row[f"{category.lower()}_{axis}_displacement_A"] = value
                row[f"{category.lower()}_{axis}_velocity_mps"] = (
                    100.0 * value / (count * duration)
                )
            reconstructed = sum(
                float(row[f"{category.lower()}_{axis}_displacement_A"])
                for category in categories
            )
            row[f"{axis}_transport_closure_residual_A"] = total_displacement - reconstructed
        output.append(row)
    return output


def _add_baseline_response(rows: list[dict[str, object]]) -> None:
    baseline = {
        (str(row["case_id"]), int(row["block_index"])): row
        for row in rows
        if str(row["direction"]) == "none"
    }
    for row in rows:
        direction = str(row["direction"])
        axis = "x" if direction in {"none", "x"} else "y"
        if direction == "none":
            row["drive_axis"] = "none"
            row["axis_velocity_mps"] = float(row["mean_x_velocity_mps"])
            row["baseline_axis_velocity_mps"] = float(row["mean_x_velocity_mps"])
            row["excess_axis_velocity_mps"] = 0.0
            row["excess_axis_crossing_rate_per_ps"] = 0.0
            continue
        reference = baseline.get((str(row["case_id"]), int(row["block_index"])))
        if reference is None:
            raise ValueError("Missing time-aligned zero-force block")
        row["drive_axis"] = axis
        row["axis_velocity_mps"] = float(row[f"mean_{axis}_velocity_mps"])
        row["baseline_axis_velocity_mps"] = float(reference[f"mean_{axis}_velocity_mps"])
        row["excess_axis_velocity_mps"] = float(row[f"mean_{axis}_velocity_mps"]) - float(
            reference[f"mean_{axis}_velocity_mps"]
        )
        row["excess_axis_crossing_rate_per_ps"] = float(
            row[f"net_{axis}_crossing_rate_per_ps"]
        ) - float(reference[f"net_{axis}_crossing_rate_per_ps"])


def _plot(blocks: Sequence[Mapping[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    driven = [row for row in blocks if row["direction"] != "none"]
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    for branch in sorted({str(row["branch_id"]) for row in driven}):
        selected = [row for row in driven if row["branch_id"] == branch]
        time = [0.001 * (float(row["start_ps"]) + float(row["end_ps"])) / 2.0 for row in selected]
        axes[0].plot(time, [row["excess_axis_velocity_mps"] for row in selected], label=branch)
        exchange = []
        for row in selected:
            axis = str(row["drive_axis"])
            exchange.append(row[f"persistent_island_transfer_{axis}_velocity_mps"])
        axes[1].plot(time, exchange, label=branch)
    for axis in axes:
        axis.axhline(0.0, color="black", lw=0.7)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Excess water velocity (m/s)")
    axes[1].set_ylabel("Transfer-attributed velocity (m/s)")
    axes[1].set_xlabel("Time (ns)")
    figure.tight_layout()
    figure.savefig(output / "directed_flux_decomposition.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    """Run the directed-flux and persistent-island transport decomposition."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    timestep_fs = float(raw.get("timestep_fs", 0.0))
    block_ps = float(raw.get("block_ps", 50.0))
    cutoff_A = float(raw.get("cluster_cutoff_A", 0.0))
    plane_count = int(raw.get("plane_count", 16))
    maximum_displacement = float(raw.get("maximum_interval_displacement_A", 25.0))
    if min(timestep_fs, block_ps, cutoff_A, maximum_displacement) <= 0.0 or plane_count <= 0:
        raise ValueError("Invalid positive analysis parameter")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    output.mkdir(parents=True)
    base = contract_path.parent
    origin = int(raw.get("time_origin_step", 0))
    intervals: list[dict[str, object]] = []
    centers: list[dict[str, object]] = []
    inputs = [{"path": str(contract_path), "size_bytes": contract_path.stat().st_size, "sha256": sha256(contract_path)}]

    for entry in cases:
        case_id = str(entry["case_id"])
        branch_id = str(entry["branch_id"])
        direction = str(entry["direction"]).lower()
        paths = [resolve_path(path, base) for path in entry["trajectories"]]
        island_dir = resolve_path(entry["island_results"], base)
        exchange_path = island_dir / "molecule_exchange.tsv"
        island_path = island_dir / "island_timeseries.tsv"
        for path in [*paths, exchange_path, island_path]:
            inputs.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
        exchange_rows = [
            row
            for row in _read_tsv(exchange_path)
            if row["case_id"] == case_id and row["branch_id"] == branch_id
        ]
        island_rows = [
            row
            for row in _read_tsv(island_path)
            if row["case_id"] == case_id and row["branch_id"] == branch_id
        ]
        changes: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in exchange_rows:
            changes[_as_int(row["step"])].append(row)
        expected_sizes = {
            (_as_int(row["step"]), _as_int(row["track_id"])): _as_int(row["size"])
            for row in island_rows
        }
        frame_iterator = iter_unwrapped_oxygen_frames(paths)
        previous = next(frame_iterator, None)
        if previous is None:
            raise ValueError(f"No frames for {case_id}/{branch_id}")
        owners = _initial_owners(previous, island_rows, cutoff_A)
        _validate_owner_sizes(owners, expected_sizes, previous.step)
        for current in frame_iterator:
            dt_ps = (current.step - previous.step) * timestep_fs / 1000.0
            if dt_ps <= 0.0:
                raise ValueError("Non-positive frame interval")
            displacement = current.unwrapped - previous.unwrapped
            if float(np.max(np.linalg.norm(displacement, axis=1))) > maximum_displacement:
                raise ValueError(f"Interval displacement gate failed at step {current.step}")
            old_owners = dict(owners)
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
            categories = np.asarray(
                [category_by_id.get(int(atom_id), "UNCHANGED_TRACK") for atom_id in current.atom_ids]
            )
            lengths = current.bounds[:, 1] - current.bounds[:, 0]
            row: dict[str, object] = {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "start_step": previous.step,
                "end_step": current.step,
                "start_time_ps": (previous.step - origin) * timestep_fs / 1000.0,
                "end_time_ps": (current.step - origin) * timestep_fs / 1000.0,
                "mid_time_ps": ((previous.step + current.step) / 2.0 - origin) * timestep_fs / 1000.0,
                "interval_ps": dt_ps,
                "oxygen_count": len(current.atom_ids),
            }
            for axis_index, axis in enumerate(("x", "y")):
                values = displacement[:, axis_index]
                row[f"total_{axis}_displacement_A"] = float(np.sum(values))
                for category in (
                    "UNCHANGED_TRACK",
                    "PERSISTENT_ISLAND_TRANSFER",
                    "LINEAGE_REASSIGNMENT",
                    "UNTRACKED_TRANSITION",
                ):
                    row[f"{category.lower()}_{axis}_displacement_A"] = float(
                        np.sum(values[categories == category])
                    )
                positive, negative, net = plane_crossing_counts(
                    previous.unwrapped[:, axis_index],
                    current.unwrapped[:, axis_index],
                    lower_bound=current.bounds[axis_index, 0],
                    box_length=lengths[axis_index],
                    plane_count=plane_count,
                )
                row[f"positive_{axis}_crossings"] = positive
                row[f"negative_{axis}_crossings"] = negative
                row[f"net_{axis}_crossings"] = net
            intervals.append(row)

            old_tracks: dict[int, set[int]] = defaultdict(set)
            new_tracks: dict[int, set[int]] = defaultdict(set)
            for atom_id, track in old_owners.items():
                old_tracks[track].add(atom_id)
            for atom_id, track in owners.items():
                new_tracks[track].add(atom_id)
            for track in sorted(set(old_tracks) & set(new_tracks)):
                center_row: dict[str, object] = {
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "direction": direction,
                    "start_step": previous.step,
                    "end_step": current.step,
                    "end_time_ps": (current.step - origin) * timestep_fs / 1000.0,
                    "interval_ps": dt_ps,
                    "track_id": track,
                }
                for axis_index, axis in enumerate(("x", "y")):
                    values = decompose_track_center(
                        old_tracks[track],
                        new_tracks[track],
                        current.atom_ids,
                        previous.unwrapped[:, axis_index],
                        current.unwrapped[:, axis_index],
                    )
                    if axis == "x":
                        center_row.update(values)
                    else:
                        center_row.update({f"{key}_y": value for key, value in values.items()})
                centers.append(center_row)
            previous = current

    blocks = _block_summary(intervals, block_ps)
    _add_baseline_response(blocks)
    write_tsv(output / "directed_flux_intervals.tsv", intervals, tuple(intervals[0]))
    write_tsv(output / "transport_blocks_50ps.tsv", blocks, tuple(blocks[0]))
    write_tsv(output / "island_center_decomposition.tsv", centers, tuple(centers[0]))
    write_tsv(output / "input_manifest.tsv", inputs, ("path", "size_bytes", "sha256"))
    if bool(raw.get("write_plots", True)):
        _plot(blocks, output)
    maximum_closure = max(
        abs(float(row[field]))
        for row in blocks
        for field in ("x_transport_closure_residual_A", "y_transport_closure_residual_A")
    )
    maximum_center_closure = max(
        max(abs(float(row["closure_residual_A"])), abs(float(row["closure_residual_A_y"])))
        for row in centers
    )
    summary = {
        "status": "PASS",
        "case_branches": len(cases),
        "interval_rows": len(intervals),
        "block_rows": len(blocks),
        "center_decomposition_rows": len(centers),
        "plane_count": plane_count,
        "maximum_transport_closure_residual_A": maximum_closure,
        "maximum_center_closure_residual_A": maximum_center_closure,
        "single_trajectory_descriptive_only": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Directed island transport decomposition\n\n"
        "Water-oxygen displacement is partitioned exactly by persistent-island transfer, "
        "split/merge lineage reassignment, unchanged-track motion, and untracked transitions. "
        "Periodic plane crossings are averaged across phase-shifted planes. Track-center motion "
        "is decomposed exactly into old-member advection and membership change.\n\n"
        "Time blocks are single-trajectory diagnostics, not independent replicas.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary
