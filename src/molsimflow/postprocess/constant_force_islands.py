"""Track persistent water-oxygen islands in constant-force trajectories."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.constant_force_oxygen import (
    connected_components,
    iter_oxygen_frames,
    periodic_xy_center,
    resolve_path,
    sha256,
    unwrap_xy_center,
    write_output_hashes,
    write_tsv,
)

FRAME_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "component_count",
    "largest_size", "largest_fraction", "second_size", "singleton_count",
)
ISLAND_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "track_id",
    "component_rank", "size", "fraction", "center_x_A", "center_y_A", "center_z_A",
    "center_x_unwrapped_A", "center_y_unwrapped_A", "vx_mps", "vy_mps",
    "matched_predecessor_track_id", "overlap_count", "retained_fraction",
    "gained_oxygen_count", "lost_oxygen_count",
)
EVENT_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "event_type",
    "source_track_ids", "target_track_ids", "overlap_oxygen_count",
)
TRACK_FIELDS = (
    "case_id", "branch_id", "direction", "track_id", "first_step", "last_step",
    "first_time_ps", "last_time_ps", "lifetime_ps", "frames", "minimum_size",
    "maximum_size", "mean_size", "mean_vx_mps", "mean_vy_mps",
    "main_rank_frames", "satellite_rank_frames", "total_gained", "total_lost",
    "inbound_oxygen_count", "outbound_oxygen_count", "net_oxygen_transfer",
    "exchange_partner_count",
)
EXCHANGE_FIELDS = (
    "case_id", "branch_id", "direction", "step", "time_ps", "interval_ps",
    "oxygen_id", "source_track_id", "target_track_id", "source_size", "target_size",
    "source_component_rank", "target_component_rank", "source_role", "target_role",
    "exchange_class",
)
EXCHANGE_SUMMARY_FIELDS = (
    "case_id", "branch_id", "direction", "source_track_id", "target_track_id",
    "source_role", "target_role", "exchange_class", "oxygen_transfer_count",
    "event_frames", "first_time_ps", "last_time_ps", "observation_duration_ps",
    "transfer_rate_per_ns",
)
BRANCH_FIELDS = (
    "case_id", "branch_id", "direction", "frames", "observation_duration_ps",
    "main_island_mean_vx_mps", "main_island_mean_vy_mps",
    "satellite_size_weighted_mean_vx_mps", "satellite_size_weighted_mean_vy_mps",
    "gross_gained_oxygen_count", "gross_lost_oxygen_count", "net_tracked_oxygen_change",
    "track_to_track_transfer_count", "persistent_island_transfer_count",
    "lineage_reassignment_count", "entry_from_untracked_count", "exit_to_untracked_count",
    "main_island_inbound_count", "main_island_outbound_count",
    "main_island_net_oxygen_transfer", "track_to_track_transfer_rate_per_ns",
)


def _validate_contract(raw: Mapping[str, object]) -> None:
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    if float(raw.get("timestep_fs", 0.0)) <= 0.0:
        raise ValueError("timestep_fs must be positive")
    if float(raw.get("cluster_cutoff_A", 0.0)) <= 0.0:
        raise ValueError("cluster_cutoff_A must be positive")
    overlap_fraction = float(raw.get("lineage_overlap_fraction", 0.25))
    if not 0.0 < overlap_fraction <= 1.0:
        raise ValueError("lineage_overlap_fraction must be in (0, 1]")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    identities = []
    for entry in cases:
        if not isinstance(entry, dict):
            raise TypeError("Each case entry must be an object")
        for key in ("case_id", "branch_id", "direction", "trajectories"):
            if key not in entry:
                raise ValueError(f"Case entry is missing {key}")
        direction = str(entry["direction"]).lower()
        if direction not in {"none", "x", "y"}:
            raise ValueError("direction must be none, x, or y")
        if not isinstance(entry["trajectories"], list) or not entry["trajectories"]:
            raise ValueError("trajectories must be a non-empty list")
        identities.append((str(entry["case_id"]), str(entry["branch_id"])))
    if len(identities) != len(set(identities)):
        raise ValueError("case_id/branch_id pairs must be unique")


def _qualified_overlap(
    overlap: int,
    previous_size: int,
    current_size: int,
    fraction: float,
    minimum_count: int,
) -> bool:
    return overlap >= minimum_count and overlap / min(previous_size, current_size) >= fraction


def match_components(
    previous: Mapping[int, set[int]],
    current: Sequence[set[int]],
    *,
    overlap_fraction: float,
) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    """Return a greedy one-to-one current-index to previous-track assignment."""

    overlaps = {
        (track_id, index): len(ids & component)
        for track_id, ids in previous.items()
        for index, component in enumerate(current)
        if ids & component
    }
    candidates = sorted(
        (
            overlap,
            overlap / min(len(previous[track]), len(current[index])),
            track,
            index,
        )
        for (track, index), overlap in overlaps.items()
        if overlap / min(len(previous[track]), len(current[index])) >= overlap_fraction
    )
    assignment: dict[int, int] = {}
    used_tracks: set[int] = set()
    for _, _, track, index in reversed(candidates):
        if track in used_tracks or index in assignment:
            continue
        assignment[index] = track
        used_tracks.add(track)
    return assignment, overlaps


def _finite_mean(values: Sequence[object]) -> float:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    return float(np.mean(array[finite])) if np.any(finite) else math.nan


def _size_weighted_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=float)
    weights = np.asarray([row["size"] for row in rows], dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[finite], weights=weights[finite])) if np.any(finite) else math.nan


def _component_role(rank: int | None) -> str:
    if rank is None:
        return "untracked"
    return "main" if rank == 1 else "satellite"


def _exchange_class(
    source: int | None,
    target: int | None,
    previous_tracks: set[int],
    current_tracks: set[int],
) -> str:
    if source is None:
        return "ENTRY_FROM_UNTRACKED"
    if target is None:
        return "EXIT_TO_UNTRACKED"
    if source in current_tracks and target in previous_tracks:
        return "PERSISTENT_ISLAND_TRANSFER"
    if source not in current_tracks and target in previous_tracks:
        return "MERGE_LINEAGE_REASSIGNMENT"
    if source in current_tracks and target not in previous_tracks:
        return "SPLIT_LINEAGE_REASSIGNMENT"
    return "LINEAGE_REASSIGNMENT"


def _plot(frame_rows: list[dict], island_rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    keys = sorted({(row["case_id"], row["branch_id"]) for row in frame_rows})
    figure, axes = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 6.2), squeeze=False)
    for column, key in enumerate(keys):
        frames = [row for row in frame_rows if (row["case_id"], row["branch_id"]) == key]
        times = np.asarray([row["time_ps"] for row in frames]) / 1000.0
        axes[0, column].plot(times, [row["largest_fraction"] for row in frames], label="largest fraction")
        secondary = axes[0, column].twinx()
        secondary.plot(times, [row["component_count"] for row in frames], color="tab:orange", alpha=0.7)
        axes[0, column].set_ylabel("Largest fraction")
        secondary.set_ylabel("Components")
        tracks = [row for row in island_rows if (row["case_id"], row["branch_id"]) == key]
        track_sizes: dict[int, int] = defaultdict(int)
        for row in tracks:
            track_sizes[int(row["track_id"])] = max(track_sizes[int(row["track_id"])], int(row["size"]))
        for track_id, _ in sorted(track_sizes.items(), key=lambda item: item[1], reverse=True)[:5]:
            selected = [row for row in tracks if int(row["track_id"]) == track_id]
            axes[1, column].plot(
                np.asarray([row["time_ps"] for row in selected]) / 1000.0,
                [row["size"] for row in selected],
                label=f"island {track_id}",
            )
        axes[1, column].set_xlabel("Time (ns)")
        axes[1, column].set_ylabel("Island size")
        axes[1, column].legend(frameon=False, fontsize=7)
        axes[0, column].set_title(f"{key[0]} / {key[1]}")
    figure.tight_layout()
    figure.savefig(output / "island_overview.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(raw)
    output.mkdir(parents=True)
    base = contract_path.parent
    timestep_fs = float(raw["timestep_fs"])
    origin = int(raw.get("time_origin_step", 0))
    cutoff = float(raw["cluster_cutoff_A"])
    overlap_fraction = float(raw.get("lineage_overlap_fraction", 0.25))
    event_overlap_fraction = float(raw.get("event_overlap_fraction", 0.25))
    event_minimum_count = int(raw.get("event_minimum_overlap_count", 2))
    frame_rows: list[dict] = []
    island_rows: list[dict] = []
    event_rows: list[dict] = []
    exchange_rows: list[dict] = []
    input_rows = [{"path": str(contract_path), "size_bytes": contract_path.stat().st_size, "sha256": sha256(contract_path)}]

    for entry in raw["cases"]:
        case_id, branch_id = str(entry["case_id"]), str(entry["branch_id"])
        direction = str(entry["direction"]).lower()
        paths = [resolve_path(path, base) for path in entry["trajectories"]]
        input_rows.extend({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths)
        previous: dict[int, set[int]] = {}
        previous_ranks: dict[int, int] = {}
        previous_centers: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        next_track = 1
        previous_step: int | None = None
        for frame in iter_oxygen_frames(paths):
            time_ps = (frame.timestep - origin) * timestep_fs / 1000.0
            component_indices = connected_components(frame.coordinates, frame.bounds, cutoff)
            components = [set(frame.atom_ids[index].tolist()) for index in component_indices]
            assignment, _ = match_components(previous, components, overlap_fraction=overlap_fraction)
            current_tracks: dict[int, set[int]] = {}
            index_to_track: dict[int, int] = {}
            for index in range(len(components)):
                track = assignment.get(index)
                if track is None:
                    track = next_track
                    next_track += 1
                index_to_track[index] = track
                current_tracks[track] = components[index]

            if previous_step is not None:
                previous_owner = {
                    oxygen_id: track_id
                    for track_id, oxygen_ids in previous.items()
                    for oxygen_id in oxygen_ids
                }
                current_owner = {
                    oxygen_id: track_id
                    for track_id, oxygen_ids in current_tracks.items()
                    for oxygen_id in oxygen_ids
                }
                current_ranks = {
                    index_to_track[index]: index + 1 for index in range(len(components))
                }
                previous_tracks = set(previous)
                current_track_ids = set(current_tracks)
                interval_ps = (frame.timestep - previous_step) * timestep_fs / 1000.0
                for oxygen_id in sorted(set(previous_owner) | set(current_owner)):
                    source = previous_owner.get(oxygen_id)
                    target = current_owner.get(oxygen_id)
                    if source == target:
                        continue
                    source_rank = previous_ranks.get(source) if source is not None else None
                    target_rank = current_ranks.get(target) if target is not None else None
                    exchange_rows.append({
                        "case_id": case_id, "branch_id": branch_id, "direction": direction,
                        "step": frame.timestep, "time_ps": time_ps, "interval_ps": interval_ps,
                        "oxygen_id": oxygen_id,
                        "source_track_id": source if source is not None else "",
                        "target_track_id": target if target is not None else "",
                        "source_size": len(previous[source]) if source is not None else "",
                        "target_size": len(current_tracks[target]) if target is not None else "",
                        "source_component_rank": source_rank if source_rank is not None else "",
                        "target_component_rank": target_rank if target_rank is not None else "",
                        "source_role": _component_role(source_rank),
                        "target_role": _component_role(target_rank),
                        "exchange_class": _exchange_class(
                            source, target, previous_tracks, current_track_ids
                        ),
                    })

            for previous_track, previous_ids in previous.items():
                targets = [
                    index_to_track[index]
                    for index, component in enumerate(components)
                    if _qualified_overlap(
                        len(previous_ids & component), len(previous_ids), len(component),
                        event_overlap_fraction, event_minimum_count,
                    )
                ]
                if len(set(targets)) > 1:
                    event_rows.append({
                        "case_id": case_id, "branch_id": branch_id, "direction": direction,
                        "step": frame.timestep, "time_ps": time_ps, "event_type": "SPLIT",
                        "source_track_ids": str(previous_track),
                        "target_track_ids": ",".join(map(str, sorted(set(targets)))),
                        "overlap_oxygen_count": sum(len(previous_ids & components[index]) for index in range(len(components)) if index_to_track[index] in targets),
                    })
            for index, component in enumerate(components):
                sources = [
                    track
                    for track, previous_ids in previous.items()
                    if _qualified_overlap(
                        len(previous_ids & component), len(previous_ids), len(component),
                        event_overlap_fraction, event_minimum_count,
                    )
                ]
                if len(sources) > 1:
                    event_rows.append({
                        "case_id": case_id, "branch_id": branch_id, "direction": direction,
                        "step": frame.timestep, "time_ps": time_ps, "event_type": "MERGE",
                        "source_track_ids": ",".join(map(str, sorted(sources))),
                        "target_track_ids": str(index_to_track[index]),
                        "overlap_oxygen_count": sum(len(previous[track] & component) for track in sources),
                    })

            total = len(frame.atom_ids)
            frame_rows.append({
                "case_id": case_id, "branch_id": branch_id, "direction": direction,
                "step": frame.timestep, "time_ps": time_ps, "component_count": len(components),
                "largest_size": len(components[0]), "largest_fraction": len(components[0]) / total,
                "second_size": len(components[1]) if len(components) > 1 else 0,
                "singleton_count": sum(len(component) == 1 for component in components),
            })
            id_to_index = {int(atom_id): index for index, atom_id in enumerate(frame.atom_ids)}
            for rank, component in enumerate(components, start=1):
                track = index_to_track[rank - 1]
                indices = np.asarray([id_to_index[atom_id] for atom_id in component])
                center = periodic_xy_center(frame.coordinates[indices], frame.bounds)
                matched = assignment.get(rank - 1)
                overlap = len(component & previous.get(matched, set())) if matched is not None else 0
                gained = len(component - previous.get(matched, set())) if matched is not None else len(component)
                lost = len(previous.get(matched, set()) - component) if matched is not None else 0
                if track in previous_centers:
                    previous_center, previous_unwrapped, previous_time = previous_centers[track]
                    unwrapped = unwrap_xy_center(center, previous_center, previous_unwrapped, frame.bounds)
                    dt_ps = time_ps - previous_time
                    vx = 100.0 * (unwrapped[0] - previous_unwrapped[0]) / dt_ps if dt_ps > 0 else math.nan
                    vy = 100.0 * (unwrapped[1] - previous_unwrapped[1]) / dt_ps if dt_ps > 0 else math.nan
                else:
                    unwrapped = center.copy()
                    vx = math.nan
                    vy = math.nan
                previous_centers[track] = (center.copy(), unwrapped.copy(), time_ps)
                row = {
                    "case_id": case_id, "branch_id": branch_id, "direction": direction,
                    "step": frame.timestep, "time_ps": time_ps, "track_id": track,
                    "component_rank": rank, "size": len(component), "fraction": len(component) / total,
                    "center_x_A": center[0], "center_y_A": center[1], "center_z_A": center[2],
                    "center_x_unwrapped_A": unwrapped[0], "center_y_unwrapped_A": unwrapped[1],
                    "vx_mps": vx, "vy_mps": vy,
                    "matched_predecessor_track_id": matched if matched is not None else "",
                    "overlap_count": overlap,
                    "retained_fraction": overlap / len(component) if component else math.nan,
                    "gained_oxygen_count": gained, "lost_oxygen_count": lost,
                }
                island_rows.append(row)
            previous = current_tracks
            previous_ranks = {
                index_to_track[index]: index + 1 for index in range(len(components))
            }
            previous_step = frame.timestep
        if previous_step is None:
            raise ValueError(f"No frames analyzed for {case_id}/{branch_id}")

    track_rows: list[dict] = []
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in island_rows:
        grouped[(row["case_id"], row["branch_id"], int(row["track_id"]))].append(row)
    for (case_id, branch_id, track_id), rows in sorted(grouped.items()):
        sizes = np.asarray([row["size"] for row in rows], dtype=float)
        inbound = [row for row in exchange_rows if row["case_id"] == case_id
                   and row["branch_id"] == branch_id and row["target_track_id"] == track_id]
        outbound = [row for row in exchange_rows if row["case_id"] == case_id
                    and row["branch_id"] == branch_id and row["source_track_id"] == track_id]
        partners = {
            int(row["source_track_id"])
            for row in inbound
            if row["source_track_id"] != ""
        } | {
            int(row["target_track_id"])
            for row in outbound
            if row["target_track_id"] != ""
        }
        track_rows.append({
            "case_id": case_id, "branch_id": branch_id, "direction": rows[0]["direction"],
            "track_id": track_id, "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
            "first_time_ps": rows[0]["time_ps"], "last_time_ps": rows[-1]["time_ps"],
            "lifetime_ps": rows[-1]["time_ps"] - rows[0]["time_ps"], "frames": len(rows),
            "minimum_size": int(np.min(sizes)), "maximum_size": int(np.max(sizes)),
            "mean_size": float(np.mean(sizes)),
            "mean_vx_mps": _finite_mean([row["vx_mps"] for row in rows]),
            "mean_vy_mps": _finite_mean([row["vy_mps"] for row in rows]),
            "main_rank_frames": sum(int(row["component_rank"]) == 1 for row in rows),
            "satellite_rank_frames": sum(int(row["component_rank"]) > 1 for row in rows),
            "total_gained": sum(int(row["gained_oxygen_count"]) for row in rows[1:]),
            "total_lost": sum(int(row["lost_oxygen_count"]) for row in rows[1:]),
            "inbound_oxygen_count": len(inbound), "outbound_oxygen_count": len(outbound),
            "net_oxygen_transfer": len(inbound) - len(outbound),
            "exchange_partner_count": len(partners),
        })

    branch_times: dict[tuple[str, str], tuple[float, float]] = {}
    for row in frame_rows:
        key = (str(row["case_id"]), str(row["branch_id"]))
        time_ps = float(row["time_ps"])
        first, last = branch_times.get(key, (time_ps, time_ps))
        branch_times[key] = (min(first, time_ps), max(last, time_ps))

    exchange_summary_rows: list[dict] = []
    exchange_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in exchange_rows:
        key = (
            row["case_id"], row["branch_id"], row["direction"],
            row["source_track_id"], row["target_track_id"],
            row["source_role"], row["target_role"], row["exchange_class"],
        )
        exchange_groups[key].append(row)
    for key, rows in sorted(exchange_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        case_id, branch_id, direction, source, target, source_role, target_role, exchange_class = key
        first_time, last_time = branch_times[(str(case_id), str(branch_id))]
        duration_ps = last_time - first_time
        exchange_summary_rows.append({
            "case_id": case_id, "branch_id": branch_id, "direction": direction,
            "source_track_id": source, "target_track_id": target,
            "source_role": source_role, "target_role": target_role,
            "exchange_class": exchange_class, "oxygen_transfer_count": len(rows),
            "event_frames": len({int(row["step"]) for row in rows}),
            "first_time_ps": min(float(row["time_ps"]) for row in rows),
            "last_time_ps": max(float(row["time_ps"]) for row in rows),
            "observation_duration_ps": duration_ps,
            "transfer_rate_per_ns": 1000.0 * len(rows) / duration_ps if duration_ps > 0 else math.nan,
        })

    branch_rows: list[dict] = []
    branch_keys = sorted(branch_times)
    for case_id, branch_id in branch_keys:
        frames = [row for row in frame_rows
                  if row["case_id"] == case_id and row["branch_id"] == branch_id]
        islands = [row for row in island_rows
                   if row["case_id"] == case_id and row["branch_id"] == branch_id]
        exchanges = [row for row in exchange_rows
                     if row["case_id"] == case_id and row["branch_id"] == branch_id]
        main = [row for row in islands if int(row["component_rank"]) == 1]
        satellites = [row for row in islands if int(row["component_rank"]) > 1]
        first_time, last_time = branch_times[(case_id, branch_id)]
        duration_ps = last_time - first_time
        track_to_track = [row for row in exchanges
                          if row["source_track_id"] != "" and row["target_track_id"] != ""]
        main_inbound = sum(
            row["target_component_rank"] == 1 and row["source_component_rank"] != 1
            for row in exchanges
        )
        main_outbound = sum(
            row["source_component_rank"] == 1 and row["target_component_rank"] != 1
            for row in exchanges
        )
        direction = str(frames[0]["direction"])
        branch_rows.append({
            "case_id": case_id, "branch_id": branch_id, "direction": direction,
            "frames": len(frames), "observation_duration_ps": duration_ps,
            "main_island_mean_vx_mps": _finite_mean([row["vx_mps"] for row in main]),
            "main_island_mean_vy_mps": _finite_mean([row["vy_mps"] for row in main]),
            "satellite_size_weighted_mean_vx_mps": _size_weighted_mean(satellites, "vx_mps"),
            "satellite_size_weighted_mean_vy_mps": _size_weighted_mean(satellites, "vy_mps"),
            "gross_gained_oxygen_count": sum(row["target_track_id"] != "" for row in exchanges),
            "gross_lost_oxygen_count": sum(row["source_track_id"] != "" for row in exchanges),
            "net_tracked_oxygen_change": (
                sum(row["target_track_id"] != "" for row in exchanges)
                - sum(row["source_track_id"] != "" for row in exchanges)
            ),
            "track_to_track_transfer_count": len(track_to_track),
            "persistent_island_transfer_count": sum(
                row["exchange_class"] == "PERSISTENT_ISLAND_TRANSFER" for row in exchanges
            ),
            "lineage_reassignment_count": sum(
                str(row["exchange_class"]).endswith("LINEAGE_REASSIGNMENT") for row in exchanges
            ),
            "entry_from_untracked_count": sum(
                row["exchange_class"] == "ENTRY_FROM_UNTRACKED" for row in exchanges
            ),
            "exit_to_untracked_count": sum(
                row["exchange_class"] == "EXIT_TO_UNTRACKED" for row in exchanges
            ),
            "main_island_inbound_count": main_inbound,
            "main_island_outbound_count": main_outbound,
            "main_island_net_oxygen_transfer": main_inbound - main_outbound,
            "track_to_track_transfer_rate_per_ns": (
                1000.0 * len(track_to_track) / duration_ps if duration_ps > 0 else math.nan
            ),
        })

    write_tsv(output / "frame_summary.tsv", frame_rows, FRAME_FIELDS)
    write_tsv(output / "island_timeseries.tsv", island_rows, ISLAND_FIELDS)
    write_tsv(output / "lineage_events.tsv", event_rows, EVENT_FIELDS)
    write_tsv(output / "molecule_exchange.tsv", exchange_rows, EXCHANGE_FIELDS)
    write_tsv(output / "track_exchange_summary.tsv", exchange_summary_rows, EXCHANGE_SUMMARY_FIELDS)
    write_tsv(output / "branch_transport_summary.tsv", branch_rows, BRANCH_FIELDS)
    write_tsv(output / "track_summary.tsv", track_rows, TRACK_FIELDS)
    write_tsv(output / "input_manifest.tsv", input_rows, ("path", "size_bytes", "sha256"))
    if bool(raw.get("write_plots", True)):
        _plot(frame_rows, island_rows, output)
    summary = {
        "status": "PASS", "case_branches": len(raw["cases"]),
        "frames": len(frame_rows), "island_rows": len(island_rows),
        "tracks": len(track_rows), "split_events": sum(row["event_type"] == "SPLIT" for row in event_rows),
        "merge_events": sum(row["event_type"] == "MERGE" for row in event_rows),
        "molecule_exchange_rows": len(exchange_rows),
        "persistent_island_transfer_rows": sum(
            row["exchange_class"] == "PERSISTENT_ISLAND_TRANSFER" for row in exchange_rows
        ),
        "lineage_reassignment_rows": sum(
            str(row["exchange_class"]).endswith("LINEAGE_REASSIGNMENT") for row in exchange_rows
        ),
        "entry_from_untracked_rows": sum(
            row["exchange_class"] == "ENTRY_FROM_UNTRACKED" for row in exchange_rows
        ),
        "exit_to_untracked_rows": sum(
            row["exchange_class"] == "EXIT_TO_UNTRACKED" for row in exchange_rows
        ),
        "connectivity": "O-O cutoff with periodic X/Y and nonperiodic Z",
        "lineage": "greedy one-to-one oxygen-ID overlap; split/merge tables retain qualified multi-lineage overlaps",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Constant-force water-island tracking\n\n"
        f"Analyzed {len(raw['cases'])} branches and {len(frame_rows)} frames. "
        f"Resolved {len(track_rows)} persistent track IDs, {summary['split_events']} split events, "
        f"{summary['merge_events']} merge events, and {len(exchange_rows)} identity-resolved "
        "oxygen transfers.\n\n"
        "Island identities use oxygen-ID overlap. Hydrogen exchange does not rename a water oxygen, "
        "and split/merge labels are geometric connectivity events. `molecule_exchange.tsv` "
        "separates transfers between persistent tracks, lineage reassignment during split/merge, "
        "and entry/exit when the selected oxygen population changes.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary
