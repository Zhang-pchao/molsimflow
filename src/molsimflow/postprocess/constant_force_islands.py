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
    "maximum_size", "mean_size", "total_gained", "total_lost",
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
            raise ValueError("Each case entry must be an object")
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
    input_rows = [{"path": str(contract_path), "size_bytes": contract_path.stat().st_size, "sha256": sha256(contract_path)}]

    for entry in raw["cases"]:
        case_id, branch_id = str(entry["case_id"]), str(entry["branch_id"])
        direction = str(entry["direction"]).lower()
        paths = [resolve_path(path, base) for path in entry["trajectories"]]
        input_rows.extend({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths)
        previous: dict[int, set[int]] = {}
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
                    unwrapped = center.copy(); vx = math.nan; vy = math.nan
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
            previous_step = frame.timestep
        if previous_step is None:
            raise ValueError(f"No frames analyzed for {case_id}/{branch_id}")

    track_rows: list[dict] = []
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in island_rows:
        grouped[(row["case_id"], row["branch_id"], int(row["track_id"]))].append(row)
    for (case_id, branch_id, track_id), rows in sorted(grouped.items()):
        sizes = np.asarray([row["size"] for row in rows], dtype=float)
        track_rows.append({
            "case_id": case_id, "branch_id": branch_id, "direction": rows[0]["direction"],
            "track_id": track_id, "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
            "first_time_ps": rows[0]["time_ps"], "last_time_ps": rows[-1]["time_ps"],
            "lifetime_ps": rows[-1]["time_ps"] - rows[0]["time_ps"], "frames": len(rows),
            "minimum_size": int(np.min(sizes)), "maximum_size": int(np.max(sizes)),
            "mean_size": float(np.mean(sizes)),
            "total_gained": sum(int(row["gained_oxygen_count"]) for row in rows[1:]),
            "total_lost": sum(int(row["lost_oxygen_count"]) for row in rows[1:]),
        })

    write_tsv(output / "frame_summary.tsv", frame_rows, FRAME_FIELDS)
    write_tsv(output / "island_timeseries.tsv", island_rows, ISLAND_FIELDS)
    write_tsv(output / "lineage_events.tsv", event_rows, EVENT_FIELDS)
    write_tsv(output / "track_summary.tsv", track_rows, TRACK_FIELDS)
    write_tsv(output / "input_manifest.tsv", input_rows, ("path", "size_bytes", "sha256"))
    if bool(raw.get("write_plots", True)):
        _plot(frame_rows, island_rows, output)
    summary = {
        "status": "PASS", "case_branches": len(raw["cases"]),
        "frames": len(frame_rows), "island_rows": len(island_rows),
        "tracks": len(track_rows), "split_events": sum(row["event_type"] == "SPLIT" for row in event_rows),
        "merge_events": sum(row["event_type"] == "MERGE" for row in event_rows),
        "connectivity": "O-O cutoff with periodic X/Y and nonperiodic Z",
        "lineage": "greedy one-to-one oxygen-ID overlap; split/merge tables retain qualified multi-lineage overlaps",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Constant-force water-island tracking\n\n"
        f"Analyzed {len(raw['cases'])} branches and {len(frame_rows)} frames. "
        f"Resolved {len(track_rows)} persistent track IDs, {summary['split_events']} split events, "
        f"and {summary['merge_events']} merge events.\n\n"
        "Island identities use oxygen-ID overlap. Hydrogen exchange does not rename a water oxygen, "
        "and split/merge labels are geometric connectivity events.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary
