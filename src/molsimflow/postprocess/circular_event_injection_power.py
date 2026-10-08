"""Measure recovery power for injected one-generation circular event kernels."""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from molsimflow.postprocess.circular_event_association import (
    BlockColumns,
    BlockSupport,
    DistanceBin,
    EventColumns,
    _bh_adjust,
    _pair_counts,
    _read_blocks,
    _read_events,
    _sha256,
    _shift_per_arc,
    _validate_inputs,
    _write_csv,
    parse_distance_bins,
)
from molsimflow.postprocess.tpcl_dynamic_state import LagWindow, parse_windows

SCIENTIFIC_STATUS = (
    "TABLE_LEVEL_INJECTION_RECOVERY_SENSITIVITY_NOT_MOLECULAR_DYNAMICS_"
    "OR_A_PHYSICAL_BRANCHING_MODEL"
)


@dataclass(frozen=True)
class CombinationTask:
    case_id: str
    blocks: tuple[BlockSupport, ...]
    events: tuple[tuple[str, tuple[float, ...], tuple[int, ...]], ...]
    arc_count: int
    distance_bins: tuple[DistanceBin, ...]
    windows: tuple[LagWindow, ...]
    target_distance: str
    target_window: str
    branching_probability: float
    frame_ps: float
    null_samples: int
    injection_replicates: int
    combination_seed: int
    family_keys: tuple[tuple[str, str, str], ...]
    baseline_p_values: tuple[float, ...]


def parse_probabilities(raw: str) -> tuple[float, ...]:
    probabilities = tuple(float(item) for item in raw.split(","))
    if not probabilities or any(item <= 0.0 or item > 1.0 for item in probabilities):
        raise ValueError("branching probabilities must be in (0, 1]")
    if len(set(probabilities)) != len(probabilities):
        raise ValueError("branching probabilities must be unique")
    return probabilities


def _read_baseline(
    path: Path,
) -> tuple[tuple[tuple[str, str, str], ...], tuple[float, ...], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case_id", "distance_bin", "window", "empirical_two_sided_p"}
    if not rows or required.difference(rows[0]):
        raise ValueError(f"{path}: missing baseline rows or columns")
    keys = tuple((row["case_id"], row["distance_bin"], row["window"]) for row in rows)
    p_values = tuple(float(row["empirical_two_sided_p"]) for row in rows)
    if len(keys) != len(set(keys)) or any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError(f"{path}: duplicate keys or invalid p values")
    return keys, p_values, rows


def _frame_index(time_ps: float, frame_ps: float) -> int:
    index = round(time_ps / frame_ps)
    if abs(time_ps - index * frame_ps) > 1.0e-7:
        raise ValueError(f"event time {time_ps} ps is not aligned to {frame_ps} ps frames")
    return index


def _inject_events(
    blocks: Sequence[BlockSupport],
    base_events: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    arc_count: int,
    distance_bin: DistanceBin,
    window: LagWindow,
    probability: float,
    frame_ps: float,
    rng: np.random.Generator,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int, int, int, int]:
    lag_frames = np.arange(
        math.floor(window.start_ps / frame_ps) + 1,
        math.floor(window.end_ps / frame_ps) + 1,
        dtype=int,
    )
    if len(lag_frames) == 0:
        raise ValueError(f"window {window.name} contains no trajectory frames")
    injected = {}
    attempted = accepted = rejected_outside = coalesced = 0
    for block in blocks:
        times, arcs = base_events.get(
            block.label, (np.asarray([], dtype=float), np.asarray([], dtype=int))
        )
        output = [(float(time), int(arc)) for time, arc in zip(times, arcs)]
        occupied = {(_frame_index(time, frame_ps), arc) for time, arc in output}
        for parent_time, _parent_arc in output.copy():
            if rng.random() >= probability:
                continue
            attempted += 1
            distance = int(rng.integers(distance_bin.start, distance_bin.end))
            direction = -1 if int(rng.integers(0, 2)) == 0 else 1
            lag_frame = int(rng.choice(lag_frames))
            child_frame = _frame_index(parent_time, frame_ps) + lag_frame
            child_time = child_frame * frame_ps
            child_arc = (_parent_arc + direction * distance) % arc_count
            if child_time >= block.end_ps:
                rejected_outside += 1
                continue
            identity = (child_frame, child_arc)
            if identity in occupied:
                coalesced += 1
                continue
            occupied.add(identity)
            output.append((child_time, child_arc))
            accepted += 1
        output.sort()
        injected[block.label] = (
            np.asarray([item[0] for item in output], dtype=float),
            np.asarray([item[1] for item in output], dtype=int),
        )
    return injected, attempted, accepted, rejected_outside, coalesced


def _association_statistics(
    blocks: Sequence[BlockSupport],
    events: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    arc_count: int,
    distance_bins: Sequence[DistanceBin],
    windows: Sequence[LagWindow],
    null_samples: int,
    rng: np.random.Generator,
) -> dict[tuple[str, str], tuple[int, float, float, float]]:
    observed = np.zeros((len(distance_bins), len(windows)), dtype=float)
    null_totals = np.zeros((null_samples, len(distance_bins), len(windows)), dtype=float)
    for block in blocks:
        times, arcs = events.get(
            block.label, (np.asarray([], dtype=float), np.asarray([], dtype=int))
        )
        observed += _pair_counts(times, arcs, arc_count, distance_bins, windows)[1:]
        for sample in range(null_samples):
            shifted = _shift_per_arc(times, arcs, block.start_ps, block.end_ps - block.start_ps, rng)
            null_totals[sample] += _pair_counts(
                shifted, arcs, arc_count, distance_bins, windows
            )[1:]
    output = {}
    for distance_index, distance_bin in enumerate(distance_bins):
        for window_index, window in enumerate(windows):
            obs = float(observed[distance_index, window_index])
            null = null_totals[:, distance_index, window_index]
            expected = float(np.mean(null))
            lower_p = (1 + np.count_nonzero(null <= obs)) / (null_samples + 1)
            upper_p = (1 + np.count_nonzero(null >= obs)) / (null_samples + 1)
            output[(distance_bin.name, window.name)] = (
                int(obs),
                expected,
                obs / expected - 1.0 if expected > 0.0 else math.nan,
                min(1.0, 2.0 * min(lower_p, upper_p)),
            )
    return output


def _run_combination(task: CombinationTask) -> list[dict[str, object]]:
    base_events = {
        label: (np.asarray(times, dtype=float), np.asarray(arcs, dtype=int))
        for label, times, arcs in task.events
    }
    distance_bin = next(item for item in task.distance_bins if item.name == task.target_distance)
    window = next(item for item in task.windows if item.name == task.target_window)
    seed_sequences = np.random.SeedSequence(task.combination_seed).spawn(
        2 * task.injection_replicates
    )
    rows = []
    for replicate in range(task.injection_replicates):
        injection_seed = int(seed_sequences[2 * replicate].generate_state(1, dtype=np.uint64)[0])
        null_seed = int(seed_sequences[2 * replicate + 1].generate_state(1, dtype=np.uint64)[0])
        injected, attempted, accepted, rejected, coalesced = _inject_events(
            task.blocks,
            base_events,
            arc_count=task.arc_count,
            distance_bin=distance_bin,
            window=window,
            probability=task.branching_probability,
            frame_ps=task.frame_ps,
            rng=np.random.default_rng(injection_seed),
        )
        statistics = _association_statistics(
            task.blocks,
            injected,
            arc_count=task.arc_count,
            distance_bins=task.distance_bins,
            windows=task.windows,
            null_samples=task.null_samples,
            rng=np.random.default_rng(null_seed),
        )
        family_p = list(task.baseline_p_values)
        for key, values in statistics.items():
            family_index = task.family_keys.index((task.case_id, *key))
            family_p[family_index] = values[3]
        family_q = _bh_adjust(family_p)
        target = statistics[(task.target_distance, task.target_window)]
        target_index = task.family_keys.index(
            (task.case_id, task.target_distance, task.target_window)
        )
        target_q = family_q[target_index]
        rows.append(
            {
                "case_id": task.case_id,
                "distance_bin": task.target_distance,
                "window": task.target_window,
                "branching_probability": task.branching_probability,
                "replicate": replicate,
                "injection_seed": injection_seed,
                "null_seed": null_seed,
                "parent_event_count": sum(len(values[0]) for values in base_events.values()),
                "attempted_child_count": attempted,
                "accepted_child_count": accepted,
                "rejected_outside_block_count": rejected,
                "coalesced_child_count": coalesced,
                "target_observed_ordered_pairs": target[0],
                "target_null_mean_ordered_pairs": target[1],
                "target_chi_signed": target[2],
                "target_empirical_two_sided_p": target[3],
                "target_bh_q_primary_family": target_q,
                "recovered": math.isfinite(target[2]) and target[2] > 0.0 and target_q <= 0.05,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    return rows


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    fraction = successes / count
    denominator = 1.0 + z * z / count
    center = (fraction + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(fraction * (1.0 - fraction) / count + z * z / (4.0 * count**2))
    return center - radius / denominator, center + radius / denominator


def run_power(
    event_table: Path,
    block_table: Path,
    observed_primary_table: Path,
    output_dir: Path,
    *,
    event_columns: EventColumns,
    block_columns: BlockColumns,
    event_time_scale_to_ps: float,
    block_time_scale_to_ps: float,
    distance_bins: Sequence[DistanceBin],
    windows: Sequence[LagWindow],
    branching_probabilities: Sequence[float],
    frame_ps: float,
    null_samples: int,
    injection_replicates: int,
    random_seed: int,
    workers: int,
) -> dict[str, object]:
    """Inject one-generation children and measure target-kernel recovery."""

    if frame_ps <= 0.0 or null_samples < 20 or injection_replicates < 1 or workers < 1:
        raise ValueError("invalid frame, resampling, replicate, or worker count")
    case_order, events, arc_counts = _read_events(
        event_table, event_columns, event_time_scale_to_ps
    )
    blocks = _read_blocks(block_table, block_columns, block_time_scale_to_ps)
    _validate_inputs(case_order, events, arc_counts, blocks, distance_bins)
    family_keys, baseline_p_values, baseline_rows = _read_baseline(observed_primary_table)
    expected_keys = {
        (case_id, distance_bin.name, window.name)
        for case_id in case_order
        for distance_bin in distance_bins
        for window in windows
    }
    if set(family_keys) != expected_keys:
        raise ValueError("observed primary family does not match configured cases/kernels")
    for times, _ in events.values():
        for time_ps in times:
            _frame_index(float(time_ps), frame_ps)

    combinations = [
        (case_id, distance_bin.name, window.name, probability)
        for case_id in case_order
        for distance_bin in distance_bins
        for window in windows
        for probability in branching_probabilities
    ]
    seeds = [
        int(item.generate_state(1, dtype=np.uint64)[0])
        for item in np.random.SeedSequence(random_seed).spawn(len(combinations))
    ]
    tasks = []
    for combination, seed in zip(combinations, seeds):
        case_id, distance_name, window_name, probability = combination
        case_blocks = tuple(block for block in blocks if block.case_id == case_id)
        case_events = tuple(
            (
                block.label,
                tuple(float(value) for value in events.get((case_id, block.label), ((), ()))[0]),
                tuple(int(value) for value in events.get((case_id, block.label), ((), ()))[1]),
            )
            for block in case_blocks
        )
        tasks.append(
            CombinationTask(
                case_id=case_id,
                blocks=case_blocks,
                events=case_events,
                arc_count=arc_counts[case_id],
                distance_bins=tuple(distance_bins),
                windows=tuple(windows),
                target_distance=distance_name,
                target_window=window_name,
                branching_probability=probability,
                frame_ps=frame_ps,
                null_samples=null_samples,
                injection_replicates=injection_replicates,
                combination_seed=seed,
                family_keys=family_keys,
                baseline_p_values=baseline_p_values,
            )
        )
    if workers == 1:
        grouped_rows = map(_run_combination, tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            grouped_rows = list(executor.map(_run_combination, tasks, chunksize=1))
    replicate_rows = [row for group in grouped_rows for row in group]

    grouped = defaultdict(list)
    for row in replicate_rows:
        grouped[
            (
                row["case_id"],
                row["distance_bin"],
                row["window"],
                float(row["branching_probability"]),
            )
        ].append(row)
    power_rows = []
    for key, rows in grouped.items():
        recoveries = sum(bool(row["recovered"]) for row in rows)
        interval = _wilson_interval(recoveries, len(rows))
        power_rows.append(
            {
                "case_id": key[0],
                "distance_bin": key[1],
                "window": key[2],
                "branching_probability": key[3],
                "replicate_count": len(rows),
                "recovery_count": recoveries,
                "recovery_probability": recoveries / len(rows),
                "recovery_wilson_ci025": interval[0],
                "recovery_wilson_ci975": interval[1],
                "mean_attempted_child_count": float(
                    np.mean([int(row["attempted_child_count"]) for row in rows])
                ),
                "mean_accepted_child_count": float(
                    np.mean([int(row["accepted_child_count"]) for row in rows])
                ),
                "mean_rejected_outside_block_count": float(
                    np.mean([int(row["rejected_outside_block_count"]) for row in rows])
                ),
                "mean_coalesced_child_count": float(
                    np.mean([int(row["coalesced_child_count"]) for row in rows])
                ),
                "recovery_at_least_80pct": recoveries / len(rows) >= 0.8,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    power_rows.sort(
        key=lambda row: (
            case_order.index(str(row["case_id"])),
            next(i for i, item in enumerate(distance_bins) if item.name == row["distance_bin"]),
            next(i for i, item in enumerate(windows) if item.name == row["window"]),
            float(row["branching_probability"]),
        )
    )
    threshold_rows = []
    for case_id in case_order:
        for distance_bin in distance_bins:
            for window in windows:
                selected = [
                    row
                    for row in power_rows
                    if row["case_id"] == case_id
                    and row["distance_bin"] == distance_bin.name
                    and row["window"] == window.name
                ]
                qualified = [
                    float(row["branching_probability"])
                    for row in selected
                    if row["recovery_at_least_80pct"]
                ]
                baseline = next(
                    row
                    for row in baseline_rows
                    if (row["case_id"], row["distance_bin"], row["window"])
                    == (case_id, distance_bin.name, window.name)
                )
                threshold_rows.append(
                    {
                        "case_id": case_id,
                        "distance_bin": distance_bin.name,
                        "window": window.name,
                        "observed_chi_signed": baseline.get("chi_signed", ""),
                        "observed_bh_q_primary_family": baseline.get(
                            "bh_q_primary_family", ""
                        ),
                        "observed_association_class": baseline.get("association_class", ""),
                        "smallest_branching_probability_80pct": (
                            min(qualified) if qualified else math.nan
                        ),
                        "scientific_status": SCIENTIFIC_STATUS,
                    }
                )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "injection_replicates.csv", replicate_rows)
    _write_csv(output / "power_summary.csv", power_rows)
    _write_csv(output / "recovery_thresholds.csv", threshold_rows)
    summary = {
        "status": "PASS",
        "case_count": len(case_order),
        "kernel_count": len(distance_bins) * len(windows),
        "branching_probability_count": len(branching_probabilities),
        "combination_count": len(combinations),
        "injection_replicates_per_combination": injection_replicates,
        "replicate_row_count": len(replicate_rows),
        "power_row_count": len(power_rows),
        "threshold_row_count": len(threshold_rows),
        "null_samples_per_replicate": null_samples,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        **summary,
        "event_table": {"path": str(event_table), "sha256": _sha256(event_table)},
        "block_table": {"path": str(block_table), "sha256": _sha256(block_table)},
        "observed_primary_table": {
            "path": str(observed_primary_table),
            "sha256": _sha256(observed_primary_table),
        },
        "event_columns": asdict(event_columns),
        "block_columns": asdict(block_columns),
        "event_time_scale_to_ps": event_time_scale_to_ps,
        "block_time_scale_to_ps": block_time_scale_to_ps,
        "distance_bins": [asdict(item) for item in distance_bins],
        "windows": [asdict(item) for item in windows],
        "branching_probabilities": list(branching_probabilities),
        "frame_ps": frame_ps,
        "random_seed": random_seed,
        "workers": workers,
        "unchanged_surface_p_values": "accepted observed-family values",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "# Circular event injection--recovery sensitivity\n\n"
        "Each accepted cluster anchor independently attempts at most one child in the "
        "configured distance/lag kernel. Children do not branch. Out-of-block children "
        "are rejected and duplicate site/frame events are coalesced. The injected case "
        "is reanalyzed with the observed-analysis circular-shift null; accepted observed "
        "p values are retained for unchanged surfaces before the same family-wide BH "
        "correction. This table-level sensitivity test is not MD or a physical branching "
        "model.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-table", type=Path, required=True)
    parser.add_argument("--block-table", type=Path, required=True)
    parser.add_argument("--observed-primary-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-case-column", default="case_id")
    parser.add_argument("--event-time-column", default="transition_time_ns")
    parser.add_argument("--event-arc-column", default="primary_arc_index")
    parser.add_argument("--event-arc-count-column", default="arc_count")
    parser.add_argument("--event-block-column", default="time_block_200ps")
    parser.add_argument("--event-time-scale-to-ps", type=float, default=1000.0)
    parser.add_argument("--block-case-column", default="case_id")
    parser.add_argument("--block-index-column", default="block_index")
    parser.add_argument("--block-start-column", default="block_start_ps")
    parser.add_argument("--block-end-column", default="block_end_ps")
    parser.add_argument("--block-event-count-column", default="event_count")
    parser.add_argument("--block-time-scale-to-ps", type=float, default=1.0)
    parser.add_argument("--distance-bins", default="near:1:4,mid:4:10,far:10:19")
    parser.add_argument("--windows", default="fast:0:5,slow:5:50")
    parser.add_argument("--branching-probabilities", default="0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--frame-ps", type=float, default=0.5)
    parser.add_argument("--null-samples", type=int, default=2000)
    parser.add_argument("--injection-replicates", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260904)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_power(
        args.event_table,
        args.block_table,
        args.observed_primary_table,
        args.output_dir,
        event_columns=EventColumns(
            case=args.event_case_column,
            time=args.event_time_column,
            arc=args.event_arc_column,
            arc_count=args.event_arc_count_column,
            block=args.event_block_column,
        ),
        block_columns=BlockColumns(
            case=args.block_case_column,
            block=args.block_index_column,
            start=args.block_start_column,
            end=args.block_end_column,
            event_count=args.block_event_count_column,
        ),
        event_time_scale_to_ps=args.event_time_scale_to_ps,
        block_time_scale_to_ps=args.block_time_scale_to_ps,
        distance_bins=parse_distance_bins(args.distance_bins),
        windows=parse_windows(args.windows),
        branching_probabilities=parse_probabilities(args.branching_probabilities),
        frame_ps=args.frame_ps,
        null_samples=args.null_samples,
        injection_replicates=args.injection_replicates,
        random_seed=args.random_seed,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
