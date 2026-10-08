"""Measure signed lagged associations between events on circular sites.

The null independently circular-shifts each site's complete event sequence
inside each declared time block.  Results are retrospective associations, not
causal triggering, propagation speeds, or replicate-level inference.
"""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from molsimflow.postprocess.tpcl_dynamic_state import (
    LagWindow,
    _bh_adjust,
    _sha256,
    _shift_per_arc,
    _write_csv,
    parse_windows,
)

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_SINGLE_TRAJECTORY_SIGNED_CIRCULAR_EVENT_ASSOCIATION_"
    "NOT_CAUSAL_OR_REPLICATE_LEVEL_EVIDENCE"
)


@dataclass(frozen=True)
class DistanceBin:
    """Half-open circular-distance bin in site units."""

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class EventColumns:
    """Column routing for an event table."""

    case: str = "case_id"
    time: str = "transition_time_ns"
    arc: str = "primary_arc_index"
    arc_count: str = "arc_count"
    block: str = "time_block_200ps"


@dataclass(frozen=True)
class BlockColumns:
    """Column routing for a block-support table."""

    case: str = "case_id"
    block: str = "block_index"
    start: str = "block_start_ps"
    end: str = "block_end_ps"
    event_count: str = "event_count"


@dataclass(frozen=True)
class BlockSupport:
    """Accepted support and source-table event count for one case/time block."""

    case_id: str
    label: str
    start_ps: float
    end_ps: float
    source_event_count: int


def parse_distance_bins(raw: str) -> tuple[DistanceBin, ...]:
    """Parse ``name:start:end`` comma-separated half-open bins."""

    bins = []
    for item in raw.split(","):
        fields = item.split(":")
        if len(fields) != 3 or not fields[0]:
            raise ValueError(f"invalid distance bin {item!r}")
        distance_bin = DistanceBin(fields[0], int(fields[1]), int(fields[2]))
        if distance_bin.start < 1 or distance_bin.end <= distance_bin.start:
            raise ValueError(f"invalid distance bin {item!r}")
        bins.append(distance_bin)
    if not bins or len({item.name for item in bins}) != len(bins):
        raise ValueError("distance bins must be nonempty and uniquely named")
    return tuple(bins)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows, set(reader.fieldnames)


def _read_blocks(
    path: Path,
    columns: BlockColumns,
    time_scale_to_ps: float,
) -> list[BlockSupport]:
    rows, fields = _read_csv(path)
    required = {columns.case, columns.block, columns.start, columns.end, columns.event_count}
    if missing := required.difference(fields):
        raise ValueError(f"{path}: missing block columns {sorted(missing)}")
    blocks: dict[tuple[str, str], BlockSupport] = {}
    for row in rows:
        case_id, label = row[columns.case], row[columns.block]
        block = BlockSupport(
            case_id=case_id,
            label=label,
            start_ps=float(row[columns.start]) * time_scale_to_ps,
            end_ps=float(row[columns.end]) * time_scale_to_ps,
            source_event_count=int(row[columns.event_count]),
        )
        if (
            not case_id
            or not label
            or block.end_ps <= block.start_ps
            or block.source_event_count < 0
        ):
            raise ValueError(f"{path}: invalid support row {row}")
        key = (case_id, label)
        if key in blocks and blocks[key] != block:
            raise ValueError(f"{path}: inconsistent duplicate support for {key}")
        blocks[key] = block
    output = sorted(blocks.values(), key=lambda item: (item.case_id, item.start_ps, item.label))
    by_case: dict[str, list[BlockSupport]] = defaultdict(list)
    for block in output:
        by_case[block.case_id].append(block)
    for case_id, case_blocks in by_case.items():
        for previous, current in zip(case_blocks, case_blocks[1:]):
            if current.start_ps < previous.end_ps:
                raise ValueError(f"{path}: overlapping blocks for {case_id}")
    return output


def _read_events(
    path: Path,
    columns: EventColumns,
    time_scale_to_ps: float,
) -> tuple[list[str], dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    rows, fields = _read_csv(path)
    required = {columns.case, columns.time, columns.arc, columns.arc_count, columns.block}
    if missing := required.difference(fields):
        raise ValueError(f"{path}: missing event columns {sorted(missing)}")
    raw: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    arc_counts: dict[str, int] = {}
    case_order = []
    seen_cases, seen_events = set(), set()
    for row in rows:
        case_id, label = row[columns.case], row[columns.block]
        time_ps = float(row[columns.time]) * time_scale_to_ps
        arc = int(row[columns.arc])
        arc_count = int(row[columns.arc_count])
        if not case_id or not label or not math.isfinite(time_ps) or arc_count < 2:
            raise ValueError(f"{path}: invalid event row {row}")
        if case_id not in seen_cases:
            seen_cases.add(case_id)
            case_order.append(case_id)
        if case_id in arc_counts and arc_counts[case_id] != arc_count:
            raise ValueError(f"{path}: inconsistent arc count for {case_id}")
        if arc < 0 or arc >= arc_count:
            raise ValueError(f"{path}: invalid arc {arc} for {case_id}")
        identity = (case_id, label, time_ps, arc)
        if identity in seen_events:
            raise ValueError(f"{path}: duplicate case/block/time/arc event {identity}")
        seen_events.add(identity)
        arc_counts[case_id] = arc_count
        raw[(case_id, label)].append((time_ps, arc))
    events = {
        key: (
            np.asarray([item[0] for item in sorted(values)], dtype=float),
            np.asarray([item[1] for item in sorted(values)], dtype=int),
        )
        for key, values in raw.items()
    }
    return case_order, events, arc_counts


def _validate_inputs(
    case_order: Sequence[str],
    events: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    arc_counts: Mapping[str, int],
    blocks: Sequence[BlockSupport],
    distance_bins: Sequence[DistanceBin],
) -> None:
    block_by_key = {(block.case_id, block.label): block for block in blocks}
    block_cases = {block.case_id for block in blocks}
    if set(case_order) != block_cases:
        raise ValueError("event-table and block-table cases differ")
    final_label = {
        case_id: max(
            (block for block in blocks if block.case_id == case_id),
            key=lambda item: item.end_ps,
        ).label
        for case_id in case_order
    }
    for key, (times, _) in events.items():
        if key not in block_by_key:
            raise ValueError(f"event table references unknown block {key}")
        block = block_by_key[key]
        if np.any(times < block.start_ps) or np.any(times > block.end_ps):
            raise ValueError(f"events fall outside declared block support {key}")
        if key[1] != final_label[key[0]] and np.any(times >= block.end_ps):
            raise ValueError(f"non-final block contains right-boundary event {key}")
    expected_left = 1
    for item in distance_bins:
        if item.start != expected_left:
            raise ValueError("distance bins must be contiguous and start at one")
        expected_left = item.end
    for case_id, arc_count in arc_counts.items():
        if expected_left != arc_count // 2 + 1:
            raise ValueError(f"distance bins do not cover all positive distances for {case_id}")


def _pair_counts(
    times_ps: np.ndarray,
    arcs: np.ndarray,
    arc_count: int,
    distance_bins: Sequence[DistanceBin],
    windows: Sequence[LagWindow],
) -> np.ndarray:
    counts = np.zeros((len(distance_bins) + 1, len(windows)), dtype=float)
    if len(times_ps) < 2:
        return counts
    lag = times_ps[None, :] - times_ps[:, None]
    direct = np.abs(arcs[None, :] - arcs[:, None])
    distance = np.minimum(direct, arc_count - direct)
    categories = [(0, 1)] + [(item.start, item.end) for item in distance_bins]
    for category_index, (left, right) in enumerate(categories):
        selected_distance = (distance >= left) & (distance < right)
        for window_index, window in enumerate(windows):
            counts[category_index, window_index] = np.count_nonzero(
                selected_distance & (lag > window.start_ps) & (lag <= window.end_ps)
            )
    return counts


def _case_analysis(
    case_id: str,
    blocks: Sequence[BlockSupport],
    events: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    *,
    arc_count: int,
    distance_bins: Sequence[DistanceBin],
    windows: Sequence[LagWindow],
    null_samples: int,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(random_seed)
    categories = [DistanceBin("same_arc", 0, 1), *distance_bins]
    observed_blocks = np.zeros((len(blocks), len(categories), len(windows)), dtype=float)
    expected_blocks = np.zeros_like(observed_blocks)
    null_totals = np.zeros((null_samples, len(categories), len(windows)), dtype=float)
    for block_index, block in enumerate(blocks):
        times, arcs = events.get(
            (case_id, block.label),
            (np.asarray([], dtype=float), np.asarray([], dtype=int)),
        )
        observed_blocks[block_index] = _pair_counts(
            times, arcs, arc_count, distance_bins, windows
        )
        block_null = np.zeros((null_samples, len(categories), len(windows)), dtype=float)
        for sample in range(null_samples):
            shifted = _shift_per_arc(times, arcs, block.start_ps, block.end_ps - block.start_ps, rng)
            block_null[sample] = _pair_counts(
                shifted, arcs, arc_count, distance_bins, windows
            )
        expected_blocks[block_index] = np.mean(block_null, axis=0)
        null_totals += block_null

    observed = np.sum(observed_blocks, axis=0)
    null_mean = np.mean(null_totals, axis=0)
    bootstrap_indices = rng.integers(0, len(blocks), size=(bootstrap_samples, len(blocks)))
    bootstrap_observed = np.sum(observed_blocks[bootstrap_indices], axis=1)
    bootstrap_expected = np.sum(expected_blocks[bootstrap_indices], axis=1)
    bootstrap_chi = np.divide(
        bootstrap_observed,
        bootstrap_expected,
        out=np.full_like(bootstrap_observed, np.nan),
        where=bootstrap_expected > 0.0,
    ) - 1.0

    association_rows, block_rows, null_rows = [], [], []
    for category_index, category in enumerate(categories):
        for window_index, window in enumerate(windows):
            obs = float(observed[category_index, window_index])
            expected = float(null_mean[category_index, window_index])
            lower_p = float(
                (1 + np.count_nonzero(null_totals[:, category_index, window_index] <= obs))
                / (null_samples + 1)
            )
            upper_p = float(
                (1 + np.count_nonzero(null_totals[:, category_index, window_index] >= obs))
                / (null_samples + 1)
            )
            draws = bootstrap_chi[:, category_index, window_index]
            draws = draws[np.isfinite(draws)]
            association_rows.append(
                {
                    "case_id": case_id,
                    "distance_bin": category.name,
                    "distance_start_arc": category.start,
                    "distance_end_arc": category.end,
                    "window": window.name,
                    "lag_start_ps": window.start_ps,
                    "lag_end_ps": window.end_ps,
                    "cluster_event_count": sum(
                        len(events.get((case_id, block.label), ((), ()))[0]) for block in blocks
                    ),
                    "block_count": len(blocks),
                    "observed_ordered_pairs": int(obs),
                    "null_mean_ordered_pairs": expected,
                    "chi_signed": obs / expected - 1.0 if expected > 0.0 else math.nan,
                    "chi_block_bootstrap_ci025": (
                        float(np.quantile(draws, 0.025)) if len(draws) else math.nan
                    ),
                    "chi_block_bootstrap_ci975": (
                        float(np.quantile(draws, 0.975)) if len(draws) else math.nan
                    ),
                    "null_q025_ordered_pairs": float(
                        np.quantile(null_totals[:, category_index, window_index], 0.025)
                    ),
                    "null_q975_ordered_pairs": float(
                        np.quantile(null_totals[:, category_index, window_index], 0.975)
                    ),
                    "empirical_lower_p": lower_p,
                    "empirical_upper_p": upper_p,
                    "empirical_two_sided_p": min(1.0, 2.0 * min(lower_p, upper_p)),
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
            for block_index, block in enumerate(blocks):
                block_rows.append(
                    {
                        "case_id": case_id,
                        "block_index": block.label,
                        "block_start_ps": block.start_ps,
                        "block_end_ps": block.end_ps,
                        "source_event_row_count": block.source_event_count,
                        "cluster_event_count": len(
                            events.get((case_id, block.label), ((), ()))[0]
                        ),
                        "distance_bin": category.name,
                        "window": window.name,
                        "observed_ordered_pairs": int(
                            observed_blocks[block_index, category_index, window_index]
                        ),
                        "null_mean_ordered_pairs": float(
                            expected_blocks[block_index, category_index, window_index]
                        ),
                    }
                )
            null_rows.extend(
                {
                    "case_id": case_id,
                    "distance_bin": category.name,
                    "window": window.name,
                    "null_sample": sample,
                    "ordered_pairs": int(value),
                }
                for sample, value in enumerate(null_totals[:, category_index, window_index])
            )
    return association_rows, block_rows, null_rows


def run_analysis(
    event_table: Path,
    block_table: Path,
    output_dir: Path,
    *,
    event_columns: EventColumns,
    block_columns: BlockColumns,
    event_time_scale_to_ps: float,
    block_time_scale_to_ps: float,
    distance_bins: Sequence[DistanceBin],
    windows: Sequence[LagWindow],
    null_samples: int,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, object]:
    """Run signed observed-event analysis and write auditable tables."""

    if min(event_time_scale_to_ps, block_time_scale_to_ps) <= 0.0:
        raise ValueError("time scales must be positive")
    if null_samples < 20 or bootstrap_samples < 20:
        raise ValueError("null and bootstrap sample counts must each be at least 20")
    case_order, events, arc_counts = _read_events(
        event_table, event_columns, event_time_scale_to_ps
    )
    blocks = _read_blocks(block_table, block_columns, block_time_scale_to_ps)
    _validate_inputs(case_order, events, arc_counts, blocks, distance_bins)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    all_rows, block_rows, null_rows = [], [], []
    for case_index, case_id in enumerate(case_order):
        case_blocks = [block for block in blocks if block.case_id == case_id]
        rows, case_block_rows, case_null_rows = _case_analysis(
            case_id,
            case_blocks,
            events,
            arc_count=arc_counts[case_id],
            distance_bins=distance_bins,
            windows=windows,
            null_samples=null_samples,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + case_index,
        )
        all_rows.extend(rows)
        block_rows.extend(case_block_rows)
        null_rows.extend(case_null_rows)

    primary_rows = [row for row in all_rows if row["distance_bin"] != "same_arc"]
    same_arc_rows = [row for row in all_rows if row["distance_bin"] == "same_arc"]
    q_values = _bh_adjust([float(row["empirical_two_sided_p"]) for row in primary_rows])
    for row, q_value in zip(primary_rows, q_values):
        chi = float(row["chi_signed"])
        qualified = math.isfinite(chi) and chi != 0.0 and q_value <= 0.05
        row["bh_q_primary_family"] = q_value
        row["qualified_primary_effect"] = qualified
        row["association_class"] = (
            "cooperative_event_association"
            if qualified and chi > 0.0
            else "refractory_event_deficit"
            if qualified
            else "not_qualified"
        )
    for row in same_arc_rows:
        row["diagnostic_only"] = True

    _write_csv(output / "signed_primary_associations.csv", primary_rows)
    _write_csv(output / "same_arc_dead_time_diagnostics.csv", same_arc_rows)
    _write_csv(output / "block_metrics.csv", block_rows)
    _write_csv(output / "null_totals.csv", null_rows)
    case_summaries = [
        {
            "case_id": case_id,
            "arc_count": arc_counts[case_id],
            "event_count": sum(
                len(values[0]) for (event_case, _), values in events.items() if event_case == case_id
            ),
            "source_event_row_count": sum(
                block.source_event_count for block in blocks if block.case_id == case_id
            ),
            "block_count": sum(block.case_id == case_id for block in blocks),
        }
        for case_id in case_order
    ]
    summary = {
        "status": "PASS",
        "case_count": len(case_order),
        "case_summaries": case_summaries,
        "event_count": sum(len(values[0]) for values in events.values()),
        "source_event_row_count": sum(block.source_event_count for block in blocks),
        "block_count": len(blocks),
        "primary_estimand_count": len(primary_rows),
        "qualified_cooperative_count": sum(
            row["association_class"] == "cooperative_event_association" for row in primary_rows
        ),
        "qualified_refractory_count": sum(
            row["association_class"] == "refractory_event_deficit" for row in primary_rows
        ),
        "same_arc_diagnostic_count": len(same_arc_rows),
        "null_samples": null_samples,
        "bootstrap_samples": bootstrap_samples,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        **summary,
        "event_table": {"path": str(event_table), "sha256": _sha256(event_table)},
        "block_table": {"path": str(block_table), "sha256": _sha256(block_table)},
        "event_columns": asdict(event_columns),
        "block_columns": asdict(block_columns),
        "event_time_scale_to_ps": event_time_scale_to_ps,
        "block_time_scale_to_ps": block_time_scale_to_ps,
        "distance_bins": [asdict(item) for item in distance_bins],
        "windows": [asdict(item) for item in windows],
        "random_seed": random_seed,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "# Signed circular event associations\n\n"
        "Observed ordered-pair counts are compared with independent per-site circular "
        "time shifts inside accepted blocks. Negative contrasts are retained. Same-site "
        "results are dead-time diagnostics and are excluded from multiplicity control. "
        "Block-table source event-row counts and analyzed cluster-event counts are retained "
        "separately. "
        "Whole-block bootstrap intervals are within-trajectory diagnostics. These outputs "
        "do not establish causality, propagation speed, intrinsic length, physical event "
        "rates, or replicate-level uncertainty.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-table", type=Path, required=True)
    parser.add_argument("--block-table", type=Path, required=True)
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
    parser.add_argument("--null-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260904)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(
        args.event_table,
        args.block_table,
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
        null_samples=args.null_samples,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
