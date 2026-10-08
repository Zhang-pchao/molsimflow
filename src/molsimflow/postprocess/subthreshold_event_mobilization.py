"""Relate continuous site mobilization to later discrete event formation.

The analysis joins an accepted event-aligned field table to accepted cluster
events.  A localization-noise threshold normalizes the short-lag field change;
it is not interpreted as a thermodynamic or mechanical yield threshold.
Circular shifts preserve each site's within-block event train and provide a
descriptive background conversion rate.  Results remain retrospective,
single-trajectory associations rather than causal branching probabilities.
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

from molsimflow.postprocess.circular_event_association import (
    BlockColumns,
    BlockSupport,
    DistanceBin,
    _read_blocks,
    parse_distance_bins,
)
from molsimflow.postprocess.event_aligned_circular_field import signed_arc_offsets
from molsimflow.postprocess.tpcl_dynamic_state import (
    LagWindow,
    _sha256,
    _shift_per_arc,
    _write_csv,
    parse_windows,
)

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_CONTINUOUS_MOBILIZATION_TO_DISCRETE_EVENT_ASSOCIATION_"
    "NOT_A_PHYSICAL_YIELD_THRESHOLD_CAUSAL_BRANCHING_OR_REPLICATE_EVIDENCE"
)


@dataclass(frozen=True)
class RatioBin:
    """Half-open bin for an absolute response/noise-threshold ratio."""

    name: str
    start: float
    end: float


@dataclass(frozen=True)
class ClusterEvent:
    case_id: str
    event_id: int
    time_ps: float
    primary_arc: int
    member_arcs: tuple[int, ...]
    threshold_A: float
    block: str = ""


def parse_ratio_bins(raw: str) -> tuple[RatioBin, ...]:
    """Parse ``name:start:end`` ratio bins, allowing ``inf`` as the last end."""

    bins = []
    for item in raw.split(","):
        fields = item.split(":")
        if len(fields) != 3 or not fields[0]:
            raise ValueError(f"invalid ratio bin {item!r}")
        ratio_bin = RatioBin(fields[0], float(fields[1]), float(fields[2]))
        if ratio_bin.start < 0.0 or ratio_bin.end <= ratio_bin.start:
            raise ValueError(f"invalid ratio bin {item!r}")
        bins.append(ratio_bin)
    if not bins or len({item.name for item in bins}) != len(bins):
        raise ValueError("ratio bins must be nonempty and uniquely named")
    ordered = sorted(bins, key=lambda item: item.start)
    if ordered[0].start != 0.0 or any(
        not math.isclose(left.end, right.start) for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("ratio bins must be contiguous and start at zero")
    if not math.isinf(ordered[-1].end):
        raise ValueError("the last ratio bin must end at inf")
    return tuple(ordered)


def parse_cluster_sources(raw: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``CASE=PATH`` cluster-table specifications."""

    sources: dict[str, Path] = {}
    for item in raw:
        case_id, separator, path = item.partition("=")
        if not separator or not case_id or not path or case_id in sources:
            raise ValueError(f"invalid or duplicate cluster source {item!r}")
        sources[case_id] = Path(path)
    if not sources:
        raise ValueError("at least one cluster source is required")
    return sources


def _read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows, set(reader.fieldnames)


def _load_clusters(sources: Mapping[str, Path]) -> dict[tuple[str, int], ClusterEvent]:
    required = {
        "primary_event_id",
        "transition_time_ns",
        "primary_arc_index",
        "member_arcs",
        "affected_threshold_A",
    }
    output: dict[tuple[str, int], ClusterEvent] = {}
    for case_id, path in sources.items():
        rows, fields = _read_csv(path)
        if missing := required.difference(fields):
            raise ValueError(f"{path}: missing cluster columns {sorted(missing)}")
        for row in rows:
            event_id = int(row["primary_event_id"])
            members = tuple(sorted({int(value) for value in row["member_arcs"].split(";")}))
            event = ClusterEvent(
                case_id=case_id,
                event_id=event_id,
                time_ps=float(row["transition_time_ns"]) * 1000.0,
                primary_arc=int(row["primary_arc_index"]),
                member_arcs=members,
                threshold_A=float(row["affected_threshold_A"]),
            )
            if (
                not members
                or event.primary_arc not in members
                or not math.isfinite(event.time_ps)
                or not math.isfinite(event.threshold_A)
                or event.threshold_A <= 0.0
            ):
                raise ValueError(f"{path}: invalid cluster row {row}")
            key = (case_id, event_id)
            if key in output:
                raise ValueError(f"{path}: duplicate primary event {key}")
            output[key] = event
    return output


def _assign_blocks(
    clusters: Mapping[tuple[str, int], ClusterEvent], blocks: Sequence[BlockSupport]
) -> dict[tuple[str, int], ClusterEvent]:
    by_case: dict[str, list[BlockSupport]] = defaultdict(list)
    for block in blocks:
        by_case[block.case_id].append(block)
    output = {}
    for key, event in clusters.items():
        case_blocks = sorted(by_case.get(event.case_id, ()), key=lambda item: item.start_ps)
        selected = [
            block
            for index, block in enumerate(case_blocks)
            if event.time_ps >= block.start_ps
            and (
                event.time_ps < block.end_ps
                or (index == len(case_blocks) - 1 and event.time_ps <= block.end_ps)
            )
        ]
        if len(selected) != 1:
            raise ValueError(f"cluster event {key} does not map to exactly one block")
        output[key] = ClusterEvent(**{**asdict(event), "block": selected[0].label})
    return output


def _distance_label(distance: int, bins: Sequence[DistanceBin]) -> str:
    selected = [item.name for item in bins if item.start <= distance < item.end]
    if len(selected) != 1:
        raise ValueError(f"distance {distance} does not map to exactly one bin")
    return selected[0]


def _ratio_label(value: float, bins: Sequence[RatioBin]) -> str:
    selected = [item.name for item in bins if item.start <= value < item.end]
    if len(selected) != 1:
        raise ValueError(f"ratio {value} does not map to exactly one bin")
    return selected[0]


def _load_opportunities(
    response_table: Path,
    clusters: Mapping[tuple[str, int], ClusterEvent],
    blocks: Sequence[BlockSupport],
    *,
    response_field: str,
    response_lag_ps: float,
    distance_bins: Sequence[DistanceBin],
    ratio_bins: Sequence[RatioBin],
    mobilization_threshold_ratio: float,
) -> tuple[list[dict[str, object]], dict[str, int], int]:
    rows, fields = _read_csv(response_table)
    required = {
        "case_id",
        "event_id",
        "time_block_200ps",
        "primary_arc_index",
        "field",
        "lag_ps",
        "arc_offset_signed",
        "arc_distance",
        "aligned_change",
    }
    if missing := required.difference(fields):
        raise ValueError(f"{response_table}: missing response columns {sorted(missing)}")
    selected = [
        row
        for row in rows
        if row["field"] == response_field
        and math.isclose(float(row["lag_ps"]), response_lag_ps, abs_tol=1.0e-10)
    ]
    if not selected:
        raise ValueError("no rows survive the response field/lag selection")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[(row["case_id"], int(row["event_id"]))].append(row)
    block_keys = {(block.case_id, block.label) for block in blocks}
    opportunities, arc_counts, excluded_members = [], {}, 0
    for key, event_rows in sorted(grouped.items()):
        if key not in clusters:
            raise ValueError(f"response event {key} is absent from cluster sources")
        event = clusters[key]
        offsets = sorted(int(row["arc_offset_signed"]) for row in event_rows)
        arc_count = len(offsets)
        expected = signed_arc_offsets(arc_count).tolist()
        if offsets != expected or arc_count < 4:
            raise ValueError(f"response event {key} does not contain one complete circular field")
        if key[0] in arc_counts and arc_counts[key[0]] != arc_count:
            raise ValueError(f"inconsistent arc count for {key[0]}")
        arc_counts[key[0]] = arc_count
        block_labels = {row["time_block_200ps"] for row in event_rows}
        primary_arcs = {int(row["primary_arc_index"]) for row in event_rows}
        if block_labels != {event.block} or primary_arcs != {event.primary_arc}:
            raise ValueError(f"response/cluster anchor mismatch for {key}")
        if (event.case_id, event.block) not in block_keys:
            raise ValueError(f"response event {key} references unknown block")
        seen_offsets = set()
        for row in event_rows:
            offset = int(row["arc_offset_signed"])
            distance = int(row["arc_distance"])
            change = float(row["aligned_change"])
            if offset in seen_offsets or distance != abs(offset) or not math.isfinite(change):
                raise ValueError(f"invalid or duplicate response cell for {key}")
            seen_offsets.add(offset)
            target_arc = (event.primary_arc + offset) % arc_count
            if offset == 0 or target_arc in event.member_arcs:
                excluded_members += int(offset != 0)
                continue
            ratio = abs(change) / event.threshold_A
            opportunities.append(
                {
                    "case_id": event.case_id,
                    "primary_event_id": event.event_id,
                    "time_block_200ps": event.block,
                    "transition_time_ps": event.time_ps,
                    "primary_arc_index": event.primary_arc,
                    "target_arc_index": target_arc,
                    "arc_offset_signed": offset,
                    "arc_distance": distance,
                    "distance_bin": _distance_label(distance, distance_bins),
                    "response_field": response_field,
                    "response_lag_ps": response_lag_ps,
                    "aligned_change_A": change,
                    "absolute_change_A": abs(change),
                    "localization_noise_threshold_A": event.threshold_A,
                    "mobilization_ratio": ratio,
                    "margin_band": _ratio_label(ratio, ratio_bins),
                    "mobilized": ratio >= mobilization_threshold_ratio,
                }
            )
    if not opportunities:
        raise ValueError("no cross-arc opportunities survive primary-cluster exclusion")
    for case_id, arc_count in arc_counts.items():
        expected_left = 1
        for item in distance_bins:
            if item.start != expected_left:
                raise ValueError("distance bins must be contiguous and start at one")
            expected_left = item.end
        if expected_left != arc_count // 2 + 1:
            raise ValueError(f"distance bins do not cover all positive distances for {case_id}")
        for event in (item for item in clusters.values() if item.case_id == case_id):
            if any(arc < 0 or arc >= arc_count for arc in event.member_arcs):
                raise ValueError(f"cluster event {(case_id, event.event_id)} has an invalid member arc")
    return opportunities, arc_counts, excluded_members


def _cluster_arc_index(
    clusters: Mapping[tuple[str, int], ClusterEvent]
) -> dict[tuple[str, str, int], list[tuple[float, int]]]:
    output: dict[tuple[str, str, int], list[tuple[float, int]]] = defaultdict(list)
    for event in clusters.values():
        for arc in event.member_arcs:
            output[(event.case_id, event.block, arc)].append((event.time_ps, event.event_id))
    for values in output.values():
        values.sort()
    return dict(output)


def _build_outcomes(
    opportunities: Sequence[Mapping[str, object]],
    clusters: Mapping[tuple[str, int], ClusterEvent],
    windows: Sequence[LagWindow],
) -> list[dict[str, object]]:
    index = _cluster_arc_index(clusters)
    rows = []
    for opportunity in opportunities:
        events = index.get(
            (
                str(opportunity["case_id"]),
                str(opportunity["time_block_200ps"]),
                int(opportunity["target_arc_index"]),
            ),
            (),
        )
        for window in windows:
            matches = [
                (time_ps - float(opportunity["transition_time_ps"]), event_id)
                for time_ps, event_id in events
                if event_id != int(opportunity["primary_event_id"])
                and time_ps - float(opportunity["transition_time_ps"]) > window.start_ps
                and time_ps - float(opportunity["transition_time_ps"]) <= window.end_ps
            ]
            rows.append(
                {
                    **opportunity,
                    "window": window.name,
                    "lag_start_ps": window.start_ps,
                    "lag_end_ps": window.end_ps,
                    "secondary_event_detected": bool(matches),
                    "secondary_event_count": len(matches),
                    "first_secondary_lag_ps": min(matches)[0] if matches else math.nan,
                    "first_secondary_event_id": min(matches)[1] if matches else "",
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
    return rows


def _quantile_interval(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return math.nan, math.nan
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def _bootstrap_group(
    rows: Sequence[Mapping[str, object]],
    case_blocks: Sequence[str],
    *,
    numerator: str,
    value: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float]]:
    by_block: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_block[str(row["time_block_200ps"])].append(row)
    fraction_draws, mean_draws = [], []
    for _ in range(samples):
        chosen = rng.choice(case_blocks, size=len(case_blocks), replace=True)
        selected = [row for block in chosen for row in by_block.get(str(block), ())]
        if not selected:
            continue
        fraction_draws.append(
            sum(bool(row[numerator]) for row in selected) / len(selected)
        )
        mean_draws.append(sum(float(row[value]) for row in selected) / len(selected))
    return _quantile_interval(fraction_draws), _quantile_interval(mean_draws)


def _descriptive_summaries(
    opportunities: Sequence[Mapping[str, object]],
    outcomes: Sequence[Mapping[str, object]],
    blocks: Sequence[BlockSupport],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blocks_by_case: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        blocks_by_case[block.case_id].append(block.label)
    continuous_groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in opportunities:
        continuous_groups[(str(row["case_id"]), str(row["distance_bin"]))].append(row)
    continuous_rows = []
    for (case_id, distance), rows in sorted(continuous_groups.items()):
        mobilized_ci, ratio_mean_ci = _bootstrap_group(
            rows,
            blocks_by_case[case_id],
            numerator="mobilized",
            value="mobilization_ratio",
            samples=bootstrap_samples,
            rng=rng,
        )
        ratios = np.asarray([float(row["mobilization_ratio"]) for row in rows])
        continuous_rows.append(
            {
                "case_id": case_id,
                "distance_bin": distance,
                "primary_event_count": len({int(row["primary_event_id"]) for row in rows}),
                "opportunity_count": len(rows),
                "mobilized_count": sum(bool(row["mobilized"]) for row in rows),
                "mobilized_fraction": float(np.mean([bool(row["mobilized"]) for row in rows])),
                "mobilized_fraction_block_bootstrap_ci025": mobilized_ci[0],
                "mobilized_fraction_block_bootstrap_ci975": mobilized_ci[1],
                "mean_mobilization_ratio": float(np.mean(ratios)),
                "mean_ratio_block_bootstrap_ci025": ratio_mean_ci[0],
                "mean_ratio_block_bootstrap_ci975": ratio_mean_ci[1],
                "median_mobilization_ratio": float(np.median(ratios)),
                "q900_mobilization_ratio": float(np.quantile(ratios, 0.9)),
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )

    margin_groups: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in outcomes:
        margin_groups[
            (
                str(row["case_id"]),
                str(row["distance_bin"]),
                str(row["margin_band"]),
                str(row["window"]),
            )
        ].append(row)
    margin_rows = []
    for (case_id, distance, margin, window), rows in sorted(margin_groups.items()):
        event_ci, ratio_ci = _bootstrap_group(
            rows,
            blocks_by_case[case_id],
            numerator="secondary_event_detected",
            value="mobilization_ratio",
            samples=bootstrap_samples,
            rng=rng,
        )
        margin_rows.append(
            {
                "case_id": case_id,
                "distance_bin": distance,
                "margin_band": margin,
                "window": window,
                "opportunity_count": len(rows),
                "secondary_event_count": sum(bool(row["secondary_event_detected"]) for row in rows),
                "secondary_event_fraction": float(
                    np.mean([bool(row["secondary_event_detected"]) for row in rows])
                ),
                "secondary_fraction_block_bootstrap_ci025": event_ci[0],
                "secondary_fraction_block_bootstrap_ci975": event_ci[1],
                "mean_mobilization_ratio": float(
                    np.mean([float(row["mobilization_ratio"]) for row in rows])
                ),
                "mean_ratio_block_bootstrap_ci025": ratio_ci[0],
                "mean_ratio_block_bootstrap_ci975": ratio_ci[1],
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    return continuous_rows, margin_rows


def _null_conversion_summary(
    opportunities: Sequence[Mapping[str, object]],
    outcomes: Sequence[Mapping[str, object]],
    clusters: Mapping[tuple[str, int], ClusterEvent],
    blocks: Sequence[BlockSupport],
    windows: Sequence[LagWindow],
    *,
    null_samples: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    def class_for(row: Mapping[str, object]) -> str:
        return "at_or_above_threshold" if bool(row["mobilized"]) else "below_threshold"

    group_keys = sorted(
        {
            (
                str(row["case_id"]),
                str(row["distance_bin"]),
                str(row["window"]),
                class_for(row),
            )
            for row in outcomes
        }
    )
    group_index = {key: index for index, key in enumerate(group_keys)}
    block_keys = [(block.case_id, block.label) for block in blocks]
    block_index = {key: index for index, key in enumerate(block_keys)}
    opportunities_by_block: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in opportunities:
        opportunities_by_block[(str(row["case_id"]), str(row["time_block_200ps"]))].append(row)
    events_by_block: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for event in clusters.values():
        for arc in event.member_arcs:
            events_by_block[(event.case_id, event.block)].append((event.time_ps, arc))
    null_totals = np.zeros((null_samples, len(group_keys)), dtype=float)
    expected_by_block = np.zeros((len(blocks), len(group_keys)), dtype=float)
    for sample in range(null_samples):
        for block in blocks:
            key = (block.case_id, block.label)
            event_pairs = events_by_block.get(key, ())
            times = np.asarray([item[0] for item in event_pairs], dtype=float)
            arcs = np.asarray([item[1] for item in event_pairs], dtype=int)
            shifted = _shift_per_arc(
                times, arcs, block.start_ps, block.end_ps - block.start_ps, rng
            )
            shifted_by_arc = {
                int(arc): np.sort(shifted[arcs == arc]) for arc in np.unique(arcs)
            }
            block_counts = np.zeros(len(group_keys), dtype=float)
            for opportunity in opportunities_by_block.get(key, ()):
                target_times = shifted_by_arc.get(
                    int(opportunity["target_arc_index"]), np.asarray([], dtype=float)
                )
                primary_time = float(opportunity["transition_time_ps"])
                for window in windows:
                    position = np.searchsorted(
                        target_times, primary_time + window.start_ps, side="right"
                    )
                    hit = bool(
                        position < len(target_times)
                        and target_times[position] <= primary_time + window.end_ps
                    )
                    if not hit:
                        continue
                    group = (
                        str(opportunity["case_id"]),
                        str(opportunity["distance_bin"]),
                        window.name,
                        class_for(opportunity),
                    )
                    index = group_index[group]
                    null_totals[sample, index] += 1.0
                    block_counts[index] += 1.0
            expected_by_block[block_index[key]] += block_counts
    expected_by_block /= null_samples

    outcome_groups: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in outcomes:
        outcome_groups[
            (
                str(row["case_id"]),
                str(row["distance_bin"]),
                str(row["window"]),
                class_for(row),
            )
        ].append(row)
    summary_rows = []
    for group in group_keys:
        group_id = group_index[group]
        case_id, distance, window, mobilization_class = group
        rows = outcome_groups[group]
        observed_by_block = np.asarray(
            [
                sum(
                    bool(row["secondary_event_detected"])
                    for row in rows
                    if str(row["time_block_200ps"]) == block.label
                )
                if block.case_id == case_id
                else 0
                for block in blocks
            ],
            dtype=float,
        )
        denominator_by_block = np.asarray(
            [
                sum(str(row["time_block_200ps"]) == block.label for row in rows)
                if block.case_id == case_id
                else 0
                for block in blocks
            ],
            dtype=float,
        )
        expected = expected_by_block[:, group_id]
        selected_blocks = np.asarray(
            [index for index, block in enumerate(blocks) if block.case_id == case_id], dtype=int
        )
        difference_draws = []
        for _ in range(bootstrap_samples):
            chosen = rng.choice(selected_blocks, size=len(selected_blocks), replace=True)
            denominator = float(np.sum(denominator_by_block[chosen]))
            if denominator > 0.0:
                difference_draws.append(
                    float(np.sum(observed_by_block[chosen] - expected[chosen])) / denominator
                )
        denominator = len(rows)
        observed_count = sum(bool(row["secondary_event_detected"]) for row in rows)
        null_fractions = null_totals[:, group_id] / denominator
        expected_count = float(np.sum(expected))
        difference_ci = _quantile_interval(difference_draws)
        summary_rows.append(
            {
                "case_id": case_id,
                "distance_bin": distance,
                "window": window,
                "mobilization_class": mobilization_class,
                "opportunity_count": denominator,
                "observed_secondary_count": observed_count,
                "observed_secondary_fraction": observed_count / denominator,
                "null_mean_secondary_count": expected_count,
                "null_mean_secondary_fraction": expected_count / denominator,
                "null_q025_secondary_fraction": float(np.quantile(null_fractions, 0.025)),
                "null_q975_secondary_fraction": float(np.quantile(null_fractions, 0.975)),
                "observed_minus_null_fraction": (observed_count - expected_count) / denominator,
                "difference_block_bootstrap_ci025": difference_ci[0],
                "difference_block_bootstrap_ci975": difference_ci[1],
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    return summary_rows


def run_analysis(
    response_table: Path,
    cluster_sources: Mapping[str, Path],
    block_table: Path,
    output_dir: Path,
    *,
    response_field: str,
    response_lag_ps: float,
    distance_bins: Sequence[DistanceBin],
    ratio_bins: Sequence[RatioBin],
    windows: Sequence[LagWindow],
    mobilization_threshold_ratio: float,
    null_samples: int,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, object]:
    """Run continuous-mobilization versus secondary-event analysis."""

    if (
        response_lag_ps <= 0.0
        or mobilization_threshold_ratio <= 0.0
        or null_samples < 20
        or bootstrap_samples < 20
    ):
        raise ValueError("invalid response lag, threshold, or resampling count")
    blocks = _read_blocks(Path(block_table), BlockColumns(), 1.0)
    clusters = _assign_blocks(_load_clusters(cluster_sources), blocks)
    opportunities, arc_counts, excluded_members = _load_opportunities(
        Path(response_table),
        clusters,
        blocks,
        response_field=response_field,
        response_lag_ps=response_lag_ps,
        distance_bins=distance_bins,
        ratio_bins=ratio_bins,
        mobilization_threshold_ratio=mobilization_threshold_ratio,
    )
    if set(arc_counts) != set(cluster_sources) or set(arc_counts) != {
        block.case_id for block in blocks
    }:
        raise ValueError("response, cluster, and block-table cases differ")
    outcomes = _build_outcomes(opportunities, clusters, windows)
    rng = np.random.default_rng(random_seed)
    continuous_rows, margin_rows = _descriptive_summaries(
        opportunities, outcomes, blocks, bootstrap_samples=bootstrap_samples, rng=rng
    )
    conversion_rows = _null_conversion_summary(
        opportunities,
        outcomes,
        clusters,
        blocks,
        windows,
        null_samples=null_samples,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "event_arc_outcomes.csv", outcomes)
    _write_csv(output_dir / "continuous_mobilization_summary.csv", continuous_rows)
    _write_csv(output_dir / "margin_band_summary.csv", margin_rows)
    _write_csv(output_dir / "mobilization_conversion_summary.csv", conversion_rows)
    summary = {
        "status": "PASS",
        "case_count": len(arc_counts),
        "block_count": len(blocks),
        "cluster_event_count": len(clusters),
        "admitted_primary_event_count": len(
            {(row["case_id"], row["primary_event_id"]) for row in opportunities}
        ),
        "cross_arc_opportunity_count": len(opportunities),
        "outcome_row_count": len(outcomes),
        "excluded_primary_cluster_member_arc_count": excluded_members,
        "continuous_summary_row_count": len(continuous_rows),
        "margin_summary_row_count": len(margin_rows),
        "conversion_summary_row_count": len(conversion_rows),
        "null_samples": null_samples,
        "bootstrap_samples": bootstrap_samples,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    manifest = {
        **summary,
        "response_table": {"path": str(response_table), "sha256": _sha256(response_table)},
        "cluster_sources": [
            {"case_id": case_id, "path": str(path), "sha256": _sha256(path)}
            for case_id, path in cluster_sources.items()
        ],
        "block_table": {"path": str(block_table), "sha256": _sha256(block_table)},
        "response_field": response_field,
        "response_lag_ps": response_lag_ps,
        "mobilization_threshold_ratio": mobilization_threshold_ratio,
        "distance_bins": [asdict(item) for item in distance_bins],
        "ratio_bins": [
            {
                "name": item.name,
                "start": item.start,
                "end": item.end if math.isfinite(item.end) else None,
            }
            for item in ratio_bins
        ],
        "windows": [asdict(item) for item in windows],
        "random_seed": random_seed,
        "threshold_interpretation": (
            "affected_threshold_A is a localization-noise threshold used only for "
            "dimensionless response normalization; it is not a physical yield threshold"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-table", type=Path, required=True)
    parser.add_argument("--cluster-source", action="append", required=True, metavar="CASE=PATH")
    parser.add_argument("--block-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-field", default="local_residual_A")
    parser.add_argument("--response-lag-ps", type=float, default=0.5)
    parser.add_argument("--distance-bins", default="near:1:4,mid:4:10,far:10:19")
    parser.add_argument(
        "--ratio-bins", default="low:0:0.5,near_threshold:0.5:1,mobilized:1:2,strong:2:inf"
    )
    parser.add_argument("--windows", default="fast:0:5,slow:5:50")
    parser.add_argument("--mobilization-threshold-ratio", type=float, default=1.0)
    parser.add_argument("--null-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260905)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        args.response_table,
        parse_cluster_sources(args.cluster_source),
        args.block_table,
        args.output_dir,
        response_field=args.response_field,
        response_lag_ps=args.response_lag_ps,
        distance_bins=parse_distance_bins(args.distance_bins),
        ratio_bins=parse_ratio_bins(args.ratio_bins),
        windows=parse_windows(args.windows),
        mobilization_threshold_ratio=args.mobilization_threshold_ratio,
        null_samples=args.null_samples,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
