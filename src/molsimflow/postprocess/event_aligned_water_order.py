"""Event-aligned TPCL water order with three pre-registered control families.

The workflow joins an operational dwell-jump catalog to arc-resolved water
order. Same-arc matched dwells, per-arc block circular shifts, and event-time
permutations are conditional single-trajectory controls, not independent
replicates or causal interventions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVENT_STATUS = "operational_local_dwell_jump_candidate"
SCIENTIFIC_STATUS = (
    "SINGLE_TRAJECTORY_EVENT_ALIGNED_WATER_ORDER_WITH_MATCHED_AND_RANDOMIZATION_CONTROLS"
    "_NOT_CAUSAL_FREE_ENERGY_OR_PHYSICAL_RATE_EVIDENCE"
)
DEFAULT_METRICS = (
    "water_count",
    "mean_q_tet",
    "mean_lsi_A2",
    "mean_oo_coordination",
    "oo_coordination_nonfour_fraction",
    "h_coordination_defect_fraction",
    "mean_hbond_degree",
    "mean_hbond_internal_tpcl_degree",
)


@dataclass(frozen=True)
class Event:
    event_id: int
    arc_index: int
    transition_step: int
    pre_local_radius_A: float
    dwell_tolerance_A: float


@dataclass(frozen=True)
class MatchedDwell:
    event_id: int
    control_id: int
    arc_index: int
    anchor_step: int
    match_score: float


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if Path(path).suffix == ".gz":
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = tuple(reader.fieldnames or [])
        return rows, fields
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or [])
    return rows, fields


def _require(path: Path, fields: Sequence[str], required: set[str]) -> None:
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _float(value: object) -> float:
    return float(value)


def load_arc_grid(
    water_order_path: Path,
    arc_kinematics_path: Path,
    metrics: Sequence[str],
) -> tuple[list[int], list[int], dict[tuple[int, int], dict[str, float]]]:
    """Load and validate complete `(step, arc)` water-order and contour fields."""

    water_rows, water_fields = _read_csv(water_order_path)
    required = {"step", "arc_index", *metrics}
    _require(water_order_path, water_fields, required)
    track_rows, track_fields = _read_csv(arc_kinematics_path)
    _require(
        arc_kinematics_path,
        track_fields,
        {"step", "arc_index", "local_radius_A", "local_residual_A"},
    )
    tracks: dict[tuple[int, int], dict[str, float]] = {}
    for row in track_rows:
        key = (round(_float(row["step"])), round(_float(row["arc_index"])))
        if key in tracks:
            raise ValueError(f"{arc_kinematics_path}: duplicate key {key}")
        tracks[key] = {
            "local_radius_A": _float(row["local_radius_A"]),
            "local_residual_A": _float(row["local_residual_A"]),
        }
    grid: dict[tuple[int, int], dict[str, float]] = {}
    steps: set[int] = set()
    arcs: set[int] = set()
    for row in water_rows:
        step, arc = round(_float(row["step"])), round(_float(row["arc_index"]))
        key = (step, arc)
        if key in grid:
            raise ValueError(f"{water_order_path}: duplicate key {key}")
        if key not in tracks:
            raise ValueError(f"{water_order_path}: no arc kinematics for {key}")
        grid[key] = {
            **{name: _float(row[name]) for name in metrics},
            **tracks[key],
        }
        steps.add(step)
        arcs.add(arc)
    ordered_steps, ordered_arcs = sorted(steps), sorted(arcs)
    if len(ordered_steps) < 3 or ordered_arcs != list(range(len(ordered_arcs))):
        raise ValueError("water-order grid has insufficient steps or noncontiguous arcs")
    stride = ordered_steps[1] - ordered_steps[0]
    if stride <= 0 or ordered_steps != list(
        range(ordered_steps[0], ordered_steps[-1] + stride, stride)
    ):
        raise ValueError("water-order steps are not a complete regular grid")
    expected = len(ordered_steps) * len(ordered_arcs)
    if len(grid) != expected:
        raise ValueError(f"incomplete water-order grid: {len(grid)} != {expected}")
    return ordered_steps, ordered_arcs, grid


def load_events(path: Path, event_status: str) -> list[Event]:
    rows, fields = _read_csv(path)
    _require(
        path,
        fields,
        {
            "event_id",
            "arc_index",
            "transition_step",
            "quality_status",
            "pre_local_radius_A",
            "dwell_tolerance_A",
        },
    )
    events = [
        Event(
            event_id=round(_float(row["event_id"])),
            arc_index=round(_float(row["arc_index"])),
            transition_step=round(_float(row["transition_step"])),
            pre_local_radius_A=_float(row["pre_local_radius_A"]),
            dwell_tolerance_A=_float(row["dwell_tolerance_A"]),
        )
        for row in rows
        if row["quality_status"] == event_status
    ]
    events.sort(key=lambda event: (event.transition_step, event.arc_index, event.event_id))
    if len({event.event_id for event in events}) != len(events):
        raise ValueError(f"{path}: duplicate admitted event_id")
    if not events:
        raise ValueError(f"{path}: no events admitted with status {event_status!r}")
    return events


def load_event_windows(
    path: Path,
    events: Sequence[Event],
) -> tuple[dict[int, list[tuple[int, int, str]]], list[dict[str, object]]]:
    rows, fields = _read_csv(path)
    _require(path, fields, {"event_id", "arc_index", "step", "relative_frame", "phase"})
    event_by_id = {event.event_id: event for event in events}
    windows: dict[int, list[tuple[int, int, str]]] = {}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        event_id = round(_float(row["event_id"]))
        if event_id not in event_by_id:
            continue
        relative = round(_float(row["relative_frame"]))
        key = (event_id, relative)
        if key in seen:
            raise ValueError(f"{path}: duplicate event-relative row {key}")
        seen.add(key)
        if round(_float(row["arc_index"])) != event_by_id[event_id].arc_index:
            raise ValueError(f"{path}: arc mismatch for event {event_id}")
        phase = _phase(relative)
        if str(row["phase"]) != phase:
            raise ValueError(f"{path}: inconsistent phase for event {event_id}")
        windows.setdefault(event_id, []).append(
            (round(_float(row["step"])), relative, phase)
        )
    signatures: Counter[tuple[int, ...]] = Counter()
    for event_id in event_by_id:
        values = windows.get(event_id, [])
        values.sort(key=lambda item: item[1])
        signatures[tuple(item[1] for item in values)] += 1
    most_common = signatures.most_common()
    if not most_common or (len(most_common) > 1 and most_common[0][1] == most_common[1][1]):
        raise ValueError("event-window support has no unique modal offset signature")
    expected_offsets = most_common[0][0]
    maximum = max(expected_offsets, default=0)
    if expected_offsets != tuple(range(-maximum, maximum + 1)) or maximum < 1:
        raise ValueError("modal event window must be complete, symmetric, and include transition")
    eligible: dict[int, list[tuple[int, int, str]]] = {}
    eligibility: list[dict[str, object]] = []
    for event in events:
        values = windows.get(event.event_id, [])
        offsets = tuple(item[1] for item in values)
        admitted = offsets == expected_offsets
        if admitted:
            eligible[event.event_id] = values
        eligibility.append(
            {
                "event_id": event.event_id,
                "arc_index": event.arc_index,
                "transition_step": event.transition_step,
                "support_status": (
                    "complete_symmetric_event_window"
                    if admitted
                    else "excluded_incomplete_or_mismatched_event_window"
                ),
                "present_relative_frames": ";".join(str(value) for value in offsets),
                "required_relative_frames": ";".join(
                    str(value) for value in expected_offsets
                ),
            }
        )
    if not eligible:
        raise ValueError("no admitted event has complete symmetric window support")
    return eligible, eligibility


def _phase(relative_frame: int) -> str:
    if relative_frame < 0:
        return "pre"
    if relative_frame > 0:
        return "post"
    return "transition"


def build_event_samples(
    events: Sequence[Event],
    windows: Mapping[int, Sequence[tuple[int, int, str]]],
    grid: Mapping[tuple[int, int], Mapping[str, float]],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        for step, relative, supplied_phase in windows[event.event_id]:
            key = (step, event.arc_index)
            if key not in grid:
                raise ValueError(f"event {event.event_id} lacks water order at {key}")
            phase = _phase(relative)
            if supplied_phase != phase:
                raise ValueError(f"event {event.event_id}: inconsistent phase label")
            rows.append(
                {
                    "sample_kind": "event",
                    "event_id": event.event_id,
                    "control_id": 0,
                    "arc_index": event.arc_index,
                    "anchor_step": event.transition_step,
                    "step": step,
                    "relative_frame": relative,
                    "phase": phase,
                    "match_score": 0.0,
                    **{metric: grid[key][metric] for metric in metrics},
                }
            )
    return rows


def match_same_arc_dwells(
    events: Sequence[Event],
    steps: Sequence[int],
    grid: Mapping[tuple[int, int], Mapping[str, float]],
    offsets: Sequence[int],
    *,
    controls_per_event: int,
    exclusion_frames: int,
) -> list[MatchedDwell]:
    """Select deterministic same-arc quiescent windows using only pre-state fields."""

    index = {step: position for position, step in enumerate(steps)}
    lower, upper = min(offsets), max(offsets)
    pre_offsets = [offset for offset in offsets if offset < 0]
    if not pre_offsets:
        raise ValueError("matched dwell controls require pre-event frames")
    event_indices_by_arc: dict[int, list[int]] = {}
    for event in events:
        if event.transition_step not in index:
            raise ValueError(f"event {event.event_id} transition is outside the water-order grid")
        event_indices_by_arc.setdefault(event.arc_index, []).append(index[event.transition_step])
    output: list[MatchedDwell] = []
    minimum_separation = upper - lower + 1
    for event in events:
        event_index = index[event.transition_step]
        event_pre = [
            grid[(steps[event_index + offset], event.arc_index)] for offset in pre_offsets
        ]
        event_residual_std = float(
            np.std([row["local_residual_A"] for row in event_pre], ddof=0)
        )
        candidates: list[tuple[float, int]] = []
        for candidate_index in range(-lower, len(steps) - upper):
            if any(
                abs(candidate_index - observed) <= exclusion_frames
                for observed in event_indices_by_arc[event.arc_index]
            ):
                continue
            pre = [
                grid[(steps[candidate_index + offset], event.arc_index)]
                for offset in pre_offsets
            ]
            residual = np.asarray([row["local_residual_A"] for row in pre])
            if float(np.ptp(residual)) > event.dwell_tolerance_A:
                continue
            radius = float(np.mean([row["local_radius_A"] for row in pre]))
            score = abs(radius - event.pre_local_radius_A) / event.dwell_tolerance_A
            score += abs(float(np.std(residual, ddof=0)) - event_residual_std) / max(
                event.dwell_tolerance_A,
                0.5,
            )
            candidates.append((score, candidate_index))
        chosen: list[int] = []
        for score, candidate_index in sorted(candidates):
            if any(abs(candidate_index - other) < minimum_separation for other in chosen):
                continue
            chosen.append(candidate_index)
            output.append(
                MatchedDwell(
                    event_id=event.event_id,
                    control_id=len(chosen),
                    arc_index=event.arc_index,
                    anchor_step=steps[candidate_index],
                    match_score=score,
                )
            )
            if len(chosen) == controls_per_event:
                break
    return output


def build_matched_samples(
    controls: Sequence[MatchedDwell],
    event_by_id: Mapping[int, Event],
    steps: Sequence[int],
    offsets: Sequence[int],
    grid: Mapping[tuple[int, int], Mapping[str, float]],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    step_index = {step: index for index, step in enumerate(steps)}
    rows: list[dict[str, object]] = []
    for control in controls:
        anchor = step_index[control.anchor_step]
        event = event_by_id[control.event_id]
        for relative in offsets:
            step = steps[anchor + relative]
            key = (step, event.arc_index)
            rows.append(
                {
                    "sample_kind": "same_arc_matched_dwell",
                    "event_id": event.event_id,
                    "control_id": control.control_id,
                    "arc_index": event.arc_index,
                    "anchor_step": control.anchor_step,
                    "step": step,
                    "relative_frame": relative,
                    "phase": _phase(relative),
                    "match_score": control.match_score,
                    **{metric: grid[key][metric] for metric in metrics},
                }
            )
    return rows


def _finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else math.nan


def event_effect(
    anchor_index: int,
    arc_index: int,
    steps: Sequence[int],
    grid: Mapping[tuple[int, int], Mapping[str, float]],
    metric: str,
    pre_offsets: Sequence[int],
    post_offsets: Sequence[int],
    *,
    circular: bool,
) -> tuple[float, float, float]:
    def value(offset: int) -> float:
        index = anchor_index + offset
        if circular:
            index %= len(steps)
        elif not 0 <= index < len(steps):
            return math.nan
        return float(grid[(steps[index], arc_index)][metric])

    pre = _finite_mean([value(offset) for offset in pre_offsets])
    post = _finite_mean([value(offset) for offset in post_offsets])
    effect = post - pre if math.isfinite(pre) and math.isfinite(post) else math.nan
    return pre, post, effect


def observed_effect_rows(
    events: Sequence[Event],
    steps: Sequence[int],
    grid: Mapping[tuple[int, int], Mapping[str, float]],
    metrics: Sequence[str],
    pre_offsets: Sequence[int],
    post_offsets: Sequence[int],
) -> list[dict[str, object]]:
    index = {step: position for position, step in enumerate(steps)}
    rows: list[dict[str, object]] = []
    for event in events:
        for metric in metrics:
            pre, post, effect = event_effect(
                index[event.transition_step],
                event.arc_index,
                steps,
                grid,
                metric,
                pre_offsets,
                post_offsets,
                circular=False,
            )
            rows.append(
                {
                    "event_id": event.event_id,
                    "arc_index": event.arc_index,
                    "transition_step": event.transition_step,
                    "metric": metric,
                    "pre_mean": pre,
                    "post_mean": post,
                    "post_minus_pre": effect,
                }
            )
    return rows


def randomization_null_rows(
    events: Sequence[Event],
    steps: Sequence[int],
    grid: Mapping[tuple[int, int], Mapping[str, float]],
    metrics: Sequence[str],
    pre_offsets: Sequence[int],
    post_offsets: Sequence[int],
    *,
    null_samples: int,
    block_frames: int,
    random_seed: int,
) -> list[dict[str, object]]:
    """Return aggregate effects under per-arc block shifts and time permutations."""

    index = {step: position for position, step in enumerate(steps)}
    anchors = np.asarray([index[event.transition_step] for event in events], dtype=int)
    arcs = np.asarray([event.arc_index for event in events], dtype=int)
    distinct_arcs = sorted(set(arcs.tolist()))
    block_count = len(steps) // block_frames
    if null_samples < 1 or block_count < 2:
        raise ValueError("null_samples must be positive and the grid needs at least two blocks")
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for sample in range(null_samples):
        arc_shifts = {
            arc: int(rng.integers(1, block_count)) * block_frames for arc in distinct_arcs
        }
        shifted = np.asarray(
            [(anchor + arc_shifts[int(arc)]) % len(steps) for anchor, arc in zip(anchors, arcs)]
        )
        permuted = rng.permutation(anchors)
        for control_type, control_anchors in (
            ("per_arc_block_circular_shift", shifted),
            ("event_time_permutation", permuted),
        ):
            for metric in metrics:
                effects = [
                    event_effect(
                        int(anchor),
                        int(arc),
                        steps,
                        grid,
                        metric,
                        pre_offsets,
                        post_offsets,
                        circular=True,
                    )[2]
                    for anchor, arc in zip(control_anchors, arcs)
                ]
                rows.append(
                    {
                        "control_type": control_type,
                        "null_sample": sample + 1,
                        "metric": metric,
                        "event_count": int(np.count_nonzero(np.isfinite(effects))),
                        "mean_post_minus_pre": _finite_mean(effects),
                    }
                )
    return rows


def phase_contrasts(
    event_rows: Sequence[Mapping[str, object]],
    matched_rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for phase in ("pre", "transition", "post"):
        for metric in metrics:
            observed = [
                float(row[metric]) for row in event_rows if row["phase"] == phase
            ]
            matched = [
                float(row[metric]) for row in matched_rows if row["phase"] == phase
            ]
            observed_mean, matched_mean = _finite_mean(observed), _finite_mean(matched)
            rows.append(
                {
                    "phase": phase,
                    "metric": metric,
                    "event_sample_count": int(np.count_nonzero(np.isfinite(observed))),
                    "matched_sample_count": int(np.count_nonzero(np.isfinite(matched))),
                    "event_mean": observed_mean,
                    "matched_dwell_mean": matched_mean,
                    "event_minus_matched": (
                        observed_mean - matched_mean
                        if math.isfinite(observed_mean) and math.isfinite(matched_mean)
                        else math.nan
                    ),
                    "inference_status": (
                        "descriptive_same_trajectory_matched_control_not_independent_replicates"
                    ),
                }
            )
    return rows


def null_statistics(
    observed_rows: Sequence[Mapping[str, object]],
    null_rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for control_type in ("per_arc_block_circular_shift", "event_time_permutation"):
        for metric in metrics:
            observed = _finite_mean(
                [
                    float(row["post_minus_pre"])
                    for row in observed_rows
                    if row["metric"] == metric
                ]
            )
            null = np.asarray(
                [
                    float(row["mean_post_minus_pre"])
                    for row in null_rows
                    if row["control_type"] == control_type and row["metric"] == metric
                ],
                dtype=float,
            )
            null = null[np.isfinite(null)]
            null_mean = float(np.mean(null)) if len(null) else math.nan
            centered_observed = abs(observed - null_mean)
            p_value = (
                (1 + np.count_nonzero(np.abs(null - null_mean) >= centered_observed))
                / (len(null) + 1)
                if len(null) and math.isfinite(observed)
                else math.nan
            )
            rows.append(
                {
                    "control_type": control_type,
                    "metric": metric,
                    "observed_mean_post_minus_pre": observed,
                    "null_sample_count": len(null),
                    "null_mean": null_mean,
                    "null_sd": float(np.std(null, ddof=1)) if len(null) > 1 else math.nan,
                    "null_q025": float(np.quantile(null, 0.025)) if len(null) else math.nan,
                    "null_q975": float(np.quantile(null, 0.975)) if len(null) else math.nan,
                    "empirical_two_sided_p": p_value,
                    "inference_status": (
                        "within_trajectory_randomization_diagnostic_not_replicate_level_inference"
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty required table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    metrics = tuple(value.strip() for value in args.metrics.split(",") if value.strip())
    if not metrics:
        raise ValueError("at least one metric is required")
    steps, arcs, grid = load_arc_grid(args.water_order_by_arc, args.arc_kinematics, metrics)
    catalog_events = load_events(args.events, args.event_status)
    if any(event.arc_index not in arcs for event in catalog_events):
        raise ValueError("event catalog references an arc outside the water-order grid")
    windows, eligibility = load_event_windows(args.event_windows, catalog_events)
    eligible_ids = set(windows)
    events = [event for event in catalog_events if event.event_id in eligible_ids]
    offsets = [value[1] for value in windows[events[0].event_id]]
    pre_offsets = [offset for offset in offsets if offset < 0]
    post_offsets = [offset for offset in offsets if offset > 0]
    event_rows = build_event_samples(events, windows, grid, metrics)
    matched = match_same_arc_dwells(
        events,
        steps,
        grid,
        offsets,
        controls_per_event=args.matched_controls_per_event,
        exclusion_frames=args.event_exclusion_frames,
    )
    matched_event_ids = {control.event_id for control in matched}
    if matched_event_ids != {event.event_id for event in events}:
        missing = sorted({event.event_id for event in events}.difference(matched_event_ids))
        raise ValueError(f"same-arc dwell matching failed for event IDs {missing[:20]}")
    event_by_id = {event.event_id: event for event in events}
    matched_rows = build_matched_samples(
        matched,
        event_by_id,
        steps,
        offsets,
        grid,
        metrics,
    )
    effects = observed_effect_rows(
        events,
        steps,
        grid,
        metrics,
        pre_offsets,
        post_offsets,
    )
    null_rows = randomization_null_rows(
        events,
        steps,
        grid,
        metrics,
        pre_offsets,
        post_offsets,
        null_samples=args.null_samples,
        block_frames=args.block_frames,
        random_seed=args.random_seed,
    )
    contrasts = phase_contrasts(event_rows, matched_rows, metrics)
    statistics = null_statistics(effects, null_rows, metrics)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "event_window_support_eligibility.csv", eligibility)
    _write_csv(output / "event_aligned_samples.csv", event_rows)
    _write_csv(output / "matched_dwell_samples.csv", matched_rows)
    _write_csv(output / "observed_event_effects.csv", effects)
    _write_csv(output / "event_matched_phase_contrasts.csv", contrasts)
    _write_csv(output / "randomization_null_effects.csv", null_rows)
    _write_csv(output / "randomization_statistics.csv", statistics)
    summary = {
        "status": "PASS",
        "case_id": args.case_id,
        "catalog_event_count": len(catalog_events),
        "event_count": len(events),
        "support_excluded_event_count": len(catalog_events) - len(events),
        "event_aligned_rows": len(event_rows),
        "matched_dwell_count": len(matched),
        "matched_dwell_rows": len(matched_rows),
        "events_with_matched_dwells": len(matched_event_ids),
        "null_samples_per_family": args.null_samples,
        "null_effect_rows": len(null_rows),
        "metrics": list(metrics),
        "relative_frame_offsets": offsets,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    manifest = {
        "case_id": args.case_id,
        "water_order_by_arc": str(args.water_order_by_arc.resolve()),
        "arc_kinematics": str(args.arc_kinematics.resolve()),
        "events": str(args.events.resolve()),
        "event_windows": str(args.event_windows.resolve()),
        "event_status": args.event_status,
        "event_support_policy": (
            "admit_only_the_unique_modal_complete_symmetric_relative_frame_window; "
            "preserve_all_catalog_events_in_event_window_support_eligibility.csv"
        ),
        "metrics": list(metrics),
        "matched_controls_per_event": args.matched_controls_per_event,
        "same_arc_match_fields": [
            "pre_window_local_radius_mean",
            "pre_window_local_residual_std",
        ],
        "same_arc_dwell_gate": "pre_window_residual_range_le_event_dwell_tolerance",
        "event_exclusion_frames": args.event_exclusion_frames,
        "block_frames": args.block_frames,
        "null_samples": args.null_samples,
        "random_seed": args.random_seed,
        "block_null": "one_random_nonzero_block_shift_per_arc_preserving_within_arc_intervals",
        "permutation_null": "global_event_times_permuted_while_event_arc_labels_are_fixed",
        "null_boundary_policy": "circular_time_grid",
        "observed_effect": "unweighted_event_mean_of_post_mean_minus_pre_mean",
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "REPORT.md").write_text(
        "\n".join(
            (
                f"# Event-aligned water order: {args.case_id}",
                "",
                f"- Operational catalog events: {len(catalog_events)}",
                f"- Complete-window events analyzed: {len(events)}",
                f"- Support-ineligible events excluded: {len(catalog_events) - len(events)}",
                f"- Same-arc matched dwells: {len(matched)}",
                f"- Null samples per family: {args.null_samples}",
                "",
                (
                    "All comparisons are within one trajectory. Matched and randomized controls "
                    "do not establish causality, free energies, or physical event rates."
                ),
                "",
            )
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--water-order-by-arc", required=True, type=Path)
    parser.add_argument("--arc-kinematics", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--event-windows", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--event-status", default=EVENT_STATUS)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--matched-controls-per-event", type=int, default=9)
    parser.add_argument("--event-exclusion-frames", type=int, default=12)
    parser.add_argument("--block-frames", type=int, default=100)
    parser.add_argument("--null-samples", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260904)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive = (
        args.matched_controls_per_event,
        args.event_exclusion_frames,
        args.block_frames,
        args.null_samples,
    )
    if min(positive) < 1:
        raise ValueError("control counts, exclusion, block size, and null samples must be positive")
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
