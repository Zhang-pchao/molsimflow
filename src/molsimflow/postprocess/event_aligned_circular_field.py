"""Build signed event-aligned response maps on a periodic one-dimensional field.

The implementation is project-neutral: callers provide explicit source tables,
column names, fields, lag centers, and block size.  Circular-shift nulls and
whole-block bootstrap intervals are retrospective diagnostics from the supplied
records; they are not causal response functions or propagation measurements.
"""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_EVENT_ALIGNED_PERIODIC_FIELD_RESPONSE_"
    "NOT_CAUSAL_PROPAGATION_SPEED_INTRINSIC_LENGTH_OR_REPLICATE_EVIDENCE"
)


@dataclass(frozen=True)
class SourceSpec:
    case_id: str
    field_table: Path
    event_table: Path
    summary: Path


@dataclass(frozen=True)
class ColumnSpec:
    step: str = "step"
    time: str = "time_ns"
    arc: str = "arc_index"
    event_id: str = "primary_event_id"
    event_step: str = "transition_step"
    event_arc: str = "primary_arc_index"
    event_sign: str = "primary_residual_change_A"


@dataclass(frozen=True)
class AnalysisConfig:
    fields: tuple[str, ...]
    primary_field: str
    residual_field: str
    low_order_field: str
    mean_field: str
    lags_ps: tuple[float, ...]
    reference_lag_ps: float = -0.5
    block_ps: float = 200.0
    null_samples: int = 2000
    bootstrap_samples: int = 2000
    random_seed: int = 20260904
    write_event_table: bool = True


@dataclass(frozen=True)
class FieldData:
    steps: np.ndarray
    times_ps: np.ndarray
    values: np.ndarray
    arc_count: int
    cadence_ps: float
    step_to_index: Mapping[int, int]


@dataclass(frozen=True)
class EventData:
    event_ids: np.ndarray
    frame_indices: np.ndarray
    primary_arcs: np.ndarray
    signs: np.ndarray
    block_indices: np.ndarray
    block_start_indices: np.ndarray
    admitted_count: int
    excluded_boundary_count: int
    excluded_sign_count: int


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, tuple(reader.fieldnames or ())


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(path: Path, columns: Sequence[str], required: Sequence[str]) -> None:
    missing = sorted(set(required).difference(columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def _finite_float(value: object, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number, got {value!r}")
    return number


def read_sources(path: Path) -> tuple[SourceSpec, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"case_id", "arc_kinematics", "event_sizes", "propagation_summary"}
    if not rows or required.difference(rows[0]):
        raise ValueError(f"{path}: missing rows or required columns")
    sources = tuple(
        SourceSpec(
            case_id=row["case_id"],
            field_table=Path(row["arc_kinematics"]),
            event_table=Path(row["event_sizes"]),
            summary=Path(row["propagation_summary"]),
        )
        for row in rows
    )
    case_ids = [source.case_id for source in sources]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("case_id values must be nonempty and unique")
    return sources


def load_field_data(
    path: Path,
    fields: Sequence[str],
    columns: ColumnSpec,
) -> FieldData:
    """Load a complete regular `(time, circular arc, field)` table."""

    rows, fieldnames = _read_csv(path)
    _require_columns(path, fieldnames, (columns.step, columns.time, columns.arc, *fields))
    if not rows:
        raise ValueError(f"empty field table: {path}")
    grouped: dict[int, list[dict[str, str]]] = {}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        step = int(row[columns.step])
        arc = int(row[columns.arc])
        key = (step, arc)
        if key in seen:
            raise ValueError(f"{path}: duplicate sample {key}")
        seen.add(key)
        grouped.setdefault(step, []).append(row)
    steps = np.asarray(sorted(grouped), dtype=np.int64)
    if len(steps) < 3:
        raise ValueError(f"{path}: at least three frames are required")
    first_arcs = sorted(int(row[columns.arc]) for row in grouped[int(steps[0])])
    if first_arcs != list(range(len(first_arcs))) or len(first_arcs) < 4:
        raise ValueError("arc indices must be contiguous from zero with at least four arcs")
    arc_count = len(first_arcs)
    times_ps = np.empty(len(steps), dtype=float)
    values = np.empty((len(steps), arc_count, len(fields)), dtype=float)
    for frame_index, step in enumerate(steps):
        ordered = sorted(grouped[int(step)], key=lambda row: int(row[columns.arc]))
        if [int(row[columns.arc]) for row in ordered] != first_arcs:
            raise ValueError(f"{path}: incomplete circular field at step {step}")
        times = np.asarray(
            [_finite_float(row[columns.time], f"{path}/{step}/time") for row in ordered]
        )
        if not np.allclose(times, times[0], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{path}: inconsistent time within step {step}")
        times_ps[frame_index] = times[0] * 1000.0
        for field_index, field in enumerate(fields):
            values[frame_index, :, field_index] = [
                _finite_float(row[field], f"{path}/{step}/{field}") for row in ordered
            ]
    time_deltas = np.diff(times_ps)
    cadence_ps = float(np.median(time_deltas))
    if cadence_ps <= 0.0 or not np.allclose(
        time_deltas, cadence_ps, rtol=0.0, atol=max(1.0e-9, cadence_ps * 1.0e-8)
    ):
        raise ValueError(f"{path}: field time grid must be positive and regular")
    if np.any(np.diff(steps) <= 0):
        raise ValueError(f"{path}: steps must be strictly increasing")
    return FieldData(
        steps=steps,
        times_ps=times_ps,
        values=values,
        arc_count=arc_count,
        cadence_ps=cadence_ps,
        step_to_index={int(step): index for index, step in enumerate(steps)},
    )


def _lag_offsets(lags_ps: Sequence[float], cadence_ps: float) -> np.ndarray:
    raw = np.asarray(lags_ps, dtype=float) / cadence_ps
    rounded = np.rint(raw).astype(int)
    if not np.allclose(raw, rounded, rtol=0.0, atol=1.0e-8):
        raise ValueError("every lag must lie on the field cadence")
    return rounded


def load_event_data(
    path: Path,
    field: FieldData,
    columns: ColumnSpec,
    config: AnalysisConfig,
) -> EventData:
    """Load primary event anchors and apply the frozen within-block support gate."""

    rows, fieldnames = _read_csv(path)
    required = (
        columns.event_id,
        columns.event_step,
        columns.event_arc,
        columns.event_sign,
    )
    _require_columns(path, fieldnames, required)
    if not rows:
        raise ValueError(f"empty event table: {path}")
    lag_offsets = _lag_offsets(config.lags_ps, field.cadence_ps)
    reference_offset = int(_lag_offsets((config.reference_lag_ps,), field.cadence_ps)[0])
    block_frames_raw = config.block_ps / field.cadence_ps
    block_frames = round(block_frames_raw)
    if block_frames < 2 or not math.isclose(block_frames_raw, block_frames, abs_tol=1.0e-8):
        raise ValueError("block_ps must be an integer multiple of the field cadence")
    first_time_ps = float(field.times_ps[0])
    event_ids: list[int] = []
    frame_indices: list[int] = []
    primary_arcs: list[int] = []
    signs: list[float] = []
    block_indices: list[int] = []
    block_start_indices: list[int] = []
    excluded_boundary = 0
    excluded_sign = 0
    seen_ids: set[int] = set()
    for row in rows:
        event_id = int(row[columns.event_id])
        if event_id in seen_ids:
            raise ValueError(f"{path}: duplicate event id {event_id}")
        seen_ids.add(event_id)
        step = int(row[columns.event_step])
        if step not in field.step_to_index:
            raise ValueError(f"{path}: event step {step} is absent from the field table")
        frame_index = int(field.step_to_index[step])
        arc = int(row[columns.event_arc])
        if arc < 0 or arc >= field.arc_count:
            raise ValueError(f"{path}: event arc {arc} is outside the circular field")
        sign_value = float(row[columns.event_sign])
        if not math.isfinite(sign_value) or sign_value == 0.0:
            excluded_sign += 1
            continue
        event_time_ps = float(field.times_ps[frame_index])
        block_index = math.floor((event_time_ps - first_time_ps) / config.block_ps + 1.0e-10)
        block_start_index = block_index * block_frames
        relative_index = frame_index - block_start_index
        requested = np.concatenate((lag_offsets, np.asarray([reference_offset])))
        if (
            block_start_index < 0
            or block_start_index + block_frames > len(field.steps)
            or np.min(relative_index + requested) < 0
            or np.max(relative_index + requested) >= block_frames
        ):
            excluded_boundary += 1
            continue
        event_ids.append(event_id)
        frame_indices.append(frame_index)
        primary_arcs.append(arc)
        signs.append(math.copysign(1.0, sign_value))
        block_indices.append(block_index)
        block_start_indices.append(block_start_index)
    if not event_ids:
        raise ValueError(f"{path}: no events survive the support and sign gates")
    return EventData(
        event_ids=np.asarray(event_ids, dtype=np.int64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        primary_arcs=np.asarray(primary_arcs, dtype=np.int32),
        signs=np.asarray(signs, dtype=float),
        block_indices=np.asarray(block_indices, dtype=np.int32),
        block_start_indices=np.asarray(block_start_indices, dtype=np.int64),
        admitted_count=len(event_ids),
        excluded_boundary_count=excluded_boundary,
        excluded_sign_count=excluded_sign,
    )


def signed_arc_offsets(arc_count: int) -> np.ndarray:
    """Return deterministic circular offsets in a half-open centered interval."""

    if arc_count < 4:
        raise ValueError("arc_count must be at least four")
    return np.arange(-(arc_count // 2), arc_count - arc_count // 2, dtype=int)


def event_responses(
    field: FieldData,
    events: EventData,
    config: AnalysisConfig,
    *,
    circular_shifts: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `[event, lag, offset, field]` signed changes and arc offsets."""

    lag_offsets = _lag_offsets(config.lags_ps, field.cadence_ps)
    reference_offset = int(_lag_offsets((config.reference_lag_ps,), field.cadence_ps)[0])
    offsets = signed_arc_offsets(field.arc_count)
    target_arcs = (events.primary_arcs[:, None] + offsets[None, :]) % field.arc_count
    if circular_shifts is None:
        lag_indices = events.frame_indices[:, None] + lag_offsets[None, :]
        reference_indices = events.frame_indices + reference_offset
    else:
        shifts = np.asarray(circular_shifts, dtype=int)
        if shifts.shape != (events.admitted_count,):
            raise ValueError("circular_shifts must contain one shift per admitted event")
        block_frames = round(config.block_ps / field.cadence_ps)
        relative = events.frame_indices - events.block_start_indices
        lag_indices = events.block_start_indices[:, None] + np.mod(
            relative[:, None] + shifts[:, None] + lag_offsets[None, :], block_frames
        )
        reference_indices = events.block_start_indices + np.mod(
            relative + shifts + reference_offset, block_frames
        )
    lagged = field.values[lag_indices[:, :, None], target_arcs[:, None, :], :]
    reference = field.values[reference_indices[:, None], target_arcs, :]
    response = events.signs[:, None, None, None] * (lagged - reference[:, None, :, :])
    return response, offsets


def _bootstrap_maps(
    responses: np.ndarray,
    block_indices: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_blocks = np.unique(block_indices)
    block_sums = np.asarray(
        [np.sum(responses[block_indices == block], axis=0) for block in unique_blocks]
    )
    block_counts = np.asarray(
        [np.count_nonzero(block_indices == block) for block in unique_blocks], dtype=float
    )
    output = np.empty((samples, *responses.shape[1:]), dtype=float)
    for sample in range(samples):
        selected = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
        output[sample] = np.sum(block_sums[selected], axis=0) / np.sum(block_counts[selected])
    return output


def _null_maps(
    field: FieldData,
    events: EventData,
    config: AnalysisConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    group_keys = list(zip(events.block_indices.tolist(), events.primary_arcs.tolist()))
    unique_keys = {key: index for index, key in enumerate(sorted(set(group_keys)))}
    group_indices = np.asarray([unique_keys[key] for key in group_keys], dtype=int)
    block_frames = round(config.block_ps / field.cadence_ps)
    output = np.empty(
        (
            config.null_samples,
            len(config.lags_ps),
            field.arc_count,
            len(config.fields),
        ),
        dtype=float,
    )
    for sample in range(config.null_samples):
        group_shifts = rng.integers(0, block_frames, size=len(unique_keys))
        responses, _ = event_responses(
            field,
            events,
            config,
            circular_shifts=group_shifts[group_indices],
        )
        output[sample] = np.mean(responses, axis=0)
    return output


def _mean_selected(
    response_map: np.ndarray,
    lag_mask: np.ndarray,
    offset_mask: np.ndarray,
    field_index: int,
) -> float:
    selected = response_map[np.ix_(lag_mask, offset_mask, [field_index])]
    return float(np.mean(selected))


def primary_estimands(
    response_map: np.ndarray,
    offsets: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, float]:
    """Calculate the six preregistered scalar summaries from one response map."""

    lags = np.asarray(config.lags_ps, dtype=float)
    fast = (lags >= 0.5) & (lags <= 5.0)
    slow = (lags > 5.0) & (lags <= 50.0)
    distances = np.abs(offsets)
    primary = distances == 0
    near = (distances >= 1) & (distances < 4)
    far = (distances >= 4) & (distances <= offsets.size // 2)
    all_offsets = np.ones(len(offsets), dtype=bool)
    if not np.any(fast) or not np.any(slow) or not np.any(primary | near) or not np.any(far):
        raise ValueError("lag or arc grid does not cover the frozen primary estimands")
    field_index = {field: index for index, field in enumerate(config.fields)}
    for required in (
        config.primary_field,
        config.residual_field,
        config.low_order_field,
        config.mean_field,
    ):
        if required not in field_index:
            raise ValueError(f"primary estimand field {required!r} is absent")
    total = field_index[config.primary_field]
    residual = field_index[config.residual_field]
    return {
        "primary_arc_residual_persistence_fast": _mean_selected(
            response_map, fast, primary, residual
        ),
        "near_total_response_fast": _mean_selected(response_map, fast, near, total),
        "far_total_response_fast": _mean_selected(response_map, fast, far, total),
        "far_total_response_slow": _mean_selected(response_map, slow, far, total),
        "low_order_primary_arc_fast": _mean_selected(
            response_map, fast, primary, field_index[config.low_order_field]
        ),
        "mean_radius_response_fast": _mean_selected(
            response_map, fast, all_offsets, field_index[config.mean_field]
        ),
    }


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _event_rows(
    case_id: str,
    events: EventData,
    responses: np.ndarray,
    offsets: np.ndarray,
    config: AnalysisConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event_index, event_id in enumerate(events.event_ids):
        for lag_index, lag_ps in enumerate(config.lags_ps):
            for offset_index, offset in enumerate(offsets):
                for field_index, field in enumerate(config.fields):
                    rows.append(
                        {
                            "case_id": case_id,
                            "event_id": int(event_id),
                            "time_block_200ps": int(events.block_indices[event_index]),
                            "primary_arc_index": int(events.primary_arcs[event_index]),
                            "event_sign": float(events.signs[event_index]),
                            "field": field,
                            "lag_ps": lag_ps,
                            "arc_offset_signed": int(offset),
                            "arc_distance": int(abs(offset)),
                            "aligned_change": float(
                                responses[event_index, lag_index, offset_index, field_index]
                            ),
                            "scientific_status": SCIENTIFIC_STATUS,
                        }
                    )
    return rows


def analyze_source(
    source: SourceSpec,
    config: AnalysisConfig,
    columns: ColumnSpec,
    rng: np.random.Generator,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    summary = json.loads(source.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("case_id") != source.case_id:
        raise ValueError(f"{source.case_id}: source summary is not a matching PASS")
    field = load_field_data(source.field_table, config.fields, columns)
    if int(summary.get("arc_count", -1)) != field.arc_count:
        raise ValueError(f"{source.case_id}: source summary arc count does not match field")
    events = load_event_data(source.event_table, field, columns, config)
    if int(summary.get("event_cluster_count", -1)) != (
        events.admitted_count + events.excluded_boundary_count + events.excluded_sign_count
    ):
        raise ValueError(f"{source.case_id}: source event count does not match event table")
    responses, offsets = event_responses(field, events, config)
    observed = np.mean(responses, axis=0)
    bootstrap = _bootstrap_maps(
        responses,
        events.block_indices,
        config.bootstrap_samples,
        rng,
    )
    null = _null_maps(field, events, config, rng)
    null_mean = np.mean(null, axis=0)
    map_rows: list[dict[str, object]] = []
    for lag_index, lag_ps in enumerate(config.lags_ps):
        for offset_index, offset in enumerate(offsets):
            for field_index, field_name in enumerate(config.fields):
                map_rows.append(
                    {
                        "case_id": source.case_id,
                        "field": field_name,
                        "lag_ps": lag_ps,
                        "arc_offset_signed": int(offset),
                        "arc_distance": int(abs(offset)),
                        "observed_mean_aligned_change": float(
                            observed[lag_index, offset_index, field_index]
                        ),
                        "block_bootstrap_ci025": float(
                            np.quantile(bootstrap[:, lag_index, offset_index, field_index], 0.025)
                        ),
                        "block_bootstrap_ci975": float(
                            np.quantile(bootstrap[:, lag_index, offset_index, field_index], 0.975)
                        ),
                        "null_mean": float(null_mean[lag_index, offset_index, field_index]),
                        "null_q025": float(
                            np.quantile(null[:, lag_index, offset_index, field_index], 0.025)
                        ),
                        "null_q975": float(
                            np.quantile(null[:, lag_index, offset_index, field_index], 0.975)
                        ),
                        "event_minus_null": float(
                            observed[lag_index, offset_index, field_index]
                            - null_mean[lag_index, offset_index, field_index]
                        ),
                        "inference_status": "descriptive_map_not_multiplicity_controlled",
                        "scientific_status": SCIENTIFIC_STATUS,
                    }
                )
    observed_estimands = primary_estimands(observed, offsets, config)
    null_estimands = [primary_estimands(item, offsets, config) for item in null]
    bootstrap_estimands = [primary_estimands(item, offsets, config) for item in bootstrap]
    estimand_rows: list[dict[str, object]] = []
    for name, observed_value in observed_estimands.items():
        null_values = np.asarray([item[name] for item in null_estimands], dtype=float)
        null_center = float(np.mean(null_values))
        effect = observed_value - null_center
        p_value = float(
            (1 + np.count_nonzero(np.abs(null_values - null_center) >= abs(effect)))
            / (len(null_values) + 1)
        )
        bootstrap_values = np.asarray(
            [item[name] - null_center for item in bootstrap_estimands], dtype=float
        )
        estimand_rows.append(
            {
                "case_id": source.case_id,
                "estimand": name,
                "admitted_event_count": events.admitted_count,
                "observed": observed_value,
                "null_mean": null_center,
                "event_minus_null": effect,
                "effect_block_bootstrap_ci025": float(np.quantile(bootstrap_values, 0.025)),
                "effect_block_bootstrap_ci975": float(np.quantile(bootstrap_values, 0.975)),
                "null_q025": float(np.quantile(null_values, 0.025)),
                "null_q975": float(np.quantile(null_values, 0.975)),
                "empirical_two_sided_p": p_value,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    event_rows = _event_rows(source.case_id, events, responses, offsets, config)
    case_summary = {
        "case_id": source.case_id,
        "arc_count": field.arc_count,
        "frame_count": len(field.steps),
        "cadence_ps": field.cadence_ps,
        "source_event_cluster_count": int(summary["event_cluster_count"]),
        "admitted_event_count": events.admitted_count,
        "excluded_block_boundary_count": events.excluded_boundary_count,
        "excluded_nonfinite_or_zero_sign_count": events.excluded_sign_count,
        "block_count_with_admitted_events": len(np.unique(events.block_indices)),
    }
    return map_rows, estimand_rows, event_rows, case_summary


def run_analysis(
    sources_path: Path,
    output_dir: Path,
    config: AnalysisConfig,
    columns: Optional[ColumnSpec] = None,
) -> dict[str, object]:
    """Run the complete multi-case event-aligned response analysis."""

    columns = columns or ColumnSpec()
    if len(config.fields) != len(set(config.fields)) or not config.fields:
        raise ValueError("fields must be nonempty and unique")
    if tuple(sorted(config.lags_ps)) != config.lags_ps or len(set(config.lags_ps)) != len(
        config.lags_ps
    ):
        raise ValueError("lags_ps must be sorted and unique")
    if config.block_ps <= 0.0 or config.null_samples < 1 or config.bootstrap_samples < 1:
        raise ValueError("block and sample counts must be positive")
    sources = read_sources(sources_path)
    rng = np.random.default_rng(config.random_seed)
    all_maps: list[dict[str, object]] = []
    all_estimands: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    case_summaries: list[dict[str, object]] = []
    for source in sources:
        maps, estimands, events, case_summary = analyze_source(source, config, columns, rng)
        all_maps.extend(maps)
        all_estimands.extend(estimands)
        if config.write_event_table:
            all_events.extend(events)
        case_summaries.append(case_summary)
    adjusted = _bh_adjust([float(row["empirical_two_sided_p"]) for row in all_estimands])
    for row, q_value in zip(all_estimands, adjusted):
        row["bh_q_primary_family"] = q_value
        row["qualified_primary_effect"] = q_value < 0.05
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "event_aligned_map.csv", all_maps)
    _write_csv(output / "primary_estimands.csv", all_estimands)
    if config.write_event_table:
        _write_csv(output / "event_level_response.csv", all_events)
    summary = {
        "status": "PASS",
        "case_count": len(sources),
        "fields": list(config.fields),
        "lags_ps": list(config.lags_ps),
        "reference_lag_ps": config.reference_lag_ps,
        "block_ps": config.block_ps,
        "null_samples": config.null_samples,
        "bootstrap_samples": config.bootstrap_samples,
        "primary_estimand_count": len(all_estimands),
        "qualified_primary_count": sum(
            bool(row["qualified_primary_effect"]) for row in all_estimands
        ),
        "case_summaries": case_summaries,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "sources": [
            {
                "case_id": source.case_id,
                "field_table": str(source.field_table),
                "field_table_sha256": _sha256(source.field_table),
                "event_table": str(source.event_table),
                "event_table_sha256": _sha256(source.event_table),
                "summary": str(source.summary),
                "summary_sha256": _sha256(source.summary),
            }
            for source in sources
        ],
        "columns": columns.__dict__,
        "config": {
            **config.__dict__,
            "fields": list(config.fields),
            "lags_ps": list(config.lags_ps),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "# Event-aligned circular-field response\n\n"
        "Maps are sign-aligned changes relative to the configured pre-event frame. "
        "Circular-shift nulls preserve event sequences within arc and time block. "
        "Only `primary_estimands.csv` is multiplicity controlled. Results are "
        "retrospective single-trajectory associations, not causal propagation, "
        "physical speeds, intrinsic lengths, or replicate-level evidence.\n",
        encoding="utf-8",
    )
    return summary


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _parse_csv_strings(raw))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--primary-field", required=True)
    parser.add_argument("--residual-field", required=True)
    parser.add_argument("--low-order-field", required=True)
    parser.add_argument("--mean-field", required=True)
    parser.add_argument("--lags-ps", required=True)
    parser.add_argument("--reference-lag-ps", type=float, default=-0.5)
    parser.add_argument("--block-ps", type=float, default=200.0)
    parser.add_argument("--null-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--skip-event-table", action="store_true")
    parser.add_argument("--step-column", default="step")
    parser.add_argument("--time-column", default="time_ns")
    parser.add_argument("--arc-column", default="arc_index")
    parser.add_argument("--event-id-column", default="primary_event_id")
    parser.add_argument("--event-step-column", default="transition_step")
    parser.add_argument("--event-arc-column", default="primary_arc_index")
    parser.add_argument("--event-sign-column", default="primary_residual_change_A")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = AnalysisConfig(
        fields=_parse_csv_strings(args.fields),
        primary_field=args.primary_field,
        residual_field=args.residual_field,
        low_order_field=args.low_order_field,
        mean_field=args.mean_field,
        lags_ps=_parse_csv_floats(args.lags_ps),
        reference_lag_ps=args.reference_lag_ps,
        block_ps=args.block_ps,
        null_samples=args.null_samples,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.seed,
        write_event_table=not args.skip_event_table,
    )
    columns = ColumnSpec(
        step=args.step_column,
        time=args.time_column,
        arc=args.arc_column,
        event_id=args.event_id_column,
        event_step=args.event_step_column,
        event_arc=args.event_arc_column,
        event_sign=args.event_sign_column,
    )
    run_analysis(args.sources, args.output_dir, config, columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
