"""Blocked retrospective reconstruction of a global TPCL radius from local events.

The module accepts explicit table paths and column mappings.  It deliberately
uses only event timing and a local, mean-radius-free event mark as predictors.
It is a descriptive, within-trajectory reconstruction diagnostic; it does not
identify a causal impulse response, a physical rate, or a material property.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_BLOCKED_LOCAL_EVENT_TRAIN_TO_GLOBAL_MEAN_RADIUS_RECONSTRUCTION_"
    "NOT_CAUSAL_IMPULSE_RESPONSE_PHYSICAL_RATE_FREE_ENERGY_FRICTION_DISSIPATION_"
    "AVALANCHE_OR_REPLICATE_EVIDENCE"
)
BASELINE_MODEL = "baseline_mean_increment"
TIMING_MODEL = "event_timing_only"
MARK_MODEL = "event_timing_and_signed_local_residual"
EVENT_MODELS = (TIMING_MODEL, MARK_MODEL)


@dataclass(frozen=True)
class ReconstructionConfig:
    frame_interval_ps: float
    block_ps: float
    kernel_max_lag_ps: float
    ridge_penalty: float
    embargo_blocks: int
    bootstrap_samples: int
    null_samples: int
    random_seed: int
    time_tolerance_ps: float

    @property
    def kernel_lag_frames(self) -> int:
        value = self.kernel_max_lag_ps / self.frame_interval_ps
        rounded = round(value)
        if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("kernel_max_lag_ps must be an exact multiple of frame_interval_ps")
        return rounded


@dataclass(frozen=True)
class CaseSeries:
    case_id: str
    times_ns: np.ndarray
    mean_radius: np.ndarray
    event_count: np.ndarray
    signed_mark: np.ndarray
    block_ids: np.ndarray
    source_frame_count: int
    source_arc_count: int
    event_count_total: int
    event_count_mapped: int


@dataclass(frozen=True)
class RidgeModel:
    intercept: float
    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.intercept + np.asarray(features, dtype=float) @ self.coefficients


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"empty table: {path}")
    if not rows[0]:
        raise ValueError(f"missing header: {path}")
    return rows


def _number(row: dict[str, str], column: str, context: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {column} in {context}") from exc
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {column} in {context}")
    return value


def _integer(row: dict[str, str], column: str, context: str) -> int:
    value = _number(row, column, context)
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"noninteger {column} in {context}")
    return rounded


def _resolve_input_path(value: str, parent: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _source_rows(
    sources_table: Path,
    *,
    case_column: str,
    source_path_column: str,
) -> list[tuple[str, Path]]:
    rows = _read_csv(sources_table, delimiter="\t")
    items: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for row in rows:
        try:
            case_id = row[case_column]
            source = _resolve_input_path(row[source_path_column], sources_table.parent)
        except KeyError as exc:
            raise ValueError(f"source table missing required column: {exc}") from exc
        if not case_id or case_id in seen:
            raise ValueError(f"duplicate or empty case id: {case_id!r}")
        if not source.is_file():
            raise ValueError(f"missing source table: {source}")
        seen.add(case_id)
        items.append((case_id, source))
    return items


def _nearest_frame(times_ns: np.ndarray, target_ns: float, tolerance_ps: float) -> int:
    index = int(np.searchsorted(times_ns, target_ns))
    candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(times_ns)]
    if not candidates:
        raise ValueError("event is outside the radius time support")
    best = min(candidates, key=lambda candidate: abs(times_ns[candidate] - target_ns))
    if abs(times_ns[best] - target_ns) * 1000.0 > tolerance_ps:
        raise ValueError(f"event time {target_ns} ns is not aligned to source frames")
    return best


def _load_case_series(
    case_id: str,
    source_path: Path,
    event_rows: Iterable[dict[str, str]],
    config: ReconstructionConfig,
    *,
    time_column: str,
    arc_column: str,
    mean_radius_column: str,
    event_time_column: str,
    event_arc_column: str,
    event_mark_column: str,
) -> CaseSeries:
    raw_rows = _read_csv(source_path)
    by_time: dict[float, list[dict[str, str]]] = defaultdict(list)
    local_rows: dict[tuple[float, int], dict[str, str]] = {}
    for row in raw_rows:
        time_ns = _number(row, time_column, str(source_path))
        arc_index = _integer(row, arc_column, str(source_path))
        key = (time_ns, arc_index)
        if key in local_rows:
            raise ValueError(f"duplicate source frame/arc row: {case_id} {key}")
        by_time[time_ns].append(row)
        local_rows[key] = row

    ordered_times = np.asarray(sorted(by_time), dtype=float)
    if len(ordered_times) < 4:
        raise ValueError(f"insufficient source frames: {case_id}")
    cadence = np.diff(ordered_times) * 1000.0
    if not np.allclose(cadence, config.frame_interval_ps, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"nonuniform source cadence: {case_id}")

    radius: list[float] = []
    arc_sets: list[set[int]] = []
    for time_ns in ordered_times:
        rows_at_time = by_time[float(time_ns)]
        values = [_number(row, mean_radius_column, str(source_path)) for row in rows_at_time]
        if max(values) - min(values) > 1.0e-8:
            raise ValueError(f"mean-radius field is not spatially uniform: {case_id} {time_ns}")
        radius.append(float(np.mean(values)))
        arc_sets.append({_integer(row, arc_column, str(source_path)) for row in rows_at_time})
    if any(item != arc_sets[0] for item in arc_sets[1:]):
        raise ValueError(f"source arc set changes with time: {case_id}")

    event_count = np.zeros(len(ordered_times), dtype=float)
    signed_mark = np.zeros(len(ordered_times), dtype=float)
    mapped = 0
    total = 0
    for event in event_rows:
        total += 1
        time_ns = _number(event, event_time_column, f"event {case_id}")
        arc_index = _integer(event, event_arc_column, f"event {case_id}")
        frame_index = _nearest_frame(ordered_times, time_ns, config.time_tolerance_ps)
        if frame_index == 0:
            raise ValueError(f"event occurs at first source frame: {case_id}")
        source_key = (float(ordered_times[frame_index]), arc_index)
        source_row = local_rows.get(source_key)
        if source_row is None:
            raise ValueError(f"event arc is absent from source table: {case_id} {source_key}")
        mark = _number(source_row, event_mark_column, f"event mark {case_id}")
        event_count[frame_index] += 1.0
        signed_mark[frame_index] += mark
        mapped += 1

    elapsed_ps = (ordered_times - ordered_times[0]) * 1000.0
    block_ids = np.floor(np.maximum(0.0, elapsed_ps - 1.0e-9) / config.block_ps).astype(int)
    return CaseSeries(
        case_id=case_id,
        times_ns=ordered_times,
        mean_radius=np.asarray(radius, dtype=float),
        event_count=event_count,
        signed_mark=signed_mark,
        block_ids=block_ids,
        source_frame_count=len(ordered_times),
        source_arc_count=len(arc_sets[0]),
        event_count_total=total,
        event_count_mapped=mapped,
    )


def _lag_design(values: np.ndarray, lag_frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.zeros((len(values), lag_frames + 1), dtype=float)
    for lag in range(lag_frames + 1):
        output[lag:, lag] = values[: len(values) - lag]
    return output


def _features_for_model(model: str, count: np.ndarray, mark: np.ndarray, lag_frames: int) -> np.ndarray:
    if model == TIMING_MODEL:
        return _lag_design(count, lag_frames)
    if model == MARK_MODEL:
        return np.column_stack((_lag_design(count, lag_frames), _lag_design(mark, lag_frames)))
    raise ValueError(f"unknown event model: {model}")


def _fit_ridge(features: np.ndarray, target: np.ndarray, penalty: float) -> RidgeModel:
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("invalid ridge input shapes")
    if len(target) < 2:
        raise ValueError("insufficient ridge training rows")
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1.0e-12] = 1.0
    standardized = (features - center) / scale
    target_center = float(np.mean(target))
    covariance = standardized.T @ standardized
    covariance.flat[:: len(covariance) + 1] += penalty
    coefficients_standard = np.linalg.solve(covariance, standardized.T @ (target - target_center))
    coefficients = coefficients_standard / scale
    intercept = target_center - float(center @ coefficients)
    return RidgeModel(intercept=intercept, coefficients=coefficients)


def _fold_rows(
    series: CaseSeries,
    config: ReconstructionConfig,
    count: np.ndarray,
    mark: np.ndarray,
    *,
    collect_traces: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = np.diff(series.mean_radius)
    times = series.times_ns[1:]
    blocks = series.block_ids[1:]
    count = np.asarray(count[1:], dtype=float)
    mark = np.asarray(mark[1:], dtype=float)
    unique_blocks = np.unique(blocks)
    if len(unique_blocks) <= 2 * config.embargo_blocks + 1:
        raise ValueError(f"too few time blocks after embargo: {series.case_id}")

    features = {
        model: _features_for_model(model, count, mark, config.kernel_lag_frames)
        for model in EVENT_MODELS
    }
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for held_block in unique_blocks:
        test_mask = blocks == held_block
        train_mask = np.abs(blocks - held_block) > config.embargo_blocks
        if int(np.sum(test_mask)) == 0 or int(np.sum(train_mask)) < 2:
            raise ValueError(f"invalid blocked split: {series.case_id} {held_block}")
        train_target = target[train_mask]
        baseline_increment = float(np.mean(train_target))
        predictions: dict[str, np.ndarray] = {BASELINE_MODEL: np.full(int(np.sum(test_mask)), baseline_increment)}
        for model in EVENT_MODELS:
            fitted = _fit_ridge(features[model][train_mask], train_target, config.ridge_penalty)
            predictions[model] = fitted.predict(features[model][test_mask])

        observed_increment = target[test_mask]
        observed_radius = np.cumsum(observed_increment)
        baseline_radius = np.cumsum(predictions[BASELINE_MODEL])
        baseline_delta_sse = float(np.sum((observed_increment - predictions[BASELINE_MODEL]) ** 2))
        baseline_radius_sse = float(np.sum((observed_radius - baseline_radius) ** 2))
        for model, predicted_increment in predictions.items():
            predicted_radius = np.cumsum(predicted_increment)
            delta_sse = float(np.sum((observed_increment - predicted_increment) ** 2))
            radius_sse = float(np.sum((observed_radius - predicted_radius) ** 2))
            rows.append(
                {
                    "case_id": series.case_id,
                    "held_block_id": int(held_block),
                    "model": model,
                    "frame_count": int(np.sum(test_mask)),
                    "delta_sse_A2": delta_sse,
                    "radius_sse_A2": radius_sse,
                    "baseline_delta_sse_A2": baseline_delta_sse,
                    "baseline_radius_sse_A2": baseline_radius_sse,
                    "radius_sse_improvement_vs_baseline": (
                        0.0 if model == BASELINE_MODEL else 1.0 - radius_sse / baseline_radius_sse
                    ),
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
        if collect_traces:
            indices = np.flatnonzero(test_mask)
            for local_index, global_index in enumerate(indices):
                trace_rows.append(
                    {
                        "case_id": series.case_id,
                        "held_block_id": int(held_block),
                        "time_ns": float(times[global_index]),
                        "event_count": float(count[global_index]),
                        "signed_local_residual_A": float(mark[global_index]),
                        "observed_relative_mean_radius_A": float(observed_radius[local_index]),
                        "baseline_relative_mean_radius_A": float(baseline_radius[local_index]),
                        "timing_relative_mean_radius_A": float(
                            np.cumsum(predictions[TIMING_MODEL])[local_index]
                        ),
                        "marked_relative_mean_radius_A": float(
                            np.cumsum(predictions[MARK_MODEL])[local_index]
                        ),
                        "scientific_status": SCIENTIFIC_STATUS,
                    }
                )
    return rows, trace_rows


def _aggregate_rows(
    rows: Sequence[dict[str, Any]],
    config: ReconstructionConfig,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for model in (BASELINE_MODEL, *EVENT_MODELS):
        group = by_model.get(model, [])
        if not group:
            raise ValueError(f"missing fold rows for {model}")
        radius_sse = np.asarray([float(item["radius_sse_A2"]) for item in group], dtype=float)
        baseline_sse = np.asarray(
            [float(item["baseline_radius_sse_A2"]) for item in group], dtype=float
        )
        improvement = 0.0 if model == BASELINE_MODEL else 1.0 - float(np.sum(radius_sse)) / float(np.sum(baseline_sse))
        if model == BASELINE_MODEL:
            ci_low = 0.0
            ci_high = 0.0
        else:
            selected = rng.integers(0, len(group), size=(config.bootstrap_samples, len(group)))
            numerator = np.sum(radius_sse[selected], axis=1)
            denominator = np.sum(baseline_sse[selected], axis=1)
            bootstrap = 1.0 - numerator / denominator
            ci_low, ci_high = (float(item) for item in np.quantile(bootstrap, [0.025, 0.975]))
        output.append(
            {
                "model": model,
                "held_block_count": len(group),
                "held_frame_count": int(sum(int(item["frame_count"]) for item in group)),
                "radius_sse_A2": float(np.sum(radius_sse)),
                "baseline_radius_sse_A2": float(np.sum(baseline_sse)),
                "radius_sse_improvement_vs_baseline": improvement,
                "block_bootstrap_ci025": ci_low,
                "block_bootstrap_ci975": ci_high,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    return output


def _shift_features(
    count: np.ndarray,
    mark: np.ndarray,
    block_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_count = np.empty_like(count)
    shifted_mark = np.empty_like(mark)
    for block in np.unique(block_ids):
        indices = np.flatnonzero(block_ids == block)
        offset = int(rng.integers(0, len(indices)))
        shifted_count[indices] = np.roll(count[indices], offset)
        shifted_mark[indices] = np.roll(mark[indices], offset)
    return shifted_count, shifted_mark


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reversed_rank, index in enumerate(order[::-1], start=1):
        rank = count - reversed_rank + 1
        running = min(running, count * float(p_values[index]) / rank)
        adjusted[index] = running
    return [float(value) for value in adjusted]


def _full_kernel_rows(series: CaseSeries, config: ReconstructionConfig) -> list[dict[str, Any]]:
    target = np.diff(series.mean_radius)
    count = series.event_count[1:]
    mark = series.signed_mark[1:]
    rows: list[dict[str, Any]] = []
    for model in EVENT_MODELS:
        fitted = _fit_ridge(
            _features_for_model(model, count, mark, config.kernel_lag_frames), target, config.ridge_penalty
        )
        feature_names = ["event_count"] * (config.kernel_lag_frames + 1)
        if model == MARK_MODEL:
            feature_names += ["signed_local_residual_A"] * (config.kernel_lag_frames + 1)
        for position, coefficient in enumerate(fitted.coefficients):
            lag_frame = position % (config.kernel_lag_frames + 1)
            rows.append(
                {
                    "case_id": series.case_id,
                    "model": model,
                    "feature": feature_names[position],
                    "lag_frame": lag_frame,
                    "lag_ps": lag_frame * config.frame_interval_ps,
                    "coefficient_A_per_feature_unit": float(coefficient),
                    "full_trajectory_descriptive_fit_only": True,
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"inconsistent table columns: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _report(summary_rows: Sequence[dict[str, Any]], qualified: int) -> str:
    lines = [
        "# Local event-train to global mean-radius reconstruction",
        "",
        "This package performs a retrospective blocked reconstruction from accepted table inputs.",
        "It is not a causal impulse response, a physical rate, free-energy, friction, dissipation,",
        "avalanche, or replicate-level analysis.",
        "",
        "Each 200 ps held block is predicted by a ridge kernel trained outside that block and its",
        "adjacent embargo blocks. The null circularly shifts the coupled timing/mark event stream",
        "within every time block and refits the complete blocked procedure.",
        "",
        f"Qualified event-model/case reconstructions: {qualified}.",
        "",
        "| case | model | radius-MSE improvement vs baseline | bootstrap 95% diagnostic interval |",
        "| --- | --- | ---: | --- |",
    ]
    for row in summary_rows:
        if row["model"] == BASELINE_MODEL:
            continue
        lines.append(
            "| {case_id} | {model} | {score:.6g} | [{low:.6g}, {high:.6g}] |".format(
                case_id=row["case_id"],
                model=row["model"],
                score=float(row["radius_sse_improvement_vs_baseline"]),
                low=float(row["block_bootstrap_ci025"]),
                high=float(row["block_bootstrap_ci975"]),
            )
        )
    lines.append("")
    lines.append("A nonqualified model is not evidence that an event-associated component is absent.")
    return "\n".join(lines) + "\n"


def run_reconstruction(args: argparse.Namespace) -> dict[str, Any]:
    config = ReconstructionConfig(
        frame_interval_ps=float(args.frame_interval_ps),
        block_ps=float(args.block_ps),
        kernel_max_lag_ps=float(args.kernel_max_lag_ps),
        ridge_penalty=float(args.ridge_penalty),
        embargo_blocks=int(args.embargo_blocks),
        bootstrap_samples=int(args.bootstrap_samples),
        null_samples=int(args.null_samples),
        random_seed=int(args.random_seed),
        time_tolerance_ps=float(args.time_tolerance_ps),
    )
    if min(config.frame_interval_ps, config.block_ps, config.kernel_max_lag_ps) <= 0.0:
        raise ValueError("time values must be positive")
    if config.ridge_penalty < 0.0 or config.embargo_blocks < 0:
        raise ValueError("invalid ridge penalty or embargo")
    if config.bootstrap_samples < 1 or config.null_samples < 1:
        raise ValueError("resampling counts must be positive")
    if config.block_ps / config.frame_interval_ps < 2.0:
        raise ValueError("a time block must contain at least two frames")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    sources_table = Path(args.sources_table).resolve()
    events_table = Path(args.events_table).resolve()
    source_items = _source_rows(
        sources_table,
        case_column=args.case_column,
        source_path_column=args.source_path_column,
    )
    events = _read_csv(events_table)
    events_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in events:
        try:
            events_by_case[row[args.event_case_column]].append(row)
        except KeyError as exc:
            raise ValueError(f"event table missing case column: {args.event_case_column}") from exc

    cases: list[CaseSeries] = []
    for case_id, source in source_items:
        case_events = events_by_case.pop(case_id, [])
        if not case_events:
            raise ValueError(f"case has no events: {case_id}")
        cases.append(
            _load_case_series(
                case_id,
                source,
                case_events,
                config,
                time_column=args.time_column,
                arc_column=args.arc_column,
                mean_radius_column=args.mean_radius_column,
                event_time_column=args.event_time_column,
                event_arc_column=args.event_arc_column,
                event_mark_column=args.event_mark_column,
            )
        )
    if events_by_case:
        raise ValueError(f"events reference unknown cases: {sorted(events_by_case)}")

    rng = np.random.default_rng(config.random_seed)
    coverage_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    observed_by_key: dict[tuple[str, str], float] = {}

    for series in cases:
        coverage_rows.append(
            {
                "case_id": series.case_id,
                "source_frame_count": series.source_frame_count,
                "source_arc_count": series.source_arc_count,
                "time_start_ns": float(series.times_ns[0]),
                "time_stop_ns": float(series.times_ns[-1]),
                "time_block_count": len(np.unique(series.block_ids)),
                "event_count_total": series.event_count_total,
                "event_count_mapped": series.event_count_mapped,
                "event_mark_sum_A": float(np.sum(series.signed_mark)),
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
        case_folds, case_traces = _fold_rows(
            series, config, series.event_count, series.signed_mark, collect_traces=True
        )
        fold_rows.extend(case_folds)
        trace_rows.extend(case_traces)
        aggregates = _aggregate_rows(case_folds, config, rng)
        for row in aggregates:
            combined = {"case_id": series.case_id, **row}
            summary_rows.append(combined)
            if row["model"] in EVENT_MODELS:
                observed_by_key[(series.case_id, str(row["model"]))] = float(
                    row["radius_sse_improvement_vs_baseline"]
                )
        kernel_rows.extend(_full_kernel_rows(series, config))

        blocks = series.block_ids[1:]
        count = series.event_count[1:]
        mark = series.signed_mark[1:]
        for draw in range(config.null_samples):
            shifted_count, shifted_mark = _shift_features(count, mark, blocks, rng)
            padded_count = np.concatenate(([0.0], shifted_count))
            padded_mark = np.concatenate(([0.0], shifted_mark))
            null_folds, _ = _fold_rows(
                series, config, padded_count, padded_mark, collect_traces=False
            )
            for row in _aggregate_rows(null_folds, config, rng):
                if row["model"] in EVENT_MODELS:
                    null_rows.append(
                        {
                            "case_id": series.case_id,
                            "model": row["model"],
                            "draw": draw,
                            "radius_sse_improvement_vs_baseline": row[
                                "radius_sse_improvement_vs_baseline"
                            ],
                            "null_kind": "coupled_event_timing_and_mark_circular_shift_within_time_block",
                            "scientific_status": SCIENTIFIC_STATUS,
                        }
                    )

    null_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in null_rows:
        null_by_key[(str(row["case_id"]), str(row["model"]))].append(
            float(row["radius_sse_improvement_vs_baseline"])
        )
    primary_rows = [row for row in summary_rows if row["model"] in EVENT_MODELS]
    p_values: list[float] = []
    for row in primary_rows:
        key = (str(row["case_id"]), str(row["model"]))
        values = null_by_key[key]
        observed = observed_by_key[key]
        p_values.append((1.0 + sum(value >= observed for value in values)) / (1.0 + len(values)))
    for row, p_value, q_value in zip(primary_rows, p_values, _bh_adjust(p_values)):
        row["permutation_p"] = p_value
        row["bh_q"] = q_value
        row["qualified_reconstruction_information"] = int(
            float(row["radius_sse_improvement_vs_baseline"]) > 0.0
            and float(row["block_bootstrap_ci025"]) > 0.0
            and q_value <= 0.05
        )
    for row in summary_rows:
        if row["model"] == BASELINE_MODEL:
            row["permutation_p"] = math.nan
            row["bh_q"] = math.nan
            row["qualified_reconstruction_information"] = 0

    qualified = sum(int(row["qualified_reconstruction_information"]) for row in primary_rows)
    source_manifest = [
        {"case_id": case_id, "path": str(path), "sha256": _sha256(path)}
        for case_id, path in source_items
    ]
    manifest = {
        "status": "PASS",
        "scientific_status": SCIENTIFIC_STATUS,
        "sources_table": {"path": str(sources_table), "sha256": _sha256(sources_table)},
        "events_table": {"path": str(events_table), "sha256": _sha256(events_table)},
        "source_tables": source_manifest,
        "config": asdict(config),
        "column_mapping": {
            "case_column": args.case_column,
            "source_path_column": args.source_path_column,
            "time_column": args.time_column,
            "arc_column": args.arc_column,
            "mean_radius_column": args.mean_radius_column,
            "event_case_column": args.event_case_column,
            "event_time_column": args.event_time_column,
            "event_arc_column": args.event_arc_column,
            "event_mark_column": args.event_mark_column,
        },
    }
    summary = {
        "status": "PASS",
        "scientific_status": SCIENTIFIC_STATUS,
        "case_count": len(cases),
        "case_ids": [series.case_id for series in cases],
        "total_event_count": int(sum(series.event_count_total for series in cases)),
        "total_mapped_event_count": int(sum(series.event_count_mapped for series in cases)),
        "primary_test_count": len(primary_rows),
        "qualified_reconstruction_count": qualified,
        "null_control_row_count": len(null_rows),
        "block_bootstrap_note": "within-trajectory time-block diagnostic, not replicate uncertainty",
    }
    _write_csv(output / "case_coverage.csv", coverage_rows)
    _write_csv(output / "fold_reconstruction.csv", fold_rows)
    _write_csv(output / "heldout_traces.csv", trace_rows)
    _write_csv(output / "reconstruction_summary.csv", summary_rows)
    _write_csv(output / "kernel_coefficients.csv", kernel_rows)
    _write_csv(output / "null_controls.csv", null_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(_report(summary_rows, qualified), encoding="utf-8")
    return summary


def _plot_rows(path: Path) -> list[dict[str, str]]:
    return _read_csv(path)


def plot_reconstruction(results_dir: Path, output_dir: Path, font_path: Path | None = None) -> None:
    """Render generic review figures from a completed reconstruction result directory."""
    import matplotlib.pyplot as plt

    results_dir = results_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if font_path is not None and font_path.is_file():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    summary = _plot_rows(results_dir / "reconstruction_summary.csv")
    traces = _plot_rows(results_dir / "heldout_traces.csv")
    kernels = _plot_rows(results_dir / "kernel_coefficients.csv")
    coverage = _plot_rows(results_dir / "case_coverage.csv")
    cases = list(dict.fromkeys(row["case_id"] for row in summary))
    colors = {TIMING_MODEL: "#2b6cb0", MARK_MODEL: "#c05621"}

    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    positions = np.arange(len(cases), dtype=float)
    width = 0.34
    for index, model in enumerate(EVENT_MODELS):
        selected = [next(row for row in summary if row["case_id"] == case and row["model"] == model) for case in cases]
        scores = np.asarray([float(row["radius_sse_improvement_vs_baseline"]) for row in selected])
        lows = np.asarray([float(row["block_bootstrap_ci025"]) for row in selected])
        highs = np.asarray([float(row["block_bootstrap_ci975"]) for row in selected])
        axis.bar(positions + (index - 0.5) * width, scores, width=width, color=colors[model], label=model)
        axis.errorbar(
            positions + (index - 0.5) * width,
            scores,
            yerr=np.vstack((scores - lows, highs - scores)),
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=0.8,
        )
        for position, score, row in zip(positions + (index - 0.5) * width, scores, selected):
            if int(float(row["qualified_reconstruction_information"])):
                axis.text(position, score, "*", ha="center", va="bottom", fontsize=12)
    axis.axhline(0.0, color="black", lw=0.8)
    axis.set_xticks(positions, cases, rotation=20, ha="right")
    axis.set_ylabel("Held-out $m=0$ SSE improvement")
    axis.set_title("Blocked event-train reconstruction (diagnostic intervals)")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "01_blocked_reconstruction_score.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 5.6), sharex=False)
    for axis, case in zip(axes.ravel(), cases):
        available = sorted({int(float(row["held_block_id"])) for row in traces if row["case_id"] == case})
        block = available[len(available) // 2]
        selected = [row for row in traces if row["case_id"] == case and int(float(row["held_block_id"])) == block]
        time_ps = [(float(row["time_ns"]) - float(selected[0]["time_ns"])) * 1000.0 for row in selected]
        axis.plot(time_ps, [float(row["observed_relative_mean_radius_A"]) for row in selected], color="black", lw=1.2, label="observed")
        axis.plot(time_ps, [float(row["baseline_relative_mean_radius_A"]) for row in selected], color="#666666", lw=1.0, label="baseline")
        axis.plot(time_ps, [float(row["timing_relative_mean_radius_A"]) for row in selected], color=colors[TIMING_MODEL], lw=1.0, label="timing")
        axis.plot(time_ps, [float(row["marked_relative_mean_radius_A"]) for row in selected], color=colors[MARK_MODEL], lw=1.0, label="marked")
        axis.set_title(f"{case}, held block {block}", fontsize=9)
        axis.set_xlabel("Time within block (ps)")
        axis.set_ylabel("Relative mean radius (Å)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle("Predefined median held-out blocks; traces are retrospective diagnostics", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_dir / "02_heldout_reconstruction_traces.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 5.6), sharex=True)
    for axis, case in zip(axes.ravel(), cases):
        selected = [row for row in kernels if row["case_id"] == case and row["model"] == MARK_MODEL]
        for feature, color in (("event_count", colors[TIMING_MODEL]), ("signed_local_residual_A", colors[MARK_MODEL])):
            values = [row for row in selected if row["feature"] == feature]
            axis.plot(
                [float(row["lag_ps"]) for row in values],
                [float(row["coefficient_A_per_feature_unit"]) for row in values],
                marker="o",
                ms=2.5,
                lw=1.0,
                color=color,
                label=feature,
            )
        axis.axhline(0.0, color="black", lw=0.6)
        axis.set_title(case, fontsize=9)
        axis.set_xlabel("Lag (ps)")
        axis.set_ylabel("Full-fit increment coefficient (Å)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle("Descriptive full-trajectory kernels; not causal response functions", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_dir / "03_descriptive_event_kernels.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    mapped = [int(float(next(row for row in coverage if row["case_id"] == case)["event_count_mapped"])) for case in cases]
    block_counts = [int(float(next(row for row in coverage if row["case_id"] == case)["time_block_count"])) for case in cases]
    axes[0].bar(cases, mapped, color="#4a5568")
    axes[0].set_ylabel("Mapped accepted events")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(cases, block_counts, color="#718096")
    axes[1].set_ylabel("200 ps time blocks")
    axes[1].tick_params(axis="x", rotation=20)
    figure.suptitle("Fixed accepted-table coverage", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_dir / "04_event_train_coverage.png", dpi=300)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sources-table", type=Path)
    group.add_argument("--plot-results-dir", type=Path)
    parser.add_argument("--events-table", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-output-dir", type=Path)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--source-path-column", default="arc_kinematics")
    parser.add_argument("--time-column", default="time_ns")
    parser.add_argument("--arc-column", default="arc_index")
    parser.add_argument("--mean-radius-column", default="mean_radius_component_A")
    parser.add_argument("--event-case-column", default="case_id")
    parser.add_argument("--event-time-column", default="transition_time_ns")
    parser.add_argument("--event-arc-column", default="primary_arc_index")
    parser.add_argument("--event-mark-column", default="local_residual_displacement_A")
    parser.add_argument("--frame-interval-ps", type=float, default=0.5)
    parser.add_argument("--block-ps", type=float, default=200.0)
    parser.add_argument("--kernel-max-lag-ps", type=float, default=20.0)
    parser.add_argument("--ridge-penalty", type=float, default=1.0)
    parser.add_argument("--embargo-blocks", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--null-samples", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260908)
    parser.add_argument("--time-tolerance-ps", type=float, default=1.0e-6)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.plot_results_dir is not None:
        if args.plot_output_dir is None:
            raise ValueError("--plot-output-dir is required with --plot-results-dir")
        plot_reconstruction(args.plot_results_dir, args.plot_output_dir, args.font_path)
        return 0
    if args.events_table is None or args.output_dir is None:
        raise ValueError("--events-table and --output-dir are required for reconstruction")
    print(json.dumps(run_reconstruction(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
