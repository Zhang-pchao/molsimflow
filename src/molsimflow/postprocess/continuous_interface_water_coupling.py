"""Couple continuous interface fluctuations to local-water observables.

The same engine supports nanobubble footprint/shape series and nanodroplet
shape series.  It analyzes aligned first differences with block circular-shift
controls and may tabulate nearest-site residence segments.  It does not detect
slip events or establish causality.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

SCIENTIFIC_STATUS = (
    "SINGLE_TRAJECTORY_CONTINUOUS_INTERFACE_WATER_COFLUCTUATION_"
    "NOT_EVENT_CAUSALITY_SLIP_FREE_ENERGY_OR_PHYSICAL_RATE_EVIDENCE"
)


def _csv_rows(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if Path(path).suffix == ".gz":
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            return rows, tuple(reader.fieldnames or ())
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, tuple(reader.fieldnames or ())


def _require(path: Path, fields: Sequence[str], required: set[str]) -> None:
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty required table: {path}")
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


def _metrics(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("metric lists must be nonempty and contain no duplicates")
    return values


def load_frame_series(
    path: Path, metrics: Sequence[str]
) -> tuple[list[int], np.ndarray, dict[str, np.ndarray]]:
    rows, fields = _csv_rows(path)
    _require(path, fields, {"step", "time_ns", *metrics})
    if not rows:
        raise ValueError(f"{path}: empty frame table")
    seen: set[int] = set()
    parsed: list[tuple[int, float, dict[str, float]]] = []
    for row in rows:
        step = int(row["step"])
        if step in seen:
            raise ValueError(f"{path}: duplicate step {step}")
        seen.add(step)
        parsed.append(
            (step, float(row["time_ns"]), {metric: float(row[metric]) for metric in metrics})
        )
    parsed.sort(key=lambda item: item[0])
    steps = [item[0] for item in parsed]
    times = np.asarray([item[1] for item in parsed], dtype=float)
    values = {
        metric: np.asarray([item[2][metric] for item in parsed], dtype=float)
        for metric in metrics
    }
    return steps, times, values


def load_aggregated_series(
    path: Path,
    metrics: Sequence[str],
    *,
    residence_group: str | None = None,
    residence_id: str | None = None,
    residence_type: str | None = None,
) -> tuple[
    list[int],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, list[tuple[int, float, str, str]]],
]:
    rows, fields = _csv_rows(path)
    required = {"step", "time_ns", *metrics}
    if residence_group or residence_id or residence_type:
        if not residence_group or not residence_id:
            raise ValueError("residence_group and residence_id must be supplied together")
        required.update({residence_group, residence_id})
        if residence_type:
            required.add(residence_type)
    _require(path, fields, required)
    sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    times: dict[int, float] = {}
    residence: dict[str, list[tuple[int, float, str, str]]] = defaultdict(list)
    for row in rows:
        step, time = int(row["step"]), float(row["time_ns"])
        if step in times and not math.isclose(times[step], time, abs_tol=1.0e-12):
            raise ValueError(f"{path}: inconsistent time at step {step}")
        times[step] = time
        for metric in metrics:
            value = float(row[metric])
            if math.isfinite(value):
                sums[step][metric] += value
                counts[step][metric] += 1
        if residence_group and residence_id:
            group = str(row[residence_group])
            site_type = str(row[residence_type]) if residence_type else ""
            residence[group].append((step, time, str(row[residence_id]), site_type))
    steps = sorted(times)
    if not steps:
        raise ValueError(f"{path}: empty response table")
    values: dict[str, np.ndarray] = {}
    for metric in metrics:
        series = np.asarray(
            [
                sums[step][metric] / counts[step][metric]
                if counts[step][metric]
                else math.nan
                for step in steps
            ],
            dtype=float,
        )
        if not np.any(np.isfinite(series)):
            raise ValueError(f"{path}: metric {metric!r} has no finite aggregate")
        values[metric] = series
    return steps, np.asarray([times[step] for step in steps]), values, residence


def validate_grid(left_steps: Sequence[int], right_steps: Sequence[int], times: np.ndarray) -> float:
    if list(left_steps) != list(right_steps):
        raise ValueError("driver and response timestep supports differ")
    if len(left_steps) < 4 or len(times) != len(left_steps):
        raise ValueError("at least four aligned frames are required")
    deltas = np.diff(times) * 1000.0
    if np.any(~np.isfinite(deltas)) or np.any(deltas <= 0.0):
        raise ValueError("frame times must increase")
    interval = float(np.median(deltas))
    if not np.allclose(deltas, interval, rtol=0.0, atol=max(1.0e-9, interval * 1.0e-6)):
        raise ValueError("frame times are not regular")
    return interval


def summarize_series(
    role: str, series: Mapping[str, np.ndarray]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric, values in series.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        output.append(
            {
                "role": role,
                "metric": metric,
                "finite_frame_count": len(finite),
                "mean": float(np.mean(finite)),
                "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan,
                "q025": float(np.quantile(finite, 0.025)),
                "median": float(np.median(finite)),
                "q975": float(np.quantile(finite, 0.975)),
            }
        )
    return output


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[int, float]:
    finite = np.isfinite(left) & np.isfinite(right)
    count = int(np.count_nonzero(finite))
    if count < 3:
        return count, math.nan
    x, y = left[finite], right[finite]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return count, math.nan
    return count, float(np.corrcoef(x, y)[0, 1])


def _lag_pair(left: np.ndarray, right: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if abs(lag) >= len(left) - 2:
        raise ValueError("lag leaves fewer than three frames")
    if lag > 0:
        return left[:-lag], right[lag:]
    if lag < 0:
        return left[-lag:], right[:lag]
    return left, right


def lagged_increment_coupling(
    drivers: Mapping[str, np.ndarray],
    responses: Mapping[str, np.ndarray],
    lags: Sequence[int],
    *,
    frame_interval_ps: float,
    block_frames: int,
    null_samples: int,
    random_seed: int,
) -> list[dict[str, object]]:
    """Compare lagged first-difference correlations to circular-shift controls."""

    length = len(next(iter(drivers.values()))) - 1
    block_count = length // block_frames
    if block_frames < 1 or block_count < 2 or null_samples < 1:
        raise ValueError("increment grid needs at least two positive-size null blocks")
    rng = np.random.default_rng(random_seed)
    shifts = rng.integers(1, block_count, size=null_samples) * block_frames
    output: list[dict[str, object]] = []
    for driver_name, driver_values in drivers.items():
        driver = np.diff(np.asarray(driver_values, dtype=float))
        for response_name, response_values in responses.items():
            response = np.diff(np.asarray(response_values, dtype=float))
            shifted_responses = [np.roll(response, int(shift)) for shift in shifts]
            for lag in lags:
                left, right = _lag_pair(driver, response, lag)
                count, observed = _correlation(left, right)
                null = np.asarray(
                    [_correlation(*_lag_pair(driver, shifted, lag))[1] for shifted in shifted_responses],
                    dtype=float,
                )
                null = null[np.isfinite(null)]
                null_mean = float(np.mean(null)) if len(null) else math.nan
                p_value = (
                    (1 + np.count_nonzero(np.abs(null - null_mean) >= abs(observed - null_mean)))
                    / (len(null) + 1)
                    if len(null) and math.isfinite(observed)
                    else math.nan
                )
                output.append(
                    {
                        "driver_metric": driver_name,
                        "response_metric": response_name,
                        "lag_frames": lag,
                        "lag_ps": lag * frame_interval_ps,
                        "paired_increment_count": count,
                        "observed_pearson_r": observed,
                        "null_sample_count": len(null),
                        "null_mean_r": null_mean,
                        "null_q025_r": float(np.quantile(null, 0.025)) if len(null) else math.nan,
                        "null_q975_r": float(np.quantile(null, 0.975)) if len(null) else math.nan,
                        "empirical_two_sided_p": p_value,
                        "outside_null_q95": (
                            bool(observed < np.quantile(null, 0.025) or observed > np.quantile(null, 0.975))
                            if len(null) and math.isfinite(observed)
                            else False
                        ),
                        "lag_convention": "positive_lag_means_response_follows_driver",
                        "inference_status": (
                            "within_trajectory_increment_correlation_with_circular_shift_control"
                        ),
                    }
                )
    return output


def residence_segments(
    records: Mapping[str, Sequence[tuple[int, float, str, str]]], frame_interval_ps: float
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for group, values in sorted(records.items()):
        ordered = sorted(values)
        if len({step for step, *_ in ordered}) != len(ordered):
            raise ValueError(f"residence group {group!r} has duplicate steps")
        start = 0
        for index in range(1, len(ordered) + 1):
            boundary = index == len(ordered)
            if not boundary:
                previous, current = ordered[index - 1], ordered[index]
                consecutive = math.isclose(
                    (current[1] - previous[1]) * 1000.0,
                    frame_interval_ps,
                    rel_tol=0.0,
                    abs_tol=max(1.0e-9, frame_interval_ps * 1.0e-6),
                )
                boundary = current[2] != previous[2] or not consecutive
            if not boundary:
                continue
            segment = ordered[start:index]
            output.append(
                {
                    "group": group,
                    "site_id": segment[0][2],
                    "site_type": segment[0][3],
                    "start_step": segment[0][0],
                    "end_step": segment[-1][0],
                    "start_time_ns": segment[0][1],
                    "end_time_ns": segment[-1][1],
                    "frame_count": len(segment),
                    "residence_ps": len(segment) * frame_interval_ps,
                    "interpretation": (
                        "continuous_nearest_site_identity_segment_not_a_pinning_or_slip_event"
                    ),
                }
            )
            start = index
    return output


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    driver_metrics, response_metrics = _metrics(args.driver_metrics), _metrics(
        args.response_metrics
    )
    lags = tuple(int(value.strip()) for value in str(args.lag_frames).split(","))
    if not lags or len(set(lags)) != len(lags):
        raise ValueError("lag_frames must be a nonempty unique integer list")
    driver_path, response_path = Path(args.driver_table), Path(args.response_table)
    left_steps, left_times, drivers = load_frame_series(driver_path, driver_metrics)
    right_steps, right_times, responses, residence_records = load_aggregated_series(
        response_path,
        response_metrics,
        residence_group=args.residence_group,
        residence_id=args.residence_id,
        residence_type=args.residence_type,
    )
    if not np.allclose(left_times, right_times, rtol=0.0, atol=1.0e-12):
        raise ValueError("driver and response time values differ")
    frame_interval_ps = validate_grid(left_steps, right_steps, left_times)
    series_summary = summarize_series("driver", drivers) + summarize_series(
        "response", responses
    )
    coupling = lagged_increment_coupling(
        drivers,
        responses,
        lags,
        frame_interval_ps=frame_interval_ps,
        block_frames=args.block_frames,
        null_samples=args.null_samples,
        random_seed=args.random_seed,
    )
    residence = residence_segments(residence_records, frame_interval_ps)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "continuous_metric_summary.csv", series_summary)
    _write_csv(output / "lagged_increment_coupling.csv", coupling)
    if residence:
        _write_csv(output / "nearest_site_residence_segments.csv", residence)
    finite_rows = [row for row in coupling if math.isfinite(float(row["observed_pearson_r"]))]
    strongest = max(finite_rows, key=lambda row: abs(float(row["observed_pearson_r"])))
    residence_values = np.asarray([float(row["residence_ps"]) for row in residence])
    summary = {
        "status": "PASS",
        "case_id": args.case_id,
        "system_kind": args.system_kind,
        "frame_count": len(left_steps),
        "first_step": left_steps[0],
        "last_step": left_steps[-1],
        "frame_interval_ps": frame_interval_ps,
        "driver_metrics": list(driver_metrics),
        "response_metrics": list(response_metrics),
        "lag_frames": list(lags),
        "coupling_rows": len(coupling),
        "coupling_rows_outside_null_q95": sum(
            bool(row["outside_null_q95"]) for row in coupling
        ),
        "strongest_absolute_increment_correlation": strongest,
        "residence_segment_count": len(residence),
        "median_nearest_site_residence_ps": (
            float(np.median(residence_values)) if len(residence_values) else None
        ),
        "q95_nearest_site_residence_ps": (
            float(np.quantile(residence_values, 0.95)) if len(residence_values) else None
        ),
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "case_id": args.case_id,
        "system_kind": args.system_kind,
        "inputs": {
            "driver_table": {
                "path": str(driver_path.resolve()),
                "sha256": _sha256(driver_path),
            },
            "response_table": {
                "path": str(response_path.resolve()),
                "sha256": _sha256(response_path),
            },
        },
        "driver_metrics": list(driver_metrics),
        "response_metrics": list(response_metrics),
        "transform": "first_difference",
        "response_aggregation": "unweighted_finite_mean_per_step",
        "lag_convention": "positive_lag_means_response_follows_driver",
        "block_frames": args.block_frames,
        "null_samples": args.null_samples,
        "random_seed": args.random_seed,
        "residence_group": args.residence_group,
        "residence_id": args.residence_id,
        "residence_type": args.residence_type,
        "residence_definition": (
            "consecutive_frames_with_unchanged_nearest_site_identity_not_an_event_catalog"
            if args.residence_id
            else None
        ),
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "REPORT.md").write_text(
        "\n".join(
            (
                f"# Continuous interface-water coupling: {args.case_id}",
                "",
                f"- System: {args.system_kind}",
                f"- Aligned frames: {len(left_steps)}",
                f"- Lagged increment comparisons: {len(coupling)}",
                f"- Nearest-site residence segments: {len(residence)}",
                "",
                (
                    "The analysis uses continuous fluctuations and within-trajectory controls. "
                    "It does not define slip events, causality, free energies, or physical rates."
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--system-kind", choices=("nanobubble", "nanodroplet"), required=True)
    parser.add_argument("--driver-table", type=Path, required=True)
    parser.add_argument("--response-table", type=Path, required=True)
    parser.add_argument("--driver-metrics", required=True)
    parser.add_argument("--response-metrics", required=True)
    parser.add_argument("--lag-frames", default="-20,-10,0,10,20")
    parser.add_argument("--block-frames", type=int, default=200)
    parser.add_argument("--null-samples", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260904)
    parser.add_argument("--residence-group")
    parser.add_argument("--residence-id")
    parser.add_argument("--residence-type")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.block_frames < 1 or args.null_samples < 1:
        raise ValueError("block_frames and null_samples must be positive")
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
