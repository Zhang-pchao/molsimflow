"""Summarize the shape of an accepted event-aligned circular response.

The analysis subtracts an accepted cellwise null mean, averages complete
events within whole time blocks, and reports threshold-free spatial and
sampled-lag shape descriptors.  The outputs are retrospective stability
diagnostics, not causal or replicate-level susceptibilities.
"""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import numpy as np

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_NULL_CENTERED_RESIDUAL_RESPONSE_SHAPE_NOT_CAUSAL_"
    "PROPAGATION_INTRINSIC_LENGTH_OR_REPLICATE_LEVEL_EVIDENCE"
)
METRICS = (
    "fast_spatial_extent_arcs",
    "fast_far_response_fraction",
    "far_slow_response_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_window(raw: str) -> tuple[float, float]:
    fields = raw.split(":")
    if len(fields) != 2:
        raise ValueError(f"invalid lag window: {raw!r}")
    start, end = map(float, fields)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"invalid lag window: {raw!r}")
    return start, end


def _window_name(
    lag: float,
    fast_window: tuple[float, float],
    slow_window: tuple[float, float],
) -> Optional[str]:
    selected = []
    if fast_window[0] < lag <= fast_window[1]:
        selected.append("fast")
    if slow_window[0] < lag <= slow_window[1]:
        selected.append("slow")
    if len(selected) > 1:
        raise ValueError(f"lag {lag} belongs to overlapping windows")
    return selected[0] if selected else None


def _read_null_means(
    path: Path,
    *,
    field: str,
    fast_window: tuple[float, float],
    slow_window: tuple[float, float],
    minimum_distance: int,
) -> dict[tuple[str, float, int], float]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "case_id",
            "field",
            "lag_ps",
            "arc_offset_signed",
            "arc_distance",
            "null_mean",
        }
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        output = {}
        for row in reader:
            if row["field"] != field:
                continue
            lag = float(row["lag_ps"])
            window = _window_name(lag, fast_window, slow_window)
            offset = int(row["arc_offset_signed"])
            distance = int(row["arc_distance"])
            if window is None or distance < minimum_distance:
                continue
            if abs(offset) != distance:
                raise ValueError(f"{path}: offset/distance mismatch")
            value = float(row["null_mean"])
            if not math.isfinite(value):
                raise ValueError(f"{path}: nonfinite null_mean")
            key = (row["case_id"], lag, offset)
            if key in output:
                raise ValueError(f"{path}: duplicate null cell {key}")
            output[key] = value
    if not output:
        raise ValueError(f"{path}: no selected null cells")
    return output


def _read_event_blocks(
    path: Path,
    *,
    field: str,
    fast_window: tuple[float, float],
    slow_window: tuple[float, float],
    minimum_distance: int,
) -> tuple[
    dict[tuple[str, int, float, int], list[float]],
    dict[str, set[int]],
    dict[str, set[tuple[int, int]]],
]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "case_id",
            "event_id",
            "time_block_200ps",
            "field",
            "lag_ps",
            "arc_offset_signed",
            "arc_distance",
            "aligned_change",
        }
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        aggregates: dict[tuple[str, int, float, int], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        blocks: dict[str, set[int]] = defaultdict(set)
        events: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for row in reader:
            if row["field"] != field:
                continue
            lag = float(row["lag_ps"])
            window = _window_name(lag, fast_window, slow_window)
            offset = int(row["arc_offset_signed"])
            distance = int(row["arc_distance"])
            if window is None or distance < minimum_distance:
                continue
            if abs(offset) != distance:
                raise ValueError(f"{path}: offset/distance mismatch")
            value = float(row["aligned_change"])
            if not math.isfinite(value):
                raise ValueError(f"{path}: nonfinite aligned_change")
            case_id = row["case_id"]
            event_id = int(row["event_id"])
            block = int(row["time_block_200ps"])
            key = (case_id, block, lag, offset)
            aggregates[key][0] += value
            aggregates[key][1] += 1.0
            blocks[case_id].add(block)
            events[case_id].add((event_id, block))
    if not aggregates:
        raise ValueError(f"{path}: no selected event rows")
    return dict(aggregates), dict(blocks), dict(events)


def _metrics(
    residual: np.ndarray,
    *,
    distances: np.ndarray,
    fast_mask: np.ndarray,
    slow_mask: np.ndarray,
    far_mask: np.ndarray,
    denominator_floor: float,
) -> dict[str, np.ndarray]:
    amplitude = np.abs(residual)
    fast = amplitude[..., fast_mask]
    fast_distances = distances[fast_mask]
    fast_total = np.sum(fast, axis=-1)
    fast_far = np.sum(amplitude[..., fast_mask & far_mask], axis=-1)
    slow_far = np.sum(amplitude[..., slow_mask & far_mask], axis=-1)
    far_total = fast_far + slow_far
    return {
        "fast_spatial_extent_arcs": np.divide(
            np.sum(fast * fast_distances, axis=-1),
            fast_total,
            out=np.full_like(fast_total, np.nan, dtype=float),
            where=fast_total > denominator_floor,
        ),
        "fast_far_response_fraction": np.divide(
            fast_far,
            fast_total,
            out=np.full_like(fast_total, np.nan, dtype=float),
            where=fast_total > denominator_floor,
        ),
        "far_slow_response_fraction": np.divide(
            slow_far,
            far_total,
            out=np.full_like(far_total, np.nan, dtype=float),
            where=far_total > denominator_floor,
        ),
    }


def _quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if len(finite) else math.nan


def run_analysis(
    event_response_table: Path,
    map_table: Path,
    output_dir: Path,
    *,
    field: str,
    fast_window: tuple[float, float],
    slow_window: tuple[float, float],
    minimum_distance: int,
    far_minimum_distance: int,
    bootstrap_samples: int,
    random_seed: int,
    denominator_floor: float = 1.0e-12,
) -> dict[str, object]:
    """Compute null-centered response-shape metrics with block resampling."""

    if fast_window[1] > slow_window[0]:
        raise ValueError("fast and slow windows overlap")
    if minimum_distance < 0 or far_minimum_distance <= minimum_distance:
        raise ValueError("distance bounds are invalid")
    if bootstrap_samples < 20 or denominator_floor <= 0.0:
        raise ValueError("bootstrap_samples or denominator_floor is invalid")

    null_means = _read_null_means(
        map_table,
        field=field,
        fast_window=fast_window,
        slow_window=slow_window,
        minimum_distance=minimum_distance,
    )
    aggregates, case_blocks, case_events = _read_event_blocks(
        event_response_table,
        field=field,
        fast_window=fast_window,
        slow_window=slow_window,
        minimum_distance=minimum_distance,
    )
    cases = sorted(case_blocks)
    if set(cases) != {key[0] for key in null_means}:
        raise ValueError("event and null tables contain different case sets")

    seed_sequences = np.random.SeedSequence(random_seed).spawn(len(cases))
    case_rows = []
    profile_rows = []
    observed_by_case = {}
    bootstrap_by_case = {}
    for case_id, seed_sequence in zip(cases, seed_sequences):
        blocks = sorted(case_blocks[case_id])
        if len(blocks) < 2:
            raise ValueError(f"{case_id}: fewer than two populated blocks")
        cells = sorted({(key[2], key[3]) for key in aggregates if key[0] == case_id})
        expected_cells = {(key[1], key[2]) for key in null_means if key[0] == case_id}
        if set(cells) != expected_cells:
            raise ValueError(f"{case_id}: event and null cell sets differ")
        block_index = {value: index for index, value in enumerate(blocks)}
        cell_index = {value: index for index, value in enumerate(cells)}
        sums = np.zeros((len(blocks), len(cells)), dtype=float)
        counts = np.zeros_like(sums)
        for key, (total, count) in aggregates.items():
            if key[0] != case_id:
                continue
            row = block_index[key[1]]
            column = cell_index[(key[2], key[3])]
            sums[row, column] = total
            counts[row, column] = count
        if np.any(counts <= 0.0) or np.any(np.ptp(counts, axis=1) > 0.0):
            raise ValueError(f"{case_id}: incomplete event-cell coverage within a block")

        null = np.asarray([null_means[(case_id, lag, offset)] for lag, offset in cells])
        lags = np.asarray([lag for lag, _ in cells], dtype=float)
        distances = np.asarray([abs(offset) for _, offset in cells], dtype=float)
        fast_mask = (fast_window[0] < lags) & (lags <= fast_window[1])
        slow_mask = (slow_window[0] < lags) & (lags <= slow_window[1])
        far_mask = distances >= far_minimum_distance

        observed_residual = np.sum(sums, axis=0) / np.sum(counts, axis=0) - null
        observed_metrics = _metrics(
            observed_residual,
            distances=distances,
            fast_mask=fast_mask,
            slow_mask=slow_mask,
            far_mask=far_mask,
            denominator_floor=denominator_floor,
        )
        rng = np.random.default_rng(seed_sequence)
        selected = rng.integers(0, len(blocks), size=(bootstrap_samples, len(blocks)))
        selected_sums = np.sum(sums[selected], axis=1)
        selected_counts = np.sum(counts[selected], axis=1)
        bootstrap_residual = selected_sums / selected_counts - null
        bootstrap_metrics = _metrics(
            bootstrap_residual,
            distances=distances,
            fast_mask=fast_mask,
            slow_mask=slow_mask,
            far_mask=far_mask,
            denominator_floor=denominator_floor,
        )
        observed_by_case[case_id] = {
            metric: float(values) for metric, values in observed_metrics.items()
        }
        bootstrap_by_case[case_id] = bootstrap_metrics
        for metric in METRICS:
            samples = bootstrap_metrics[metric]
            case_rows.append(
                {
                    "case_id": case_id,
                    "field": field,
                    "metric": metric,
                    "value": observed_by_case[case_id][metric],
                    "block_bootstrap_ci025": _quantile(samples, 0.025),
                    "block_bootstrap_ci975": _quantile(samples, 0.975),
                    "finite_bootstrap_count": int(np.count_nonzero(np.isfinite(samples))),
                    "bootstrap_samples": bootstrap_samples,
                    "event_count": len(case_events[case_id]),
                    "block_count": len(blocks),
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
        for window, mask in (("fast", fast_mask), ("slow", slow_mask)):
            for distance in sorted(set(distances[mask])):
                selected_cells = mask & (distances == distance)
                values = observed_residual[selected_cells]
                profile_rows.append(
                    {
                        "case_id": case_id,
                        "field": field,
                        "window": window,
                        "arc_distance": int(distance),
                        "cell_count": int(np.count_nonzero(selected_cells)),
                        "mean_null_centered_response_A": float(np.mean(values)),
                        "mean_absolute_null_centered_response_A": float(np.mean(np.abs(values))),
                        "scientific_status": SCIENTIFIC_STATUS,
                    }
                )

    contrast_rows = []
    for case_a, case_b in itertools.combinations(cases, 2):
        for metric in METRICS:
            samples = bootstrap_by_case[case_a][metric] - bootstrap_by_case[case_b][metric]
            contrast_rows.append(
                {
                    "case_a": case_a,
                    "case_b": case_b,
                    "field": field,
                    "metric": metric,
                    "case_a_minus_case_b": (
                        observed_by_case[case_a][metric] - observed_by_case[case_b][metric]
                    ),
                    "block_bootstrap_ci025": _quantile(samples, 0.025),
                    "block_bootstrap_ci975": _quantile(samples, 0.975),
                    "finite_bootstrap_count": int(np.count_nonzero(np.isfinite(samples))),
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "response_shape_metrics.csv", case_rows)
    _write_csv(output_dir / "response_shape_contrasts.csv", contrast_rows)
    _write_csv(output_dir / "response_distance_profile.csv", profile_rows)
    summary = {
        "status": "PASS",
        "case_count": len(cases),
        "case_ids": cases,
        "field": field,
        "event_count": sum(len(values) for values in case_events.values()),
        "metric_count": len(case_rows),
        "contrast_count": len(contrast_rows),
        "profile_row_count": len(profile_rows),
        "metrics": list(METRICS),
        "fast_window": list(fast_window),
        "slow_window": list(slow_window),
        "minimum_distance": minimum_distance,
        "far_minimum_distance": far_minimum_distance,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "event_response_table": {
            "path": str(Path(event_response_table).resolve()),
            "sha256": _sha256(event_response_table),
        },
        "map_table": {
            "path": str(Path(map_table).resolve()),
            "sha256": _sha256(map_table),
        },
        "field": field,
        "fast_window": list(fast_window),
        "slow_window": list(slow_window),
        "minimum_distance": minimum_distance,
        "far_minimum_distance": far_minimum_distance,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "denominator_floor": denominator_floor,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(
            (
                "# Null-centered residual response shape",
                "",
                f"- Cases: {len(cases)}",
                f"- Accepted event anchors: {summary['event_count']}",
                f"- Field: `{field}`",
                f"- Whole-block bootstrap draws: {bootstrap_samples}",
                "",
                (
                    "Metrics describe the magnitude-weighted spatial and sampled-lag "
                    "shape of an accepted response after subtraction of its cellwise "
                    "circular-shift null mean. They are not causal propagation, an "
                    "intrinsic length/time, or replicate-level uncertainty."
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-response-table", type=Path, required=True)
    parser.add_argument("--map-table", type=Path, required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--fast-window", default="0:5")
    parser.add_argument("--slow-window", default="5:50")
    parser.add_argument("--minimum-distance", type=int, default=1)
    parser.add_argument("--far-minimum-distance", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260904)
    parser.add_argument("--denominator-floor", type=float, default=1.0e-12)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        args.event_response_table,
        args.map_table,
        args.output_dir,
        field=args.field,
        fast_window=_parse_window(args.fast_window),
        slow_window=_parse_window(args.slow_window),
        minimum_distance=args.minimum_distance,
        far_minimum_distance=args.far_minimum_distance,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
        denominator_floor=args.denominator_floor,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
