"""Synthesize TPCL pair excess, global modes, and event-local water response.

All outputs are conditional summaries from one trajectory.  The response
surface is an observed-minus-circular-shift event-pair count, not a causal
response function or an avalanche propagator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

SCIENTIFIC_STATUS = (
    "SINGLE_TRAJECTORY_CONDITIONAL_CASCADE_RESPONSE_SYNTHESIS_"
    "NOT_CAUSAL_AVALANCHE_INTRINSIC_LENGTH_OR_PHYSICAL_RATE_EVIDENCE"
)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def build_response_surface(
    pair_rows: Sequence[Mapping[str, object]], arc_length_A: float
) -> list[dict[str, object]]:
    """Return the pair-count excess surface on the existing P1 bins."""

    if not math.isfinite(arc_length_A) or arc_length_A <= 0.0:
        raise ValueError("arc_length_A must be positive and finite")
    output: list[dict[str, object]] = []
    seen: set[tuple[int, int, float, float]] = set()
    for row in pair_rows:
        arc_start = int(row["arc_distance_start_bins"])
        arc_end = int(row["arc_distance_end_bins_exclusive"])
        lag_start = float(row["lag_start_ps"])
        lag_end = float(row["lag_end_ps"])
        key = (arc_start, arc_end, lag_start, lag_end)
        if key in seen:
            raise ValueError(f"duplicate response bin {key}")
        seen.add(key)
        if arc_start < 0 or arc_end <= arc_start or lag_start < 0.0 or lag_end <= lag_start:
            raise ValueError(f"invalid response bin {key}")
        observed = float(row["observed_pairs"])
        null_mean = float(row["null_mean_pairs"])
        null_q025 = float(row["null_q025_pairs"])
        null_q975 = float(row["null_q975_pairs"])
        informative = _as_bool(row["informative_null_count"])
        excess = observed - null_mean
        output.append(
            {
                "arc_distance_start_bins": arc_start,
                "arc_distance_end_bins_exclusive": arc_end,
                "arc_distance_midpoint_bins": 0.5 * (arc_start + arc_end),
                "arc_distance_midpoint_A": 0.5 * (arc_start + arc_end) * arc_length_A,
                "lag_start_ps": lag_start,
                "lag_end_ps": lag_end,
                "lag_midpoint_ps": 0.5 * (lag_start + lag_end),
                "observed_pairs": observed,
                "null_mean_pairs": null_mean,
                "null_q025_pairs": null_q025,
                "null_q975_pairs": null_q975,
                "pair_excess_R": excess,
                "fractional_excess": excess / null_mean if null_mean > 0.0 else math.nan,
                "poisson_scaled_excess": excess / math.sqrt(null_mean) if null_mean > 0.0 else math.nan,
                "informative_null_count": informative,
                "above_null_q975": informative and observed > null_q975,
                "inference_status": "descriptive_within_trajectory_circular_shift_comparison",
            }
        )
    if not output:
        raise ValueError("pair response surface is empty")
    return output


def summarize_response_scales(
    surface: Sequence[Mapping[str, object]], arc_length_A: float
) -> dict[str, object]:
    """Summarize positive cross-arc excess without fitting an intrinsic scale."""

    positive = [
        row
        for row in surface
        if int(row["arc_distance_start_bins"]) >= 1
        and bool(row["informative_null_count"])
        and float(row["pair_excess_R"]) > 0.0
    ]
    total = sum(float(row["pair_excess_R"]) for row in positive)
    weighted_bins = (
        sum(
            float(row["pair_excess_R"]) * float(row["arc_distance_midpoint_bins"])
            for row in positive
        )
        / total
        if total > 0.0
        else math.nan
    )
    weighted_time = (
        sum(
            float(row["pair_excess_R"]) * float(row["lag_midpoint_ps"])
            for row in positive
        )
        / total
        if total > 0.0
        else math.nan
    )
    supported = [row for row in surface if bool(row["above_null_q975"])]
    cross_supported = [
        row for row in supported if int(row["arc_distance_start_bins"]) >= 1
    ]
    return {
        "positive_informative_cross_arc_bin_count": len(positive),
        "positive_cross_arc_pair_excess": total,
        "response_weighted_arc_distance_bins": weighted_bins,
        "response_weighted_arc_distance_A": weighted_bins * arc_length_A,
        "response_weighted_lag_ps": weighted_time,
        "cross_arc_bins_above_null_q975": len(cross_supported),
        "above_null_arc_upper_exclusive_bins": (
            max(int(row["arc_distance_end_bins_exclusive"]) for row in cross_supported)
            if cross_supported
            else None
        ),
        "above_null_arc_upper_exclusive_A": (
            max(int(row["arc_distance_end_bins_exclusive"]) for row in cross_supported)
            * arc_length_A
            if cross_supported
            else None
        ),
        "above_null_lag_upper_ps": (
            max(float(row["lag_end_ps"]) for row in cross_supported)
            if cross_supported
            else None
        ),
        "support_status": (
            "CROSS_ARC_BINS_ABOVE_NULL_Q975_PRESENT"
            if cross_supported
            else "NO_CROSS_ARC_BIN_ABOVE_NULL_Q975"
        ),
        "scale_interpretation": (
            "response-weighted descriptors and a bin-resolution support envelope; "
            "not fitted intrinsic propagation length, lifetime, or speed"
        ),
    }


def summarize_modes(
    event_rows: Sequence[Mapping[str, object]], max_mode: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in range(1, max_mode + 1):
        field = f"delta_mode_{mode}_amplitude_A"
        values = np.asarray(
            [float(row[field]) for row in event_rows if str(row.get(field, "")) != ""],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        rows.append(
            {
                "mode": mode,
                "event_cluster_count": len(values),
                "mean_delta_amplitude_A": float(np.mean(values)),
                "median_delta_amplitude_A": float(np.median(values)),
                "q025_delta_amplitude_A": float(np.quantile(values, 0.025)),
                "q975_delta_amplitude_A": float(np.quantile(values, 0.975)),
                "positive_fraction": float(np.mean(values > 0.0)),
                "inference_status": "descriptive_event_cluster_distribution",
            }
        )
    if not rows:
        raise ValueError("event-size table has no requested low-order mode response")
    return rows


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def water_response_associations(
    event_rows: Sequence[Mapping[str, object]],
    water_rows: Sequence[Mapping[str, object]],
    max_mode: int,
) -> list[dict[str, object]]:
    """Join P1 primary events to P3 local-water effects and report correlations."""

    by_event: dict[int, Mapping[str, object]] = {}
    for row in event_rows:
        event_id = int(row["primary_event_id"])
        if event_id in by_event:
            raise ValueError(f"duplicate primary_event_id {event_id}")
        by_event[event_id] = row
    water: dict[tuple[int, str], float] = {}
    for row in water_rows:
        key = (int(row["event_id"]), str(row["metric"]))
        if key in water:
            raise ValueError(f"duplicate water effect {key}")
        water[key] = float(row["post_minus_pre"])
    response_fields = ["event_size_residual_A2", "affected_arc_fraction"] + [
        f"delta_mode_{mode}_amplitude_A" for mode in range(1, max_mode + 1)
    ]
    metrics = sorted({metric for _, metric in water})
    output: list[dict[str, object]] = []
    for metric in metrics:
        for response_field in response_fields:
            pairs = [
                (float(row[response_field]), water[(event_id, metric)])
                for event_id, row in by_event.items()
                if response_field in row
                and str(row[response_field]) != ""
                and (event_id, metric) in water
                and math.isfinite(float(row[response_field]))
                and math.isfinite(water[(event_id, metric)])
            ]
            if not pairs:
                continue
            left = np.asarray([pair[0] for pair in pairs], dtype=float)
            right = np.asarray([pair[1] for pair in pairs], dtype=float)
            output.append(
                {
                    "water_metric": metric,
                    "cascade_metric": response_field,
                    "joined_primary_event_count": len(pairs),
                    "pearson_r": _correlation(left, right),
                    "spearman_rho": _correlation(_average_ranks(left), _average_ranks(right)),
                    "inference_status": (
                        "descriptive_same_trajectory_primary_event_association_not_causal"
                    ),
                }
            )
    if not output:
        raise ValueError("P1 primary events and P3 water effects have no joinable rows")
    return output


def summarize_water_nulls(
    null_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in null_rows:
        observed = float(row["observed_mean_post_minus_pre"])
        low = float(row["null_q025"])
        high = float(row["null_q975"])
        outside = observed < low or observed > high
        output.append(
            {
                "control_type": str(row["control_type"]),
                "metric": str(row["metric"]),
                "observed_mean_post_minus_pre": observed,
                "null_mean": float(row["null_mean"]),
                "null_q025": low,
                "null_q975": high,
                "empirical_two_sided_p": float(row["empirical_two_sided_p"]),
                "outside_null_q95": outside,
                "direction": "above" if observed > high else "below" if observed < low else "inside",
                "inference_status": str(row["inference_status"]),
            }
        )
    if not output:
        raise ValueError("water null-statistics table is empty")
    return output


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "pair_hazard": Path(args.pair_hazard),
        "frame_modes": Path(args.frame_modes),
        "event_sizes": Path(args.event_sizes),
        "water_effects": Path(args.water_effects),
        "water_null_statistics": Path(args.water_null_statistics),
        "propagation_summary": Path(args.propagation_summary),
    }
    pair_rows, pair_fields = _read_csv(paths["pair_hazard"])
    _require(
        paths["pair_hazard"],
        pair_fields,
        {
            "arc_distance_start_bins",
            "arc_distance_end_bins_exclusive",
            "lag_start_ps",
            "lag_end_ps",
            "observed_pairs",
            "null_mean_pairs",
            "null_q025_pairs",
            "null_q975_pairs",
            "informative_null_count",
        },
    )
    mode_frames, mode_fields = _read_csv(paths["frame_modes"])
    _require(paths["frame_modes"], mode_fields, {"step", "mean_radius_A"})
    event_rows, event_fields = _read_csv(paths["event_sizes"])
    _require(
        paths["event_sizes"],
        event_fields,
        {"primary_event_id", "event_size_residual_A2", "affected_arc_fraction"},
    )
    water_rows, water_fields = _read_csv(paths["water_effects"])
    _require(paths["water_effects"], water_fields, {"event_id", "metric", "post_minus_pre"})
    null_rows, null_fields = _read_csv(paths["water_null_statistics"])
    _require(
        paths["water_null_statistics"],
        null_fields,
        {
            "control_type",
            "metric",
            "observed_mean_post_minus_pre",
            "null_mean",
            "null_q025",
            "null_q975",
            "empirical_two_sided_p",
            "inference_status",
        },
    )
    propagation = json.loads(paths["propagation_summary"].read_text(encoding="utf-8"))
    if propagation.get("status") != "PASS":
        raise ValueError("P1 propagation summary is not PASS")
    if propagation.get("case_id") != args.case_id:
        raise ValueError("P1 propagation summary case does not match requested case")
    arc_count = int(propagation["arc_count"])
    radii = np.asarray([float(row["mean_radius_A"]) for row in mode_frames], dtype=float)
    if arc_count < 3 or not len(radii) or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("invalid arc count or frame mean radii")
    mean_radius_A = float(np.mean(radii))
    arc_length_A = 2.0 * math.pi * mean_radius_A / arc_count
    surface = build_response_surface(pair_rows, arc_length_A)
    scales = summarize_response_scales(surface, arc_length_A)
    modes = summarize_modes(event_rows, args.max_mode)
    associations = water_response_associations(event_rows, water_rows, args.max_mode)
    water_nulls = summarize_water_nulls(null_rows)
    by_metric: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in water_nulls:
        by_metric[str(row["metric"])].append(row)
    outside_both = sorted(
        metric
        for metric, rows in by_metric.items()
        if len({str(row["control_type"]) for row in rows}) >= 2
        and all(bool(row["outside_null_q95"]) for row in rows)
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "pair_excess_response_surface.csv", surface)
    _write_csv(output / "low_order_mode_response.csv", modes)
    _write_csv(output / "water_cascade_associations.csv", associations)
    _write_csv(output / "water_randomization_context.csv", water_nulls)
    summary = {
        "status": "PASS",
        "case_id": str(args.case_id),
        "arc_count": arc_count,
        "mean_radius_A": mean_radius_A,
        "mean_arc_length_A": arc_length_A,
        "response_surface_rows": len(surface),
        "mode_response_rows": len(modes),
        "water_cascade_association_rows": len(associations),
        "water_metrics_outside_both_null_q95": outside_both,
        **scales,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "case_id": str(args.case_id),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "max_mode": args.max_mode,
        "response_definition": "observed_event_pairs_minus_mean_circular_shift_null_pairs",
        "spatial_scale_definition": "positive_excess_weighted_cross_arc_bin_midpoint",
        "temporal_scale_definition": "positive_excess_weighted_lag_bin_midpoint",
        "support_envelope_definition": "informative_cross_arc_bins_above_null_q975",
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "REPORT.md").write_text(
        "\n".join(
            (
                f"# TPCL cascade-response synthesis: {args.case_id}",
                "",
                f"- Cross-arc bins above the P1 null 97.5% bound: {len([row for row in surface if bool(row['above_null_q975']) and int(row['arc_distance_start_bins']) >= 1])}",
                f"- Response-weighted arc distance: {scales['response_weighted_arc_distance_A']:.6g} A",
                f"- Response-weighted lag: {scales['response_weighted_lag_ps']:.6g} ps",
                "",
                "These are within-trajectory conditional descriptors, not an avalanche, causal propagation, intrinsic length, lifetime, speed, free-energy barrier, or physical rate.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--pair-hazard", required=True, type=Path)
    parser.add_argument("--frame-modes", required=True, type=Path)
    parser.add_argument("--event-sizes", required=True, type=Path)
    parser.add_argument("--water-effects", required=True, type=Path)
    parser.add_argument("--water-null-statistics", required=True, type=Path)
    parser.add_argument("--propagation-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-mode", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_mode < 1:
        raise ValueError("max_mode must be positive")
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
