"""Compare validated nanobubble physicochemical trajectories on a common grid.

The comparison is manifest-driven and deliberately avoids interpolation. It
uses only immutable, checksum-verified case outputs created by the
nanobubble_physchem workflow. Time blocks summarize fluctuations within one
trajectory; they are never replicate uncertainty or hypothesis tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from molsimflow.postprocess.nanobubble_physchem_aggregate import (
    CONDITION_COLORS,
    CaseData,
    _configure_matplotlib,
    _load_case,
    _read_tsv,
)

METRICS: dict[str, str] = {
    "largest_cluster_n2_count": "Largest N2 cluster count",
    "dissolved_or_disconnected_n2_count": "Disconnected N2 count",
    "bubble_height_q05_q95_A": "Bubble height (A)",
    "bubble_lateral_displacement_A": "Lateral displacement (A)",
    "relative_shape_anisotropy": "Relative shape anisotropy",
    "footprint_equivalent_radius_A": "Footprint radius proxy (A)",
}
CORE_METRICS = (
    "largest_cluster_n2_count",
    "bubble_height_q05_q95_A",
    "relative_shape_anisotropy",
)
PLOT_METRICS = (
    "largest_cluster_n2_count",
    "bubble_height_q05_q95_A",
    "footprint_equivalent_radius_A",
)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sample_index(time_ns: float, interval_ns: float) -> int | None:
    index = round(time_ns / interval_ns)
    if index < 1:
        return None
    if abs(time_ns - index * interval_ns) > max(1.0e-9, interval_ns * 1.0e-5):
        return None
    return index


def _common_values(case: CaseData, metric: str, interval_ns: float, final_index: int) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in case.geometry:
        time_ns = _as_float(row.get("time_ns", ""))
        value = _as_float(row.get(metric, ""))
        if time_ns is None or value is None:
            continue
        index = _sample_index(time_ns, interval_ns)
        if index is None or index > final_index:
            continue
        if index in values:
            raise ValueError(f"{case.case_id}: duplicate sample for {metric} at common-grid index {index}")
        values[index] = value
    return values


def _condition_label(condition: str) -> str:
    if condition.startswith("hcl_"):
        return "HCl"
    if condition.startswith("naoh_"):
        return "NaOH"
    return condition


def _block_contrasts(
    samples: dict[tuple[str, str], dict[int, float]],
    cases: list[CaseData],
    metrics: list[str],
    reference_condition: str,
    late_start_index: int,
    final_index: int,
    block_points: int,
    interval_ns: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_surface: dict[str, list[CaseData]] = defaultdict(list)
    for case in cases:
        by_surface[case.surface].append(case)
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for surface, surface_cases in by_surface.items():
        references = [case for case in surface_cases if case.condition == reference_condition]
        if len(references) != 1:
            raise ValueError(f"{surface}: expected one {reference_condition} reference case, found {len(references)}")
        reference = references[0]
        for case in surface_cases:
            if case.case_id == reference.case_id:
                continue
            for metric in metrics:
                effects: list[float] = []
                last_full_start = final_index - block_points + 1
                for block_start in range(late_start_index, last_full_start + 1, block_points):
                    block_end = block_start + block_points - 1
                    paired = [
                        (samples[(case.case_id, metric)][index], samples[(reference.case_id, metric)][index])
                        for index in range(block_start, block_end + 1)
                        if index in samples[(case.case_id, metric)] and index in samples[(reference.case_id, metric)]
                    ]
                    if not paired:
                        continue
                    effect = float(np.mean([value - ref for value, ref in paired]))
                    effects.append(effect)
                    rows.append(
                        {
                            "surface": surface,
                            "case_id": case.case_id,
                            "condition": case.condition,
                            "condition_label": _condition_label(case.condition),
                            "reference_case_id": reference.case_id,
                            "metric": metric,
                            "block_start_ns": block_start * interval_ns,
                            "block_end_ns": block_end * interval_ns,
                            "grid_points_paired": len(paired),
                            "effect_condition_minus_reference": effect,
                        }
                    )
                if effects:
                    summary.append(
                        {
                            "surface": surface,
                            "case_id": case.case_id,
                            "condition": case.condition,
                            "condition_label": _condition_label(case.condition),
                            "reference_case_id": reference.case_id,
                            "metric": metric,
                            "temporal_block_count": len(effects),
                            "mean_effect_condition_minus_reference": float(np.mean(effects)),
                            "temporal_block_std": float(np.std(effects, ddof=1)) if len(effects) > 1 else 0.0,
                            "claim_boundary": "within_trajectory_temporal_blocks_not_replicate_uncertainty",
                        }
                    )
    return rows, summary


def _plot_timeseries(
    cases: list[CaseData],
    samples: dict[tuple[str, str], dict[int, float]],
    output: Path,
    metrics: list[str],
    interval_ns: float,
    final_index: int,
    surfaces: list[str],
) -> None:
    from matplotlib import pyplot as plt

    selected = [metric for metric in ("largest_cluster_n2_count", "bubble_height_q05_q95_A") if metric in metrics]
    figure, axes = plt.subplots(len(surfaces), len(selected), figsize=(8.4, 2.25 * len(surfaces)), sharex=True, squeeze=False)
    for row_index, surface in enumerate(surfaces):
        for column, metric in enumerate(selected):
            axis = axes[row_index, column]
            for case in [candidate for candidate in cases if candidate.surface == surface]:
                values = samples[(case.case_id, metric)]
                if values:
                    indexes = sorted(values)
                    axis.plot(
                        [index * interval_ns for index in indexes],
                        [values[index] for index in indexes],
                        color=CONDITION_COLORS.get(case.condition, "#777777"),
                        lw=0.8,
                        label=_condition_label(case.condition),
                    )
            axis.set_ylabel(f"{surface}\n{METRICS[metric]}")
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(METRICS[metric])
            if column == 0:
                axis.legend(frameon=False, fontsize=7, ncol=3)
    for axis in axes[-1]:
        axis.set_xlabel("Time (ns)")
        axis.set_xlim(0.0, final_index * interval_ns)
    figure.suptitle("Common-grid trajectories; one realization per surface-condition", y=0.995, fontsize=10)
    figure.tight_layout()
    figure.savefig(output / "01_common_grid_geometry.png", dpi=300)
    plt.close(figure)


def _plot_contrasts(summary: list[dict[str, object]], output: Path, surfaces: list[str], metrics: list[str]) -> None:
    from matplotlib import pyplot as plt

    selected = [metric for metric in PLOT_METRICS if metric in metrics]
    conditions = list(dict.fromkeys(str(row["condition"]) for row in summary))
    if not summary:
        raise ValueError("no late-window contrast rows available")
    figure, axes = plt.subplots(len(conditions), len(selected), figsize=(3.4 * len(selected), 2.7 * len(conditions)), squeeze=False)
    positions = {surface: index for index, surface in enumerate(surfaces)}
    for row_index, condition in enumerate(conditions):
        for column, metric in enumerate(selected):
            axis = axes[row_index, column]
            rows = [row for row in summary if row["condition"] == condition and row["metric"] == metric]
            if rows:
                axis.errorbar(
                    [positions[str(row["surface"])] for row in rows],
                    [float(row["mean_effect_condition_minus_reference"]) for row in rows],
                    yerr=[float(row["temporal_block_std"]) for row in rows],
                    color=CONDITION_COLORS.get(condition, "#777777"),
                    fmt="o",
                    capsize=3,
                )
            axis.axhline(0.0, color="#555555", lw=0.8)
            axis.set_xticks(range(len(surfaces)), surfaces, rotation=25, ha="right")
            axis.grid(axis="y", alpha=0.2)
            if row_index == 0:
                axis.set_title(METRICS[metric])
            if column == 0:
                axis.set_ylabel(f"{_condition_label(condition)} - pure water")
    figure.suptitle("8--10 ns block contrasts; bars are temporal block SD, not replicate uncertainty", y=1.01, fontsize=10)
    figure.tight_layout()
    figure.savefig(output / "02_late_window_contrasts.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_coverage(coverage: list[dict[str, object]], output: Path) -> None:
    from matplotlib import pyplot as plt

    labels = ["Surface", "Condition", "Metric", "Status", "Available/common", "Fraction"]
    values = [
        [
            str(row["surface"]),
            str(row["condition"]),
            str(row["metric"]),
            str(row["status"]),
            f"{row['available_grid_points']}/{row['expected_grid_points']}",
            f"{float(row['coverage_fraction']):.3f}",
        ]
        for row in coverage
    ]
    figure, axis = plt.subplots(figsize=(10.5, max(3.0, 0.23 * len(values) + 1.0)))
    axis.axis("off")
    table = axis.table(cellText=values, colLabels=labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)
    for index, row in enumerate(coverage, start=1):
        if row["status"] == "DEFERRED":
            for column in range(len(labels)):
                table[(index, column)].set_facecolor("#f1f1f1")
    axis.set_title("Common-grid coverage; unavailable values remain unavailable", pad=12)
    figure.tight_layout()
    figure.savefig(output / "03_common_grid_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if args.grid_interval_ps <= 0.0 or args.required_end_ns <= 0.0:
        raise ValueError("grid interval and required end must be positive")
    if args.late_start_ns < 0.0 or args.late_end_ns <= args.late_start_ns:
        raise ValueError("late window must satisfy 0 <= start < end")
    if args.late_end_ns > args.required_end_ns + 1.0e-9:
        raise ValueError("late window exceeds required coverage")
    interval_ns = args.grid_interval_ps * 1.0e-3
    final_index = round(args.required_end_ns / interval_ns)
    if not math.isclose(final_index * interval_ns, args.required_end_ns, abs_tol=1.0e-9):
        raise ValueError("required end must be divisible by the common-grid interval")
    late_start_index = math.ceil(args.late_start_ns / interval_ns - 1.0e-9)
    block_points = round(args.block_ns / interval_ns)
    if block_points < 1 or not math.isclose(block_points * interval_ns, args.block_ns, abs_tol=1.0e-9):
        raise ValueError("block duration must be a positive multiple of the common-grid interval")
    metrics = args.metric or list(METRICS)
    if len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be unique")
    rows = _read_tsv(Path(args.case_manifest))
    deferred = [row for row in rows if row["status"] == "DEFERRED"]
    if deferred and not args.allow_incomplete:
        raise ValueError("manifest contains deferred cases; rerun explicitly with --allow-incomplete")
    cases = [_load_case(row, args.max_initial_time_ns, args.required_end_ns) for row in rows if row["status"] == "ACCEPTED"]
    if not cases:
        raise ValueError("no accepted cases")
    surfaces = list(dict.fromkeys(row["surface"] for row in rows))
    samples: dict[tuple[str, str], dict[int, float]] = {}
    grid_rows: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for case in cases:
        for metric in metrics:
            values = _common_values(case, metric, interval_ns, final_index)
            samples[(case.case_id, metric)] = values
            for index in range(1, final_index + 1):
                value = values.get(index)
                grid_rows.append(
                    {
                        "case_id": case.case_id,
                        "surface": case.surface,
                        "condition": case.condition,
                        "time_ns": index * interval_ns,
                        "metric": metric,
                        "value": "" if value is None else value,
                        "available": value is not None,
                    }
                )
            coverage.append(
                {
                    "case_id": case.case_id,
                    "surface": case.surface,
                    "condition": case.condition,
                    "metric": metric,
                    "status": "ACCEPTED",
                    "available_grid_points": len(values),
                    "expected_grid_points": final_index,
                    "coverage_fraction": len(values) / final_index,
                }
            )
    for row in deferred:
        for metric in metrics:
            coverage.append(
                {
                    "case_id": row["case_id"],
                    "surface": row["surface"],
                    "condition": row["condition"],
                    "metric": metric,
                    "status": "DEFERRED",
                    "available_grid_points": 0,
                    "expected_grid_points": final_index,
                    "coverage_fraction": 0.0,
                }
            )
    for case in cases:
        for metric in CORE_METRICS:
            if metric in metrics and len(samples[(case.case_id, metric)]) < final_index:
                raise ValueError(f"{case.case_id}: insufficient common-grid coverage for core metric {metric}")
    _write_csv(output / "common_grid_timeseries.csv", grid_rows, ["case_id", "surface", "condition", "time_ns", "metric", "value", "available"])
    _write_csv(output / "coverage_ledger.csv", coverage, ["case_id", "surface", "condition", "metric", "status", "available_grid_points", "expected_grid_points", "coverage_fraction"])
    block_rows, summary = _block_contrasts(samples, cases, metrics, args.reference_condition, late_start_index, final_index, block_points, interval_ns)
    _write_csv(output / "late_block_contrasts.csv", block_rows, ["surface", "case_id", "condition", "condition_label", "reference_case_id", "metric", "block_start_ns", "block_end_ns", "grid_points_paired", "effect_condition_minus_reference"])
    _write_csv(output / "late_contrast_summary.csv", summary, ["surface", "case_id", "condition", "condition_label", "reference_case_id", "metric", "temporal_block_count", "mean_effect_condition_minus_reference", "temporal_block_std", "claim_boundary"])
    if not args.no_plots:
        _configure_matplotlib(args.font_path)
        figures = output / "figures"
        figures.mkdir()
        _plot_timeseries(cases, samples, figures, metrics, interval_ns, final_index, surfaces)
        _plot_contrasts(summary, figures, surfaces, metrics)
        _plot_coverage(coverage, figures)
    validation = {
        "status": "PASS",
        "accepted_case_count": len(cases),
        "deferred_case_count": len(deferred),
        "coverage_complete": not deferred,
        "grid_interval_ps": args.grid_interval_ps,
        "required_end_ns": args.required_end_ns,
        "late_window_ns": [args.late_start_ns, args.late_end_ns],
        "block_ns": args.block_ns,
        "reference_condition": args.reference_condition,
        "metrics": metrics,
        "claim_boundary": [
            "No interpolation, imputation, or substitution of unavailable values is performed.",
            "Each condition is one trajectory; temporal blocks are descriptive within-trajectory variation, not replicate uncertainty or hypothesis tests.",
            "Contrasts are condition-minus-reference associations, not causal ion, pH, free-energy, friction, or dissipation laws.",
            "This module reports molecular-center geometry proxies and gas-integrity metrics, not a density-based contact angle, local stress, or surface tension.",
        ],
    }
    (output / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--max-initial-time-ns", type=float, default=0.02)
    parser.add_argument("--required-end-ns", type=float, default=10.0)
    parser.add_argument("--grid-interval-ps", type=float, default=10.0)
    parser.add_argument("--late-start-ns", type=float, default=8.0)
    parser.add_argument("--late-end-ns", type=float, default=10.0)
    parser.add_argument("--block-ns", type=float, default=0.2)
    parser.add_argument("--reference-condition", default="pure_water")
    parser.add_argument("--metric", choices=sorted(METRICS), action="append")
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
