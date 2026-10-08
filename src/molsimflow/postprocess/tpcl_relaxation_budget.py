"""Quantify event rate, event amplitude, and unsigned relaxation-budget proxies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BudgetConfig:
    region: str
    window: str
    block_duration_ns: float
    arc_count: int
    bootstrap_samples: int
    seed: int


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def collect_block_rows(
    rows: Sequence[dict[str, str]], config: BudgetConfig
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["region"] != config.region or row["window"] != config.window:
            continue
        grouped[(row["case_id"], int(row["time_block_200ps"]))].append(
            abs(number(row, "mean_radius_component_A"))
        )
    if not grouped:
        raise ValueError("no rows match the requested region/window")
    output = []
    for (case_id, block), amplitudes in sorted(grouped.items()):
        count = len(amplitudes)
        total = float(sum(amplitudes))
        output.append(
            {
                "case_id": case_id,
                "time_block": block,
                "selected_event_count": count,
                "selected_event_rate_per_arc_ns": count
                / (config.block_duration_ns * config.arc_count),
                "mean_unsigned_mean_radius_component_A": total / count,
                "total_unsigned_mean_radius_component_A": total,
                "unsigned_relaxation_flux_per_arc_ns": total
                / (config.block_duration_ns * config.arc_count),
            }
        )
    return output


def bootstrap_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    if not len(values):
        raise ValueError("cannot bootstrap an empty block series")
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_cases(
    block_rows: Sequence[dict[str, object]], config: BudgetConfig
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in block_rows:
        grouped[str(row["case_id"])].append(row)
    rng = np.random.default_rng(config.seed)
    fields = (
        "selected_event_rate_per_arc_ns",
        "mean_unsigned_mean_radius_component_A",
        "unsigned_relaxation_flux_per_arc_ns",
    )
    summaries = []
    for case_id, rows in sorted(grouped.items()):
        result: dict[str, object] = {
            "case_id": case_id,
            "block_count": len(rows),
            "selected_event_count": sum(int(row["selected_event_count"]) for row in rows),
        }
        for field in fields:
            values = np.asarray([float(row[field]) for row in rows])
            low, high = bootstrap_interval(values, config.bootstrap_samples, rng)
            result[f"{field}_mean"] = float(np.mean(values))
            result[f"{field}_temporal_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            result[f"{field}_bootstrap_ci025"] = low
            result[f"{field}_bootstrap_ci975"] = high
        summaries.append(result)
    return summaries


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")


def plot_rate_amplitude(rows: Sequence[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        selected = [row for row in rows if row["case_id"] == case_id]
        axis.scatter(
            [float(row["selected_event_rate_per_arc_ns"]) for row in selected],
            [float(row["mean_unsigned_mean_radius_component_A"]) for row in selected],
            alpha=0.75,
            label=case_id,
        )
    axis.set_xlabel("Selected event frequency (arc-1 ns-1)")
    axis.set_ylabel("Mean |mean-radius component| (A)")
    axis.set_title("Within-trajectory event packetization")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_flux(summary: Sequence[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    ordered = sorted(summary, key=lambda row: str(row["case_id"]))
    labels = [str(row["case_id"]) for row in ordered]
    means = [float(row["unsigned_relaxation_flux_per_arc_ns_mean"]) for row in ordered]
    lower = [
        mean - float(row["unsigned_relaxation_flux_per_arc_ns_bootstrap_ci025"])
        for mean, row in zip(means, ordered)
    ]
    upper = [
        float(row["unsigned_relaxation_flux_per_arc_ns_bootstrap_ci975"]) - mean
        for mean, row in zip(means, ordered)
    ]
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.errorbar(range(len(means)), means, yerr=[lower, upper], fmt="o", capsize=3)
    axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axis.set_ylabel("Unsigned component flux (A arc-1 ns-1)")
    axis.set_title("Descriptive time-block relaxation-budget proxy")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_coverage(rows: Sequence[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: str(row["case_id"]))
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.bar(
        range(len(ordered)),
        [int(row["selected_event_count"]) for row in ordered],
    )
    axis.set_xticks(
        range(len(ordered)),
        [str(row["case_id"]) for row in ordered],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Events with far-fast mode attribution")
    axis.set_title("Relaxation-budget input coverage")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    config = BudgetConfig(
        args.region,
        args.window,
        args.block_duration_ns,
        args.arc_count,
        args.bootstrap_samples,
        args.seed,
    )
    if config.block_duration_ns <= 0 or config.arc_count < 1 or config.bootstrap_samples < 100:
        raise ValueError("invalid duration, arc count, or bootstrap sample count")
    block_rows = collect_block_rows(read_rows(args.event_mode_attribution), config)
    summary_rows = summarize_cases(block_rows, config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "case_block_budget.csv", block_rows)
    write_csv(output / "case_budget_summary.csv", summary_rows)
    figures = output / "figures"
    figures.mkdir()
    plot_rate_amplitude(block_rows, figures / "01_rate_amplitude_packetization.png")
    plot_flux(summary_rows, figures / "02_relaxation_budget_proxy.png")
    plot_coverage(summary_rows, figures / "03_relaxation_budget_coverage.png")
    result = {
        "status": "PASS",
        "case_count": len(summary_rows),
        "block_count": len(block_rows),
        "region": config.region,
        "window": config.window,
        "claim_boundary": [
            "The flux is an unsigned retrospective additive-component proxy, not a conserved physical flux or relaxation rate.",
            "Time-block bootstrap intervals are within-trajectory diagnostics, not replicate uncertainty or cross-surface hypothesis tests.",
            "Rate-amplitude patterns do not establish pH, chemistry, free energy, friction, dissipation, or causal event-to-global coupling.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-mode-attribution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--region", default="far")
    parser.add_argument("--window", default="fast")
    parser.add_argument("--block-duration-ns", type=float, default=0.2)
    parser.add_argument("--arc-count", type=int, default=36)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
