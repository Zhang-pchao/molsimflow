"""Separate TPCL water-node renewal from retained-network edge rewiring."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_number(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}")
    return value


def collect_block_rows(
    rows: Sequence[dict[str, str]],
    *,
    node_metric: str,
    retained_edge_metric: str,
) -> list[dict[str, object]]:
    """Pair two finite event-level metrics, then average each 200-ps block."""

    event_values: dict[tuple[str, int, int, int, float], dict[str, float]] = {}
    for row in rows:
        if row["metric"] not in {node_metric, retained_edge_metric}:
            continue
        if row["finite"] != "1":
            continue
        key = (
            row["case_id"],
            int(row["primary_event_id"]),
            int(row["time_block_200ps"]),
            int(row["lag_frames"]),
            finite_number(row, "lag_ps"),
        )
        values = event_values.setdefault(key, {})
        metric = row["metric"]
        if metric in values:
            raise ValueError(f"duplicate {metric} entry for event key {key}")
        values[metric] = finite_number(row, "topology_did")
    grouped: dict[tuple[str, int, int, float], list[tuple[float, float]]] = defaultdict(list)
    for (case_id, _event_id, block, lag_frames, lag_ps), values in event_values.items():
        if set(values) != {node_metric, retained_edge_metric}:
            continue
        grouped[(case_id, block, lag_frames, lag_ps)].append(
            (values[node_metric], values[retained_edge_metric])
        )
    if not grouped:
        raise ValueError("no complete finite node/retained-edge event pairs")
    block_rows: list[dict[str, object]] = []
    for (case_id, block, lag_frames, lag_ps), values in sorted(grouped.items()):
        node_did = float(np.mean([value[0] for value in values]))
        edge_did = float(np.mean([value[1] for value in values]))
        node_loss = -node_did
        edge_loss = -edge_did
        block_rows.append(
            {
                "case_id": case_id,
                "time_block_200ps": block,
                "lag_frames": lag_frames,
                "lag_ps": lag_ps,
                "complete_event_pair_count": len(values),
                "node_survival_did": node_did,
                "retained_edge_survival_did": edge_did,
                "node_loss_excess": node_loss,
                "retained_edge_loss_excess": edge_loss,
            }
        )
    return block_rows


def bootstrap_mean(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    if not len(values):
        raise ValueError("cannot summarize an empty block series")
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(values)), float(low), float(high)


def summarize_lags(
    block_rows: Sequence[dict[str, object]], samples: int, seed: int
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in block_rows:
        grouped[(str(row["case_id"]), float(row["lag_ps"]))].append(row)
    output: list[dict[str, object]] = []
    for index, ((case_id, lag_ps), rows) in enumerate(sorted(grouped.items())):
        rng = np.random.default_rng(seed + index)
        for metric in ("node_loss_excess", "retained_edge_loss_excess"):
            values = np.asarray([float(row[metric]) for row in rows])
            mean, low, high = bootstrap_mean(values, samples, rng)
            output.append(
                {
                    "case_id": case_id,
                    "lag_ps": lag_ps,
                    "component": metric,
                    "block_count": len(rows),
                    "mean_loss_excess": mean,
                    "bootstrap_ci025": low,
                    "bootstrap_ci975": high,
                }
            )
    return output


def summarize_central_lag(
    block_rows: Sequence[dict[str, object]],
    *,
    central_lag_ps: float,
    tolerance_ps: float,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in block_rows:
        if math.isclose(float(row["lag_ps"]), central_lag_ps, abs_tol=tolerance_ps):
            grouped[str(row["case_id"])].append(row)
    if not grouped:
        raise ValueError("central lag has no complete finite blocks")
    output: list[dict[str, object]] = []
    for index, (case_id, rows) in enumerate(sorted(grouped.items())):
        rng = np.random.default_rng(seed + 1000 + index)
        node = np.asarray([float(row["node_loss_excess"]) for row in rows])
        edge = np.asarray([float(row["retained_edge_loss_excess"]) for row in rows])
        node_mean, node_low, node_high = bootstrap_mean(node, samples, rng)
        edge_mean, edge_low, edge_high = bootstrap_mean(edge, samples, rng)
        output.append(
            {
                "case_id": case_id,
                "central_lag_ps": central_lag_ps,
                "block_count": len(rows),
                "complete_event_pair_count": sum(
                    int(row["complete_event_pair_count"]) for row in rows
                ),
                "node_loss_excess_mean": node_mean,
                "node_loss_excess_bootstrap_ci025": node_low,
                "node_loss_excess_bootstrap_ci975": node_high,
                "retained_edge_loss_excess_mean": edge_mean,
                "retained_edge_loss_excess_bootstrap_ci025": edge_low,
                "retained_edge_loss_excess_bootstrap_ci975": edge_high,
            }
        )
    return output


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


def plot_central_components(rows: Sequence[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: str(row["case_id"]))
    labels = [str(row["case_id"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    positions = np.arange(len(ordered))
    width = 0.36
    components = (
        ("node_loss_excess", "Water-node renewal excess", -width / 2),
        ("retained_edge_loss_excess", "Retained-water edge rewiring excess", width / 2),
    )
    for prefix, label, offset in components:
        mean = np.asarray([float(row[f"{prefix}_mean"]) for row in ordered])
        lower = mean - np.asarray(
            [float(row[f"{prefix}_bootstrap_ci025"]) for row in ordered]
        )
        upper = np.asarray(
            [float(row[f"{prefix}_bootstrap_ci975"]) for row in ordered]
        ) - mean
        axis.bar(positions + offset, mean, width, yerr=[lower, upper], capsize=3, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Event-minus-pre loss excess")
    axis.set_title("Water renewal dominates over retained-network rewiring")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_central_block_scatter(
    block_rows: Sequence[dict[str, object]],
    *,
    central_lag_ps: float,
    tolerance_ps: float,
    path: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in block_rows
        if math.isclose(float(row["lag_ps"]), central_lag_ps, abs_tol=tolerance_ps)
    ]
    if not selected:
        raise ValueError("central lag has no block rows for scatter plot")
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    for case_id in sorted({str(row["case_id"]) for row in selected}):
        rows = [row for row in selected if row["case_id"] == case_id]
        axis.scatter(
            [float(row["node_loss_excess"]) for row in rows],
            [float(row["retained_edge_loss_excess"]) for row in rows],
            alpha=0.75,
            label=case_id,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Water-node loss excess")
    axis.set_ylabel("Retained-edge loss excess")
    axis.set_title("Block-resolved renewal versus retained-edge rewiring")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_lag_profiles(rows: Sequence[dict[str, object]], path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    case_ids = sorted({str(row["case_id"]) for row in rows})
    columns = min(2, len(case_ids))
    row_count = math.ceil(len(case_ids) / columns)
    figure, axes = plt.subplots(
        row_count, columns, figsize=(4.0 * columns, 3.0 * row_count), sharex=True, sharey=True
    )
    axis_list = np.atleast_1d(axes).ravel()
    for axis, case_id in zip(axis_list, case_ids):
        selected = [row for row in rows if row["case_id"] == case_id]
        for component, label in (
            ("node_loss_excess", "node renewal"),
            ("retained_edge_loss_excess", "retained-edge rewiring"),
        ):
            series = sorted(
                (row for row in selected if row["component"] == component),
                key=lambda row: float(row["lag_ps"]),
            )
            x = np.asarray([float(row["lag_ps"]) for row in series])
            y = np.asarray([float(row["mean_loss_excess"]) for row in series])
            low = np.asarray([float(row["bootstrap_ci025"]) for row in series])
            high = np.asarray([float(row["bootstrap_ci975"]) for row in series])
            axis.plot(x, y, marker="o", label=label)
            axis.fill_between(x, low, high, alpha=0.18)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(case_id, fontsize=9)
        axis.set_xlabel("Lag (ps)")
        axis.set_ylabel("Event-minus-pre loss excess")
    for axis in axis_list[len(case_ids) :]:
        axis.set_visible(False)
    axis_list[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    if args.bootstrap_samples < 100 or args.central_lag_ps <= 0 or args.lag_tolerance_ps < 0:
        raise ValueError("invalid bootstrap or lag parameters")
    block_rows = collect_block_rows(
        read_rows(args.event_topology_did),
        node_metric=args.node_metric,
        retained_edge_metric=args.retained_edge_metric,
    )
    central_rows = summarize_central_lag(
        block_rows,
        central_lag_ps=args.central_lag_ps,
        tolerance_ps=args.lag_tolerance_ps,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    lag_rows = summarize_lags(block_rows, args.bootstrap_samples, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "block_decomposition.csv", block_rows)
    write_csv(output / "central_lag_summary.csv", central_rows)
    write_csv(output / "lag_profile_summary.csv", lag_rows)
    figures = output / "figures"
    figures.mkdir()
    plot_central_components(central_rows, figures / "01_node_vs_retained_edge_excess.png")
    plot_central_block_scatter(
        block_rows,
        central_lag_ps=args.central_lag_ps,
        tolerance_ps=args.lag_tolerance_ps,
        path=figures / "02_node_vs_retained_edge_block_scatter.png",
    )
    plot_lag_profiles(lag_rows, figures / "03_renewal_rewiring_lag_profiles.png")
    result = {
        "status": "PASS",
        "case_count": len(central_rows),
        "block_row_count": len(block_rows),
        "central_lag_ps": args.central_lag_ps,
        "node_metric": args.node_metric,
        "retained_edge_metric": args.retained_edge_metric,
        "claim_boundary": [
            "The decomposition is a retrospective event-aligned difference-in-difference summary, not causal molecular triggering or a reaction coordinate.",
            "Node-loss and retained-edge-loss excess are conditional topology contrasts, not conserved fractions or physical network fluxes.",
            "Block bootstrap intervals are within-trajectory diagnostics, not replicate uncertainty or cross-surface hypothesis tests.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-topology-did", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--node-metric", default="node_survival_fraction")
    parser.add_argument(
        "--retained-edge-metric", default="retained_node_edge_survival_fraction"
    )
    parser.add_argument("--central-lag-ps", type=float, default=1.0)
    parser.add_argument("--lag-tolerance-ps", type=float, default=1.0e-9)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
