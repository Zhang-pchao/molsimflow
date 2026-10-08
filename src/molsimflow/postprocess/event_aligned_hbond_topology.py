"""Compare event-centered TPCL H-bond topology with equal-duration pre baselines."""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, TextIO

import numpy as np

from molsimflow.postprocess.tpcl_snapshot_contrasts import (
    _bh_adjust,
    _sha256,
    _sign_flip_p,
    _stable_seed,
    _write_csv,
)

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_EVENT_ALIGNED_HBOND_TOPOLOGY_NOT_CAUSAL_PROPAGATION_"
    "PHYSICAL_RATE_FREE_ENERGY_OR_REPLICATE_EVIDENCE"
)
INTERVALS = (
    (1, -2, -1, -1, 0),
    (2, -3, -1, -1, 1),
    (3, -4, -1, -1, 2),
)
METRICS = (
    "node_survival_fraction",
    "induced_edge_survival_fraction",
    "retained_node_edge_survival_fraction",
    "directed_edge_survival_fraction",
    "induced_edge_turnover_jaccard",
    "largest_component_node_fraction_change",
    "largest_component_arc_fraction_change",
)
PRIMARY_METRICS = (
    "node_survival_fraction",
    "retained_node_edge_survival_fraction",
)


def _open_csv_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def _integer(raw: object, name: str) -> int:
    value = float(raw)
    rounded = round(value)
    if not math.isfinite(value) or not math.isclose(
        value, rounded, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(f"{name} must be a finite integer")
    return int(rounded)


def _load_sources(path: Path) -> list[dict[str, object]]:
    required = {
        "case_id",
        "edge_table",
        "frame_table",
        "node_table",
        "event_window_table",
    }
    sources = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if missing := required.difference(reader.fieldnames or []):
            raise ValueError(f"source table is missing columns: {sorted(missing)}")
        for row in reader:
            case_id = row["case_id"]
            if not case_id:
                raise ValueError("case_id must be nonempty")
            sources.append(
                {
                    "case_id": case_id,
                    **{name: Path(row[name]) for name in required if name != "case_id"},
                }
            )
    case_ids = [str(row["case_id"]) for row in sources]
    if not sources or len(case_ids) != len(set(case_ids)):
        raise ValueError("source cases must be nonempty and unique")
    return sources


def _load_event_blocks(path: Path) -> dict[str, dict[int, str]]:
    required = {"case_id", "primary_event_id", "time_block_200ps"}
    output: dict[str, dict[int, str]] = defaultdict(dict)
    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if missing := required.difference(reader.fieldnames or []):
            raise ValueError(f"event table is missing columns: {sorted(missing)}")
        for row in reader:
            case_id = row["case_id"]
            event_id = _integer(row["primary_event_id"], "primary_event_id")
            if event_id in output[case_id]:
                raise ValueError(f"duplicate primary event {(case_id, event_id)}")
            output[case_id][event_id] = row["time_block_200ps"]
    return dict(output)


def _load_event_steps(
    path: Path, admitted_events: Mapping[int, str]
) -> dict[int, dict[int, int]]:
    required = {"event_id", "step", "relative_frame"}
    output: dict[int, dict[int, int]] = defaultdict(dict)
    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if missing := required.difference(reader.fieldnames or []):
            raise ValueError(f"event-window table is missing columns: {sorted(missing)}")
        for row in reader:
            event_id = _integer(row["event_id"], "event_id")
            if event_id not in admitted_events:
                continue
            relative = _integer(row["relative_frame"], "relative_frame")
            if relative in output[event_id]:
                raise ValueError(f"duplicate event relative frame {(event_id, relative)}")
            output[event_id][relative] = _integer(row["step"], "step")
    required_frames = set(range(-4, 3))
    if set(output) != set(admitted_events):
        raise ValueError("event windows do not cover every admitted primary event")
    for event_id, frames in output.items():
        if not required_frames.issubset(frames):
            raise ValueError(f"event {event_id} lacks required relative frames")
    return dict(output)


def _load_frame_support(path: Path) -> set[int]:
    output = set()
    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if "step" not in set(reader.fieldnames or []):
            raise ValueError(f"frame table lacks step: {path}")
        for row in reader:
            step = _integer(row["step"], "frame step")
            if step in output:
                raise ValueError(f"duplicate frame step {step}")
            output.add(step)
    if not output:
        raise ValueError(f"frame table is empty: {path}")
    return output


def _load_nodes(path: Path, selected_steps: set[int]) -> dict[int, dict[int, int]]:
    required = {"step", "oxygen_id", "arc_index"}
    output: dict[int, dict[int, int]] = defaultdict(dict)
    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if missing := required.difference(reader.fieldnames or []):
            raise ValueError(f"node table is missing columns: {sorted(missing)}")
        for row in reader:
            step = _integer(row["step"], "node step")
            if step not in selected_steps:
                continue
            oxygen_id = _integer(row["oxygen_id"], "oxygen_id")
            if oxygen_id in output[step]:
                raise ValueError(f"duplicate node {(step, oxygen_id)}")
            output[step][oxygen_id] = _integer(row["arc_index"], "arc_index")
    missing_steps = selected_steps.difference(output)
    if missing_steps:
        raise ValueError(f"node table lacks {len(missing_steps)} selected steps")
    return dict(output)


def _load_edges(
    path: Path, selected_steps: set[int]
) -> tuple[dict[int, set[tuple[int, int]]], dict[int, set[tuple[int, int]]]]:
    required = {"step", "donor_id", "acceptor_id", "edge_scope"}
    undirected: dict[int, set[tuple[int, int]]] = defaultdict(set)
    directed: dict[int, set[tuple[int, int]]] = defaultdict(set)
    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if missing := required.difference(reader.fieldnames or []):
            raise ValueError(f"edge table is missing columns: {sorted(missing)}")
        for row in reader:
            step = _integer(row["step"], "edge step")
            if step not in selected_steps or row["edge_scope"] != "induced_tpcl":
                continue
            donor = _integer(row["donor_id"], "donor_id")
            acceptor = _integer(row["acceptor_id"], "acceptor_id")
            if donor == acceptor:
                raise ValueError("self H-bond edge is invalid")
            directed[step].add((donor, acceptor))
            undirected[step].add(tuple(sorted((donor, acceptor))))
    return dict(undirected), dict(directed)


def _component_fractions(
    nodes: Mapping[int, int], edges: set[tuple[int, int]]
) -> tuple[float, float]:
    if not nodes:
        raise ValueError("component metrics require at least one TPCL node")
    neighbors = {node: set() for node in nodes}
    for left, right in edges:
        if left not in nodes or right not in nodes:
            raise ValueError("induced edge endpoint is absent from TPCL nodes")
        neighbors[left].add(right)
        neighbors[right].add(left)
    components = []
    unseen = set(nodes)
    while unseen:
        pending = [unseen.pop()]
        component = set(pending)
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        components.append(component)
    largest = max(components, key=lambda item: (len(item), -min(item)))
    occupied_arcs = set(nodes.values())
    largest_arcs = {nodes[node] for node in largest}
    return len(largest) / len(nodes), len(largest_arcs) / len(occupied_arcs)


def _safe_fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def _interval_metrics(
    start_nodes: Mapping[int, int],
    end_nodes: Mapping[int, int],
    start_edges: set[tuple[int, int]],
    end_edges: set[tuple[int, int]],
    start_directed_edges: set[tuple[int, int]],
    end_directed_edges: set[tuple[int, int]],
) -> dict[str, float]:
    start_ids, end_ids = set(start_nodes), set(end_nodes)
    retained_edges = {
        edge for edge in start_edges if edge[0] in end_ids and edge[1] in end_ids
    }
    union = start_edges | end_edges
    start_component = _component_fractions(start_nodes, start_edges)
    end_component = _component_fractions(end_nodes, end_edges)
    return {
        "node_survival_fraction": len(start_ids & end_ids) / len(start_ids),
        "induced_edge_survival_fraction": _safe_fraction(
            len(start_edges & end_edges), len(start_edges)
        ),
        "retained_node_edge_survival_fraction": _safe_fraction(
            len(start_edges & end_edges), len(retained_edges)
        ),
        "directed_edge_survival_fraction": _safe_fraction(
            len(start_directed_edges & end_directed_edges), len(start_directed_edges)
        ),
        "induced_edge_turnover_jaccard": _safe_fraction(
            len(union - (start_edges & end_edges)), len(union)
        ),
        "largest_component_node_fraction_change": end_component[0]
        - start_component[0],
        "largest_component_arc_fraction_change": end_component[1]
        - start_component[1],
    }


def _event_rows(
    case_id: str,
    event_blocks: Mapping[int, str],
    event_steps: Mapping[int, Mapping[int, int]],
    nodes: Mapping[int, Mapping[int, int]],
    edges: Mapping[int, set[tuple[int, int]]],
    directed_edges: Mapping[int, set[tuple[int, int]]],
    frame_interval_ps: float,
) -> list[dict[str, object]]:
    output = []
    for event_id, relative_steps in sorted(event_steps.items()):
        for lag, pre_start, pre_end, event_start, event_end in INTERVALS:
            pre = _interval_metrics(
                nodes[relative_steps[pre_start]],
                nodes[relative_steps[pre_end]],
                edges.get(relative_steps[pre_start], set()),
                edges.get(relative_steps[pre_end], set()),
                directed_edges.get(relative_steps[pre_start], set()),
                directed_edges.get(relative_steps[pre_end], set()),
            )
            event = _interval_metrics(
                nodes[relative_steps[event_start]],
                nodes[relative_steps[event_end]],
                edges.get(relative_steps[event_start], set()),
                edges.get(relative_steps[event_end], set()),
                directed_edges.get(relative_steps[event_start], set()),
                directed_edges.get(relative_steps[event_end], set()),
            )
            for metric in METRICS:
                pre_value, event_value = pre[metric], event[metric]
                did = (
                    event_value - pre_value
                    if math.isfinite(pre_value) and math.isfinite(event_value)
                    else math.nan
                )
                output.append(
                    {
                        "case_id": case_id,
                        "primary_event_id": event_id,
                        "time_block_200ps": event_blocks[event_id],
                        "lag_frames": lag,
                        "lag_ps": lag * frame_interval_ps,
                        "metric": metric,
                        "pre_interval_value": pre_value,
                        "event_interval_value": event_value,
                        "topology_did": did,
                        "finite": int(math.isfinite(did)),
                        "scientific_status": SCIENTIFIC_STATUS,
                    }
                )
    return output


def _bootstrap_block_means(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    estimates = np.mean(values[indices], axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _summaries(
    rows: Sequence[Mapping[str, object]], bootstrap_draws: int, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), int(row["lag_frames"]), str(row["metric"]))].append(row)
    summaries = []
    for (case_id, lag, metric), group in sorted(grouped.items()):
        finite = [row for row in group if int(row["finite"])]
        block_values: dict[str, list[float]] = defaultdict(list)
        for row in finite:
            block_values[str(row["time_block_200ps"])].append(float(row["topology_did"]))
        means = np.asarray(
            [np.mean(values) for _, values in sorted(block_values.items())], dtype=float
        )
        low, high = (math.nan, math.nan)
        if len(means):
            low, high = _bootstrap_block_means(
                means,
                bootstrap_draws,
                _stable_seed(seed, case_id, lag, metric, "bootstrap"),
            )
        summaries.append(
            {
                "case_id": case_id,
                "lag_frames": lag,
                "lag_ps": float(group[0]["lag_ps"]),
                "metric": metric,
                "event_count": len(group),
                "finite_event_count": len(finite),
                "finite_block_count": len(means),
                "mean_topology_did": float(np.mean([float(row["topology_did"]) for row in finite]))
                if finite
                else math.nan,
                "median_topology_did": float(
                    np.median([float(row["topology_did"]) for row in finite])
                )
                if finite
                else math.nan,
                "block_bootstrap_ci025": low,
                "block_bootstrap_ci975": high,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
    primary = []
    for row in summaries:
        if int(row["lag_frames"]) != 2 or row["metric"] not in PRIMARY_METRICS:
            continue
        group = grouped[(str(row["case_id"]), 2, str(row["metric"]))]
        by_block: dict[str, list[float]] = defaultdict(list)
        for item in group:
            if int(item["finite"]):
                by_block[str(item["time_block_200ps"])].append(float(item["topology_did"]))
        block_means = np.asarray(
            [np.mean(values) for _, values in sorted(by_block.items())], dtype=float
        )
        p_value = _sign_flip_p(block_means) if len(block_means) else math.nan
        primary.append(
            {
                **row,
                "sign_flip_p": p_value,
                "bh_q": "",
                "within_trajectory_qualified": 0,
            }
        )
    finite_indices = [
        index
        for index, row in enumerate(primary)
        if math.isfinite(float(row["sign_flip_p"]))
    ]
    q_values = _bh_adjust([float(primary[index]["sign_flip_p"]) for index in finite_indices])
    for index, q_value in zip(finite_indices, q_values):
        row = primary[index]
        row["bh_q"] = q_value
        row["within_trajectory_qualified"] = int(
            q_value <= 0.05
            and (
                float(row["block_bootstrap_ci025"]) > 0.0
                or float(row["block_bootstrap_ci975"]) < 0.0
            )
        )
    return summaries, primary


def run_analysis(
    sources_table: Path,
    event_table: Path,
    output_dir: Path,
    *,
    frame_interval_ps: float = 0.5,
    bootstrap_draws: int = 2000,
    seed: int = 20260905,
) -> dict[str, object]:
    if frame_interval_ps <= 0.0 or bootstrap_draws < 100:
        raise ValueError("invalid frame interval or bootstrap count")
    sources = _load_sources(sources_table)
    event_blocks = _load_event_blocks(event_table)
    rows = []
    source_manifest = []
    for source in sources:
        case_id = str(source["case_id"])
        if case_id not in event_blocks:
            raise ValueError(f"event table lacks case {case_id}")
        steps = _load_event_steps(source["event_window_table"], event_blocks[case_id])
        selected_steps = {step for frames in steps.values() for step in frames.values()}
        frame_support = _load_frame_support(source["frame_table"])
        if not selected_steps.issubset(frame_support):
            raise ValueError(f"frame table lacks selected steps for {case_id}")
        nodes = _load_nodes(source["node_table"], selected_steps)
        edges, directed_edges = _load_edges(source["edge_table"], selected_steps)
        rows.extend(
            _event_rows(
                case_id,
                event_blocks[case_id],
                steps,
                nodes,
                edges,
                directed_edges,
                frame_interval_ps,
            )
        )
        source_manifest.append(
            {
                "case_id": case_id,
                **{
                    name: {"path": str(source[name]), "sha256": _sha256(source[name])}
                    for name in (
                        "edge_table",
                        "frame_table",
                        "node_table",
                        "event_window_table",
                    )
                },
            }
        )
    summaries, primary = _summaries(rows, bootstrap_draws, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "event_topology_did.csv", rows)
    _write_csv(output / "topology_summary.csv", summaries)
    _write_csv(output / "primary_topology_tests.csv", primary)
    coverage = [
        {
            "case_id": row["case_id"],
            "lag_ps": row["lag_ps"],
            "metric": row["metric"],
            "event_count": row["event_count"],
            "finite_event_count": row["finite_event_count"],
            "finite_fraction": int(row["finite_event_count"]) / int(row["event_count"]),
            "scientific_status": SCIENTIFIC_STATUS,
        }
        for row in summaries
    ]
    _write_csv(output / "finite_coverage.csv", coverage)
    summary = {
        "status": "PASS",
        "case_count": len(sources),
        "event_count": len({(row["case_id"], row["primary_event_id"]) for row in rows}),
        "event_metric_row_count": len(rows),
        "summary_row_count": len(summaries),
        "primary_test_count": len(primary),
        "primary_qualified_count": sum(int(row["within_trajectory_qualified"]) for row in primary),
        "frame_interval_ps": frame_interval_ps,
        "bootstrap_draws": bootstrap_draws,
        "seed": seed,
        "metrics": list(METRICS),
        "primary_metrics": list(PRIMARY_METRICS),
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "sources_table": {"path": str(sources_table), "sha256": _sha256(sources_table)},
                "event_table": {"path": str(event_table), "sha256": _sha256(event_table)},
                "sources": source_manifest,
                "intervals": [list(item) for item in INTERVALS],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        "# Event-aligned TPCL H-bond topology\n\n"
        "Equal-duration pre-event intervals provide the internal turnover baseline. "
        "Node loss and edge rewiring among retained waters are reported separately. "
        "Results are retrospective single-trajectory topology associations, not "
        "causal, propagation, rate, free-energy, or replicate-level evidence.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-table", type=Path, required=True)
    parser.add_argument("--event-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-interval-ps", type=float, default=0.5)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            run_analysis(
                args.sources_table,
                args.event_table,
                args.output_dir,
                frame_interval_ps=args.frame_interval_ps,
                bootstrap_draws=args.bootstrap_draws,
                seed=args.seed,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
