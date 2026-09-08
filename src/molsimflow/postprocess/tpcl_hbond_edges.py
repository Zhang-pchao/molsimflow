"""Extract explicit TPCL H-bond edges from an existing trajectory.

The frozen local-water sample table defines TPCL node membership and arc
coordinates. This module reuses the same water selection, O-H assignment, and
H-bond geometry as :mod:`local_water_order`; it does not redefine the TPCL or
run molecular dynamics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from molsimflow.io.lammps_dump import iter_lammps_dump_records
from molsimflow.postprocess.local_water_order import (
    SelectedFrame,
    assigned_water_oh_vectors,
    parse_range,
    select_frame,
    water_hbond_edges,
)

SCIENTIFIC_STATUS = (
    "EXPLICIT_EVENT_WINDOW_TPCL_HBOND_EDGES_FROM_EXISTING_TRAJECTORY_"
    "NOT_CAUSAL_FREE_ENERGY_OR_PHYSICAL_RATE_EVIDENCE"
)
SAMPLE_FIELDS = {
    "step",
    "time_ns",
    "oxygen_id",
    "arc_index",
    "theta_deg",
    "normal_distance_A",
    "tangential_offset_A",
    "surface_distance_A",
    "hbond_donor_count",
    "hbond_acceptor_count",
    "hbond_degree",
    "hbond_internal_tpcl_degree",
}
EDGE_FIELDS = (
    "step",
    "time_ns",
    "donor_id",
    "acceptor_id",
    "hbond_type",
    "donor_species",
    "acceptor_species",
    "edge_scope",
    "donor_in_tpcl",
    "acceptor_in_tpcl",
    "donor_arc_index",
    "acceptor_arc_index",
    "donor_theta_deg",
    "acceptor_theta_deg",
    "donor_normal_distance_A",
    "acceptor_normal_distance_A",
    "donor_tangential_offset_A",
    "acceptor_tangential_offset_A",
    "donor_surface_distance_A",
    "acceptor_surface_distance_A",
)
FRAME_FIELDS = (
    "step",
    "time_ns",
    "tpcl_node_count",
    "incident_undirected_edge_count",
    "induced_undirected_edge_count",
    "directed_hbond_count",
    "zero_induced_edges",
    "sample_metric_parity_pass",
)


@dataclass(frozen=True)
class TpclNode:
    oxygen_id: int
    time_ns: float
    arc_index: int
    theta_deg: float
    normal_distance_A: float
    tangential_offset_A: float
    surface_distance_A: float
    hbond_donor_count: int
    hbond_acceptor_count: int
    hbond_degree: int
    hbond_internal_tpcl_degree: int


def _finite(raw: object, name: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(raw: object, name: str) -> int:
    value = _finite(raw, name)
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must be integer-valued")
    return int(rounded)


def _open_csv_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def load_event_steps(path: Path) -> set[int]:
    """Load the unique trajectory steps named by an event-window table."""

    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if "step" not in set(reader.fieldnames or []):
            raise ValueError(f"{path}: missing step column")
        steps = {_integer(row["step"], "event step") for row in reader}
    if not steps:
        raise ValueError(f"{path}: no event-window steps")
    return steps


def iter_tpcl_node_frames(
    path: Path,
    selected_steps: set[int] | None = None,
) -> Iterator[tuple[int, dict[int, TpclNode]]]:
    """Stream step-grouped TPCL nodes from a frozen local-water sample table."""

    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        missing = SAMPLE_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        current_step: int | None = None
        current_nodes: dict[int, TpclNode] = {}
        previous_step: int | None = None
        seen_steps: set[int] = set()
        yielded = 0
        for row in reader:
            step = _integer(row["step"], "sample step")
            if previous_step is not None and step < previous_step:
                raise ValueError(f"{path}: sample steps are not sorted")
            previous_step = step
            if current_step is not None and step != current_step:
                if current_nodes:
                    yielded += 1
                    yield current_step, current_nodes
                current_nodes = {}
            current_step = step
            if selected_steps is not None and step not in selected_steps:
                continue
            seen_steps.add(step)
            oxygen_id = _integer(row["oxygen_id"], "oxygen_id")
            if oxygen_id in current_nodes:
                raise ValueError(f"{path}: duplicate oxygen {oxygen_id} at step {step}")
            current_nodes[oxygen_id] = TpclNode(
                oxygen_id=oxygen_id,
                time_ns=_finite(row["time_ns"], "time_ns"),
                arc_index=_integer(row["arc_index"], "arc_index"),
                theta_deg=_finite(row["theta_deg"], "theta_deg"),
                normal_distance_A=_finite(
                    row["normal_distance_A"], "normal_distance_A"
                ),
                tangential_offset_A=_finite(
                    row["tangential_offset_A"], "tangential_offset_A"
                ),
                surface_distance_A=_finite(
                    row["surface_distance_A"], "surface_distance_A"
                ),
                hbond_donor_count=_integer(
                    row["hbond_donor_count"], "hbond_donor_count"
                ),
                hbond_acceptor_count=_integer(
                    row["hbond_acceptor_count"], "hbond_acceptor_count"
                ),
                hbond_degree=_integer(row["hbond_degree"], "hbond_degree"),
                hbond_internal_tpcl_degree=_integer(
                    row["hbond_internal_tpcl_degree"],
                    "hbond_internal_tpcl_degree",
                ),
            )
        if current_step is not None and current_nodes:
            yielded += 1
            yield current_step, current_nodes
        if yielded == 0:
            raise ValueError(f"{path}: no selected TPCL node frames")
        if selected_steps is not None and seen_steps != selected_steps:
            missing_steps = sorted(selected_steps.difference(seen_steps))
            raise ValueError(f"{path}: missing selected steps {missing_steps[:10]}")


def _node_value(node: TpclNode | None, name: str) -> object:
    return getattr(node, name) if node is not None else ""


def analyze_hbond_frame(
    frame: SelectedFrame,
    nodes: Mapping[int, TpclNode],
    *,
    oh_cutoff_A: float,
    oo_cutoff_A: float,
    hbond_angle_deg: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract edges and verify exact parity with frozen per-node metrics."""

    oxygen_index = {
        int(oxygen_id): index
        for index, oxygen_id in enumerate(frame.water_oxygen_ids)
    }
    missing = set(nodes).difference(oxygen_index)
    if missing:
        raise ValueError(f"step {frame.step}: selected oxygen IDs absent from trajectory")
    selected = np.asarray([oxygen_index[oxygen_id] for oxygen_id in nodes], dtype=int)
    oh_vectors = assigned_water_oh_vectors(frame, oh_cutoff_A)
    directed_edges = water_hbond_edges(
        frame.water_oxygen,
        oh_vectors,
        selected,
        frame.bounds,
        oo_cutoff_A=oo_cutoff_A,
        angle_cutoff_deg=hbond_angle_deg,
    )

    selected_ids = set(nodes)
    donor_counts = {oxygen_id: 0 for oxygen_id in nodes}
    acceptor_counts = {oxygen_id: 0 for oxygen_id in nodes}
    incident_pairs: set[tuple[int, int]] = set()
    rows: list[dict[str, object]] = []
    for donor_index, acceptor_index in directed_edges:
        donor_id = int(frame.water_oxygen_ids[donor_index])
        acceptor_id = int(frame.water_oxygen_ids[acceptor_index])
        donor_node = nodes.get(donor_id)
        acceptor_node = nodes.get(acceptor_id)
        if donor_node is not None:
            donor_counts[donor_id] += 1
        if acceptor_node is not None:
            acceptor_counts[acceptor_id] += 1
        incident_pairs.add(tuple(sorted((donor_id, acceptor_id))))
        rows.append(
            {
                "step": frame.step,
                "time_ns": next(iter(nodes.values())).time_ns,
                "donor_id": donor_id,
                "acceptor_id": acceptor_id,
                "hbond_type": "water_water",
                "donor_species": "h2o",
                "acceptor_species": "h2o",
                "edge_scope": (
                    "induced_tpcl"
                    if donor_id in selected_ids and acceptor_id in selected_ids
                    else "incident_tpcl"
                ),
                "donor_in_tpcl": donor_node is not None,
                "acceptor_in_tpcl": acceptor_node is not None,
                "donor_arc_index": _node_value(donor_node, "arc_index"),
                "acceptor_arc_index": _node_value(acceptor_node, "arc_index"),
                "donor_theta_deg": _node_value(donor_node, "theta_deg"),
                "acceptor_theta_deg": _node_value(acceptor_node, "theta_deg"),
                "donor_normal_distance_A": _node_value(
                    donor_node, "normal_distance_A"
                ),
                "acceptor_normal_distance_A": _node_value(
                    acceptor_node, "normal_distance_A"
                ),
                "donor_tangential_offset_A": _node_value(
                    donor_node, "tangential_offset_A"
                ),
                "acceptor_tangential_offset_A": _node_value(
                    acceptor_node, "tangential_offset_A"
                ),
                "donor_surface_distance_A": _node_value(
                    donor_node, "surface_distance_A"
                ),
                "acceptor_surface_distance_A": _node_value(
                    acceptor_node, "surface_distance_A"
                ),
            }
        )

    induced_pairs = {
        pair
        for pair in incident_pairs
        if pair[0] in selected_ids and pair[1] in selected_ids
    }
    degrees = {oxygen_id: 0 for oxygen_id in nodes}
    internal_degrees = {oxygen_id: 0 for oxygen_id in nodes}
    for left, right in incident_pairs:
        if left in degrees:
            degrees[left] += 1
        if right in degrees:
            degrees[right] += 1
    for left, right in induced_pairs:
        internal_degrees[left] += 1
        internal_degrees[right] += 1
    for oxygen_id, node in nodes.items():
        observed = (
            donor_counts[oxygen_id],
            acceptor_counts[oxygen_id],
            degrees[oxygen_id],
            internal_degrees[oxygen_id],
        )
        expected = (
            node.hbond_donor_count,
            node.hbond_acceptor_count,
            node.hbond_degree,
            node.hbond_internal_tpcl_degree,
        )
        if observed != expected:
            raise ValueError(
                f"step {frame.step} oxygen {oxygen_id}: H-bond metric parity mismatch "
                f"observed={observed} expected={expected}"
            )

    time_values = {node.time_ns for node in nodes.values()}
    if len(time_values) != 1:
        raise ValueError(f"step {frame.step}: inconsistent sample times")
    frame_row = {
        "step": frame.step,
        "time_ns": time_values.pop(),
        "tpcl_node_count": len(nodes),
        "incident_undirected_edge_count": len(incident_pairs),
        "induced_undirected_edge_count": len(induced_pairs),
        "directed_hbond_count": len(directed_edges),
        "zero_induced_edges": not induced_pairs,
        "sample_metric_parity_pass": True,
    }
    return rows, frame_row


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    if len(args.trajectory) != len(args.trajectory_end_step):
        raise ValueError("each trajectory requires one declared end step")
    selected_steps = load_event_steps(args.event_windows) if args.event_windows else None
    node_frames = iter_tpcl_node_frames(args.water_samples, selected_steps)
    next_group: tuple[int, dict[int, TpclNode]] | None = next(node_frames)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    processed: set[int] = set()
    edge_count = 0
    frame_rows: list[dict[str, object]] = []
    stop = False
    with gzip.open(
        output / "tpcl_hbond_edges.csv.gz", "wt", newline="", encoding="utf-8"
    ) as edge_handle, (output / "tpcl_hbond_frames.csv").open(
        "w", newline="", encoding="utf-8"
    ) as frame_handle:
        edge_writer = csv.DictWriter(edge_handle, fieldnames=list(EDGE_FIELDS))
        frame_writer = csv.DictWriter(frame_handle, fieldnames=list(FRAME_FIELDS))
        edge_writer.writeheader()
        frame_writer.writeheader()
        for segment_index, (trajectory, end_step) in enumerate(
            zip(args.trajectory, args.trajectory_end_step)
        ):
            reached_end = False
            for raw_frame in iter_lammps_dump_records(trajectory):
                if next_group is None:
                    stop = True
                    break
                step = raw_frame.timestep
                if step > end_step:
                    raise ValueError(f"{trajectory}: passed declared end step {end_step}")
                if step == end_step:
                    reached_end = True
                replace_with_later_segment = (
                    segment_index < len(args.trajectory) - 1 and step == end_step
                )
                wanted_step, nodes = next_group
                if step > wanted_step:
                    raise ValueError(f"trajectory omitted selected sample step {wanted_step}")
                if step == wanted_step and not replace_with_later_segment:
                    frame = select_frame(
                        raw_frame,
                        args.surface_range,
                        args.water_range,
                        oxygen_type=args.oxygen_type,
                        hydrogen_type=args.hydrogen_type,
                    )
                    edge_rows, frame_row = analyze_hbond_frame(
                        frame,
                        nodes,
                        oh_cutoff_A=args.oh_cutoff_A,
                        oo_cutoff_A=args.oo_cutoff_A,
                        hbond_angle_deg=args.hbond_angle_deg,
                    )
                    edge_writer.writerows(edge_rows)
                    frame_writer.writerow(frame_row)
                    edge_count += len(edge_rows)
                    frame_rows.append(frame_row)
                    processed.add(step)
                    if args.max_frames is not None and len(processed) >= args.max_frames:
                        stop = True
                        break
                    try:
                        next_group = next(node_frames)
                    except StopIteration:
                        next_group = None
                        stop = True
                        break
                if step == end_step:
                    break
            if stop:
                break
            if not reached_end:
                raise ValueError(f"{trajectory}: declared end step {end_step} was not found")
    if not frame_rows:
        raise ValueError("no H-bond frames were analyzed")
    if args.max_frames is None and next_group is not None:
        missing_step, _ = next_group
        raise ValueError(f"trajectory omitted selected sample step {missing_step}")
    if (
        args.max_frames is None
        and selected_steps is not None
        and processed != selected_steps
    ):
        raise ValueError("processed frame support differs from event-window support")

    summary: dict[str, object] = {
        "status": "PASS",
        "case_id": args.case_id,
        "analyzed_frames": len(frame_rows),
        "first_step": int(frame_rows[0]["step"]),
        "last_step": int(frame_rows[-1]["step"]),
        "tpcl_node_frame_rows": sum(int(row["tpcl_node_count"]) for row in frame_rows),
        "directed_hbond_rows": edge_count,
        "frames_with_zero_induced_edges": sum(
            bool(row["zero_induced_edges"]) for row in frame_rows
        ),
        "sample_metric_parity_pass": all(
            bool(row["sample_metric_parity_pass"]) for row in frame_rows
        ),
        "scientific_status": SCIENTIFIC_STATUS,
    }
    manifest = {
        "case_id": args.case_id,
        "trajectories": [str(Path(path).resolve()) for path in args.trajectory],
        "trajectory_end_steps": args.trajectory_end_step,
        "water_samples": str(args.water_samples.resolve()),
        "event_windows": str(args.event_windows.resolve()) if args.event_windows else None,
        "surface_atom_range": list(args.surface_range),
        "water_atom_range": list(args.water_range),
        "oxygen_type": args.oxygen_type,
        "hydrogen_type": args.hydrogen_type,
        "oh_assignment_cutoff_A": args.oh_cutoff_A,
        "hbond_oo_cutoff_A": args.oo_cutoff_A,
        "hbond_angle_cutoff_deg": args.hbond_angle_deg,
        "edge_scope": "all_directed_water_water_edges_incident_to_frozen_tpcl_nodes",
        "node_membership_source": "frozen_local_water_order_sample_table",
        "restart_policy": "later_segment_replaces_duplicate_boundary_step",
        "max_frames": args.max_frames,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "\n".join(
            (
                "# Explicit TPCL H-bond edges",
                "",
                f"- Case: {args.case_id}",
                f"- Analyzed event-window frames: {summary['analyzed_frames']}",
                f"- Directed H-bond rows: {summary['directed_hbond_rows']}",
                "- Frozen sample-metric parity: PASS",
                "",
                (
                    "These are existing-trajectory structural records, not causal, "
                    "free-energy, replicate-level, or physical-rate evidence."
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
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-end-step", type=int, action="append", required=True)
    parser.add_argument("--water-samples", type=Path, required=True)
    parser.add_argument("--event-windows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-range", type=parse_range, required=True)
    parser.add_argument("--water-range", type=parse_range, required=True)
    parser.add_argument("--oxygen-type", type=int, default=2)
    parser.add_argument("--hydrogen-type", type=int, default=1)
    parser.add_argument("--oh-cutoff-A", type=float, default=1.25)
    parser.add_argument("--oo-cutoff-A", type=float, default=3.5)
    parser.add_argument("--hbond-angle-deg", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.oh_cutoff_A, args.oo_cutoff_A, args.hbond_angle_deg) <= 0.0:
        raise ValueError("H-bond cutoffs must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
