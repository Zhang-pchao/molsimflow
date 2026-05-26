"""Generic contact-graph topology summaries from explicit edge tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


EDGE_COLUMNS = (
    "case_label",
    "frame",
    "time",
    "source_id",
    "target_id",
    "edge_type",
    "source_role",
    "target_role",
    "source_region",
    "target_region",
    "source_s_A",
    "target_s_A",
    "surface_gap_A",
    "source_row_index",
)


@dataclass(frozen=True)
class ContactEdgeRow:
    """One contact edge at a frame/time."""

    frame: int
    time: float
    source_id: str
    target_id: str
    edge_type: str = ""
    source_role: str = ""
    target_role: str = ""
    source_region: str = ""
    target_region: str = ""
    source_s_A: float = math.nan
    target_s_A: float = math.nan
    surface_gap_A: float = math.nan
    source_row_index: int = -1

    @property
    def undirected_edge(self) -> Tuple[str, str]:
        return tuple(sorted((self.source_id, self.target_id)))


@dataclass(frozen=True)
class ContactGraphConfig:
    """Settings for contact-graph summaries."""

    bridge_s_min_A: float = -10.0
    bridge_s_max_A: float = 10.0
    side_thickness_A: float = 1.0
    gap_bin_width_A: float = 2.0
    min_bin_count: int = 1

    def validate(self) -> None:
        if self.bridge_s_max_A <= self.bridge_s_min_A:
            raise ValueError("bridge_s_max_A must be greater than bridge_s_min_A")
        if self.side_thickness_A < 0.0:
            raise ValueError("side_thickness_A must be non-negative")
        if self.gap_bin_width_A <= 0.0:
            raise ValueError("gap_bin_width_A must be positive")
        if self.min_bin_count < 1:
            raise ValueError("min_bin_count must be >= 1")


def _read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return [dict(row) for row in reader], [str(field) for field in reader.fieldnames]


def _write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = _ordered_fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _ordered_fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "case_label",
        "frame",
        "time",
        "surface_gap_A",
        "n_nodes",
        "n_edges",
        "n_components",
        "largest_component_size",
        "largest_component_fraction",
        "largest_component_fraction_bridge",
        "bridge_spanning_flag",
        "avg_degree",
        "avg_degree_water",
        "avg_degree_ion",
        "avg_degree_surface",
        "ion_mediated_edge_fraction",
        "surface_mediated_edge_fraction",
        "articulation_node_fraction",
        "articulation_water_count",
        "articulation_ion_count",
        "articulation_surface_count",
        "cycle_rank",
        "network_fracture_index",
        "gap_bin_left_A",
        "gap_bin_right_A",
        "gap_bin_center_A",
        "n_frames",
        "metric",
        "value",
    ]
    ordered = [key for key in preferred if key in keys]
    ordered.extend(key for key in keys if key not in ordered)
    return ordered


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: object, default: int = -1) -> int:
    number = _as_float(value)
    return int(round(number)) if math.isfinite(number) else default


def _format_float(value: float) -> object:
    return value if math.isfinite(value) else ""


def _pick_column(fieldnames: Sequence[str], candidates: Sequence[Optional[str]], required: bool = True) -> Optional[str]:
    available = set(fieldnames)
    for candidate in candidates:
        if candidate and candidate in available:
            return candidate
    if required:
        requested = ", ".join(str(candidate) for candidate in candidates if candidate)
        raise ValueError(f"Could not find required column among [{requested}]; available columns: {', '.join(fieldnames)}")
    return None


def load_contact_edge_rows(
    path: Path,
    frame_column: str = "frame",
    time_column: str = "time",
    source_column: str = "source_id",
    target_column: str = "target_id",
    edge_type_column: Optional[str] = "edge_type",
    source_role_column: Optional[str] = "source_role",
    target_role_column: Optional[str] = "target_role",
    source_region_column: Optional[str] = "source_region",
    target_region_column: Optional[str] = "target_region",
    source_s_column: Optional[str] = None,
    target_s_column: Optional[str] = None,
    gap_column: Optional[str] = None,
) -> List[ContactEdgeRow]:
    """Load contact graph edges from a CSV table."""

    raw_rows, fieldnames = _read_csv_rows(path)
    frame_src = _pick_column(fieldnames, [frame_column, "frame", "Frame", "frame_index", "timestep", "step"])
    time_src = _pick_column(fieldnames, [time_column, "time", "time_ns", "Time", "Time(ns)", "t"], required=False)
    source_src = _pick_column(fieldnames, [source_column, "source_id", "atom_i", "node_i", "i"])
    target_src = _pick_column(fieldnames, [target_column, "target_id", "atom_j", "node_j", "j"])
    edge_type_src = _pick_column(fieldnames, [edge_type_column, "edge_type", "contact_type", "type"], required=False)
    source_role_src = _pick_column(fieldnames, [source_role_column, "source_role", "role_i"], required=False)
    target_role_src = _pick_column(fieldnames, [target_role_column, "target_role", "role_j"], required=False)
    source_region_src = _pick_column(fieldnames, [source_region_column, "source_region", "region_i"], required=False)
    target_region_src = _pick_column(fieldnames, [target_region_column, "target_region", "region_j"], required=False)
    source_s_src = _pick_column(fieldnames, [source_s_column, "source_s_A", "source_s", "s_i"], required=False)
    target_s_src = _pick_column(fieldnames, [target_s_column, "target_s_A", "target_s", "s_j"], required=False)
    gap_src = _pick_column(fieldnames, [gap_column, "surface_gap_A", "surface_gap_estimate_A", "gap_A"], required=False)

    rows: List[ContactEdgeRow] = []
    for index, raw in enumerate(raw_rows):
        source_id = str(raw.get(source_src, "")).strip()
        target_id = str(raw.get(target_src, "")).strip()
        if not source_id or not target_id:
            raise ValueError(f"Missing contact source/target id at source row {index}")
        rows.append(
            ContactEdgeRow(
                frame=_as_int(raw.get(frame_src)),
                time=_as_float(raw.get(time_src)) if time_src is not None else math.nan,
                source_id=source_id,
                target_id=target_id,
                edge_type=str(raw.get(edge_type_src, "")).strip() if edge_type_src else "",
                source_role=str(raw.get(source_role_src, "")).strip() if source_role_src else "",
                target_role=str(raw.get(target_role_src, "")).strip() if target_role_src else "",
                source_region=str(raw.get(source_region_src, "")).strip() if source_region_src else "",
                target_region=str(raw.get(target_region_src, "")).strip() if target_region_src else "",
                source_s_A=_as_float(raw.get(source_s_src)) if source_s_src else math.nan,
                target_s_A=_as_float(raw.get(target_s_src)) if target_s_src else math.nan,
                surface_gap_A=_as_float(raw.get(gap_src)) if gap_src else math.nan,
                source_row_index=index,
            )
        )
    rows.sort(key=lambda row: (row.frame, row.time if math.isfinite(row.time) else row.frame, row.source_id, row.target_id))
    return rows


def _edge_rows_to_csv_rows(rows: Sequence[ContactEdgeRow], case_label: str) -> List[Dict[str, object]]:
    return [
        {
            "case_label": case_label,
            "frame": row.frame,
            "time": _format_float(row.time),
            "source_id": row.source_id,
            "target_id": row.target_id,
            "edge_type": row.edge_type,
            "source_role": row.source_role,
            "target_role": row.target_role,
            "source_region": row.source_region,
            "target_region": row.target_region,
            "source_s_A": _format_float(row.source_s_A),
            "target_s_A": _format_float(row.target_s_A),
            "surface_gap_A": _format_float(row.surface_gap_A),
            "source_row_index": row.source_row_index,
        }
        for row in rows
    ]


def _adjacency(nodes: Set[str], edges: Iterable[Tuple[str, str]]) -> Dict[str, List[str]]:
    adjacency: Dict[str, List[str]] = {node: [] for node in nodes}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].append(b)
            adjacency[b].append(a)
    return adjacency


def _connected_components(nodes: Set[str], edges: Iterable[Tuple[str, str]]) -> List[Set[str]]:
    adjacency = _adjacency(nodes, edges)
    seen: Set[str] = set()
    components: List[Set[str]] = []
    for node in sorted(adjacency):
        if node in seen:
            continue
        queue: deque[str] = deque([node])
        seen.add(node)
        component: Set[str] = set()
        while queue:
            item = queue.popleft()
            component.add(item)
            for neighbor in adjacency[item]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _articulation_points(nodes: Set[str], edges: Set[Tuple[str, str]]) -> Set[str]:
    if len(nodes) <= 2:
        return set()
    base_count = len(_connected_components(nodes, edges))
    out: Set[str] = set()
    for node in nodes:
        remaining_nodes = set(nodes)
        remaining_nodes.remove(node)
        remaining_edges = {edge for edge in edges if node not in edge}
        if len(_connected_components(remaining_nodes, remaining_edges)) > base_count:
            out.add(node)
    return out


def _mean(values: Sequence[object]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def _median(values: Sequence[object]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else math.nan


def _sem(values: Sequence[object]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    if arr.size == 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(float(arr.size)))


def _spanning_flag(
    components: Sequence[Set[str]],
    axial_by_id: Mapping[str, float],
    lower: float,
    upper: float,
    side_thickness: float,
) -> bool:
    for component in components:
        axials = np.asarray([_as_float(axial_by_id.get(atom_id)) for atom_id in component], dtype=float)
        axials = axials[np.isfinite(axials)]
        if axials.size == 0:
            continue
        if np.any(axials <= float(lower) + float(side_thickness)) and np.any(axials >= float(upper) - float(side_thickness)):
            return True
    return False


def _is_ion_role(role: str) -> bool:
    return str(role).lower() in {"ion", "cation", "anion", "na", "cl", "h3o", "oh"}


def _is_surface_role(role: str) -> bool:
    return str(role).lower() in {"surface", "surface_o", "surface_oh", "tio2", "solid"}


def build_contact_graph_summary(
    rows: Sequence[ContactEdgeRow],
    case_label: str = "",
    config: ContactGraphConfig = ContactGraphConfig(),
) -> List[Dict[str, object]]:
    """Compute per-frame contact graph topology metrics."""

    config.validate()
    grouped: Dict[int, List[ContactEdgeRow]] = defaultdict(list)
    for row in rows:
        grouped[row.frame].append(row)

    out: List[Dict[str, object]] = []
    for frame in sorted(grouped):
        chunk = grouped[frame]
        nodes = {row.source_id for row in chunk} | {row.target_id for row in chunk}
        edges = {row.undirected_edge for row in chunk}
        components = _connected_components(nodes, edges)
        largest = max((len(component) for component in components), default=0)
        role_by_node: Dict[str, str] = {}
        region_by_node: Dict[str, str] = {}
        axial_by_node: Dict[str, float] = {}
        for row in chunk:
            if row.source_role:
                role_by_node[row.source_id] = row.source_role
            if row.target_role:
                role_by_node[row.target_id] = row.target_role
            if row.source_region:
                region_by_node[row.source_id] = row.source_region
            if row.target_region:
                region_by_node[row.target_id] = row.target_region
            if math.isfinite(row.source_s_A):
                axial_by_node[row.source_id] = row.source_s_A
            if math.isfinite(row.target_s_A):
                axial_by_node[row.target_id] = row.target_s_A

        bridge_nodes = {node for node in nodes if region_by_node.get(node, "bridge") == "bridge"}
        largest_bridge = 0
        for component in components:
            largest_bridge = max(largest_bridge, len(component & bridge_nodes))
        degrees = {node: 0 for node in nodes}
        for a, b in edges:
            degrees[a] += 1
            degrees[b] += 1

        def mean_degree_for_role(role: str) -> float:
            values = [degrees[node] for node, node_role in role_by_node.items() if node_role == role]
            return float(np.mean(values)) if values else 0.0

        ion_edges = 0
        surface_edges = 0
        for a, b in edges:
            roles = {role_by_node.get(a, ""), role_by_node.get(b, "")}
            if any(_is_ion_role(role) for role in roles):
                ion_edges += 1
            if any(_is_surface_role(role) for role in roles):
                surface_edges += 1

        articulation = _articulation_points(nodes, edges)
        n_components = len(components)
        cycle_rank = max(0, len(edges) - len(nodes) + n_components)
        gap_values = [row.surface_gap_A for row in chunk if math.isfinite(row.surface_gap_A)]
        time_values = [row.time for row in chunk if math.isfinite(row.time)]
        item = {
            "case_label": case_label,
            "frame": frame,
            "time": _format_float(time_values[0] if time_values else math.nan),
            "surface_gap_A": _format_float(float(np.mean(gap_values)) if gap_values else math.nan),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_components": n_components,
            "largest_component_size": largest,
            "largest_component_fraction": largest / max(len(nodes), 1),
            "largest_component_fraction_bridge": largest_bridge / max(len(bridge_nodes), 1),
            "bridge_spanning_flag": int(
                _spanning_flag(
                    components,
                    axial_by_id=axial_by_node,
                    lower=config.bridge_s_min_A,
                    upper=config.bridge_s_max_A,
                    side_thickness=config.side_thickness_A,
                )
            ),
            "avg_degree": 2.0 * len(edges) / max(len(nodes), 1),
            "avg_degree_water": mean_degree_for_role("water"),
            "avg_degree_ion": mean_degree_for_role("ion"),
            "avg_degree_surface": mean_degree_for_role("surface"),
            "ion_mediated_edge_fraction": ion_edges / max(len(edges), 1),
            "surface_mediated_edge_fraction": surface_edges / max(len(edges), 1),
            "articulation_node_fraction": len(articulation) / max(len(nodes), 1),
            "articulation_water_count": sum(1 for node in articulation if role_by_node.get(node) == "water"),
            "articulation_ion_count": sum(1 for node in articulation if role_by_node.get(node) == "ion"),
            "articulation_surface_count": sum(1 for node in articulation if role_by_node.get(node) == "surface"),
            "cycle_rank": cycle_rank,
            "network_fracture_index": n_components / max(len(nodes), 1) + len(articulation) / max(len(nodes), 1),
        }
        out.append(item)
    return out


def summarize_contact_graph_by_gap(
    frame_summary: Sequence[Mapping[str, object]],
    gap_bin_width_A: float = 2.0,
    min_bin_count: int = 1,
) -> List[Dict[str, object]]:
    """Summarize frame-level contact graph metrics by surface-gap bins."""

    gap_values = [_as_float(row.get("surface_gap_A")) for row in frame_summary]
    finite = [value for value in gap_values if math.isfinite(value)]
    if not finite:
        return []
    low = math.floor(min(finite) / float(gap_bin_width_A)) * float(gap_bin_width_A)
    high = math.ceil(max(finite) / float(gap_bin_width_A)) * float(gap_bin_width_A)
    if math.isclose(low, high):
        high = low + float(gap_bin_width_A)
    edges = np.arange(low, high + float(gap_bin_width_A) * 0.5, float(gap_bin_width_A))
    metric_columns = [
        "n_nodes",
        "n_edges",
        "n_components",
        "largest_component_fraction",
        "largest_component_fraction_bridge",
        "bridge_spanning_flag",
        "avg_degree",
        "avg_degree_water",
        "avg_degree_ion",
        "avg_degree_surface",
        "ion_mediated_edge_fraction",
        "surface_mediated_edge_fraction",
        "articulation_node_fraction",
        "cycle_rank",
        "network_fracture_index",
    ]
    out: List[Dict[str, object]] = []
    for index in range(len(edges) - 1):
        left = float(edges[index])
        right = float(edges[index + 1])
        if index == len(edges) - 2:
            chunk = [
                row
                for row, gap in zip(frame_summary, gap_values)
                if math.isfinite(gap) and ((gap >= left and gap < right) or math.isclose(gap, right))
            ]
        else:
            chunk = [row for row, gap in zip(frame_summary, gap_values) if math.isfinite(gap) and gap >= left and gap < right]
        if len(chunk) < int(min_bin_count):
            continue
        item: Dict[str, object] = {
            "gap_bin_left_A": left,
            "gap_bin_right_A": right,
            "gap_bin_center_A": 0.5 * (left + right),
            "n_frames": len(chunk),
            "surface_gap_A_mean": _mean([row.get("surface_gap_A") for row in chunk]),
        }
        for column in metric_columns:
            values = [row.get(column) for row in chunk]
            item[f"{column}_mean"] = _format_float(_mean(values))
            item[f"{column}_median"] = _format_float(_median(values))
            item[f"{column}_sem"] = _format_float(_sem(values))
        out.append(item)
    return out


def analyze_contact_graph(
    input_csv: Path,
    output_dir: Path,
    case_label: str = "",
    config: ContactGraphConfig = ContactGraphConfig(),
    frame_column: str = "frame",
    time_column: str = "time",
    source_column: str = "source_id",
    target_column: str = "target_id",
    edge_type_column: Optional[str] = "edge_type",
    source_role_column: Optional[str] = "source_role",
    target_role_column: Optional[str] = "target_role",
    source_region_column: Optional[str] = "source_region",
    target_region_column: Optional[str] = "target_region",
    source_s_column: Optional[str] = None,
    target_s_column: Optional[str] = None,
    gap_column: Optional[str] = None,
) -> Dict[str, Path]:
    """Run contact graph topology summaries and write CSV outputs."""

    rows = load_contact_edge_rows(
        input_csv,
        frame_column=frame_column,
        time_column=time_column,
        source_column=source_column,
        target_column=target_column,
        edge_type_column=edge_type_column,
        source_role_column=source_role_column,
        target_role_column=target_role_column,
        source_region_column=source_region_column,
        target_region_column=target_region_column,
        source_s_column=source_s_column,
        target_s_column=target_s_column,
        gap_column=gap_column,
    )
    frame_summary = build_contact_graph_summary(rows, case_label=case_label, config=config)
    gap_summary = summarize_contact_graph_by_gap(
        frame_summary,
        gap_bin_width_A=config.gap_bin_width_A,
        min_bin_count=config.min_bin_count,
    )
    output_dir = Path(output_dir)
    outputs = {
        "edge_table": output_dir / "contact_graph_edges.csv",
        "frame_summary": output_dir / "contact_graph_frame_summary.csv",
        "gap_summary": output_dir / "contact_graph_gap_summary.csv",
        "state_statistics": output_dir / "state_statistics.csv",
    }
    _write_csv_rows(outputs["edge_table"], _edge_rows_to_csv_rows(rows, case_label), fieldnames=EDGE_COLUMNS)
    _write_csv_rows(outputs["frame_summary"], frame_summary)
    _write_csv_rows(outputs["gap_summary"], gap_summary)
    _write_csv_rows(
        outputs["state_statistics"],
        [
            {"metric": "input_csv", "value": str(input_csv)},
            {"metric": "case_label", "value": case_label},
            {"metric": "n_edge_rows", "value": len(rows)},
            {"metric": "n_frames", "value": len({row.frame for row in rows})},
            {"metric": "edge_types", "value": ",".join(sorted({row.edge_type for row in rows if row.edge_type}))},
            {"metric": "bridge_s_min_A", "value": config.bridge_s_min_A},
            {"metric": "bridge_s_max_A", "value": config.bridge_s_max_A},
            {"metric": "side_thickness_A", "value": config.side_thickness_A},
            {"metric": "gap_bin_width_A", "value": config.gap_bin_width_A},
            {"metric": "min_bin_count", "value": config.min_bin_count},
        ],
        fieldnames=["metric", "value"],
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize contact graph topology from explicit edge tables")
    parser.add_argument("--input", type=Path, required=True, help="Input contact edge CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-label", default="")
    parser.add_argument("--frame-column", default="frame")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--source-column", default="source_id")
    parser.add_argument("--target-column", default="target_id")
    parser.add_argument("--edge-type-column", default="edge_type")
    parser.add_argument("--source-role-column", default="source_role")
    parser.add_argument("--target-role-column", default="target_role")
    parser.add_argument("--source-region-column", default="source_region")
    parser.add_argument("--target-region-column", default="target_region")
    parser.add_argument("--source-s-column")
    parser.add_argument("--target-s-column")
    parser.add_argument("--gap-column")
    parser.add_argument("--bridge-s-min-A", type=float, default=-10.0)
    parser.add_argument("--bridge-s-max-A", type=float, default=10.0)
    parser.add_argument("--side-thickness-A", type=float, default=1.0)
    parser.add_argument("--gap-bin-width-A", type=float, default=2.0)
    parser.add_argument("--min-bin-count", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outputs = analyze_contact_graph(
            input_csv=args.input,
            output_dir=args.output_dir,
            case_label=args.case_label,
            config=ContactGraphConfig(
                bridge_s_min_A=args.bridge_s_min_A,
                bridge_s_max_A=args.bridge_s_max_A,
                side_thickness_A=args.side_thickness_A,
                gap_bin_width_A=args.gap_bin_width_A,
                min_bin_count=args.min_bin_count,
            ),
            frame_column=args.frame_column,
            time_column=args.time_column,
            source_column=args.source_column,
            target_column=args.target_column,
            edge_type_column=args.edge_type_column,
            source_role_column=args.source_role_column,
            target_role_column=args.target_role_column,
            source_region_column=args.source_region_column,
            target_region_column=args.target_region_column,
            source_s_column=args.source_s_column,
            target_s_column=args.target_s_column,
            gap_column=args.gap_column,
        )
    except Exception as exc:
        print(f"Contact graph analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
