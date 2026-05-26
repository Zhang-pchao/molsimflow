"""Hydrogen-bond network summaries from explicit edge tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


H_BOND_TYPES = (
    "water_water",
    "water_ion_oxygen",
    "water_h3o",
    "water_oh",
    "water_cl",
    "water_tio2_surfaceO",
    "surfaceOH_water",
    "h3o_water",
    "oh_water",
)

EDGE_COLUMNS = (
    "case_label",
    "frame",
    "time",
    "donor_id",
    "acceptor_id",
    "hbond_type",
    "donor_species",
    "acceptor_species",
    "donor_s_A",
    "acceptor_s_A",
    "surface_gap_A",
    "source_row_index",
)


@dataclass(frozen=True)
class HbondEdgeRow:
    """One hydrogen-bond edge at a frame/time."""

    frame: int
    time: float
    donor_id: str
    acceptor_id: str
    hbond_type: str
    donor_species: str = ""
    acceptor_species: str = ""
    donor_s_A: float = math.nan
    acceptor_s_A: float = math.nan
    surface_gap_A: float = math.nan
    source_row_index: int = -1

    @property
    def undirected_edge(self) -> Tuple[str, str]:
        return tuple(sorted((self.donor_id, self.acceptor_id)))


@dataclass(frozen=True)
class HbondNetworkConfig:
    """Settings for H-bond network summaries."""

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
        "n_frames",
        "n_nodes",
        "n_edges",
        "n_hbond_total",
        "largest_hbond_component_size",
        "largest_hbond_component_fraction",
        "hbond_network_avg_degree",
        "hbond_network_spanning_axial",
        "hbond_type",
        "n_unique_runs",
        "dt_per_processed_frame_ps",
        "lifetime_mean_ps",
        "lifetime_median_ps",
        "lifetime_p90_ps",
        "lifetime_max_ps",
        "gap_bin_left_A",
        "gap_bin_right_A",
        "gap_bin_center_A",
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


def classify_hbond_type(donor_species: str, acceptor_species: str) -> Optional[str]:
    """Classify a donor/acceptor species pair into a migrated H-bond type."""

    donor = str(donor_species).strip()
    acceptor = str(acceptor_species).strip()
    if donor == "h2o" and acceptor == "h2o":
        return "water_water"
    if donor == "h3o" and acceptor == "h2o":
        return "h3o_water"
    if donor == "h2o" and acceptor == "h3o":
        return "water_h3o"
    if donor == "h2o" and acceptor == "oh":
        return "water_oh"
    if donor == "oh" and acceptor == "h2o":
        return "oh_water"
    if donor == "h2o" and acceptor == "cl":
        return "water_cl"
    if donor == "h2o" and acceptor == "surface_o":
        return "water_tio2_surfaceO"
    if donor == "surface_oh" and acceptor == "h2o":
        return "surfaceOH_water"
    return None


def load_hbond_edge_rows(
    path: Path,
    frame_column: str = "frame",
    time_column: str = "time",
    donor_column: str = "donor_id",
    acceptor_column: str = "acceptor_id",
    hbond_type_column: Optional[str] = "hbond_type",
    donor_species_column: Optional[str] = "donor_species",
    acceptor_species_column: Optional[str] = "acceptor_species",
    donor_s_column: Optional[str] = None,
    acceptor_s_column: Optional[str] = None,
    gap_column: Optional[str] = None,
) -> List[HbondEdgeRow]:
    """Load an explicit H-bond edge table."""

    raw_rows, fieldnames = _read_csv_rows(path)
    frame_src = _pick_column(fieldnames, [frame_column, "frame", "Frame", "frame_index", "timestep", "step"])
    time_src = _pick_column(fieldnames, [time_column, "time", "time_ns", "Time", "Time(ns)", "t"], required=False)
    donor_src = _pick_column(fieldnames, [donor_column, "donor_id", "donor_atom_id", "source_id"])
    acceptor_src = _pick_column(fieldnames, [acceptor_column, "acceptor_id", "acceptor_atom_id", "target_id"])
    type_src = _pick_column(fieldnames, [hbond_type_column, "hbond_type", "type", "edge_type"], required=False)
    donor_species_src = _pick_column(fieldnames, [donor_species_column, "donor_species", "source_species"], required=False)
    acceptor_species_src = _pick_column(
        fieldnames,
        [acceptor_species_column, "acceptor_species", "target_species"],
        required=False,
    )
    donor_s_src = _pick_column(fieldnames, [donor_s_column, "donor_s_A", "donor_s", "source_s_A"], required=False)
    acceptor_s_src = _pick_column(
        fieldnames,
        [acceptor_s_column, "acceptor_s_A", "acceptor_s", "target_s_A"],
        required=False,
    )
    gap_src = _pick_column(fieldnames, [gap_column, "surface_gap_A", "surface_gap_estimate_A", "gap_A"], required=False)

    rows: List[HbondEdgeRow] = []
    for index, raw in enumerate(raw_rows):
        donor_id = str(raw.get(donor_src, "")).strip()
        acceptor_id = str(raw.get(acceptor_src, "")).strip()
        if not donor_id or not acceptor_id:
            raise ValueError(f"Missing donor/acceptor id at source row {index}")
        donor_species = str(raw.get(donor_species_src, "")).strip() if donor_species_src else ""
        acceptor_species = str(raw.get(acceptor_species_src, "")).strip() if acceptor_species_src else ""
        if type_src is not None and str(raw.get(type_src, "")).strip():
            hbond_type = str(raw.get(type_src, "")).strip()
        else:
            hbond_type = classify_hbond_type(donor_species, acceptor_species) or "unknown"
        rows.append(
            HbondEdgeRow(
                frame=_as_int(raw.get(frame_src)),
                time=_as_float(raw.get(time_src)) if time_src is not None else math.nan,
                donor_id=donor_id,
                acceptor_id=acceptor_id,
                hbond_type=hbond_type,
                donor_species=donor_species,
                acceptor_species=acceptor_species,
                donor_s_A=_as_float(raw.get(donor_s_src)) if donor_s_src is not None else math.nan,
                acceptor_s_A=_as_float(raw.get(acceptor_s_src)) if acceptor_s_src is not None else math.nan,
                surface_gap_A=_as_float(raw.get(gap_src)) if gap_src is not None else math.nan,
                source_row_index=index,
            )
        )
    rows.sort(key=lambda row: (row.frame, row.time if math.isfinite(row.time) else row.frame, row.donor_id, row.acceptor_id))
    return rows


def _edge_rows_to_csv_rows(rows: Sequence[HbondEdgeRow], case_label: str) -> List[Dict[str, object]]:
    return [
        {
            "case_label": case_label,
            "frame": row.frame,
            "time": _format_float(row.time),
            "donor_id": row.donor_id,
            "acceptor_id": row.acceptor_id,
            "hbond_type": row.hbond_type,
            "donor_species": row.donor_species,
            "acceptor_species": row.acceptor_species,
            "donor_s_A": _format_float(row.donor_s_A),
            "acceptor_s_A": _format_float(row.acceptor_s_A),
            "surface_gap_A": _format_float(row.surface_gap_A),
            "source_row_index": row.source_row_index,
        }
        for row in rows
    ]


def _connected_components(nodes: Set[str], edges: Iterable[Tuple[str, str]]) -> List[Set[str]]:
    adjacency: Dict[str, List[str]] = {node: [] for node in nodes}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].append(b)
            adjacency[b].append(a)
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


def build_frame_network_summary(
    rows: Sequence[HbondEdgeRow],
    case_label: str = "",
    config: HbondNetworkConfig = HbondNetworkConfig(),
) -> List[Dict[str, object]]:
    """Compute per-frame H-bond network summaries."""

    config.validate()
    grouped: Dict[int, List[HbondEdgeRow]] = defaultdict(list)
    for row in rows:
        grouped[row.frame].append(row)
    all_types = list(dict.fromkeys(list(H_BOND_TYPES) + sorted({row.hbond_type for row in rows})))

    out: List[Dict[str, object]] = []
    for frame in sorted(grouped):
        chunk = grouped[frame]
        nodes = {row.donor_id for row in chunk} | {row.acceptor_id for row in chunk}
        primary_edges = {row.undirected_edge for row in chunk}
        components = _connected_components(nodes, primary_edges)
        largest = max((len(component) for component in components), default=0)
        axial_by_id: Dict[str, float] = {}
        for row in chunk:
            if math.isfinite(row.donor_s_A):
                axial_by_id[row.donor_id] = row.donor_s_A
            if math.isfinite(row.acceptor_s_A):
                axial_by_id[row.acceptor_id] = row.acceptor_s_A
        gap_values = [row.surface_gap_A for row in chunk if math.isfinite(row.surface_gap_A)]
        time_values = [row.time for row in chunk if math.isfinite(row.time)]
        item: Dict[str, object] = {
            "case_label": case_label,
            "frame": frame,
            "time": _format_float(time_values[0] if time_values else math.nan),
            "surface_gap_A": _format_float(float(np.mean(gap_values)) if gap_values else math.nan),
            "n_nodes": len(nodes),
            "n_edges": len(primary_edges),
            "n_hbond_total": len(chunk),
            "largest_hbond_component_size": largest,
            "largest_hbond_component_fraction": largest / max(len(nodes), 1),
            "hbond_network_avg_degree": 2.0 * len(primary_edges) / max(len(nodes), 1),
            "hbond_network_spanning_axial": int(
                _spanning_flag(
                    components,
                    axial_by_id=axial_by_id,
                    lower=config.bridge_s_min_A,
                    upper=config.bridge_s_max_A,
                    side_thickness=config.side_thickness_A,
                )
            ),
        }
        for hbond_type in all_types:
            item[f"n_hbond_{hbond_type}"] = sum(1 for row in chunk if row.hbond_type == hbond_type)
        out.append(item)
    return out


def summarize_hbond_lifetimes(rows: Sequence[HbondEdgeRow]) -> List[Dict[str, object]]:
    """Summarize consecutive-frame lifetimes for each H-bond type."""

    frame_ids = sorted({row.frame for row in rows})
    frame_times: Dict[int, float] = {}
    for frame in frame_ids:
        times = [row.time for row in rows if row.frame == frame and math.isfinite(row.time)]
        if times:
            frame_times[frame] = float(times[0])
    time_values = [frame_times[frame] for frame in frame_ids if frame in frame_times]
    if len(time_values) > 1:
        dt_ps = float(np.nanmedian(np.diff(np.asarray(time_values, dtype=float))) * 1000.0)
    else:
        dt_ps = 0.0

    frame_sets: Dict[int, Dict[str, Set[Tuple[str, str]]]] = {frame: defaultdict(set) for frame in frame_ids}
    for row in rows:
        frame_sets[row.frame][row.hbond_type].add(row.undirected_edge)
    all_types = list(dict.fromkeys(list(H_BOND_TYPES) + sorted({row.hbond_type for row in rows})))

    out: List[Dict[str, object]] = []
    for hbond_type in all_types:
        runs: List[int] = []
        active: Dict[Tuple[str, str], int] = {}
        for frame in frame_ids:
            current = set(frame_sets[frame].get(hbond_type, set()))
            for edge in list(active):
                if edge not in current:
                    runs.append(active.pop(edge))
            for edge in current:
                active[edge] = active.get(edge, 0) + 1
        runs.extend(active.values())
        values = np.asarray(runs, dtype=float) * dt_ps
        out.append(
            {
                "hbond_type": hbond_type,
                "n_unique_runs": int(len(values)),
                "dt_per_processed_frame_ps": dt_ps,
                "lifetime_mean_ps": _format_float(float(np.nanmean(values)) if values.size else math.nan),
                "lifetime_median_ps": _format_float(float(np.nanmedian(values)) if values.size else math.nan),
                "lifetime_p90_ps": _format_float(float(np.nanpercentile(values, 90.0)) if values.size else math.nan),
                "lifetime_max_ps": _format_float(float(np.nanmax(values)) if values.size else math.nan),
            }
        )
    return out


def _numeric_mean(values: Sequence[object]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def _numeric_median(values: Sequence[object]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else math.nan


def _numeric_sem(values: Sequence[object]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    if arr.size == 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(float(arr.size)))


def summarize_hbond_by_gap(
    frame_summary: Sequence[Mapping[str, object]],
    gap_bin_width_A: float = 2.0,
    min_bin_count: int = 1,
) -> List[Dict[str, object]]:
    """Summarize frame-level network metrics by surface-gap bins."""

    gap_values = [_as_float(row.get("surface_gap_A")) for row in frame_summary]
    finite = [value for value in gap_values if math.isfinite(value)]
    if not finite:
        return []
    low = math.floor(min(finite) / float(gap_bin_width_A)) * float(gap_bin_width_A)
    high = math.ceil(max(finite) / float(gap_bin_width_A)) * float(gap_bin_width_A)
    if math.isclose(low, high):
        high = low + float(gap_bin_width_A)
    edges = np.arange(low, high + float(gap_bin_width_A) * 0.5, float(gap_bin_width_A))
    metric_columns = sorted(
        {
            key
            for row in frame_summary
            for key in row.keys()
            if key.startswith("n_hbond_")
            or key
            in {
                "n_nodes",
                "n_edges",
                "n_hbond_total",
                "largest_hbond_component_fraction",
                "hbond_network_avg_degree",
                "hbond_network_spanning_axial",
            }
        }
    )
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
            "surface_gap_A_mean": _numeric_mean([row.get("surface_gap_A") for row in chunk]),
        }
        for column in metric_columns:
            values = [row.get(column) for row in chunk]
            item[f"{column}_mean"] = _format_float(_numeric_mean(values))
            item[f"{column}_median"] = _format_float(_numeric_median(values))
            item[f"{column}_sem"] = _format_float(_numeric_sem(values))
        out.append(item)
    return out


def analyze_hbond_network(
    input_csv: Path,
    output_dir: Path,
    case_label: str = "",
    config: HbondNetworkConfig = HbondNetworkConfig(),
    frame_column: str = "frame",
    time_column: str = "time",
    donor_column: str = "donor_id",
    acceptor_column: str = "acceptor_id",
    hbond_type_column: Optional[str] = "hbond_type",
    donor_species_column: Optional[str] = "donor_species",
    acceptor_species_column: Optional[str] = "acceptor_species",
    donor_s_column: Optional[str] = None,
    acceptor_s_column: Optional[str] = None,
    gap_column: Optional[str] = None,
) -> Dict[str, Path]:
    """Run H-bond network summaries and write CSV outputs."""

    rows = load_hbond_edge_rows(
        input_csv,
        frame_column=frame_column,
        time_column=time_column,
        donor_column=donor_column,
        acceptor_column=acceptor_column,
        hbond_type_column=hbond_type_column,
        donor_species_column=donor_species_column,
        acceptor_species_column=acceptor_species_column,
        donor_s_column=donor_s_column,
        acceptor_s_column=acceptor_s_column,
        gap_column=gap_column,
    )
    frame_summary = build_frame_network_summary(rows, case_label=case_label, config=config)
    lifetimes = summarize_hbond_lifetimes(rows)
    gap_summary = summarize_hbond_by_gap(
        frame_summary,
        gap_bin_width_A=config.gap_bin_width_A,
        min_bin_count=config.min_bin_count,
    )

    output_dir = Path(output_dir)
    outputs = {
        "edge_table": output_dir / "hbond_edge_table.csv",
        "frame_summary": output_dir / "hbond_frame_summary.csv",
        "lifetime_summary": output_dir / "hbond_lifetime_summary.csv",
        "gap_summary": output_dir / "hbond_gap_summary.csv",
        "state_statistics": output_dir / "state_statistics.csv",
    }
    _write_csv_rows(outputs["edge_table"], _edge_rows_to_csv_rows(rows, case_label), fieldnames=EDGE_COLUMNS)
    _write_csv_rows(outputs["frame_summary"], frame_summary)
    _write_csv_rows(outputs["lifetime_summary"], lifetimes)
    _write_csv_rows(outputs["gap_summary"], gap_summary)
    _write_csv_rows(
        outputs["state_statistics"],
        [
            {"metric": "input_csv", "value": str(input_csv)},
            {"metric": "case_label", "value": case_label},
            {"metric": "n_edge_rows", "value": len(rows)},
            {"metric": "n_frames", "value": len({row.frame for row in rows})},
            {"metric": "hbond_types", "value": ",".join(sorted({row.hbond_type for row in rows}))},
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
    parser = argparse.ArgumentParser(description="Summarize H-bond networks from explicit edge tables")
    parser.add_argument("--input", type=Path, required=True, help="Input H-bond edge CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-label", default="")
    parser.add_argument("--frame-column", default="frame")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--donor-column", default="donor_id")
    parser.add_argument("--acceptor-column", default="acceptor_id")
    parser.add_argument("--hbond-type-column", default="hbond_type")
    parser.add_argument("--donor-species-column", default="donor_species")
    parser.add_argument("--acceptor-species-column", default="acceptor_species")
    parser.add_argument("--donor-s-column")
    parser.add_argument("--acceptor-s-column")
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
        outputs = analyze_hbond_network(
            input_csv=args.input,
            output_dir=args.output_dir,
            case_label=args.case_label,
            config=HbondNetworkConfig(
                bridge_s_min_A=args.bridge_s_min_A,
                bridge_s_max_A=args.bridge_s_max_A,
                side_thickness_A=args.side_thickness_A,
                gap_bin_width_A=args.gap_bin_width_A,
                min_bin_count=args.min_bin_count,
            ),
            frame_column=args.frame_column,
            time_column=args.time_column,
            donor_column=args.donor_column,
            acceptor_column=args.acceptor_column,
            hbond_type_column=args.hbond_type_column,
            donor_species_column=args.donor_species_column,
            acceptor_species_column=args.acceptor_species_column,
            donor_s_column=args.donor_s_column,
            acceptor_s_column=args.acceptor_s_column,
            gap_column=args.gap_column,
        )
    except Exception as exc:
        print(f"H-bond network analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
