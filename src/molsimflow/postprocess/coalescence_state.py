"""Two-bubble coalescence state assignment from table outputs.

This module provides a dependency-light rewrite of the legacy coalescence-state
workflow.  It keeps the reusable state-assignment logic while removing private
case paths, plotting side effects, and legacy package imports.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


STATE_COLUMNS: Tuple[str, ...] = (
    "sample_index",
    "time_ns",
    "time_ps",
    "d3d_all",
    "surface_gap_contact_distance_A",
    "surface_gap_estimate_A",
    "bridge_cyl_env.sum",
    "bridge_cyl_env.mean",
    "n2A_num",
    "n2B_num",
    "sumA_cn.sum",
    "sumB_cn.sum",
    "Bubble1Size",
    "Bubble2Size",
    "TotalSize",
    "dominant_bubble_size",
    "minor_bubble_size",
    "dominant_fraction_of_initial_total",
    "minor_fraction_of_initial_total",
    "both_large_fraction_of_initial_single",
    "raw_state",
    "state",
    "state_confidence",
    "state_basis",
)

DEFAULT_COLVAR_COLUMNS: Tuple[str, ...] = (
    "d3d_all",
    "bridge_cyl_env.sum",
    "opes.bias",
    "opes_e.bias",
)

DEFAULT_POST_COLUMNS: Tuple[str, ...] = (
    "n2A_num",
    "n2B_num",
    "sumA_cn.sum",
    "sumB_cn.sum",
    "bridge_cyl_env.mean",
    "wallA.bias",
    "wallB.bias",
)


@dataclass(frozen=True)
class CoalescenceStateConfig:
    """Thresholds and time controls for two-bubble state assignment."""

    start_ns: float = 0.0
    end_ns: Optional[float] = None
    sample_interval_ns: float = 0.001
    time_tolerance_ns: float = 0.00051
    bubble_time_tolerance_ns: float = 0.00051
    nominal_radius_A: float = 19.0
    surface_contact_distance_A: Optional[float] = None
    close_gap_A: float = 0.0
    separated_min_single_fraction: float = 0.60
    merged_major_total_fraction: float = 0.85
    merged_minor_total_fraction: float = 0.10
    min_persist_samples: int = 3
    cv_bins: int = 40

    @property
    def contact_distance_A(self) -> float:
        if self.surface_contact_distance_A is not None:
            return float(self.surface_contact_distance_A)
        return 2.0 * float(self.nominal_radius_A)


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _finite_values(values: Iterable[object]) -> List[float]:
    out = []
    for value in values:
        number = _as_float(value)
        if math.isfinite(number):
            out.append(number)
    return out


def _median(values: Iterable[object]) -> float:
    finite = _finite_values(values)
    return float(np.median(finite)) if finite else math.nan


def time_scale_to_ns(unit: str) -> float:
    """Return conversion factor from a time unit to nanoseconds."""

    value = unit.lower()
    if value == "ns":
        return 1.0
    if value == "ps":
        return 1.0e-3
    if value == "fs":
        return 1.0e-6
    raise ValueError(f"Unsupported time unit: {unit}")


def _read_whitespace_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    fields: Optional[List[str]] = None
    rows: List[List[str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.lstrip("#").strip().split()
                if parts and parts[0] == "!" and len(parts) >= 3 and parts[1] == "FIELDS":
                    fields = parts[2:]
                elif parts and parts[0] == "FIELDS" and len(parts) >= 2:
                    fields = parts[1:]
                continue
            parts = line.split()
            if fields is None and any(not _looks_numeric(item) for item in parts):
                fields = parts
                continue
            rows.append(parts)
    if fields is None:
        raise ValueError(f"Could not infer table fields from {path}")
    return fields, rows


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def read_plumed_table(
    path: Path,
    time_unit: str = "ps",
    optional_columns: Sequence[str] = DEFAULT_COLVAR_COLUMNS,
) -> List[Dict[str, object]]:
    """Read a PLUMED-style whitespace table with a `#! FIELDS` header."""

    fields, raw_rows = _read_whitespace_table(path)
    if "time" not in fields:
        raise ValueError(f"PLUMED table is missing required time column: {path}")
    selected = ["time"] + [column for column in optional_columns if column in fields]
    indices = {column: fields.index(column) for column in selected}
    scale = time_scale_to_ns(time_unit)
    rows: List[Dict[str, object]] = []
    for raw in raw_rows:
        if len(raw) < len(fields):
            continue
        row: Dict[str, object] = {}
        for column, index in indices.items():
            row[column] = _as_float(raw[index])
        time_value = _as_float(row.get("time"))
        if not math.isfinite(time_value):
            continue
        row["time_ns"] = time_value * scale
        rows.append(row)
    rows.sort(key=lambda item: float(item["time_ns"]))
    return _drop_duplicate_times(rows)


def read_bubble_evolution(path: Path) -> List[Dict[str, object]]:
    """Read a legacy bubble-evolution table if it has two-bubble size columns."""

    data_rows: List[List[str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            data_rows.append(line.split())
    if not data_rows or len(data_rows[0]) < 9:
        return []

    fields = [
        "FrameIndex",
        "time_ns",
        "Bubble1Size",
        "Bubble2Size",
        "TotalSize",
        "Bubble1Pct",
        "Bubble2Pct",
        "TotalPct",
        "TimePeriod",
    ]
    rows: List[Dict[str, object]] = []
    for raw in data_rows:
        row = {field: _as_float(raw[index]) for index, field in enumerate(fields) if index < len(raw)}
        if math.isfinite(_as_float(row.get("time_ns"))):
            rows.append(row)
    rows.sort(key=lambda item: float(item["time_ns"]))
    return _drop_duplicate_times(rows)


def _drop_duplicate_times(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for row in rows:
        time_ns = _as_float(row.get("time_ns"))
        if not math.isfinite(time_ns):
            continue
        key = round(time_ns, 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def regular_sample_times(start_ns: float, end_ns: float, interval_ns: float) -> np.ndarray:
    """Return inclusive regular sample times."""

    if interval_ns <= 0.0:
        raise ValueError("interval_ns must be positive")
    if end_ns < start_ns:
        raise ValueError("end_ns must be >= start_ns")
    n = int(math.floor((float(end_ns) - float(start_ns)) / float(interval_ns) + 1e-9)) + 1
    values = float(start_ns) + np.arange(max(n, 1), dtype=float) * float(interval_ns)
    if values.size == 0 or values[-1] < end_ns - 1e-9:
        values = np.append(values, float(end_ns))
    return values


def _nearest_row(
    rows: Sequence[Mapping[str, object]],
    time_ns: float,
    tolerance_ns: float,
) -> Optional[Mapping[str, object]]:
    if not rows:
        return None
    times = np.asarray([_as_float(row.get("time_ns")) for row in rows], dtype=float)
    if times.size == 0:
        return None
    index = int(np.searchsorted(times, time_ns))
    candidates = []
    if index < times.size:
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(float(times[item]) - float(time_ns)))
    if abs(float(times[best]) - float(time_ns)) <= float(tolerance_ns):
        return rows[best]
    return None


def nearest_merge_rows(
    anchor_rows: Sequence[Mapping[str, object]],
    optional_rows: Sequence[Mapping[str, object]],
    tolerance_ns: float,
    columns: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Nearest-time merge optional rows into anchor rows."""

    optional_sorted = sorted(optional_rows, key=lambda row: _as_float(row.get("time_ns")))
    out: List[Dict[str, object]] = []
    for anchor in anchor_rows:
        row = dict(anchor)
        time_ns = _as_float(row.get("time_ns"))
        match = _nearest_row(optional_sorted, time_ns, tolerance_ns) if math.isfinite(time_ns) else None
        if match is not None:
            selected_columns = columns or [key for key in match.keys() if key != "time_ns"]
            for column in selected_columns:
                if column in match:
                    row[column] = match[column]
        out.append(row)
    return out


def _estimate_initial_sizes(rows: Sequence[Mapping[str, object]]) -> Tuple[float, float]:
    bubble_rows = [
        row
        for row in rows
        if math.isfinite(_as_float(row.get("Bubble1Size"))) and math.isfinite(_as_float(row.get("Bubble2Size")))
    ]
    if bubble_rows:
        head = bubble_rows[: min(20, len(bubble_rows))]
        return _median(row.get("Bubble1Size") for row in head), _median(row.get("Bubble2Size") for row in head)

    n2_rows = [
        row
        for row in rows
        if math.isfinite(_as_float(row.get("n2A_num"))) and math.isfinite(_as_float(row.get("n2B_num")))
    ]
    if n2_rows:
        head = n2_rows[: min(200, len(n2_rows))]
        return _median(row.get("n2A_num") for row in head), _median(row.get("n2B_num") for row in head)
    return math.nan, math.nan


def assign_raw_state(
    row: Mapping[str, object],
    config: CoalescenceStateConfig,
    initial_single: float,
    initial_total: float,
) -> Tuple[str, float, str]:
    """Assign an unfiltered coalescence state to one row."""

    b1 = _as_float(row.get("Bubble1Size"))
    b2 = _as_float(row.get("Bubble2Size"))
    d3d = _as_float(row.get("d3d_all"))
    n2a = _as_float(row.get("n2A_num"))
    n2b = _as_float(row.get("n2B_num"))
    gap = d3d - config.contact_distance_A if math.isfinite(d3d) else math.nan

    has_bubble_sizes = math.isfinite(b1) and math.isfinite(b2) and math.isfinite(initial_total) and initial_total > 0
    if has_bubble_sizes:
        dominant = max(b1, b2)
        minor = min(b1, b2)
        major_frac = dominant / initial_total
        minor_frac = minor / initial_total
        both_large_frac = min(b1, b2) / max(initial_single, 1.0)
        if major_frac >= config.merged_major_total_fraction and minor_frac <= config.merged_minor_total_fraction:
            return "merged_like", 0.90, "bubble_size_dominant_cluster"
        if both_large_frac >= config.separated_min_single_fraction and (not math.isfinite(gap) or gap > config.close_gap_A):
            return "separated", 0.80, "two_large_bubble_clusters"
        if both_large_frac >= config.separated_min_single_fraction and math.isfinite(gap) and gap <= config.close_gap_A:
            return "transition_like", 0.65, "two_large_clusters_close_centers"
        return "transition_like", 0.55, "bubble_size_intermediate"

    if math.isfinite(n2a) and math.isfinite(n2b) and math.isfinite(initial_single) and initial_single > 0:
        integrity = min(n2a, n2b) / initial_single
        if integrity >= config.separated_min_single_fraction and (not math.isfinite(gap) or gap > config.close_gap_A):
            return "separated", 0.55, "plumed_cluster_integrity_only"
        if math.isfinite(gap) and gap <= config.close_gap_A:
            return "transition_like", 0.45, "close_centers_without_bubble_sizes"

    if math.isfinite(gap) and gap <= config.close_gap_A:
        return "transition_like", 0.35, "close_centers_only"
    return "ambiguous", 0.20, "insufficient_state_observables"


def apply_persistence_filter(states: Sequence[str], min_persist: int) -> List[str]:
    """Replace short transition/merged runs by `ambiguous`."""

    if min_persist <= 1:
        return list(states)
    out = list(states)
    start = 0
    n_states = len(states)
    while start < n_states:
        end = start + 1
        while end < n_states and states[end] == states[start]:
            end += 1
        if states[start] in {"merged_like", "transition_like"} and end - start < int(min_persist):
            for index in range(start, end):
                out[index] = "ambiguous"
        start = end
    return out


def assign_states(
    rows: Sequence[Mapping[str, object]],
    config: CoalescenceStateConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Assign coalescence states and derived size/gap columns."""

    init_a, init_b = _estimate_initial_sizes(rows)
    if math.isfinite(init_a) and math.isfinite(init_b):
        initial_single = float(np.nanmedian([init_a, init_b]))
        initial_total = float(init_a + init_b)
    else:
        initial_single = math.nan
        initial_total = math.nan

    raw_states: List[str] = []
    assigned_rows: List[Dict[str, object]] = []
    for source in rows:
        row = dict(source)
        d3d = _as_float(row.get("d3d_all"))
        b1 = _as_float(row.get("Bubble1Size"))
        b2 = _as_float(row.get("Bubble2Size"))
        row["surface_gap_contact_distance_A"] = config.contact_distance_A
        row["surface_gap_estimate_A"] = d3d - config.contact_distance_A if math.isfinite(d3d) else math.nan
        row["dominant_bubble_size"] = max(b1, b2) if math.isfinite(b1) and math.isfinite(b2) else math.nan
        row["minor_bubble_size"] = min(b1, b2) if math.isfinite(b1) and math.isfinite(b2) else math.nan
        row["dominant_fraction_of_initial_total"] = (
            row["dominant_bubble_size"] / initial_total
            if math.isfinite(_as_float(row["dominant_bubble_size"])) and math.isfinite(initial_total) and initial_total > 0
            else math.nan
        )
        row["minor_fraction_of_initial_total"] = (
            row["minor_bubble_size"] / initial_total
            if math.isfinite(_as_float(row["minor_bubble_size"])) and math.isfinite(initial_total) and initial_total > 0
            else math.nan
        )
        row["both_large_fraction_of_initial_single"] = (
            row["minor_bubble_size"] / initial_single
            if math.isfinite(_as_float(row["minor_bubble_size"])) and math.isfinite(initial_single) and initial_single > 0
            else math.nan
        )
        state, confidence, basis = assign_raw_state(row, config, initial_single, initial_total)
        row["raw_state"] = state
        row["state_confidence"] = confidence
        row["state_basis"] = basis
        raw_states.append(state)
        assigned_rows.append(row)

    filtered_states = apply_persistence_filter(raw_states, config.min_persist_samples)
    for row, state in zip(assigned_rows, filtered_states):
        row["state"] = state

    stats = {
        "initial_bubble_A_size": init_a,
        "initial_bubble_B_size": init_b,
        "initial_single_size_estimate": initial_single,
        "initial_total_size_estimate": initial_total,
        "nominal_radius_A": config.nominal_radius_A,
        "surface_contact_distance_A": config.contact_distance_A,
        "surface_gap_definition": "surface_gap_estimate_A = d3d_all - surface_contact_distance_A",
        "close_gap_A": config.close_gap_A,
        "min_persist_samples": config.min_persist_samples,
    }
    return assigned_rows, stats


def build_state_table(
    colvar_rows: Sequence[Mapping[str, object]],
    post_rows: Sequence[Mapping[str, object]] = (),
    bubble_rows: Sequence[Mapping[str, object]] = (),
    config: CoalescenceStateConfig = CoalescenceStateConfig(),
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Merge input tables and assign states."""

    if not colvar_rows:
        raise ValueError("At least one COLVAR row is required")
    start_ns = float(config.start_ns)
    available_end = max(_as_float(row.get("time_ns")) for row in colvar_rows)
    end_ns = available_end if config.end_ns is None else min(float(config.end_ns), available_end)
    if end_ns < start_ns:
        raise ValueError("Requested start_ns is beyond available COLVAR time")

    filtered_colvar = [
        dict(row) for row in colvar_rows if start_ns <= _as_float(row.get("time_ns")) <= end_ns
    ]
    if config.sample_interval_ns > 0:
        anchor_rows = [
            {"time_ns": float(time_ns), "time_ps": float(time_ns) * 1000.0}
            for time_ns in regular_sample_times(start_ns, end_ns, config.sample_interval_ns)
        ]
        colvar_tolerance = max(config.time_tolerance_ns, config.sample_interval_ns / 2.0)
        merged = nearest_merge_rows(
            anchor_rows,
            filtered_colvar,
            tolerance_ns=colvar_tolerance,
            columns=["time", *DEFAULT_COLVAR_COLUMNS],
        )
    else:
        merged = [dict(row) for row in filtered_colvar]
        for row in merged:
            row["time_ps"] = _as_float(row.get("time_ns")) * 1000.0

    for index, row in enumerate(merged):
        row["sample_index"] = index
        row.setdefault("time_ps", _as_float(row.get("time_ns")) * 1000.0)

    if post_rows:
        post_filtered = [row for row in post_rows if start_ns <= _as_float(row.get("time_ns")) <= end_ns]
        merged = nearest_merge_rows(
            merged,
            post_filtered,
            tolerance_ns=config.time_tolerance_ns,
            columns=DEFAULT_POST_COLUMNS,
        )
    if bubble_rows:
        bubble_filtered = [row for row in bubble_rows if start_ns <= _as_float(row.get("time_ns")) <= end_ns]
        merged = nearest_merge_rows(
            merged,
            bubble_filtered,
            tolerance_ns=config.bubble_time_tolerance_ns,
            columns=["FrameIndex", "Bubble1Size", "Bubble2Size", "TotalSize", "TimePeriod"],
        )

    assigned, stats = assign_states(merged, config)
    stats.update(
        {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "available_end_ns": available_end,
            "sample_interval_ns": config.sample_interval_ns,
            "n_output_rows": len(assigned),
        }
    )
    return assigned, stats


def summarize_states(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Summarize state counts and time ranges."""

    total = max(len(rows), 1)
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("state", "ambiguous")), []).append(row)
    summary: List[Dict[str, object]] = []
    for state in sorted(grouped):
        group = grouped[state]
        d3d = _finite_values(row.get("d3d_all") for row in group)
        times = _finite_values(row.get("time_ns") for row in group)
        summary.append(
            {
                "state": state,
                "count": len(group),
                "fraction": len(group) / total,
                "d3d_all_mean": float(np.mean(d3d)) if d3d else math.nan,
                "time_ns_min": min(times) if times else math.nan,
                "time_ns_max": max(times) if times else math.nan,
            }
        )
    return summary


def summarize_by_cv(rows: Sequence[Mapping[str, object]], bins: int) -> List[Dict[str, object]]:
    """Summarize state probabilities by d3d_all bins."""

    valid = [row for row in rows if math.isfinite(_as_float(row.get("d3d_all")))]
    if not valid:
        return []
    values = np.asarray([_as_float(row.get("d3d_all")) for row in valid], dtype=float)
    if float(np.nanmax(values)) == float(np.nanmin(values)):
        edges = np.asarray([float(values[0]) - 0.5, float(values[0]) + 0.5], dtype=float)
    else:
        edges = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), int(bins) + 1)
    states = ("separated", "transition_like", "merged_like", "ambiguous")
    out: List[Dict[str, object]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = [
            row
            for row in valid
            if _as_float(row.get("d3d_all")) >= float(left)
            and (_as_float(row.get("d3d_all")) < float(right) or right == edges[-1])
        ]
        if not in_bin:
            continue
        item: Dict[str, object] = {
            "cv_bin_left": float(left),
            "cv_bin_right": float(right),
            "cv_bin_center": 0.5 * (float(left) + float(right)),
            "count": len(in_bin),
            "time_ns_min": min(_finite_values(row.get("time_ns") for row in in_bin)),
            "time_ns_max": max(_finite_values(row.get("time_ns") for row in in_bin)),
        }
        for state in states:
            item[f"p_{state}"] = sum(1 for row in in_bin if row.get("state") == state) / len(in_bin)
        for column in ("bridge_cyl_env.sum", "bridge_cyl_env.mean", "n2A_num", "n2B_num", "Bubble1Size", "Bubble2Size"):
            finite = _finite_values(row.get(column) for row in in_bin)
            if finite:
                item[column.replace(".", "_") + "_mean"] = float(np.mean(finite))
        out.append(item)
    return out


def _ordered_fieldnames(rows: Sequence[Mapping[str, object]], preferred: Sequence[str] = STATE_COLUMNS) -> List[str]:
    keys = {key for row in rows for key in row.keys()}
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(key for key in keys if key not in ordered))
    return ordered


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    """Write rows to CSV, including a header for empty outputs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = _ordered_fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_statistics(path: Path, stats: Mapping[str, object]) -> None:
    """Write key-value state statistics."""

    with Path(path).open("w", encoding="utf-8") as handle:
        for key in sorted(stats):
            handle.write(f"{key}={stats[key]}\n")


def analyze_coalescence_state(
    colvar: Path,
    output_dir: Path,
    colvar_post: Optional[Path] = None,
    bubble_evolution: Optional[Path] = None,
    colvar_time_unit: str = "ps",
    rebase_colvar_time_zero: bool = False,
    config: CoalescenceStateConfig = CoalescenceStateConfig(),
) -> Dict[str, Path]:
    """Read inputs, assign states, and write output tables."""

    colvar_rows = read_plumed_table(colvar, time_unit=colvar_time_unit, optional_columns=DEFAULT_COLVAR_COLUMNS)
    post_rows = (
        read_plumed_table(colvar_post, time_unit=colvar_time_unit, optional_columns=DEFAULT_POST_COLUMNS)
        if colvar_post is not None
        else []
    )
    if rebase_colvar_time_zero and colvar_rows:
        origin = min(_as_float(row.get("time_ns")) for row in colvar_rows)
        for row in colvar_rows:
            row["time_ns"] = _as_float(row.get("time_ns")) - origin
        for row in post_rows:
            row["time_ns"] = _as_float(row.get("time_ns")) - origin
    else:
        origin = 0.0
    bubble_rows = read_bubble_evolution(bubble_evolution) if bubble_evolution is not None else []
    table, stats = build_state_table(colvar_rows, post_rows=post_rows, bubble_rows=bubble_rows, config=config)
    stats.update(
        {
            "colvar": str(colvar),
            "colvar_post": str(colvar_post) if colvar_post is not None else "",
            "bubble_evolution": str(bubble_evolution) if bubble_evolution is not None else "",
            "colvar_time_origin_ns": origin,
            "rebase_colvar_time_zero": bool(rebase_colvar_time_zero),
        }
    )

    outputs = {
        "state_table": Path(output_dir) / "coalescence_state_table.csv",
        "state_summary": Path(output_dir) / "coalescence_state_summary.csv",
        "cv_summary": Path(output_dir) / "coalescence_state_by_d3d_all.csv",
        "statistics": Path(output_dir) / "state_statistics.txt",
    }
    write_csv_rows(outputs["state_table"], table)
    write_csv_rows(
        outputs["state_summary"],
        summarize_states(table),
        ["state", "count", "fraction", "d3d_all_mean", "time_ns_min", "time_ns_max"],
    )
    write_csv_rows(outputs["cv_summary"], summarize_by_cv(table, config.cv_bins))
    write_statistics(outputs["statistics"], stats)
    return outputs


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign provisional two-bubble coalescence states")
    parser.add_argument("--colvar", type=Path, required=True, help="PLUMED COLVAR path with a time column")
    parser.add_argument("--colvar-post", type=Path, help="Optional secondary COLVAR table with cluster counters")
    parser.add_argument("--bubble-evolution", type=Path, help="Optional bubble evolution table")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-ns", type=float, default=0.0)
    parser.add_argument("--end-ns", type=float)
    parser.add_argument("--sample-interval-ns", type=float, default=0.001, help="Use 0 to keep every COLVAR row")
    parser.add_argument("--time-tolerance-ns", type=float, default=0.00051)
    parser.add_argument("--bubble-time-tolerance-ns", type=float, default=0.00051)
    parser.add_argument("--colvar-time-unit", choices=["fs", "ps", "ns"], default="ps")
    parser.add_argument("--rebase-colvar-time-zero", action="store_true")
    parser.add_argument("--nominal-radius-A", type=float, default=19.0)
    parser.add_argument("--surface-contact-distance-A", type=float)
    parser.add_argument("--close-gap-A", type=float, default=0.0)
    parser.add_argument("--separated-min-single-fraction", type=float, default=0.60)
    parser.add_argument("--merged-major-total-fraction", type=float, default=0.85)
    parser.add_argument("--merged-minor-total-fraction", type=float, default=0.10)
    parser.add_argument("--min-persist-samples", type=int, default=3)
    parser.add_argument("--cv-bins", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    try:
        outputs = analyze_coalescence_state(
            colvar=args.colvar,
            colvar_post=args.colvar_post,
            bubble_evolution=args.bubble_evolution,
            output_dir=args.output_dir,
            colvar_time_unit=args.colvar_time_unit,
            rebase_colvar_time_zero=args.rebase_colvar_time_zero,
            config=CoalescenceStateConfig(
                start_ns=args.start_ns,
                end_ns=args.end_ns,
                sample_interval_ns=args.sample_interval_ns,
                time_tolerance_ns=args.time_tolerance_ns,
                bubble_time_tolerance_ns=args.bubble_time_tolerance_ns,
                nominal_radius_A=args.nominal_radius_A,
                surface_contact_distance_A=args.surface_contact_distance_A,
                close_gap_A=args.close_gap_A,
                separated_min_single_fraction=args.separated_min_single_fraction,
                merged_major_total_fraction=args.merged_major_total_fraction,
                merged_minor_total_fraction=args.merged_minor_total_fraction,
                min_persist_samples=args.min_persist_samples,
                cv_bins=args.cv_bins,
            ),
        )
    except Exception as exc:
        print(f"Coalescence state assignment failed: {exc}")
        return 1

    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
