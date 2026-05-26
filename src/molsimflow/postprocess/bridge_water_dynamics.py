"""Bridge-water dynamics summaries from trace-metrics tables.

This module migrates the reusable table algorithms from the legacy
bridge-water entry/exit flux and seed-water survival workflows.  Inputs are
explicit CSV files rather than project-specific case directories.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_GAP_WINDOWS: Tuple[Tuple[str, float, float], ...] = (
    ("merged_or_overlap_gap_le_0A", -math.inf, 0.0),
    ("thin_bridge_0_5A", 0.0, 5.0),
    ("near_bridge_5_10A", 5.0, 10.0),
    ("open_bridge_10_20A", 10.0, 20.0),
    ("wide_gap_gt_20A", 20.0, math.inf),
)

FLUX_REQUIRED_COLUMNS = (
    "time_ns",
    "n_current_bridge_waters",
    "n_newly_tracked_waters_this_frame",
)

FLUX_OPTIONAL_COLUMNS = (
    "dynamic_surface_gap_est_A",
    "state",
    "n_seed_waters",
    "n_seed_retained_in_bridge",
    "seed_retention_fraction",
    "n_new_bridge_waters",
    "new_bridge_water_fraction",
    "n_tracked_waters_so_far",
    "n_tracked_entrants_so_far",
    "n_tracked_entrants_in_bridge",
    "n_untracked_current_bridge_waters",
    "median_seed_displacement_A",
    "p90_seed_displacement_A",
)

SEED_REQUIRED_COLUMNS = (
    "time_ns",
    "n_seed_waters",
    "n_seed_retained_in_bridge",
    "seed_retention_fraction",
)

SEED_OPTIONAL_COLUMNS = (
    "dynamic_surface_gap_est_A",
    "state",
    "n_current_bridge_waters",
    "n_new_bridge_waters",
    "new_bridge_water_fraction",
    "mean_seed_displacement_A",
    "median_seed_displacement_A",
    "p90_seed_displacement_A",
)

FLUX_VALUE_COLUMNS = (
    "n_current_bridge_waters",
    "entry_count_proxy",
    "exit_count_proxy",
    "net_drainage_count",
    "net_filling_count",
    "turnover_count_proxy",
    "replacement_count_proxy",
    "entry_rate_per_ps_proxy",
    "exit_rate_per_ps_proxy",
    "net_drainage_rate_per_ps",
    "turnover_rate_per_ps_proxy",
    "replacement_rate_per_ps_proxy",
    "entry_fraction_of_previous_bridge",
    "exit_fraction_of_previous_bridge",
    "replacement_fraction_of_previous_bridge",
    "turnover_fraction_of_previous_bridge",
    "depletion_vs_turnover_index",
    "exchange_dominant_flag",
    "depletion_dominant_flag",
    "new_bridge_water_fraction",
    "seed_retention_fraction",
)

SEED_VALUE_COLUMNS = (
    "seed_retention_fraction",
    "seed_survival_fraction_monotonic_proxy",
    "n_seed_waters",
    "n_seed_retained_in_bridge",
    "monotonic_seed_retained_count_proxy",
    "seed_lost_count_instant",
    "seed_lost_fraction_instant",
    "monotonic_seed_lost_count_proxy",
    "seed_exit_proxy_count_this_frame",
    "median_seed_displacement_A",
    "p90_seed_displacement_A",
    "n_current_bridge_waters",
    "new_bridge_water_fraction",
)


@dataclass(frozen=True)
class TraceInputSpec:
    """One bridge-water trace metrics input."""

    case_label: str
    trace_metrics: Path
    state_table: Optional[Path] = None


@dataclass(frozen=True)
class BridgeWaterDynamicsConfig:
    """Shared configuration for bridge-water dynamics summaries."""

    start_time_ns: Optional[float] = None
    end_time_ns: Optional[float] = None
    gap_source: str = "coalescence"
    gap_bin_width_A: float = 2.0
    min_bin_count: int = 1
    state_time_tolerance_ns: float = 0.0015

    def validate(self) -> None:
        if self.gap_source not in {"trace", "coalescence"}:
            raise ValueError("gap_source must be 'trace' or 'coalescence'")
        if self.gap_bin_width_A <= 0.0:
            raise ValueError("gap_bin_width_A must be positive")
        if self.min_bin_count < 1:
            raise ValueError("min_bin_count must be at least 1")


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
        "time_ns",
        "surface_gap_A",
        "gap_window",
        "gap_low_A",
        "gap_high_A",
        "gap_bin_left_A",
        "gap_bin_right_A",
        "gap_bin_center_A",
        "n_frames",
        "trace_metrics",
        "state_table",
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


def _safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return math.nan
    return numerator / denominator


def _truthy_numeric(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = _as_float(value)
    if math.isfinite(number):
        return number
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y"}:
        return 1.0
    if text in {"false", "f", "no", "n"}:
        return 0.0
    return math.nan


def _finite_values(values: Iterable[object]) -> List[float]:
    out: List[float] = []
    for value in values:
        number = _truthy_numeric(value)
        if math.isfinite(number):
            out.append(float(number))
    return out


def _value_stats(values: Iterable[object]) -> Dict[str, float]:
    finite = _finite_values(values)
    if not finite:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "sem": math.nan, "sum": 0.0}
    data = np.asarray(finite, dtype=float)
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    sem = float(std / math.sqrt(data.size)) if data.size > 1 else 0.0
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": std,
        "sem": sem,
        "sum": float(np.sum(data)),
    }


def _resolve_manifest_path(raw_path: object, manifest_path: Path) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def load_trace_input_manifest(path: Path) -> List[TraceInputSpec]:
    """Load trace input specs from a CSV manifest.

    Required columns are `case_label` and `trace_metrics`.  `label` is accepted
    as a case-label fallback.  `state_table` is optional.  Relative paths are
    resolved relative to the manifest file.
    """

    rows, fieldnames = _read_csv_rows(path)
    label_column = "case_label" if "case_label" in fieldnames else "label" if "label" in fieldnames else ""
    if not label_column or "trace_metrics" not in fieldnames:
        raise ValueError("Manifest requires case_label or label, plus trace_metrics")
    out: List[TraceInputSpec] = []
    for row in rows:
        label = str(row.get(label_column, "")).strip()
        raw_trace = str(row.get("trace_metrics", "")).strip()
        if not label or not raw_trace:
            continue
        raw_state = str(row.get("state_table", "")).strip()
        state_table = _resolve_manifest_path(raw_state, Path(path)) if raw_state else None
        out.append(
            TraceInputSpec(
                case_label=label,
                trace_metrics=_resolve_manifest_path(raw_trace, Path(path)),
                state_table=state_table,
            )
        )
    if not out:
        raise ValueError(f"No trace inputs found in manifest: {path}")
    return out


def load_trace_inputs(
    manifest: Optional[Path] = None,
    trace_metrics: Optional[Path] = None,
    case_label: str = "",
    state_table: Optional[Path] = None,
) -> List[TraceInputSpec]:
    """Load one or more trace inputs from either a manifest or explicit paths."""

    if manifest is not None:
        return load_trace_input_manifest(manifest)
    if trace_metrics is None:
        raise ValueError("Provide --manifest or --trace-metrics")
    label = case_label.strip() or Path(trace_metrics).stem
    return [TraceInputSpec(case_label=label, trace_metrics=Path(trace_metrics), state_table=state_table)]


def _validate_columns(path: Path, fieldnames: Sequence[str], required: Sequence[str]) -> None:
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")


def _read_trace_rows(path: Path, required: Sequence[str]) -> List[Dict[str, str]]:
    rows, fieldnames = _read_csv_rows(path)
    _validate_columns(path, fieldnames, required)
    return rows


def _read_state_rows(path: Optional[Path]) -> List[Dict[str, object]]:
    if path is None:
        return []
    rows, fieldnames = _read_csv_rows(path)
    if "time_ns" not in fieldnames or "surface_gap_estimate_A" not in fieldnames:
        return []
    out: List[Dict[str, object]] = []
    for row in rows:
        time_ns = _as_float(row.get("time_ns"))
        gap_A = _as_float(row.get("surface_gap_estimate_A"))
        if not math.isfinite(time_ns):
            continue
        out.append({"time_ns": time_ns, "surface_gap_estimate_A": gap_A, "state": row.get("state", "")})
    return sorted(out, key=lambda item: float(item["time_ns"]))


def _nearest_state_row(
    state_rows: Sequence[Mapping[str, object]],
    time_ns: float,
    tolerance_ns: float,
) -> Optional[Mapping[str, object]]:
    if not state_rows or not math.isfinite(time_ns):
        return None
    times = np.asarray([_as_float(row.get("time_ns")) for row in state_rows], dtype=float)
    index = int(np.searchsorted(times, float(time_ns)))
    candidates: List[int] = []
    if index < times.size:
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(float(times[item]) - float(time_ns)))
    if abs(float(times[best]) - float(time_ns)) <= float(tolerance_ns):
        return state_rows[best]
    return None


def _surface_gap_for_row(
    row: Mapping[str, object],
    state_row: Optional[Mapping[str, object]],
    gap_source: str,
) -> Tuple[float, float, object]:
    trace_gap = _as_float(row.get("dynamic_surface_gap_est_A"))
    state_gap = _as_float(state_row.get("surface_gap_estimate_A") if state_row is not None else math.nan)
    state_value = state_row.get("state", "") if state_row is not None else ""
    if gap_source == "coalescence" and math.isfinite(state_gap):
        return state_gap, state_gap, state_value
    if math.isfinite(trace_gap):
        return trace_gap, state_gap, state_value
    return state_gap, state_gap, state_value


def _filter_and_prepare_rows(
    spec: TraceInputSpec,
    required: Sequence[str],
    config: BridgeWaterDynamicsConfig,
) -> List[Dict[str, object]]:
    trace_rows = _read_trace_rows(spec.trace_metrics, required=required)
    state_rows = _read_state_rows(spec.state_table)
    prepared: List[Dict[str, object]] = []
    for row in trace_rows:
        time_ns = _as_float(row.get("time_ns"))
        if not math.isfinite(time_ns):
            continue
        if config.start_time_ns is not None and time_ns < float(config.start_time_ns):
            continue
        if config.end_time_ns is not None and time_ns > float(config.end_time_ns):
            continue
        state_row = _nearest_state_row(state_rows, time_ns, config.state_time_tolerance_ns)
        surface_gap_A, coalescence_gap_A, coalescence_state = _surface_gap_for_row(
            row,
            state_row,
            gap_source=config.gap_source,
        )
        if not math.isfinite(surface_gap_A):
            continue
        item: Dict[str, object] = dict(row)
        item.update(
            {
                "case_label": spec.case_label,
                "time_ns": time_ns,
                "surface_gap_A": surface_gap_A,
                "coalescence_surface_gap_est_A": coalescence_gap_A,
                "coalescence_state": coalescence_state,
                "trace_state": row.get("state", ""),
                "trace_metrics": str(spec.trace_metrics),
                "state_table": str(spec.state_table) if spec.state_table is not None else "",
            }
        )
        prepared.append(item)
    return sorted(prepared, key=lambda item: float(item["time_ns"]))


def _median_dt_ns(rows: Sequence[Mapping[str, object]]) -> float:
    times = [_as_float(row.get("time_ns")) for row in rows]
    deltas = [b - a for a, b in zip(times[:-1], times[1:]) if math.isfinite(a) and math.isfinite(b) and b > a]
    if not deltas:
        return 0.001
    return float(np.median(np.asarray(deltas, dtype=float)))


def build_bridge_water_flux_frame_table(
    inputs: Sequence[TraceInputSpec],
    config: BridgeWaterDynamicsConfig = BridgeWaterDynamicsConfig(),
) -> List[Dict[str, object]]:
    """Build per-frame bridge-water entry/exit flux proxy rows."""

    config.validate()
    frames: List[Dict[str, object]] = []
    for spec in inputs:
        rows = _filter_and_prepare_rows(spec, required=FLUX_REQUIRED_COLUMNS, config=config)
        if not rows:
            continue
        median_dt_ns = _median_dt_ns(rows)
        previous_current = math.nan
        for index, row in enumerate(rows):
            current = _as_float(row.get("n_current_bridge_waters"))
            newly_tracked = _as_float(row.get("n_newly_tracked_waters_this_frame"))
            if not math.isfinite(current) or not math.isfinite(newly_tracked):
                continue
            time_ns = _as_float(row.get("time_ns"))
            if index == 0:
                dt_ns = median_dt_ns
                bridge_delta = 0.0
                entry_count = 0.0
            else:
                dt_ns = time_ns - _as_float(rows[index - 1].get("time_ns"))
                if not math.isfinite(dt_ns) or dt_ns <= 0.0:
                    dt_ns = median_dt_ns
                bridge_delta = current - previous_current
                entry_count = max(newly_tracked, 0.0)
            dt_ps = dt_ns * 1000.0
            exit_count = max(entry_count - bridge_delta, 0.0)
            net_drainage = max(-bridge_delta, 0.0)
            net_filling = max(bridge_delta, 0.0)
            turnover = entry_count + exit_count
            replacement = min(entry_count, exit_count)
            previous_for_fraction = previous_current if index > 0 else math.nan
            out: Dict[str, object] = {
                "case_label": spec.case_label,
                "time_ns": time_ns,
                "surface_gap_A": row["surface_gap_A"],
                "dynamic_surface_gap_est_A": _as_float(row.get("dynamic_surface_gap_est_A")),
                "coalescence_surface_gap_est_A": row.get("coalescence_surface_gap_est_A", math.nan),
                "trace_state": row.get("trace_state", ""),
                "coalescence_state": row.get("coalescence_state", ""),
                "n_current_bridge_waters": current,
                "n_newly_tracked_waters_this_frame": newly_tracked,
                "dt_ns": dt_ns,
                "dt_ps": dt_ps,
                "bridge_count_delta": bridge_delta,
                "entry_count_proxy": entry_count,
                "exit_count_proxy": exit_count,
                "net_drainage_count": net_drainage,
                "net_filling_count": net_filling,
                "turnover_count_proxy": turnover,
                "replacement_count_proxy": replacement,
                "entry_fraction_of_previous_bridge": _safe_div(entry_count, previous_for_fraction),
                "exit_fraction_of_previous_bridge": _safe_div(exit_count, previous_for_fraction),
                "replacement_fraction_of_previous_bridge": _safe_div(replacement, previous_for_fraction),
                "turnover_fraction_of_previous_bridge": _safe_div(turnover, previous_for_fraction),
                "entry_rate_per_ps_proxy": _safe_div(entry_count, dt_ps),
                "exit_rate_per_ps_proxy": _safe_div(exit_count, dt_ps),
                "net_drainage_rate_per_ps": _safe_div(net_drainage, dt_ps),
                "turnover_rate_per_ps_proxy": _safe_div(turnover, dt_ps),
                "replacement_rate_per_ps_proxy": _safe_div(replacement, dt_ps),
                "entry_exit_balance_count": entry_count - exit_count,
                "depletion_vs_turnover_index": _safe_div(net_drainage, turnover),
                "exchange_dominant_flag": bool(replacement > 0.0 and net_drainage <= replacement),
                "depletion_dominant_flag": bool(net_drainage > replacement),
                "trace_metrics": str(spec.trace_metrics),
                "state_table": str(spec.state_table) if spec.state_table is not None else "",
            }
            for column in FLUX_OPTIONAL_COLUMNS:
                if column not in out and column in row:
                    out[column] = row.get(column) if column == "state" else _as_float(row.get(column))
            frames.append(out)
            previous_current = current
    if not frames:
        raise ValueError("No bridge-water flux rows remain after filtering")
    return frames


def build_seed_water_survival_frame_table(
    inputs: Sequence[TraceInputSpec],
    config: BridgeWaterDynamicsConfig = BridgeWaterDynamicsConfig(),
) -> List[Dict[str, object]]:
    """Build per-frame seed-water survival proxy rows."""

    config.validate()
    frames: List[Dict[str, object]] = []
    for spec in inputs:
        rows = _filter_and_prepare_rows(spec, required=SEED_REQUIRED_COLUMNS, config=config)
        if not rows:
            continue
        monotonic_retained = math.inf
        previous_monotonic = math.nan
        for index, row in enumerate(rows):
            n_seed = _as_float(row.get("n_seed_waters"))
            retained = _as_float(row.get("n_seed_retained_in_bridge"))
            retention = _as_float(row.get("seed_retention_fraction"))
            if not math.isfinite(n_seed) or not math.isfinite(retained) or not math.isfinite(retention):
                continue
            monotonic_retained = min(monotonic_retained, retained)
            if index == 0 or not math.isfinite(previous_monotonic):
                exit_proxy = 0.0
            else:
                exit_proxy = max(previous_monotonic - monotonic_retained, 0.0)
            out: Dict[str, object] = {
                "case_label": spec.case_label,
                "time_ns": row["time_ns"],
                "surface_gap_A": row["surface_gap_A"],
                "dynamic_surface_gap_est_A": _as_float(row.get("dynamic_surface_gap_est_A")),
                "coalescence_surface_gap_est_A": row.get("coalescence_surface_gap_est_A", math.nan),
                "trace_state": row.get("trace_state", ""),
                "coalescence_state": row.get("coalescence_state", ""),
                "n_seed_waters": n_seed,
                "n_seed_retained_in_bridge": retained,
                "seed_retention_fraction": retention,
                "seed_lost_count_instant": n_seed - retained,
                "seed_lost_fraction_instant": 1.0 - retention,
                "monotonic_seed_retained_count_proxy": monotonic_retained,
                "seed_survival_fraction_monotonic_proxy": _safe_div(monotonic_retained, n_seed),
                "monotonic_seed_lost_count_proxy": n_seed - monotonic_retained,
                "seed_exit_proxy_count_this_frame": exit_proxy,
                "trace_metrics": str(spec.trace_metrics),
                "state_table": str(spec.state_table) if spec.state_table is not None else "",
            }
            for column in SEED_OPTIONAL_COLUMNS:
                if column not in out and column in row:
                    out[column] = row.get(column) if column == "state" else _as_float(row.get(column))
            frames.append(out)
            previous_monotonic = monotonic_retained
    if not frames:
        raise ValueError("No seed-water survival rows remain after filtering")
    return frames


def _case_order(rows: Sequence[Mapping[str, object]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for row in rows:
        label = str(row.get("case_label", ""))
        if label and label not in seen:
            out.append(label)
            seen.add(label)
    return out


def summarize_by_gap_bins(
    frame_rows: Sequence[Mapping[str, object]],
    value_columns: Sequence[str],
    gap_bin_width_A: float,
    min_bin_count: int = 1,
) -> List[Dict[str, object]]:
    """Summarize numeric values by case and surface-gap bin."""

    if gap_bin_width_A <= 0.0:
        raise ValueError("gap_bin_width_A must be positive")
    gaps = [_as_float(row.get("surface_gap_A")) for row in frame_rows]
    gaps = [gap for gap in gaps if math.isfinite(gap)]
    if not gaps:
        raise ValueError("No finite surface_gap_A values found")
    left_edge = math.floor(min(gaps) / gap_bin_width_A) * gap_bin_width_A
    right_edge = math.ceil(max(gaps) / gap_bin_width_A) * gap_bin_width_A
    if right_edge <= left_edge:
        right_edge = left_edge + gap_bin_width_A
    n_bins = max(1, int(math.ceil((right_edge - left_edge) / gap_bin_width_A)))
    out: List[Dict[str, object]] = []
    for case_label in _case_order(frame_rows):
        case_rows = [row for row in frame_rows if str(row.get("case_label", "")) == case_label]
        for bin_index in range(n_bins):
            bin_left = left_edge + bin_index * gap_bin_width_A
            bin_right = bin_left + gap_bin_width_A
            chunk = [
                row
                for row in case_rows
                if _as_float(row.get("surface_gap_A")) >= bin_left
                and (_as_float(row.get("surface_gap_A")) < bin_right or bin_index == n_bins - 1)
            ]
            if len(chunk) < int(min_bin_count):
                continue
            row_out: Dict[str, object] = {
                "case_label": case_label,
                "gap_bin_left_A": bin_left,
                "gap_bin_right_A": bin_right,
                "gap_bin_center_A": 0.5 * (bin_left + bin_right),
                "n_frames": len(chunk),
                "time_min_ns": min(_as_float(row.get("time_ns")) for row in chunk),
                "time_max_ns": max(_as_float(row.get("time_ns")) for row in chunk),
                "mean_surface_gap_A": float(np.mean([_as_float(row.get("surface_gap_A")) for row in chunk])),
            }
            for column in value_columns:
                stats = _value_stats(row.get(column) for row in chunk)
                for suffix, value in stats.items():
                    row_out[f"{column}_{suffix}"] = value
            out.append(row_out)
    if not out:
        raise ValueError("No gap bins retained; lower min_bin_count or check input data")
    return out


def summarize_gap_windows(
    frame_rows: Sequence[Mapping[str, object]],
    value_columns: Sequence[str],
    windows: Sequence[Tuple[str, float, float]] = DEFAULT_GAP_WINDOWS,
) -> List[Dict[str, object]]:
    """Summarize numeric values by named surface-gap windows."""

    out: List[Dict[str, object]] = []
    for case_label in _case_order(frame_rows):
        case_rows = [row for row in frame_rows if str(row.get("case_label", "")) == case_label]
        for window_name, low, high in windows:
            chunk = []
            for row in case_rows:
                gap = _as_float(row.get("surface_gap_A"))
                if not math.isfinite(gap):
                    continue
                if not math.isinf(low) and gap < low:
                    continue
                if not math.isinf(high) and gap >= high:
                    continue
                chunk.append(row)
            row_out: Dict[str, object] = {
                "case_label": case_label,
                "gap_window": window_name,
                "gap_low_A": low,
                "gap_high_A": high,
                "n_frames": len(chunk),
            }
            if chunk:
                row_out["surface_gap_A_mean"] = float(
                    np.mean([_as_float(row.get("surface_gap_A")) for row in chunk])
                )
                row_out["surface_gap_A_min"] = min(_as_float(row.get("surface_gap_A")) for row in chunk)
                row_out["surface_gap_A_max"] = max(_as_float(row.get("surface_gap_A")) for row in chunk)
                row_out["time_min_ns"] = min(_as_float(row.get("time_ns")) for row in chunk)
                row_out["time_max_ns"] = max(_as_float(row.get("time_ns")) for row in chunk)
                for column in value_columns:
                    stats = _value_stats(row.get(column) for row in chunk)
                    row_out[f"{column}_mean"] = stats["mean"]
                    row_out[f"{column}_median"] = stats["median"]
                    row_out[f"{column}_sum"] = stats["sum"]
            out.append(row_out)
    return out


def build_seed_exit_proxy_events(frame_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Return rows where the monotonic seed-survival proxy drops."""

    columns = [
        "case_label",
        "time_ns",
        "surface_gap_A",
        "dynamic_surface_gap_est_A",
        "coalescence_surface_gap_est_A",
        "seed_exit_proxy_count_this_frame",
        "seed_survival_fraction_monotonic_proxy",
        "seed_retention_fraction",
        "trace_metrics",
        "state_table",
    ]
    out: List[Dict[str, object]] = []
    for row in frame_rows:
        if _as_float(row.get("seed_exit_proxy_count_this_frame")) <= 0.0:
            continue
        out.append({column: row.get(column, math.nan) for column in columns})
    return out


def _input_rows(inputs: Sequence[TraceInputSpec]) -> List[Dict[str, object]]:
    return [
        {
            "case_label": item.case_label,
            "trace_metrics": str(item.trace_metrics),
            "state_table": str(item.state_table) if item.state_table is not None else "",
        }
        for item in inputs
    ]


def analyze_bridge_water_flux(
    inputs: Sequence[TraceInputSpec],
    output_dir: Path,
    config: BridgeWaterDynamicsConfig = BridgeWaterDynamicsConfig(),
) -> Dict[str, Path]:
    """Write bridge-water flux proxy summary tables."""

    frame_rows = build_bridge_water_flux_frame_table(inputs, config=config)
    binned = summarize_by_gap_bins(
        frame_rows,
        value_columns=FLUX_VALUE_COLUMNS,
        gap_bin_width_A=config.gap_bin_width_A,
        min_bin_count=config.min_bin_count,
    )
    windows = summarize_gap_windows(frame_rows, value_columns=FLUX_VALUE_COLUMNS)
    output_dir = Path(output_dir)
    outputs = {
        "frame_table": output_dir / "bridge_water_flux_frame_table.csv",
        "binned": output_dir / "bridge_water_flux_binned.csv",
        "window_summary": output_dir / "bridge_water_flux_window_summary.csv",
        "inputs": output_dir / "bridge_water_dynamics_inputs.csv",
    }
    _write_csv_rows(outputs["frame_table"], frame_rows)
    _write_csv_rows(outputs["binned"], binned)
    _write_csv_rows(outputs["window_summary"], windows)
    _write_csv_rows(outputs["inputs"], _input_rows(inputs), fieldnames=["case_label", "trace_metrics", "state_table"])
    return outputs


def analyze_seed_water_survival(
    inputs: Sequence[TraceInputSpec],
    output_dir: Path,
    config: BridgeWaterDynamicsConfig = BridgeWaterDynamicsConfig(),
) -> Dict[str, Path]:
    """Write seed-water survival proxy summary tables."""

    frame_rows = build_seed_water_survival_frame_table(inputs, config=config)
    binned = summarize_by_gap_bins(
        frame_rows,
        value_columns=SEED_VALUE_COLUMNS,
        gap_bin_width_A=config.gap_bin_width_A,
        min_bin_count=config.min_bin_count,
    )
    windows = summarize_gap_windows(frame_rows, value_columns=SEED_VALUE_COLUMNS)
    events = build_seed_exit_proxy_events(frame_rows)
    output_dir = Path(output_dir)
    outputs = {
        "frame_table": output_dir / "seed_water_survival_frame_table.csv",
        "binned": output_dir / "seed_water_survival_binned.csv",
        "window_summary": output_dir / "seed_water_survival_window_summary.csv",
        "exit_events": output_dir / "seed_water_exit_proxy_events.csv",
        "inputs": output_dir / "seed_water_survival_inputs.csv",
    }
    _write_csv_rows(outputs["frame_table"], frame_rows)
    _write_csv_rows(outputs["binned"], binned)
    _write_csv_rows(outputs["window_summary"], windows)
    _write_csv_rows(
        outputs["exit_events"],
        events,
        fieldnames=[
            "case_label",
            "time_ns",
            "surface_gap_A",
            "dynamic_surface_gap_est_A",
            "coalescence_surface_gap_est_A",
            "seed_exit_proxy_count_this_frame",
            "seed_survival_fraction_monotonic_proxy",
            "seed_retention_fraction",
            "trace_metrics",
            "state_table",
        ],
    )
    _write_csv_rows(outputs["inputs"], _input_rows(inputs), fieldnames=["case_label", "trace_metrics", "state_table"])
    return outputs


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path, help="CSV with case_label,trace_metrics,state_table columns")
    input_group.add_argument("--trace-metrics", type=Path, help="Single bridge_water_trace_metrics.csv input")
    parser.add_argument("--case-label", default="", help="Case label for --trace-metrics")
    parser.add_argument("--state-table", type=Path, help="Optional coalescence_state_table.csv for --trace-metrics")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-time-ns", type=float)
    parser.add_argument("--end-time-ns", type=float)
    parser.add_argument("--gap-source", choices=["trace", "coalescence"], default="coalescence")
    parser.add_argument("--gap-bin-width-A", type=float, default=2.0)
    parser.add_argument("--min-bin-count", type=int, default=1)
    parser.add_argument("--state-time-tolerance-ns", type=float, default=0.0015)


def _config_from_args(args: argparse.Namespace) -> BridgeWaterDynamicsConfig:
    return BridgeWaterDynamicsConfig(
        start_time_ns=args.start_time_ns,
        end_time_ns=args.end_time_ns,
        gap_source=args.gap_source,
        gap_bin_width_A=args.gap_bin_width_A,
        min_bin_count=args.min_bin_count,
        state_time_tolerance_ns=args.state_time_tolerance_ns,
    )


def _inputs_from_args(args: argparse.Namespace) -> List[TraceInputSpec]:
    return load_trace_inputs(
        manifest=args.manifest,
        trace_metrics=args.trace_metrics,
        case_label=args.case_label,
        state_table=args.state_table,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge-water dynamics table analyses")
    subparsers = parser.add_subparsers(dest="command", required=True)

    flux = subparsers.add_parser("flux", help="Compute entry/exit flux proxy summaries")
    _add_common_args(flux)

    seed = subparsers.add_parser("seed-survival", help="Compute seed-water survival proxy summaries")
    _add_common_args(seed)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        inputs = _inputs_from_args(args)
        config = _config_from_args(args)
        if args.command == "flux":
            outputs = analyze_bridge_water_flux(inputs, output_dir=args.output_dir, config=config)
        elif args.command == "seed-survival":
            outputs = analyze_seed_water_survival(inputs, output_dir=args.output_dir, config=config)
        else:  # pragma: no cover
            parser.error(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"Bridge-water dynamics failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
