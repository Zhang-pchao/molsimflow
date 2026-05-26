"""Bridge-water and bridge-ion descriptor utilities.

The functions here extract reusable descriptor tables from legacy bridge-water
and bridge-ion workflows without carrying over case-specific path discovery or
plotting layers.
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

ION_SPECIES_ORDER = ("h3o", "oh_bulk", "oh_surface", "oh", "na", "cl")
ION_SPECIES_CHARGE = {
    "h3o": 1.0,
    "na": 1.0,
    "oh_bulk": -1.0,
    "oh_surface": -1.0,
    "oh": -1.0,
    "cl": -1.0,
}

BRIDGE_WATER_VALUE_COLUMNS = (
    "bridge_water_count_proxy",
    "bridge_water_density_proxy_per_nm3",
    "bridge_env_mean_proxy",
)

BRIDGE_ION_VALUE_COLUMNS = (
    "n_bridge_h3o",
    "n_bridge_oh",
    "n_bridge_oh_bulk",
    "n_bridge_oh_surface",
    "n_bridge_na",
    "n_bridge_cl",
    "n_bridge_total_ions",
    "n_bridge_cations",
    "n_bridge_anions",
    "bridge_net_charge_e",
    "n_bridge_h3o_per_nm3",
    "n_bridge_oh_per_nm3",
    "n_bridge_na_per_nm3",
    "n_bridge_cl_per_nm3",
    "n_bridge_total_ions_per_nm3",
    "bridge_charge_density_e_per_nm3",
)


@dataclass(frozen=True)
class BridgeCylinder:
    """Cylinder used to define a bridge region in an orthorhombic box."""

    center: np.ndarray
    axis: np.ndarray
    radius_A: float
    length_A: float

    def __post_init__(self) -> None:
        if self.radius_A <= 0.0:
            raise ValueError("radius_A must be positive")
        if self.length_A <= 0.0:
            raise ValueError("length_A must be positive")
        axis_norm = float(np.linalg.norm(self.axis))
        if not math.isfinite(axis_norm) or axis_norm <= 0.0:
            raise ValueError("axis must have nonzero length")

    @property
    def unit_axis(self) -> np.ndarray:
        return np.asarray(self.axis, dtype=float) / float(np.linalg.norm(self.axis))

    @property
    def volume_nm3(self) -> float:
        return cylinder_volume_nm3(self.radius_A, self.length_A)

    def contains(self, points: Sequence[Sequence[float]], box_dims: Optional[Sequence[float]] = None) -> np.ndarray:
        """Return a boolean mask for points inside the cylinder."""

        coords = np.asarray(points, dtype=float)
        if coords.ndim == 1:
            coords = coords.reshape(1, 3)
        delta = coords - np.asarray(self.center, dtype=float)
        if box_dims is not None:
            delta = minimum_image_delta(delta, box_dims)
        axis = self.unit_axis
        axial = delta @ axis
        radial_vec = delta - axial[:, None] * axis[None, :]
        radial = np.linalg.norm(radial_vec, axis=1)
        return (np.abs(axial) <= 0.5 * float(self.length_A)) & (radial <= float(self.radius_A))


def cylinder_volume_nm3(radius_A: float, length_A: float) -> float:
    """Return cylinder volume in nm^3 from Angstrom dimensions."""

    volume_A3 = math.pi * float(radius_A) ** 2 * float(length_A)
    if volume_A3 <= 0.0:
        raise ValueError("Bridge cylinder volume must be positive")
    return volume_A3 / 1000.0


def minimum_image_delta(delta: np.ndarray, box_dims: Sequence[float]) -> np.ndarray:
    """Apply an orthorhombic minimum-image transform to displacement vectors."""

    values = np.asarray(delta, dtype=float)
    box = np.asarray(box_dims, dtype=float)
    out = np.array(values, copy=True)
    for dim in range(min(3, box.shape[0])):
        length = box[dim]
        if length > 0.0:
            out[..., dim] = out[..., dim] - length * np.round(out[..., dim] / length)
    return out


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "case_label",
        "time_ns",
        "surface_gap_A",
        "gap_window",
        "gap_bin_left_A",
        "gap_bin_right_A",
        "gap_bin_center_A",
        "n_frames",
    ]
    ordered = [key for key in preferred if key in fieldnames]
    ordered.extend([key for key in fieldnames if key not in ordered])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _truthy(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "in", "inside"}


def _mean(values: Sequence[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(data)) if data else math.nan


def _median(values: Sequence[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(data)) if data else math.nan


def _std(values: Sequence[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.std(data, ddof=1)) if len(data) > 1 else 0.0


def _sem(values: Sequence[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.std(data, ddof=1) / math.sqrt(len(data))) if len(data) > 1 else 0.0


def canonical_ion_species(raw: object) -> str:
    """Normalize common ion species labels used in legacy bridge traces."""

    text = str(raw).strip().lower()
    aliases = {
        "h3o+": "h3o",
        "h3o": "h3o",
        "na+": "na",
        "na": "na",
        "cl-": "cl",
        "cl": "cl",
        "bulk oh-": "oh_bulk",
        "bulk_oh": "oh_bulk",
        "oh_bulk": "oh_bulk",
        "solution_bulk_oh": "oh_bulk",
        "surface oh-": "oh_surface",
        "surface_oh": "oh_surface",
        "oh_surface": "oh_surface",
        "solution_surface_oh": "oh_surface",
        "oh": "oh_bulk",
        "oh-": "oh_bulk",
    }
    return aliases.get(text, text)


def build_bridge_water_frame_table(
    rows: Iterable[Mapping[str, object]],
    bridge_radius_A: float,
    bridge_length_A: float,
    case_label: str = "",
    time_column: str = "time_ns",
    gap_column: str = "surface_gap_estimate_A",
    water_count_column: str = "bridge_cyl_env.sum",
    water_mean_column: str = "bridge_cyl_env.mean",
    state_column: str = "state",
    start_time_ns: Optional[float] = None,
    end_time_ns: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Build per-frame bridge-water descriptor rows from a state/metrics table."""

    volume_nm3 = cylinder_volume_nm3(bridge_radius_A, bridge_length_A)
    out: List[Dict[str, object]] = []
    for row in rows:
        time_ns = _as_float(row.get(time_column))
        gap_A = _as_float(row.get(gap_column))
        water_count = _as_float(row.get(water_count_column))
        if not (math.isfinite(time_ns) and math.isfinite(gap_A) and math.isfinite(water_count)):
            continue
        if start_time_ns is not None and time_ns < float(start_time_ns):
            continue
        if end_time_ns is not None and time_ns > float(end_time_ns):
            continue
        mean_proxy = _as_float(row.get(water_mean_column))
        out.append(
            {
                "case_label": case_label,
                "time_ns": time_ns,
                "surface_gap_A": gap_A,
                "state": row.get(state_column, ""),
                "bridge_water_count_proxy": water_count,
                "bridge_water_density_proxy_per_nm3": water_count / volume_nm3,
                "bridge_env_mean_proxy": mean_proxy,
                "bridge_cylinder_radius_A": float(bridge_radius_A),
                "bridge_cylinder_length_A": float(bridge_length_A),
                "bridge_cylinder_volume_nm3": volume_nm3,
            }
        )
    if not out:
        raise ValueError("No bridge-water rows remain after filtering")
    return out


def _select_species_column(row: Mapping[str, object], explicit: Optional[str] = None) -> Optional[str]:
    candidates = [explicit] if explicit else []
    candidates.extend(["current_trace_species", "current_trace_species_label", "species", "ion_species"])
    for column in candidates:
        if column and column in row and str(row.get(column, "")).strip():
            return column
    return None


def _nearest_gap(time_ns: float, gap_rows: Sequence[Mapping[str, object]], tolerance_ns: float) -> Tuple[float, str]:
    best_gap = math.nan
    best_state = ""
    best_dt = math.inf
    for row in gap_rows:
        dt = abs(float(row["time_ns"]) - time_ns)
        if dt < best_dt:
            best_dt = dt
            best_gap = float(row["surface_gap_A"])
            best_state = str(row.get("state", ""))
    if best_dt <= float(tolerance_ns):
        return best_gap, best_state
    return math.nan, ""


def _load_gap_rows(
    rows: Iterable[Mapping[str, object]],
    time_column: str,
    gap_column: str,
    state_column: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        time_ns = _as_float(row.get(time_column))
        gap_A = _as_float(row.get(gap_column))
        if math.isfinite(time_ns) and math.isfinite(gap_A):
            out.append({"time_ns": time_ns, "surface_gap_A": gap_A, "state": row.get(state_column, "")})
    return sorted(out, key=lambda row: float(row["time_ns"]))


def build_bridge_ion_occupancy_table(
    position_rows: Iterable[Mapping[str, object]],
    bridge_radius_A: float,
    bridge_length_A: float,
    case_label: str = "",
    gap_rows: Optional[Sequence[Mapping[str, object]]] = None,
    time_column: str = "time_ns",
    species_column: Optional[str] = None,
    in_bridge_column: str = "in_bridge_region",
    time_tolerance_ns: float = 0.0015,
    start_time_ns: Optional[float] = None,
    end_time_ns: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Build per-frame bridge-ion occupancy and charge rows."""

    volume_nm3 = cylinder_volume_nm3(bridge_radius_A, bridge_length_A)
    counts_by_time: Dict[float, Dict[str, float]] = {}
    all_times = set()
    for row in position_rows:
        time_ns = _as_float(row.get(time_column))
        if not math.isfinite(time_ns):
            continue
        if start_time_ns is not None and time_ns < float(start_time_ns):
            continue
        if end_time_ns is not None and time_ns > float(end_time_ns):
            continue
        all_times.add(time_ns)
        if in_bridge_column in row and not _truthy(row.get(in_bridge_column)):
            continue
        selected_species_column = _select_species_column(row, explicit=species_column)
        if selected_species_column is None:
            continue
        species = canonical_ion_species(row.get(selected_species_column))
        if species not in ION_SPECIES_CHARGE:
            continue
        counts = counts_by_time.setdefault(time_ns, {sp: 0.0 for sp in ION_SPECIES_ORDER})
        counts[species] += 1.0

    normalized_gap_rows = list(gap_rows or [])
    if normalized_gap_rows:
        all_times.update(float(row["time_ns"]) for row in normalized_gap_rows)
    if not all_times:
        raise ValueError("No bridge-ion rows remain after filtering")

    out: List[Dict[str, object]] = []
    for time_ns in sorted(all_times):
        if start_time_ns is not None and time_ns < float(start_time_ns):
            continue
        if end_time_ns is not None and time_ns > float(end_time_ns):
            continue
        counts = counts_by_time.get(time_ns, {sp: 0.0 for sp in ION_SPECIES_ORDER})
        gap_A, state = _nearest_gap(time_ns, normalized_gap_rows, time_tolerance_ns) if normalized_gap_rows else (math.nan, "")
        n_oh = counts.get("oh", 0.0) + counts.get("oh_bulk", 0.0) + counts.get("oh_surface", 0.0)
        n_h3o = counts.get("h3o", 0.0)
        n_na = counts.get("na", 0.0)
        n_cl = counts.get("cl", 0.0)
        cations = n_h3o + n_na
        anions = n_oh + n_cl
        total = cations + anions
        net_charge = n_h3o + n_na - n_oh - n_cl
        row = {
            "case_label": case_label,
            "time_ns": time_ns,
            "surface_gap_A": gap_A,
            "state": state,
            "n_bridge_h3o": n_h3o,
            "n_bridge_oh": n_oh,
            "n_bridge_oh_bulk": counts.get("oh_bulk", 0.0),
            "n_bridge_oh_surface": counts.get("oh_surface", 0.0),
            "n_bridge_na": n_na,
            "n_bridge_cl": n_cl,
            "n_bridge_total_ions": total,
            "n_bridge_cations": cations,
            "n_bridge_anions": anions,
            "bridge_net_charge_e": net_charge,
            "bridge_cylinder_radius_A": float(bridge_radius_A),
            "bridge_cylinder_length_A": float(bridge_length_A),
            "bridge_cylinder_volume_nm3": volume_nm3,
            "bridge_charge_density_e_per_nm3": net_charge / volume_nm3,
        }
        for species in ["h3o", "oh", "na", "cl"]:
            row[f"n_bridge_{species}_per_nm3"] = row[f"n_bridge_{species}"] / volume_nm3
        row["n_bridge_total_ions_per_nm3"] = total / volume_nm3
        out.append(row)
    return out


def _gap_edges(values: Sequence[float], gap_bin_width_A: float) -> np.ndarray:
    if gap_bin_width_A <= 0.0:
        raise ValueError("gap_bin_width_A must be positive")
    finite = [value for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError("No finite surface gaps available for binning")
    low = math.floor(min(finite) / gap_bin_width_A) * gap_bin_width_A
    high = math.ceil(max(finite) / gap_bin_width_A) * gap_bin_width_A
    edges = np.arange(low, high + gap_bin_width_A * 1.5, gap_bin_width_A)
    if len(edges) < 2:
        edges = np.array([low, low + gap_bin_width_A])
    return edges


def _bin_index(value: float, edges: np.ndarray) -> Optional[int]:
    if not math.isfinite(value):
        return None
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    if idx < 0:
        return None
    if idx >= len(edges) - 1:
        idx = len(edges) - 2 if math.isclose(value, float(edges[-1])) else idx
    return idx if 0 <= idx < len(edges) - 1 else None


def summarize_by_gap_bins(
    frame_rows: Sequence[Mapping[str, object]],
    value_columns: Sequence[str],
    gap_bin_width_A: float = 2.0,
    min_bin_count: int = 1,
) -> List[Dict[str, object]]:
    """Aggregate frame rows by case label and fixed-width surface-gap bins."""

    edges = _gap_edges([_as_float(row.get("surface_gap_A")) for row in frame_rows], gap_bin_width_A)
    groups: Dict[Tuple[str, int], List[Mapping[str, object]]] = {}
    for row in frame_rows:
        idx = _bin_index(_as_float(row.get("surface_gap_A")), edges)
        if idx is None:
            continue
        key = (str(row.get("case_label", "")), idx)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (case_label, idx), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        if len(rows) < int(min_bin_count):
            continue
        left = float(edges[idx])
        right = float(edges[idx + 1])
        summary: Dict[str, object] = {
            "case_label": case_label,
            "gap_bin_left_A": left,
            "gap_bin_right_A": right,
            "gap_bin_center_A": 0.5 * (left + right),
            "n_frames": len(rows),
            "surface_gap_A_mean": _mean([_as_float(row.get("surface_gap_A")) for row in rows]),
            "time_min_ns": min(_as_float(row.get("time_ns")) for row in rows),
            "time_max_ns": max(_as_float(row.get("time_ns")) for row in rows),
        }
        for column in value_columns:
            values = [_as_float(row.get(column)) for row in rows]
            summary[f"{column}_mean"] = _mean(values)
            summary[f"{column}_median"] = _median(values)
            summary[f"{column}_std"] = _std(values)
            summary[f"{column}_sem"] = _sem(values)
            summary[f"{column}_sum"] = float(np.nansum(values))
        out.append(summary)
    if not out:
        raise ValueError("No gap bins retained; lower min_bin_count or check input data")
    return out


def summarize_by_gap_windows(
    frame_rows: Sequence[Mapping[str, object]],
    value_columns: Sequence[str],
    windows: Sequence[Tuple[str, float, float]] = DEFAULT_GAP_WINDOWS,
) -> List[Dict[str, object]]:
    """Aggregate frame rows by named surface-gap windows."""

    case_labels = sorted({str(row.get("case_label", "")) for row in frame_rows})
    out: List[Dict[str, object]] = []
    for case_label in case_labels:
        case_rows = [row for row in frame_rows if str(row.get("case_label", "")) == case_label]
        for name, low, high in windows:
            selected = []
            for row in case_rows:
                gap = _as_float(row.get("surface_gap_A"))
                if not math.isfinite(gap):
                    continue
                if not math.isinf(low) and gap < low:
                    continue
                if not math.isinf(high) and gap >= high:
                    continue
                selected.append(row)
            summary: Dict[str, object] = {
                "case_label": case_label,
                "gap_window": name,
                "gap_low_A": low,
                "gap_high_A": high,
                "n_frames": len(selected),
            }
            if selected:
                summary["surface_gap_A_mean"] = _mean([_as_float(row.get("surface_gap_A")) for row in selected])
                summary["surface_gap_A_min"] = min(_as_float(row.get("surface_gap_A")) for row in selected)
                summary["surface_gap_A_max"] = max(_as_float(row.get("surface_gap_A")) for row in selected)
                summary["time_min_ns"] = min(_as_float(row.get("time_ns")) for row in selected)
                summary["time_max_ns"] = max(_as_float(row.get("time_ns")) for row in selected)
                for column in value_columns:
                    values = [_as_float(row.get(column)) for row in selected]
                    summary[f"{column}_mean"] = _mean(values)
                    summary[f"{column}_median"] = _median(values)
                    summary[f"{column}_sum"] = float(np.nansum(values))
            out.append(summary)
    return out


def analyze_bridge_water_density(
    input_csv: Path,
    output_dir: Path,
    bridge_radius_A: float = 8.0,
    bridge_length_A: float = 20.0,
    case_label: str = "",
    time_column: str = "time_ns",
    gap_column: str = "surface_gap_estimate_A",
    water_count_column: str = "bridge_cyl_env.sum",
    water_mean_column: str = "bridge_cyl_env.mean",
    gap_bin_width_A: float = 2.0,
    min_bin_count: int = 1,
) -> Dict[str, Path]:
    """Analyze bridge-water density proxy from a CSV state/metrics table."""

    rows = _read_csv_rows(input_csv)
    frame_rows = build_bridge_water_frame_table(
        rows,
        bridge_radius_A=bridge_radius_A,
        bridge_length_A=bridge_length_A,
        case_label=case_label,
        time_column=time_column,
        gap_column=gap_column,
        water_count_column=water_count_column,
        water_mean_column=water_mean_column,
    )
    binned = summarize_by_gap_bins(frame_rows, BRIDGE_WATER_VALUE_COLUMNS, gap_bin_width_A, min_bin_count)
    windows = summarize_by_gap_windows(frame_rows, BRIDGE_WATER_VALUE_COLUMNS)
    output_dir = Path(output_dir)
    outputs = {
        "frame_table": output_dir / "bridge_water_density_frame_table.csv",
        "binned": output_dir / "bridge_water_density_binned.csv",
        "window_summary": output_dir / "bridge_water_density_window_summary.csv",
    }
    _write_csv_rows(outputs["frame_table"], frame_rows)
    _write_csv_rows(outputs["binned"], binned)
    _write_csv_rows(outputs["window_summary"], windows)
    return outputs


def analyze_bridge_ion_occupancy(
    positions_csv: Path,
    output_dir: Path,
    bridge_radius_A: float = 8.0,
    bridge_length_A: float = 20.0,
    case_label: str = "",
    gap_table: Optional[Path] = None,
    time_column: str = "time_ns",
    species_column: Optional[str] = None,
    in_bridge_column: str = "in_bridge_region",
    gap_time_column: str = "time_ns",
    gap_column: str = "surface_gap_estimate_A",
    gap_state_column: str = "state",
    gap_bin_width_A: float = 2.0,
    min_bin_count: int = 1,
    time_tolerance_ns: float = 0.0015,
) -> Dict[str, Path]:
    """Analyze strict bridge-ion occupancy and net charge from tracked positions."""

    position_rows = _read_csv_rows(positions_csv)
    gap_rows = None
    if gap_table is not None:
        gap_rows = _load_gap_rows(_read_csv_rows(gap_table), gap_time_column, gap_column, gap_state_column)
    elif any(gap_column in row for row in position_rows):
        gap_rows = _load_gap_rows(position_rows, gap_time_column, gap_column, gap_state_column)
    frame_rows = build_bridge_ion_occupancy_table(
        position_rows,
        bridge_radius_A=bridge_radius_A,
        bridge_length_A=bridge_length_A,
        case_label=case_label,
        gap_rows=gap_rows,
        time_column=time_column,
        species_column=species_column,
        in_bridge_column=in_bridge_column,
        time_tolerance_ns=time_tolerance_ns,
    )
    binned = summarize_by_gap_bins(frame_rows, BRIDGE_ION_VALUE_COLUMNS, gap_bin_width_A, min_bin_count)
    windows = summarize_by_gap_windows(frame_rows, BRIDGE_ION_VALUE_COLUMNS)
    output_dir = Path(output_dir)
    outputs = {
        "frame_table": output_dir / "bridge_ion_occupancy_frame_table.csv",
        "binned": output_dir / "bridge_ion_occupancy_binned.csv",
        "window_summary": output_dir / "bridge_ion_occupancy_window_summary.csv",
    }
    _write_csv_rows(outputs["frame_table"], frame_rows)
    _write_csv_rows(outputs["binned"], binned)
    _write_csv_rows(outputs["window_summary"], windows)
    return outputs


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge-water and bridge-ion descriptor utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    water = subparsers.add_parser("water-density", help="Analyze bridge-water density proxy")
    water.add_argument("--input", type=Path, required=True, help="Input state/metrics CSV")
    water.add_argument("--output-dir", type=Path, required=True)
    water.add_argument("--case-label", default="")
    water.add_argument("--bridge-radius-A", type=float, default=8.0)
    water.add_argument("--bridge-length-A", type=float, default=20.0)
    water.add_argument("--time-column", default="time_ns")
    water.add_argument("--gap-column", default="surface_gap_estimate_A")
    water.add_argument("--water-count-column", default="bridge_cyl_env.sum")
    water.add_argument("--water-mean-column", default="bridge_cyl_env.mean")
    water.add_argument("--gap-bin-width-A", type=float, default=2.0)
    water.add_argument("--min-bin-count", type=int, default=1)

    ion = subparsers.add_parser("ion-occupancy", help="Analyze strict bridge-ion occupancy")
    ion.add_argument("--positions", type=Path, required=True, help="tracked_bridge_ion_positions.csv")
    ion.add_argument("--output-dir", type=Path, required=True)
    ion.add_argument("--case-label", default="")
    ion.add_argument("--gap-table", type=Path, help="Optional state table with surface gap")
    ion.add_argument("--bridge-radius-A", type=float, default=8.0)
    ion.add_argument("--bridge-length-A", type=float, default=20.0)
    ion.add_argument("--time-column", default="time_ns")
    ion.add_argument("--species-column", default=None)
    ion.add_argument("--in-bridge-column", default="in_bridge_region")
    ion.add_argument("--gap-time-column", default="time_ns")
    ion.add_argument("--gap-column", default="surface_gap_estimate_A")
    ion.add_argument("--gap-state-column", default="state")
    ion.add_argument("--gap-bin-width-A", type=float, default=2.0)
    ion.add_argument("--min-bin-count", type=int, default=1)
    ion.add_argument("--time-tolerance-ns", type=float, default=0.0015)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    try:
        if args.command == "water-density":
            outputs = analyze_bridge_water_density(
                input_csv=args.input,
                output_dir=args.output_dir,
                bridge_radius_A=args.bridge_radius_A,
                bridge_length_A=args.bridge_length_A,
                case_label=args.case_label,
                time_column=args.time_column,
                gap_column=args.gap_column,
                water_count_column=args.water_count_column,
                water_mean_column=args.water_mean_column,
                gap_bin_width_A=args.gap_bin_width_A,
                min_bin_count=args.min_bin_count,
            )
        elif args.command == "ion-occupancy":
            outputs = analyze_bridge_ion_occupancy(
                positions_csv=args.positions,
                output_dir=args.output_dir,
                bridge_radius_A=args.bridge_radius_A,
                bridge_length_A=args.bridge_length_A,
                case_label=args.case_label,
                gap_table=args.gap_table,
                time_column=args.time_column,
                species_column=args.species_column,
                in_bridge_column=args.in_bridge_column,
                gap_time_column=args.gap_time_column,
                gap_column=args.gap_column,
                gap_state_column=args.gap_state_column,
                gap_bin_width_A=args.gap_bin_width_A,
                min_bin_count=args.min_bin_count,
                time_tolerance_ns=args.time_tolerance_ns,
            )
        else:  # pragma: no cover - argparse enforces valid commands
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        print(f"Bridge descriptor analysis failed: {exc}")
        return 1

    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
