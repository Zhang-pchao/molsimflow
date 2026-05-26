"""Double-bubble bridge microstate table adapters.

This module is intentionally workflow-specific.  It converts legacy-style
double-bubble trace tables into explicit CSV products consumed by generic
`molsimflow.postprocess` modules.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MICROSTATE_COLUMNS = (
    "global_frame",
    "segment_label",
    "segment_frame",
    "segment_path",
    "time_ns",
    "time_ps",
    "d3d_all",
    "bridge_cyl_env_sum",
    "bridge_cyl_env_mean",
    "coalescence_state",
    "d3d_bin",
    "gap_window_label",
    "window_label",
    "source_quality_flag",
    "water_segment",
    "water_local_frame",
    "ion_segment",
    "ion_local_frame",
    "bridge_center_x_A",
    "bridge_center_y_A",
    "bridge_center_z_A",
    "bubble_center_distance_A",
    "dynamic_surface_gap_est_A",
    "Nw_bridge_core",
    "Nw_seed_retained_bridge",
    "water_seed_retention_fraction",
    "Nw_new_bridge",
    "water_new_bridge_fraction",
    "Nw_untracked_bridge",
    "Nion_bridge_current",
    "Nion_trace_region_current",
    "Nion_bridge_region_current",
    "Nion_bubble_near_current",
    "Nion_new_bridge",
    "N_Na_plus_bridge",
    "N_Cl_minus_bridge",
    "N_OH_minus_bulk_bridge",
    "N_OH_minus_surf_bridge",
    "N_H3O_plus_bridge",
    "N_OH_minus_total_bridge",
    "N_H_plus_surf_bridge",
    "H_plus_surf_source_available",
    "ion_new_bridge_fraction",
    "ion_seed_retention_fraction",
    "bridge_net_charge_proxy_e",
    "bridge_ion_water_ratio",
    "bridge_core_volume_A3",
    "bridge_water_density_proxy_A3",
    "bridge_ion_density_proxy_A3",
    "bridge_s_min_A",
    "bridge_s_max_A",
    "bridge_rho_max_A",
    "water_trace_matched",
    "ion_trace_matched",
    "microstate_source_quality_flag",
    "Nw_geometry_bridge_from_tracked_positions",
    "Nw_geometry_shell_only_from_tracked_positions",
    "water_geometry_minus_trace_count",
)


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
    preferred = list(MICROSTATE_COLUMNS) + [
        "species_canonical",
        "current_trace_region",
        "n_atoms",
        "mean_distance_to_bridge_center_A",
        "metric",
        "value",
        "status",
        "note",
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


def _as_int(value: object, default: int = 0) -> int:
    number = _as_float(value)
    return int(round(number)) if math.isfinite(number) else default


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    number = _as_float(value)
    if math.isfinite(number):
        return bool(round(number))
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y"}:
        return True
    if text in {"false", "f", "no", "n"}:
        return False
    return default


def _format_float(value: float) -> object:
    return value if math.isfinite(value) else ""


def _first_existing(row: Mapping[str, object], names: Sequence[str], default: object = "") -> object:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        number = _as_float(value)
        if math.isfinite(number):
            return number
        return value
    return default


def _safe_ratio(numerator: object, denominator: object) -> object:
    num = _as_float(numerator)
    den = _as_float(denominator)
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-14:
        return ""
    return num / den


def _prefix_trace_columns(row: Mapping[str, object], prefix: str, keep: Sequence[str]) -> Dict[str, object]:
    keep_set = set(keep)
    out: Dict[str, object] = {}
    for key, value in row.items():
        if key in keep_set:
            out[key] = value
        else:
            out[f"{prefix}{key}"] = value
    return out


def _index_by_keys(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> Dict[Tuple[str, ...], Mapping[str, object]]:
    out: Dict[Tuple[str, ...], Mapping[str, object]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in keys)
        if any(part == "" for part in key):
            continue
        out[key] = row
    return out


def load_and_merge_frame_tables(
    frame_index_path: Path,
    water_trace_path: Path,
    ion_trace_path: Path,
    max_frames: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Merge frame index rows with water and ion trace tables."""

    frame_rows, _ = _read_csv_rows(frame_index_path)
    water_rows, _ = _read_csv_rows(water_trace_path)
    ion_rows, _ = _read_csv_rows(ion_trace_path)
    frame_rows = sorted(frame_rows, key=lambda row: _as_int(row.get("global_frame"), default=10**12))
    if max_frames is not None and max_frames > 0:
        frame_rows = frame_rows[: int(max_frames)]

    water_prefixed = []
    for row in water_rows:
        local = dict(row)
        if "segment" in local:
            local["water_segment"] = local.pop("segment")
        if "local_frame" in local:
            local["water_local_frame"] = local.pop("local_frame")
        water_prefixed.append(_prefix_trace_columns(local, "water_trace_", ["water_segment", "water_local_frame"]))

    ion_prefixed = []
    for row in ion_rows:
        local = dict(row)
        if "segment" in local:
            local["ion_segment"] = local.pop("segment")
        if "local_frame" in local:
            local["ion_local_frame"] = local.pop("local_frame")
        ion_prefixed.append(_prefix_trace_columns(local, "ion_trace_", ["ion_segment", "ion_local_frame"]))

    water_by_key = _index_by_keys(water_prefixed, ["water_segment", "water_local_frame"])
    ion_by_key = _index_by_keys(ion_prefixed, ["ion_segment", "ion_local_frame"])

    merged: List[Dict[str, object]] = []
    for row in frame_rows:
        item: Dict[str, object] = dict(row)
        water_key = (str(row.get("water_segment", "")), str(row.get("water_local_frame", "")))
        ion_key = (str(row.get("ion_segment", "")), str(row.get("ion_local_frame", "")))
        water_match = water_by_key.get(water_key)
        ion_match = ion_by_key.get(ion_key)
        if water_match:
            item.update(water_match)
        if ion_match:
            item.update(ion_match)
        item["water_trace_matched"] = bool(water_match)
        item["ion_trace_matched"] = bool(ion_match)
        merged.append(item)
    return merged


def _canonical_species(raw: object) -> str:
    text = str(raw).strip().lower()
    if "h3o" in text:
        return "H3O+"
    if "oh" in text and "surface" in text:
        return "OH-_surface"
    if "oh" in text:
        return "OH-_bulk"
    if "na" in text:
        return "Na+"
    if "cl" in text:
        return "Cl-"
    if text:
        return str(raw)
    return "unknown"


def _charge_for_species(raw: object) -> float:
    species = _canonical_species(raw)
    if species in {"H3O+", "Na+"}:
        return 1.0
    if species in {"OH-_surface", "OH-_bulk", "Cl-"}:
        return -1.0
    return 0.0


def build_microstate_frame_rows(
    merged_rows: Sequence[Mapping[str, object]],
    bridge_rho_max_A: float,
    bridge_s_min_A: float,
    bridge_s_max_A: float,
) -> List[Dict[str, object]]:
    """Build bridge-local water/ion microstate rows from merged trace rows."""

    volume = math.pi * float(bridge_rho_max_A) * float(bridge_rho_max_A) * (
        float(bridge_s_max_A) - float(bridge_s_min_A)
    )
    rows: List[Dict[str, object]] = []
    for source in merged_rows:
        row: Dict[str, object] = {}
        for column in [
            "global_frame",
            "segment_label",
            "segment_frame",
            "segment_path",
            "time_ns",
            "time_ps",
            "d3d_all",
            "bridge_cyl_env_sum",
            "bridge_cyl_env_mean",
            "coalescence_state",
            "d3d_bin",
            "gap_window_label",
            "window_label",
            "source_quality_flag",
            "water_segment",
            "water_local_frame",
            "ion_segment",
            "ion_local_frame",
        ]:
            if column in source:
                row[column] = source[column]

        row["bridge_center_x_A"] = _first_existing(source, ["water_trace_bridge_center_x_A", "ion_trace_bridge_center_x_A"])
        row["bridge_center_y_A"] = _first_existing(source, ["water_trace_bridge_center_y_A", "ion_trace_bridge_center_y_A"])
        row["bridge_center_z_A"] = _first_existing(source, ["water_trace_bridge_center_z_A", "ion_trace_bridge_center_z_A"])
        row["bubble_center_distance_A"] = _first_existing(
            source,
            ["water_trace_bubble_center_distance_A", "ion_trace_bubble_center_distance_A"],
        )
        row["dynamic_surface_gap_est_A"] = _first_existing(
            source,
            [
                "water_trace_dynamic_surface_gap_est_A",
                "ion_trace_dynamic_surface_gap_est_A",
                "surface_gap_estimate_A",
            ],
        )
        count_columns = {
            "Nw_bridge_core": ["water_trace_n_current_bridge_waters", "water_n_current_bridge_waters"],
            "Nw_seed_retained_bridge": ["water_trace_n_seed_retained_in_bridge", "water_n_seed_retained_in_bridge"],
            "Nw_new_bridge": ["water_trace_n_new_bridge_waters", "water_n_new_bridge_waters"],
            "Nw_untracked_bridge": [
                "water_trace_n_untracked_current_bridge_waters",
                "water_n_untracked_current_bridge_waters",
            ],
            "Nion_bridge_current": ["ion_trace_n_current_bridge_ions", "ion_n_current_bridge_ions"],
            "Nion_trace_region_current": ["ion_trace_n_current_trace_region_ions", "ion_n_current_trace_region_ions"],
            "Nion_bridge_region_current": ["ion_trace_n_current_bridge_region_ions", "ion_n_current_bridge_region_ions"],
            "Nion_bubble_near_current": ["ion_trace_n_current_bubble_near_ions", "ion_n_current_bubble_near_ions"],
            "Nion_new_bridge": ["ion_trace_n_new_bridge_ions", "ion_n_new_bridge_ions"],
            "N_Na_plus_bridge": ["ion_trace_n_current_na", "ion_n_current_na"],
            "N_Cl_minus_bridge": ["ion_trace_n_current_cl", "ion_n_current_cl"],
            "N_OH_minus_bulk_bridge": ["ion_trace_n_current_oh_bulk", "ion_n_current_oh_bulk"],
            "N_OH_minus_surf_bridge": ["ion_trace_n_current_oh_surface", "ion_n_current_oh_surface"],
            "N_H3O_plus_bridge": ["ion_trace_n_current_h3o", "ion_n_current_h3o"],
            "N_OH_minus_total_bridge": ["ion_trace_n_current_oh", "ion_n_current_oh"],
        }
        for column, candidates in count_columns.items():
            row[column] = _as_int(_first_existing(source, candidates, 0), default=0)

        row["water_seed_retention_fraction"] = _first_existing(
            source,
            ["water_trace_seed_retention_fraction", "water_seed_retention_fraction"],
        )
        row["water_new_bridge_fraction"] = _first_existing(
            source,
            ["water_trace_new_bridge_water_fraction", "water_new_bridge_water_fraction"],
        )
        row["ion_new_bridge_fraction"] = _first_existing(
            source,
            ["ion_trace_new_bridge_ion_fraction", "ion_new_bridge_ion_fraction"],
        )
        row["ion_seed_retention_fraction"] = _first_existing(
            source,
            ["ion_trace_seed_retention_fraction", "ion_seed_retention_fraction"],
        )
        row["N_H_plus_surf_bridge"] = ""
        row["H_plus_surf_source_available"] = False
        row["bridge_net_charge_proxy_e"] = (
            _as_int(row["N_H3O_plus_bridge"])
            + _as_int(row["N_Na_plus_bridge"])
            - _as_int(row["N_Cl_minus_bridge"])
            - _as_int(row["N_OH_minus_bulk_bridge"])
            - _as_int(row["N_OH_minus_surf_bridge"])
        )
        row["bridge_ion_water_ratio"] = _safe_ratio(row["Nion_bridge_current"], row["Nw_bridge_core"])
        row["bridge_core_volume_A3"] = volume
        row["bridge_water_density_proxy_A3"] = _safe_ratio(row["Nw_bridge_core"], volume)
        row["bridge_ion_density_proxy_A3"] = _safe_ratio(row["Nion_bridge_current"], volume)
        row["bridge_s_min_A"] = bridge_s_min_A
        row["bridge_s_max_A"] = bridge_s_max_A
        row["bridge_rho_max_A"] = bridge_rho_max_A
        row["water_trace_matched"] = _as_bool(source.get("water_trace_matched"), default=True)
        row["ion_trace_matched"] = _as_bool(source.get("ion_trace_matched"), default=True)
        row["microstate_source_quality_flag"] = "pass" if row["water_trace_matched"] and row["ion_trace_matched"] else "check"
        rows.append(row)
    return rows


def build_ion_position_rows(
    ion_position_rows: Sequence[Mapping[str, object]],
    frame_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Join tracked ion positions with microstate frame rows and add species/QC columns."""

    frame_by_key: Dict[Tuple[str, str, str], Mapping[str, object]] = {}
    for row in frame_rows:
        key = (str(row.get("time_ns", "")), str(row.get("ion_segment", "")), str(row.get("ion_local_frame", "")))
        frame_by_key[key] = row

    out: List[Dict[str, object]] = []
    for source in ion_position_rows:
        key = (str(source.get("time_ns", "")), str(source.get("segment", "")), str(source.get("local_frame", "")))
        frame = frame_by_key.get(key)
        if frame is None:
            continue
        row = dict(source)
        for column in [
            "global_frame",
            "bridge_center_x_A",
            "bridge_center_y_A",
            "bridge_center_z_A",
            "d3d_all",
            "coalescence_state",
            "window_label",
        ]:
            if column in frame:
                row[column] = frame[column]
        species_raw = row.get("current_trace_species", row.get("current_bridge_species", row.get("first_species", "")))
        row["species_canonical"] = _canonical_species(species_raw)
        row["species_charge_e"] = _charge_for_species(species_raw)
        dx = _as_float(row.get("x_A")) - _as_float(row.get("bridge_center_x_A"))
        dy = _as_float(row.get("y_A")) - _as_float(row.get("bridge_center_y_A"))
        dz = _as_float(row.get("z_A")) - _as_float(row.get("bridge_center_z_A"))
        row["distance_to_bridge_center_A"] = math.sqrt(dx * dx + dy * dy + dz * dz) if all(
            math.isfinite(value) for value in [dx, dy, dz]
        ) else ""
        out.append(row)
    out.sort(key=lambda row: (_as_int(row.get("global_frame")), str(row.get("atom_id", ""))))
    return out


def build_species_region_summary(ion_positions: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Summarize ion species by region and frame."""

    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in ion_positions:
        key = (
            row.get("global_frame", ""),
            row.get("time_ns", ""),
            row.get("segment", ""),
            row.get("local_frame", ""),
            row.get("coalescence_state", ""),
            row.get("d3d_all", ""),
            row.get("species_canonical", ""),
            row.get("current_trace_region", ""),
            row.get("in_bridge", ""),
            row.get("in_trace_region", ""),
            row.get("in_bridge_region", ""),
        )
        grouped[key].append(row)
    rows: List[Dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        chunk = grouped[key]
        distances = [_as_float(row.get("distance_to_bridge_center_A")) for row in chunk]
        distances = [value for value in distances if math.isfinite(value)]
        rows.append(
            {
                "global_frame": key[0],
                "time_ns": key[1],
                "segment": key[2],
                "local_frame": key[3],
                "coalescence_state": key[4],
                "d3d_all": key[5],
                "species_canonical": key[6],
                "current_trace_region": key[7],
                "in_bridge": key[8],
                "in_trace_region": key[9],
                "in_bridge_region": key[10],
                "n_atoms": len({str(row.get("atom_id", "")) for row in chunk}),
                "mean_distance_to_bridge_center_A": float(np.mean(distances)) if distances else "",
            }
        )
    return rows


def update_frame_rows_with_position_qc(
    frame_rows: Sequence[Mapping[str, object]],
    water_position_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Add tracked-water geometry count QC to microstate frame rows."""

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"bridge": 0, "shell_only": 0})
    for row in water_position_rows:
        frame = str(row.get("global_frame", ""))
        if _as_bool(row.get("in_bridge")):
            counts[frame]["bridge"] += 1
        if _as_bool(row.get("in_shell_only")):
            counts[frame]["shell_only"] += 1

    out: List[Dict[str, object]] = []
    for source in frame_rows:
        row = dict(source)
        frame_counts = counts.get(str(row.get("global_frame", "")), {"bridge": 0, "shell_only": 0})
        row["Nw_geometry_bridge_from_tracked_positions"] = frame_counts["bridge"]
        row["Nw_geometry_shell_only_from_tracked_positions"] = frame_counts["shell_only"]
        row["water_geometry_minus_trace_count"] = frame_counts["bridge"] - _as_int(row.get("Nw_bridge_core"))
        out.append(row)
    return out


def build_microstate_qc_rows(
    frame_rows: Sequence[Mapping[str, object]],
    ion_positions: Sequence[Mapping[str, object]],
    water_positions: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Build simple QC rows for microstate outputs."""

    rows = [
        {
            "metric": "frame_rows",
            "value": len(frame_rows),
            "status": "pass" if frame_rows else "fail",
            "note": "Rows in bridge_microstate_frame_table.csv",
        },
        {
            "metric": "unique_global_frames",
            "value": len({str(row.get("global_frame", "")) for row in frame_rows}),
            "status": "pass",
            "note": "Unique global frames in selected frame subset",
        },
        {
            "metric": "ion_position_rows",
            "value": len(ion_positions),
            "status": "pass" if ion_positions else "check",
            "note": "Rows copied from tracked bridge ion positions for selected frames",
        },
        {
            "metric": "water_position_rows",
            "value": len(water_positions),
            "status": "pass" if water_positions else "check",
            "note": "Tracked water oxygen position rows used for geometry QC",
        },
    ]
    diffs = [_as_float(row.get("water_geometry_minus_trace_count")) for row in frame_rows]
    diffs = [value for value in diffs if math.isfinite(value)]
    if diffs:
        rows.append(
            {
                "metric": "water_geometry_count_diff_abs_max",
                "value": max(abs(value) for value in diffs),
                "status": "check",
                "note": "Geometry reclassification minus existing water trace count",
            }
        )
    return rows


def analyze_bridge_microstate(
    frame_index: Path,
    water_trace: Path,
    ion_trace: Path,
    output_dir: Path,
    bridge_rho_max_A: float,
    bridge_s_min_A: float,
    bridge_s_max_A: float,
    ion_positions: Optional[Path] = None,
    water_positions: Optional[Path] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, Path]:
    """Build double-bubble microstate CSV products."""

    merged = load_and_merge_frame_tables(frame_index, water_trace, ion_trace, max_frames=max_frames)
    frame_rows = build_microstate_frame_rows(
        merged,
        bridge_rho_max_A=bridge_rho_max_A,
        bridge_s_min_A=bridge_s_min_A,
        bridge_s_max_A=bridge_s_max_A,
    )

    ion_position_rows: List[Dict[str, object]] = []
    species_region_rows: List[Dict[str, object]] = []
    if ion_positions is not None:
        raw_ions, _ = _read_csv_rows(ion_positions)
        ion_position_rows = build_ion_position_rows(raw_ions, frame_rows)
        species_region_rows = build_species_region_summary(ion_position_rows)

    water_position_rows: List[Dict[str, object]] = []
    if water_positions is not None:
        water_position_rows, _ = _read_csv_rows(water_positions)
        frame_rows = update_frame_rows_with_position_qc(frame_rows, water_position_rows)

    qc_rows = build_microstate_qc_rows(frame_rows, ion_position_rows, water_position_rows)

    output_dir = Path(output_dir)
    outputs = {
        "frame_table": output_dir / "bridge_microstate_frame_table.csv",
        "ion_positions": output_dir / "bridge_species_position_table.csv",
        "species_region_summary": output_dir / "bridge_species_region_summary.csv",
        "qc": output_dir / "bridge_microstate_qc.csv",
    }
    _write_csv_rows(outputs["frame_table"], frame_rows, fieldnames=[column for column in MICROSTATE_COLUMNS if any(column in row for row in frame_rows)])
    _write_csv_rows(outputs["ion_positions"], ion_position_rows)
    _write_csv_rows(outputs["species_region_summary"], species_region_rows)
    _write_csv_rows(outputs["qc"], qc_rows, fieldnames=["metric", "value", "status", "note"])
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build double-bubble bridge microstate tables")
    parser.add_argument("--frame-index", type=Path, required=True)
    parser.add_argument("--water-trace", type=Path, required=True)
    parser.add_argument("--ion-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-rho-max-A", type=float, default=8.0)
    parser.add_argument("--bridge-s-min-A", type=float, default=-10.0)
    parser.add_argument("--bridge-s-max-A", type=float, default=10.0)
    parser.add_argument("--ion-positions", type=Path)
    parser.add_argument("--water-positions", type=Path)
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outputs = analyze_bridge_microstate(
            frame_index=args.frame_index,
            water_trace=args.water_trace,
            ion_trace=args.ion_trace,
            output_dir=args.output_dir,
            bridge_rho_max_A=args.bridge_rho_max_A,
            bridge_s_min_A=args.bridge_s_min_A,
            bridge_s_max_A=args.bridge_s_max_A,
            ion_positions=args.ion_positions,
            water_positions=args.water_positions,
            max_frames=args.max_frames,
        )
    except Exception as exc:
        print(f"Double-bubble microstate analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
