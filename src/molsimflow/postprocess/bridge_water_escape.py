"""Bridge-water escape direction analysis from explicit seed-position tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.io.lammps_dump import minimum_image_vectors


EVENT_COLUMNS = (
    "case_label",
    "atom_id",
    "status",
    "escape_direction",
    "initial_frame",
    "exit_frame",
    "destination_frame",
    "initial_time",
    "exit_time",
    "destination_time",
    "residence_time",
    "initial_state",
    "exit_state",
    "destination_state",
    "initial_gap_A",
    "exit_gap_A",
    "destination_gap_A",
    "initial_x_A",
    "initial_y_A",
    "initial_z_A",
    "exit_x_A",
    "exit_y_A",
    "exit_z_A",
    "destination_x_A",
    "destination_y_A",
    "destination_z_A",
    "exit_disp_x_A",
    "exit_disp_y_A",
    "exit_disp_z_A",
    "exit_lateral_xy_disp_A",
    "destination_disp_x_A",
    "destination_disp_y_A",
    "destination_disp_z_A",
    "destination_lateral_xy_disp_A",
    "n_track_rows",
)

DIRECTION_LABELS = (
    "toward_bulk_or_zplus",
    "toward_TiO2_or_zminus",
    "lateral_xy",
    "unresolved",
    "retained",
)

GAP_SUMMARY_COLUMNS = (
    "gap_bin_left",
    "gap_bin_right",
    "gap_bin_center",
    "n_events",
    "n_exited",
    "exit_fraction",
    "toward_bulk_or_zplus_fraction",
    "toward_TiO2_or_zminus_fraction",
    "lateral_xy_fraction",
    "unresolved_fraction",
    "retained_fraction",
)


@dataclass(frozen=True)
class SeedPositionRow:
    """One tracked seed-water position at a frame/time."""

    atom_id: str
    frame: int
    time: float
    position: np.ndarray
    in_bridge: bool
    state: str = ""
    gap_A: float = math.nan
    source_row_index: int = -1


@dataclass(frozen=True)
class BridgeWaterEscapeConfig:
    """Settings for seed-water escape classification."""

    exit_confirm_frames: int = 1
    destination_lag_frames: int = 0
    direction_z_threshold_A: float = 2.0
    direction_lateral_threshold_A: float = 2.0
    direction_z_dominance_ratio: float = 1.2
    gap_bin_width_A: float = 2.0
    min_bin_count: int = 1

    def validate(self) -> None:
        if self.exit_confirm_frames < 1:
            raise ValueError("exit_confirm_frames must be >= 1")
        if self.destination_lag_frames < 0:
            raise ValueError("destination_lag_frames must be non-negative")
        if self.direction_z_threshold_A < 0.0 or self.direction_lateral_threshold_A < 0.0:
            raise ValueError("direction thresholds must be non-negative")
        if self.direction_z_dominance_ratio < 1.0:
            raise ValueError("direction_z_dominance_ratio must be >= 1")
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
        "status",
        "escape_direction",
        "count",
        "fraction_all",
        "fraction_exited",
        "gap_bin_left",
        "gap_bin_right",
        "gap_bin_center",
        "n_events",
        "n_exited",
        "exit_fraction",
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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    number = _as_float(value)
    if math.isfinite(number):
        return bool(round(number))
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "in", "bridge"}:
        return True
    if text in {"false", "f", "no", "n", "out", "outside"}:
        return False
    raise ValueError(f"Cannot parse boolean bridge membership: {value!r}")


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


def _parse_box_lengths(values: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("box_lengths must contain three positive finite values")
    return arr


def _minimum_delta(to_coord: np.ndarray, from_coord: np.ndarray, box_lengths: Optional[np.ndarray]) -> np.ndarray:
    delta = np.asarray(to_coord, dtype=float) - np.asarray(from_coord, dtype=float)
    if box_lengths is None:
        return delta
    return minimum_image_vectors(delta, box_lengths)


def load_seed_position_rows(
    path: Path,
    atom_column: str = "atom_id",
    frame_column: str = "frame",
    time_column: str = "time",
    x_column: str = "x",
    y_column: str = "y",
    z_column: str = "z",
    in_bridge_column: str = "in_bridge",
    state_column: Optional[str] = None,
    gap_column: Optional[str] = None,
) -> List[SeedPositionRow]:
    """Load a long-form seed-water position and bridge-membership table."""

    raw_rows, fieldnames = _read_csv_rows(path)
    atom_src = _pick_column(fieldnames, [atom_column, "atom_id", "oxygen_id", "entity_id"])
    frame_src = _pick_column(fieldnames, [frame_column, "frame", "Frame", "frame_index", "timestep", "step"])
    time_src = _pick_column(fieldnames, [time_column, "time", "time_ns", "Time", "Time(ns)", "t"], required=False)
    x_src = _pick_column(fieldnames, [x_column, "x", "x_A", "pos_x", "position_x"])
    y_src = _pick_column(fieldnames, [y_column, "y", "y_A", "pos_y", "position_y"])
    z_src = _pick_column(fieldnames, [z_column, "z", "z_A", "pos_z", "position_z"])
    bridge_src = _pick_column(
        fieldnames,
        [in_bridge_column, "in_bridge", "in_bridge_region", "bridge_member", "seed_in_bridge"],
    )
    state_src = _pick_column(fieldnames, [state_column, "state", "coalescence_state"], required=False)
    gap_src = _pick_column(
        fieldnames,
        [gap_column, "surface_gap_A", "surface_gap_estimate_A", "gap_A", "d3d_all"],
        required=False,
    )

    rows: List[SeedPositionRow] = []
    for index, raw in enumerate(raw_rows):
        atom_id = str(raw.get(atom_src, "")).strip()
        if not atom_id:
            raise ValueError(f"Missing atom id at source row {index}")
        position = np.asarray([_as_float(raw.get(x_src)), _as_float(raw.get(y_src)), _as_float(raw.get(z_src))], dtype=float)
        if not np.all(np.isfinite(position)):
            raise ValueError(f"Missing or invalid position at source row {index}")
        rows.append(
            SeedPositionRow(
                atom_id=atom_id,
                frame=_as_int(raw.get(frame_src)),
                time=_as_float(raw.get(time_src)) if time_src is not None else math.nan,
                position=position,
                in_bridge=_as_bool(raw.get(bridge_src)),
                state=str(raw.get(state_src, "")).strip() if state_src is not None else "",
                gap_A=_as_float(raw.get(gap_src)) if gap_src is not None else math.nan,
                source_row_index=index,
            )
        )
    rows.sort(key=lambda row: (row.atom_id, row.time if math.isfinite(row.time) else row.frame, row.frame))
    return rows


def classify_escape_direction(
    displacement: np.ndarray,
    status: str,
    z_threshold_A: float = 2.0,
    lateral_threshold_A: float = 2.0,
    z_dominance_ratio: float = 1.2,
) -> str:
    """Classify an escaped seed-water displacement direction."""

    if status != "exited":
        return status
    vec = np.asarray(displacement, dtype=float)
    if vec.size != 3 or not np.all(np.isfinite(vec)):
        return "unresolved"
    lateral_xy = float(math.hypot(float(vec[0]), float(vec[1])))
    dz = float(vec[2])
    abs_z = abs(dz)
    dominance = max(1.0, float(z_dominance_ratio))
    if abs_z >= float(z_threshold_A) and abs_z >= lateral_xy * dominance:
        return "toward_bulk_or_zplus" if dz > 0.0 else "toward_TiO2_or_zminus"
    if lateral_xy >= float(lateral_threshold_A) and lateral_xy >= abs_z / dominance:
        return "lateral_xy"
    if abs_z >= float(z_threshold_A):
        return "toward_bulk_or_zplus" if dz > 0.0 else "toward_TiO2_or_zminus"
    return "unresolved"


def _confirmed_exit_index(rows: Sequence[SeedPositionRow], confirm_frames: int) -> Optional[int]:
    confirm_frames = max(1, int(confirm_frames))
    for index, row in enumerate(rows):
        if row.in_bridge:
            continue
        end = min(len(rows), index + confirm_frames)
        if all(not rows[pos].in_bridge for pos in range(index, end)):
            return index
    return None


def build_seed_escape_events(
    rows: Sequence[SeedPositionRow],
    case_label: str = "",
    config: BridgeWaterEscapeConfig = BridgeWaterEscapeConfig(),
    box_lengths: Optional[Sequence[float]] = None,
) -> List[Dict[str, object]]:
    """Build one escape event row per tracked seed atom."""

    config.validate()
    lengths = _parse_box_lengths(box_lengths)
    grouped: Dict[str, List[SeedPositionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.atom_id].append(row)

    events: List[Dict[str, object]] = []
    for atom_id in sorted(grouped):
        track = sorted(grouped[atom_id], key=lambda row: (row.time if math.isfinite(row.time) else row.frame, row.frame))
        if not track:
            continue
        initial = track[0]
        exit_index = _confirmed_exit_index(track, config.exit_confirm_frames)
        status = "exited" if exit_index is not None else "retained"
        if exit_index is None:
            exit_index = len(track) - 1
        exit_row = track[int(exit_index)]
        destination_index = min(len(track) - 1, int(exit_index) + int(config.destination_lag_frames)) if status == "exited" else len(track) - 1
        destination = track[destination_index]

        exit_disp = _minimum_delta(exit_row.position, initial.position, lengths)
        destination_disp = _minimum_delta(destination.position, initial.position, lengths)
        direction = classify_escape_direction(
            destination_disp,
            status=status,
            z_threshold_A=config.direction_z_threshold_A,
            lateral_threshold_A=config.direction_lateral_threshold_A,
            z_dominance_ratio=config.direction_z_dominance_ratio,
        )
        residence_time = exit_row.time - initial.time if status == "exited" and math.isfinite(exit_row.time) and math.isfinite(initial.time) else math.nan
        events.append(
            {
                "case_label": case_label,
                "atom_id": atom_id,
                "status": status,
                "escape_direction": direction,
                "initial_frame": initial.frame,
                "exit_frame": exit_row.frame if status == "exited" else "",
                "destination_frame": destination.frame,
                "initial_time": _format_float(initial.time),
                "exit_time": _format_float(exit_row.time) if status == "exited" else "",
                "destination_time": _format_float(destination.time),
                "residence_time": _format_float(residence_time),
                "initial_state": initial.state,
                "exit_state": exit_row.state if status == "exited" else "retained",
                "destination_state": destination.state,
                "initial_gap_A": _format_float(initial.gap_A),
                "exit_gap_A": _format_float(exit_row.gap_A) if status == "exited" else "",
                "destination_gap_A": _format_float(destination.gap_A),
                "initial_x_A": float(initial.position[0]),
                "initial_y_A": float(initial.position[1]),
                "initial_z_A": float(initial.position[2]),
                "exit_x_A": float(exit_row.position[0]),
                "exit_y_A": float(exit_row.position[1]),
                "exit_z_A": float(exit_row.position[2]),
                "destination_x_A": float(destination.position[0]),
                "destination_y_A": float(destination.position[1]),
                "destination_z_A": float(destination.position[2]),
                "exit_disp_x_A": float(exit_disp[0]),
                "exit_disp_y_A": float(exit_disp[1]),
                "exit_disp_z_A": float(exit_disp[2]),
                "exit_lateral_xy_disp_A": float(math.hypot(float(exit_disp[0]), float(exit_disp[1]))),
                "destination_disp_x_A": float(destination_disp[0]),
                "destination_disp_y_A": float(destination_disp[1]),
                "destination_disp_z_A": float(destination_disp[2]),
                "destination_lateral_xy_disp_A": float(math.hypot(float(destination_disp[0]), float(destination_disp[1]))),
                "n_track_rows": len(track),
            }
        )
    return events


def summarize_escape_events(events: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Summarize escape directions and retained/exited fractions."""

    total = max(len(events), 1)
    total_exited = max(sum(1 for event in events if str(event.get("status")) == "exited"), 1)
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for event in events:
        key = (str(event.get("case_label", "")), str(event.get("status", "")), str(event.get("escape_direction", "")))
        counts[key] += 1
    out: List[Dict[str, object]] = []
    for key in sorted(counts):
        case_label, status, direction = key
        count = counts[key]
        out.append(
            {
                "case_label": case_label,
                "status": status,
                "escape_direction": direction,
                "count": count,
                "fraction_all": count / total,
                "fraction_exited": count / total_exited if status == "exited" else "",
            }
        )
    return out


def summarize_escape_by_gap(
    events: Sequence[Mapping[str, object]],
    gap_bin_width_A: float = 2.0,
    min_bin_count: int = 1,
) -> List[Dict[str, object]]:
    """Summarize escape fractions by destination-gap bins when gaps exist."""

    gap_values = [_as_float(event.get("destination_gap_A")) for event in events]
    finite = [value for value in gap_values if math.isfinite(value)]
    if not finite:
        return []
    low = math.floor(min(finite) / float(gap_bin_width_A)) * float(gap_bin_width_A)
    high = math.ceil(max(finite) / float(gap_bin_width_A)) * float(gap_bin_width_A)
    if math.isclose(low, high):
        high = low + float(gap_bin_width_A)
    edges = np.arange(low, high + float(gap_bin_width_A) * 0.5, float(gap_bin_width_A))
    out: List[Dict[str, object]] = []
    for index in range(len(edges) - 1):
        left = float(edges[index])
        right = float(edges[index + 1])
        if index == len(edges) - 2:
            chunk = [
                event
                for event, gap in zip(events, gap_values)
                if math.isfinite(gap) and ((gap >= left and gap < right) or math.isclose(gap, right))
            ]
        else:
            chunk = [event for event, gap in zip(events, gap_values) if math.isfinite(gap) and gap >= left and gap < right]
        if len(chunk) < int(min_bin_count):
            continue
        n_events = len(chunk)
        n_exited = sum(1 for event in chunk if str(event.get("status")) == "exited")
        row: Dict[str, object] = {
            "gap_bin_left": left,
            "gap_bin_right": right,
            "gap_bin_center": 0.5 * (left + right),
            "n_events": n_events,
            "n_exited": n_exited,
            "exit_fraction": n_exited / max(n_events, 1),
        }
        for direction in DIRECTION_LABELS:
            row[f"{direction}_fraction"] = sum(
                1 for event in chunk if str(event.get("escape_direction")) == direction
            ) / max(n_events, 1)
        out.append(row)
    return out


def analyze_bridge_water_escape(
    input_csv: Path,
    output_dir: Path,
    case_label: str = "",
    config: BridgeWaterEscapeConfig = BridgeWaterEscapeConfig(),
    box_lengths: Optional[Sequence[float]] = None,
    atom_column: str = "atom_id",
    frame_column: str = "frame",
    time_column: str = "time",
    x_column: str = "x",
    y_column: str = "y",
    z_column: str = "z",
    in_bridge_column: str = "in_bridge",
    state_column: Optional[str] = None,
    gap_column: Optional[str] = None,
) -> Dict[str, Path]:
    """Run escape-direction analysis and write CSV outputs."""

    rows = load_seed_position_rows(
        input_csv,
        atom_column=atom_column,
        frame_column=frame_column,
        time_column=time_column,
        x_column=x_column,
        y_column=y_column,
        z_column=z_column,
        in_bridge_column=in_bridge_column,
        state_column=state_column,
        gap_column=gap_column,
    )
    events = build_seed_escape_events(rows, case_label=case_label, config=config, box_lengths=box_lengths)
    summary = summarize_escape_events(events)
    gap_summary = summarize_escape_by_gap(
        events,
        gap_bin_width_A=config.gap_bin_width_A,
        min_bin_count=config.min_bin_count,
    )
    output_dir = Path(output_dir)
    outputs = {
        "events": output_dir / "bridge_water_escape_events.csv",
        "direction_summary": output_dir / "bridge_water_escape_direction_summary.csv",
        "gap_summary": output_dir / "bridge_water_escape_gap_summary.csv",
        "state_statistics": output_dir / "state_statistics.csv",
    }
    _write_csv_rows(outputs["events"], events, fieldnames=EVENT_COLUMNS)
    _write_csv_rows(outputs["direction_summary"], summary)
    _write_csv_rows(outputs["gap_summary"], gap_summary, fieldnames=GAP_SUMMARY_COLUMNS)
    _write_csv_rows(
        outputs["state_statistics"],
        [
            {"metric": "input_csv", "value": str(input_csv)},
            {"metric": "case_label", "value": case_label},
            {"metric": "n_input_rows", "value": len(rows)},
            {"metric": "n_seed_atoms", "value": len({row.atom_id for row in rows})},
            {"metric": "n_events", "value": len(events)},
            {"metric": "n_exited", "value": sum(1 for event in events if str(event.get("status")) == "exited")},
            {"metric": "exit_confirm_frames", "value": config.exit_confirm_frames},
            {"metric": "destination_lag_frames", "value": config.destination_lag_frames},
            {"metric": "direction_z_threshold_A", "value": config.direction_z_threshold_A},
            {"metric": "direction_lateral_threshold_A", "value": config.direction_lateral_threshold_A},
            {"metric": "direction_z_dominance_ratio", "value": config.direction_z_dominance_ratio},
            {"metric": "box_lengths", "value": ",".join(str(value) for value in box_lengths) if box_lengths is not None else ""},
        ],
        fieldnames=["metric", "value"],
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify tracked bridge-water escape directions")
    parser.add_argument("--input", type=Path, required=True, help="Input seed-position CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-label", default="")
    parser.add_argument("--atom-column", default="atom_id")
    parser.add_argument("--frame-column", default="frame")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--x-column", default="x")
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--z-column", default="z")
    parser.add_argument("--in-bridge-column", default="in_bridge")
    parser.add_argument("--state-column")
    parser.add_argument("--gap-column")
    parser.add_argument("--box-lengths", type=float, nargs=3)
    parser.add_argument("--exit-confirm-frames", type=int, default=1)
    parser.add_argument("--destination-lag-frames", type=int, default=0)
    parser.add_argument("--direction-z-threshold-A", type=float, default=2.0)
    parser.add_argument("--direction-lateral-threshold-A", type=float, default=2.0)
    parser.add_argument("--direction-z-dominance-ratio", type=float, default=1.2)
    parser.add_argument("--gap-bin-width-A", type=float, default=2.0)
    parser.add_argument("--min-bin-count", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outputs = analyze_bridge_water_escape(
            input_csv=args.input,
            output_dir=args.output_dir,
            case_label=args.case_label,
            config=BridgeWaterEscapeConfig(
                exit_confirm_frames=args.exit_confirm_frames,
                destination_lag_frames=args.destination_lag_frames,
                direction_z_threshold_A=args.direction_z_threshold_A,
                direction_lateral_threshold_A=args.direction_lateral_threshold_A,
                direction_z_dominance_ratio=args.direction_z_dominance_ratio,
                gap_bin_width_A=args.gap_bin_width_A,
                min_bin_count=args.min_bin_count,
            ),
            box_lengths=args.box_lengths,
            atom_column=args.atom_column,
            frame_column=args.frame_column,
            time_column=args.time_column,
            x_column=args.x_column,
            y_column=args.y_column,
            z_column=args.z_column,
            in_bridge_column=args.in_bridge_column,
            state_column=args.state_column,
            gap_column=args.gap_column,
        )
    except Exception as exc:
        print(f"Bridge-water escape analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
