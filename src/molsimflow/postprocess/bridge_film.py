"""Bridge liquid-film state and residence summaries from tabular metrics."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


STATE_SUMMARY_COLUMNS = (
    "scope",
    "bridge_film_state",
    "n_frames",
    "fraction",
    "N_water_bridge_mean",
    "N_OH_bridge_mean",
    "N_H3O_bridge_mean",
    "N_Na_bridge_mean",
    "N_Cl_bridge_mean",
    "bridge_charge_density_e_per_A3_mean",
    "tio2_hydration_bridge_fraction_mean",
)

RESIDENCE_EVENT_COLUMNS = (
    "species",
    "atom_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "n_frames",
    "duration",
    "duration_unit",
)

RESIDENCE_SUMMARY_COLUMNS = (
    "species",
    "n_events",
    "duration_mean",
    "duration_median",
    "duration_std",
    "duration_q25",
    "duration_q75",
    "duration_p90",
    "duration_unit",
)

COORDINATION_SUMMARY_COLUMNS = (
    "species",
    "n_samples",
    "coordination_mean",
    "coordination_std",
    "coordination_median",
    "coordination_q25",
    "coordination_q75",
    "coordination_p90",
)


@dataclass(frozen=True)
class BridgeFilmConfig:
    """Thresholds for bridge liquid-film classification and barrier selection."""

    min_oxygen_for_film: int = 3
    min_reactive_count: int = 1
    acid_base_ratio_threshold: float = 2.0
    min_salt_ion_count: int = 1
    salt_ratio_threshold: float = 0.08
    hydration_extension_threshold: float = 0.25
    min_hydration_count: int = 2
    barrier_event_window_before: int = 2
    barrier_event_window_after: int = 2
    barrier_cv_min: Optional[float] = None
    barrier_cv_max: Optional[float] = None
    barrier_cv_quantile_width: float = 0.10
    barrier_dewet_top_fraction: float = 0.10
    time_tolerance: float = 0.0005


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
        "frame",
        "time",
        "time_ns",
        "d3d_all",
        "barrier_top_flag",
        "bridge_film_state",
        "scope",
        "species",
        "atom_id",
        "start_frame",
        "end_frame",
        "n_frames",
        "duration",
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


def _as_int(value: object) -> int:
    number = _as_float(value)
    return max(0, int(round(number))) if math.isfinite(number) else 0


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    number = _as_float(value)
    if math.isfinite(number):
        return number != 0.0
    return str(value).strip().lower() in {"true", "t", "yes", "y", "in", "inside"}


def _finite_values(values: Iterable[object]) -> List[float]:
    out: List[float] = []
    for value in values:
        number = _as_float(value)
        if math.isfinite(number):
            out.append(number)
    return out


def _mean(values: Iterable[object]) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite else math.nan


def _summary_stats(values: Iterable[object]) -> Dict[str, float]:
    finite = np.asarray(_finite_values(values), dtype=float)
    if finite.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "p90": math.nan,
        }
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan,
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def classify_bridge_film_state(row: Mapping[str, object], config: BridgeFilmConfig = BridgeFilmConfig()) -> str:
    """Classify bridge film state from per-frame composition counts."""

    n_oxygen = _as_int(row.get("N_oxygen_bridge_total", row.get("Nw_bridge", 0)))
    if n_oxygen < int(config.min_oxygen_for_film):
        return "dry_or_vapor"

    n_water = _as_int(row.get("N_water_bridge", row.get("Nw_bridge", 0)))
    n_oh = _as_int(row.get("N_OH_bridge", 0))
    n_h3o = _as_int(row.get("N_H3O_bridge", 0))
    n_na = _as_int(row.get("N_Na_bridge", 0))
    n_cl = _as_int(row.get("N_Cl_bridge", 0))
    n_hydration = _as_int(row.get("N_tio2_hydration_bridge", 0))

    hydration_fraction = n_hydration / max(1, n_oxygen)
    salinity_ratio = (n_na + n_cl) / max(1, n_oxygen)

    if n_hydration >= int(config.min_hydration_count) and hydration_fraction >= float(config.hydration_extension_threshold):
        return "tio2_hydration_extended_film"
    if n_oh >= int(config.min_reactive_count) and n_oh >= float(config.acid_base_ratio_threshold) * max(1, n_h3o):
        return "basic_film"
    if n_h3o >= int(config.min_reactive_count) and n_h3o >= float(config.acid_base_ratio_threshold) * max(1, n_oh):
        return "acidic_film"
    if (
        n_na >= int(config.min_salt_ion_count)
        and n_cl >= int(config.min_salt_ion_count)
        and salinity_ratio >= float(config.salt_ratio_threshold)
    ):
        return "saltwater_film"
    if n_water > 0:
        return "water_film"
    return "mixed_film"


def _canonical_frame_time(row: Mapping[str, object], index: int) -> Tuple[int, float]:
    frame = _as_float(row.get("frame", row.get("Frame", row.get("timestep", index))))
    time_value = _as_float(row.get("time", row.get("time_ns", row.get("Time", math.nan))))
    return int(round(frame)) if math.isfinite(frame) else index, time_value


def _load_events(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        return []
    rows, _fieldnames = _read_csv_rows(path)
    return rows


def _select_barrier_mask(
    frame_rows: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    config: BridgeFilmConfig,
) -> Tuple[List[bool], str, str]:
    n_rows = len(frame_rows)
    if n_rows == 0:
        return [], "empty", "no frames"

    frames = [_canonical_frame_time(row, index)[0] for index, row in enumerate(frame_rows)]
    times = [_canonical_frame_time(row, index)[1] for index, row in enumerate(frame_rows)]

    if events:
        mask = [False] * n_rows
        for event in events:
            event_frame = _as_float(event.get("event_frame"))
            event_time = _as_float(event.get("event_time"))
            anchor: Optional[int] = None
            if math.isfinite(event_frame):
                for index, frame in enumerate(frames):
                    if frame == int(round(event_frame)):
                        anchor = index
                        break
            if anchor is None and math.isfinite(event_time):
                deltas = [abs(time - event_time) if math.isfinite(time) else math.inf for time in times]
                best = int(np.argmin(np.asarray(deltas, dtype=float)))
                if deltas[best] <= float(config.time_tolerance):
                    anchor = best
            if anchor is None:
                continue
            left = max(0, anchor - int(config.barrier_event_window_before))
            right = min(n_rows - 1, anchor + int(config.barrier_event_window_after))
            for index in range(left, right + 1):
                mask[index] = True
        return mask, "transition_events_window", (
            f"before={config.barrier_event_window_before},after={config.barrier_event_window_after}"
        )

    d3d = [_as_float(row.get("d3d_all")) for row in frame_rows]
    finite_d3d = [value for value in d3d if math.isfinite(value)]
    if finite_d3d:
        if config.barrier_cv_min is not None and config.barrier_cv_max is not None:
            low = min(float(config.barrier_cv_min), float(config.barrier_cv_max))
            high = max(float(config.barrier_cv_min), float(config.barrier_cv_max))
            return [math.isfinite(value) and low <= value <= high for value in d3d], "cv_range", f"[{low},{high}]"
        width = float(config.barrier_cv_quantile_width)
        if width <= 0.0 or width >= 1.0:
            width = 0.10
        low_q = max(0.0, 0.5 - 0.5 * width)
        high_q = min(1.0, 0.5 + 0.5 * width)
        low = float(np.quantile(np.asarray(finite_d3d, dtype=float), low_q))
        high = float(np.quantile(np.asarray(finite_d3d, dtype=float), high_q))
        return [math.isfinite(value) and low <= value <= high for value in d3d], "cv_median_quantile_window", (
            f"q[{low_q:.3f},{high_q:.3f}]=>[{low:.6g},{high:.6g}]"
        )

    dewet = [_as_float(row.get("dewet_fraction")) for row in frame_rows]
    finite_dewet = [value for value in dewet if math.isfinite(value)]
    if finite_dewet:
        fraction = float(config.barrier_dewet_top_fraction)
        if fraction <= 0.0 or fraction >= 1.0:
            fraction = 0.10
        threshold = float(np.quantile(np.asarray(finite_dewet, dtype=float), 1.0 - fraction))
        return [math.isfinite(value) and value >= threshold for value in dewet], "top_dewet_fraction", (
            f">=quantile({1.0 - fraction:.3f})"
        )

    return [False] * n_rows, "none", "no event/cv/dewet basis"


def build_bridge_film_frame_table(
    frame_rows: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]] = (),
    config: BridgeFilmConfig = BridgeFilmConfig(),
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Add bridge film state and barrier-top flags to frame rows."""

    sorted_rows = [dict(row) for row in frame_rows]
    sorted_rows.sort(key=lambda item: (_canonical_frame_time(item, 0)[1], _canonical_frame_time(item, 0)[0]))
    barrier_mask, barrier_mode, barrier_detail = _select_barrier_mask(sorted_rows, events, config)
    out: List[Dict[str, object]] = []
    for index, row in enumerate(sorted_rows):
        frame, time_value = _canonical_frame_time(row, index)
        row["frame"] = frame
        if math.isfinite(time_value):
            row["time"] = time_value
        row["bridge_film_state"] = classify_bridge_film_state(row, config=config)
        row["barrier_top_flag"] = bool(barrier_mask[index]) if index < len(barrier_mask) else False
        if "tio2_hydration_bridge_fraction" not in row:
            n_oxygen = _as_float(row.get("N_oxygen_bridge_total"))
            n_hydration = _as_float(row.get("N_tio2_hydration_bridge"))
            row["tio2_hydration_bridge_fraction"] = (
                n_hydration / n_oxygen if math.isfinite(n_oxygen) and n_oxygen > 0 and math.isfinite(n_hydration) else math.nan
            )
        out.append(row)
    return out, {"barrier_mode": barrier_mode, "barrier_detail": barrier_detail}


def summarize_state_distribution(frame_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Summarize bridge film state fractions for all/barrier/non-barrier scopes."""

    scopes = {
        "all_frames": list(frame_rows),
        "barrier_top": [row for row in frame_rows if _truthy(row.get("barrier_top_flag"))],
        "non_barrier": [row for row in frame_rows if not _truthy(row.get("barrier_top_flag"))],
    }
    out: List[Dict[str, object]] = []
    for scope, rows in scopes.items():
        total = len(rows)
        if total == 0:
            continue
        states = sorted({str(row.get("bridge_film_state", "")) for row in rows})
        for state in states:
            chunk = [row for row in rows if str(row.get("bridge_film_state", "")) == state]
            out.append(
                {
                    "scope": scope,
                    "bridge_film_state": state,
                    "n_frames": len(chunk),
                    "fraction": len(chunk) / float(total),
                    "N_water_bridge_mean": _mean(row.get("N_water_bridge", row.get("Nw_bridge")) for row in chunk),
                    "N_OH_bridge_mean": _mean(row.get("N_OH_bridge") for row in chunk),
                    "N_H3O_bridge_mean": _mean(row.get("N_H3O_bridge") for row in chunk),
                    "N_Na_bridge_mean": _mean(row.get("N_Na_bridge") for row in chunk),
                    "N_Cl_bridge_mean": _mean(row.get("N_Cl_bridge") for row in chunk),
                    "bridge_charge_density_e_per_A3_mean": _mean(row.get("bridge_charge_density_e_per_A3") for row in chunk),
                    "tio2_hydration_bridge_fraction_mean": _mean(row.get("tio2_hydration_bridge_fraction") for row in chunk),
                }
            )
    return out


def build_residence_events(membership_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Build residence episodes from long-form in-bridge membership rows."""

    sorted_rows = sorted(
        membership_rows,
        key=lambda row: (
            str(row.get("species", "")),
            int(_as_float(row.get("atom_id"), 0)),
            _as_float(row.get("time", row.get("time_ns")), math.inf),
            _as_float(row.get("frame"), math.inf),
        ),
    )
    grouped: Dict[Tuple[str, int], List[Mapping[str, object]]] = {}
    for row in sorted_rows:
        species = str(row.get("species", "")).strip()
        atom_id = int(_as_float(row.get("atom_id"), 0))
        if not species or atom_id == 0:
            continue
        grouped.setdefault((species, atom_id), []).append(row)

    events: List[Dict[str, object]] = []
    for (species, atom_id), rows in grouped.items():
        active: Optional[Dict[str, object]] = None
        previous_frame = math.nan
        previous_time = math.nan
        for index, row in enumerate(rows):
            in_bridge = _truthy(row.get("in_bridge", row.get("in_bridge_region", row.get("bridge_member", False))))
            frame, time_value = _canonical_frame_time(row, index)
            if in_bridge and active is None:
                active = {
                    "species": species,
                    "atom_id": atom_id,
                    "start_frame": frame,
                    "end_frame": frame,
                    "start_time": time_value,
                    "end_time": time_value,
                }
            elif in_bridge and active is not None:
                active["end_frame"] = frame
                active["end_time"] = time_value
            elif (not in_bridge) and active is not None:
                active["end_frame"] = previous_frame if math.isfinite(previous_frame) else active["end_frame"]
                active["end_time"] = previous_time if math.isfinite(previous_time) else active["end_time"]
                events.append(_finalize_residence_event(active))
                active = None
            previous_frame = frame
            previous_time = time_value
        if active is not None:
            events.append(_finalize_residence_event(active))
    return events


def _finalize_residence_event(event: Mapping[str, object]) -> Dict[str, object]:
    start_frame = int(_as_float(event.get("start_frame"), 0))
    end_frame = int(_as_float(event.get("end_frame"), start_frame))
    start_time = _as_float(event.get("start_time"))
    end_time = _as_float(event.get("end_time"))
    n_frames = max(1, end_frame - start_frame + 1)
    if math.isfinite(start_time) and math.isfinite(end_time):
        duration = max(0.0, end_time - start_time)
        unit = "time"
    else:
        duration = float(n_frames)
        unit = "frames"
    return {
        "species": event.get("species", ""),
        "atom_id": int(_as_float(event.get("atom_id"), 0)),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time": start_time,
        "end_time": end_time,
        "n_frames": n_frames,
        "duration": duration,
        "duration_unit": unit,
    }


def summarize_residence_events(events: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Summarize residence episode durations by species."""

    out: List[Dict[str, object]] = []
    for species in sorted({str(event.get("species", "")) for event in events if str(event.get("species", ""))}):
        chunk = [event for event in events if str(event.get("species", "")) == species]
        stats = _summary_stats(event.get("duration") for event in chunk)
        unit = str(chunk[0].get("duration_unit", "")) if chunk else ""
        out.append(
            {
                "species": species,
                "n_events": stats["n"],
                "duration_mean": stats["mean"],
                "duration_median": stats["median"],
                "duration_std": stats["std"],
                "duration_q25": stats["q25"],
                "duration_q75": stats["q75"],
                "duration_p90": stats["p90"],
                "duration_unit": unit,
            }
        )
    return out


def summarize_coordination_samples(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Summarize long-form coordination samples by species."""

    out: List[Dict[str, object]] = []
    species_values = sorted({str(row.get("species", "")).strip() for row in rows if str(row.get("species", "")).strip()})
    for species in species_values:
        chunk = [row for row in rows if str(row.get("species", "")).strip() == species]
        stats = _summary_stats(row.get("coordination", row.get("coordination_count")) for row in chunk)
        out.append(
            {
                "species": species,
                "n_samples": stats["n"],
                "coordination_mean": stats["mean"],
                "coordination_std": stats["std"],
                "coordination_median": stats["median"],
                "coordination_q25": stats["q25"],
                "coordination_q75": stats["q75"],
                "coordination_p90": stats["p90"],
            }
        )
    return out


def analyze_bridge_film(
    frame_table: Path,
    output_dir: Path,
    config: BridgeFilmConfig = BridgeFilmConfig(),
    transition_events: Optional[Path] = None,
    residence_membership: Optional[Path] = None,
    coordination_samples: Optional[Path] = None,
) -> Dict[str, Path]:
    """Write bridge film state, barrier, residence, and coordination summaries."""

    frame_rows, _fields = _read_csv_rows(frame_table)
    event_rows = _load_events(transition_events)
    enriched, state_info = build_bridge_film_frame_table(frame_rows, events=event_rows, config=config)
    state_summary = summarize_state_distribution(enriched)
    barrier_summary = [row for row in state_summary if row.get("scope") == "barrier_top"]

    if residence_membership is not None:
        membership_rows, _membership_fields = _read_csv_rows(residence_membership)
        residence_events = build_residence_events(membership_rows)
    else:
        residence_events = []
    residence_summary = summarize_residence_events(residence_events)

    if coordination_samples is not None:
        coordination_rows, _coord_fields = _read_csv_rows(coordination_samples)
        coordination_summary = summarize_coordination_samples(coordination_rows)
    else:
        coordination_summary = []

    output_dir = Path(output_dir)
    outputs = {
        "frame_table": output_dir / "bridge_film_frame_table.csv",
        "state_summary": output_dir / "bridge_film_state_summary.csv",
        "barrier_top_summary": output_dir / "bridge_film_barrier_top_summary.csv",
        "residence_events": output_dir / "bridge_film_residence_events.csv",
        "residence_summary": output_dir / "bridge_film_residence_summary.csv",
        "coordination_summary": output_dir / "bridge_film_coordination_summary.csv",
        "state_statistics": output_dir / "state_statistics.csv",
    }
    _write_csv_rows(outputs["frame_table"], enriched)
    _write_csv_rows(outputs["state_summary"], state_summary, fieldnames=STATE_SUMMARY_COLUMNS)
    _write_csv_rows(outputs["barrier_top_summary"], barrier_summary, fieldnames=STATE_SUMMARY_COLUMNS)
    _write_csv_rows(outputs["residence_events"], residence_events, fieldnames=RESIDENCE_EVENT_COLUMNS)
    _write_csv_rows(outputs["residence_summary"], residence_summary, fieldnames=RESIDENCE_SUMMARY_COLUMNS)
    _write_csv_rows(outputs["coordination_summary"], coordination_summary, fieldnames=COORDINATION_SUMMARY_COLUMNS)
    _write_csv_rows(
        outputs["state_statistics"],
        [
            {"metric": "input_frame_table", "value": str(frame_table)},
            {"metric": "transition_events", "value": str(transition_events) if transition_events else ""},
            {"metric": "residence_membership", "value": str(residence_membership) if residence_membership else ""},
            {"metric": "coordination_samples", "value": str(coordination_samples) if coordination_samples else ""},
            {"metric": "barrier_mode", "value": state_info["barrier_mode"]},
            {"metric": "barrier_detail", "value": state_info["barrier_detail"]},
            {"metric": "total_frames", "value": len(enriched)},
            {"metric": "residence_events", "value": len(residence_events)},
        ],
        fieldnames=["metric", "value"],
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize bridge liquid-film states from frame tables")
    parser.add_argument("--frame-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition-events", type=Path)
    parser.add_argument("--residence-membership", type=Path, help="Long CSV with species,atom_id,frame/time,in_bridge")
    parser.add_argument("--coordination-samples", type=Path, help="Long CSV with species,coordination rows")
    parser.add_argument("--min-oxygen-for-film", type=int, default=3)
    parser.add_argument("--min-reactive-count", type=int, default=1)
    parser.add_argument("--acid-base-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--min-salt-ion-count", type=int, default=1)
    parser.add_argument("--salt-ratio-threshold", type=float, default=0.08)
    parser.add_argument("--hydration-extension-threshold", type=float, default=0.25)
    parser.add_argument("--min-hydration-count", type=int, default=2)
    parser.add_argument("--barrier-event-window-before", type=int, default=2)
    parser.add_argument("--barrier-event-window-after", type=int, default=2)
    parser.add_argument("--barrier-cv-min", type=float)
    parser.add_argument("--barrier-cv-max", type=float)
    parser.add_argument("--barrier-cv-quantile-width", type=float, default=0.10)
    parser.add_argument("--barrier-dewet-top-fraction", type=float, default=0.10)
    parser.add_argument("--time-tolerance", type=float, default=0.0005)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outputs = analyze_bridge_film(
            frame_table=args.frame_table,
            output_dir=args.output_dir,
            transition_events=args.transition_events,
            residence_membership=args.residence_membership,
            coordination_samples=args.coordination_samples,
            config=BridgeFilmConfig(
                min_oxygen_for_film=args.min_oxygen_for_film,
                min_reactive_count=args.min_reactive_count,
                acid_base_ratio_threshold=args.acid_base_ratio_threshold,
                min_salt_ion_count=args.min_salt_ion_count,
                salt_ratio_threshold=args.salt_ratio_threshold,
                hydration_extension_threshold=args.hydration_extension_threshold,
                min_hydration_count=args.min_hydration_count,
                barrier_event_window_before=args.barrier_event_window_before,
                barrier_event_window_after=args.barrier_event_window_after,
                barrier_cv_min=args.barrier_cv_min,
                barrier_cv_max=args.barrier_cv_max,
                barrier_cv_quantile_width=args.barrier_cv_quantile_width,
                barrier_dewet_top_fraction=args.barrier_dewet_top_fraction,
                time_tolerance=args.time_tolerance,
            ),
        )
    except Exception as exc:
        print(f"Bridge film analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
