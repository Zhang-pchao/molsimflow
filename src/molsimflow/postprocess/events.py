"""Generic event detection and event-aligned table summaries."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DERIVATIVE_COLUMNS = (
    "Nw_bridge",
    "dewet_fraction",
    "DeltaN_dewet",
    "largest_water_cluster_size_bridge",
)

EVENT_OUTPUT_COLUMNS = (
    "event_id",
    "event_anchor_index",
    "event_frame",
    "event_time",
    "event_type",
    "trigger_metric",
    "trigger_value",
    "Nw_bridge",
    "dewet_fraction",
    "water_bridge_connected_flag",
    "d3d_all",
    "notes",
)

EVENT_ALIGNED_BASE_COLUMNS = (
    "event_id",
    "event_type",
    "event_frame",
    "event_time",
    "relative_index",
    "relative_time",
    "frame",
    "time",
)


@dataclass(frozen=True)
class TransitionEventConfig:
    """Settings for transition-event detection and summaries."""

    event_method: str = "hybrid"
    connectivity_event: str = "loss"
    nw_drop_threshold: Optional[float] = None
    dewet_rate_threshold: Optional[float] = None
    dewet_fraction_threshold: Optional[float] = None
    min_event_separation: int = 3
    event_window_before: int = 20
    event_window_after: int = 20
    allow_partial_windows: bool = False
    lag_window: int = 20
    change_window: int = 10

    def validate(self) -> None:
        if self.event_method not in {"connectivity_loss", "nw_drop", "dewet_jump", "hybrid"}:
            raise ValueError("Unsupported event_method")
        if self.connectivity_event not in {"loss", "gain", "any"}:
            raise ValueError("connectivity_event must be loss, gain, or any")
        if self.min_event_separation < 0:
            raise ValueError("min_event_separation must be non-negative")
        if self.event_window_before < 0 or self.event_window_after < 0:
            raise ValueError("event windows must be non-negative")
        if self.lag_window < 0 or self.change_window < 0:
            raise ValueError("lag_window and change_window must be non-negative")


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
        "event_id",
        "event_anchor_index",
        "event_type",
        "event_frame",
        "event_time",
        "relative_index",
        "relative_time",
        "frame",
        "time",
        "feature",
        "target",
        "lag",
        "count",
        "mean",
        "median",
        "std",
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


def _bool_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = _as_float(value)
    if math.isfinite(number):
        return 1.0 if round(number) != 0 else 0.0
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y"}:
        return 1.0
    if text in {"false", "f", "no", "n"}:
        return 0.0
    return math.nan


def _finite_values(values: Iterable[object]) -> List[float]:
    out: List[float] = []
    for value in values:
        number = _as_float(value)
        if math.isfinite(number):
            out.append(number)
    return out


def _numeric_columns(rows: Sequence[Mapping[str, object]], exclude: Iterable[str] = ()) -> List[str]:
    excluded = set(exclude)
    keys = sorted({key for row in rows for key in row.keys()})
    out: List[str] = []
    for key in keys:
        if key in excluded:
            continue
        values = [_as_float(row.get(key)) for row in rows]
        if any(math.isfinite(value) for value in values):
            out.append(key)
    return out


def _pick_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    available = set(fieldnames)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def load_feature_table(
    path: Path,
    frame_column: str = "frame",
    time_column: str = "time_ns",
) -> List[Dict[str, object]]:
    """Load a feature CSV and standardize canonical `frame` and `time` keys."""

    rows, fieldnames = _read_csv_rows(path)
    frame_src = _pick_column(fieldnames, [frame_column, "frame", "Frame", "frame_index", "timestep", "step"])
    time_src = _pick_column(fieldnames, [time_column, "time", "Time", "time_ns", "Time(ns)", "t"])
    out: List[Dict[str, object]] = []
    for index, row in enumerate(rows):
        item: Dict[str, object] = dict(row)
        item["source_row_index"] = index
        if frame_src is not None:
            frame = _as_float(row.get(frame_src))
            item["frame"] = int(round(frame)) if math.isfinite(frame) else math.nan
        elif "frame" not in item:
            item["frame"] = index
        if time_src is not None:
            item["time"] = _as_float(row.get(time_src))
        elif "time" not in item:
            item["time"] = math.nan
        if "water_bridge_connected_flag" in item:
            item["water_bridge_connected_flag"] = _bool_float(item.get("water_bridge_connected_flag"))
        out.append(item)
    out.sort(key=lambda item: (_as_float(item.get("time"), math.inf), _as_float(item.get("frame"), math.inf)))
    return out


def _select_derivative_basis(rows: Sequence[Mapping[str, object]]) -> Tuple[List[float], str]:
    times = [_as_float(row.get("time")) for row in rows]
    finite_times = [value for value in times if math.isfinite(value)]
    if len(set(finite_times)) >= 2:
        return times, "time"
    frames = [_as_float(row.get("frame")) for row in rows]
    finite_frames = [value for value in frames if math.isfinite(value)]
    if len(set(finite_frames)) >= 2:
        return frames, "frame"
    return [float(index) for index in range(len(rows))], "index"


def _derivative_name(column: str) -> str:
    if column == "Nw_bridge":
        return "dNw_bridge_dt"
    if column == "dewet_fraction":
        return "ddewet_fraction_dt"
    return f"d{column}_dt"


def add_derivative_features(
    rows: Sequence[Mapping[str, object]],
    derivative_columns: Sequence[str] = DEFAULT_DERIVATIVE_COLUMNS,
) -> Tuple[List[Dict[str, object]], str, List[str]]:
    """Add backward finite-difference derivatives for selected columns."""

    out = [dict(row) for row in rows]
    basis, basis_name = _select_derivative_basis(out)
    created: List[str] = []
    for column in derivative_columns:
        if not any(column in row for row in out):
            continue
        values = [_as_float(row.get(column)) for row in out]
        dst = _derivative_name(column)
        created.append(dst)
        for index, row in enumerate(out):
            if index == 0:
                row[dst] = math.nan
                continue
            if not all(math.isfinite(value) for value in [values[index], values[index - 1], basis[index], basis[index - 1]]):
                row[dst] = math.nan
                continue
            delta = basis[index] - basis[index - 1]
            row[dst] = (values[index] - values[index - 1]) / delta if abs(delta) > 1e-14 else math.nan
    return out, basis_name, created


def _event_candidates_connectivity(
    rows: Sequence[Mapping[str, object]],
    connectivity_event: str,
) -> Dict[int, List[Tuple[str, str, float]]]:
    if not any("water_bridge_connected_flag" in row for row in rows):
        if connectivity_event:
            return {}
    flags = [_bool_float(row.get("water_bridge_connected_flag")) for row in rows]
    candidates: Dict[int, List[Tuple[str, str, float]]] = {}
    for index in range(1, len(flags)):
        if not math.isfinite(flags[index]) or not math.isfinite(flags[index - 1]):
            continue
        previous = int(round(flags[index - 1]))
        current = int(round(flags[index]))
        event_type = ""
        if connectivity_event == "loss" and previous == 1 and current == 0:
            event_type = "connectivity_loss"
        elif connectivity_event == "gain" and previous == 0 and current == 1:
            event_type = "connectivity_gain"
        elif connectivity_event == "any" and previous != current:
            event_type = "connectivity_change"
        if event_type:
            candidates.setdefault(index, []).append((event_type, "water_bridge_connected_flag", float(current)))
    return candidates


def _event_candidates_threshold(
    rows: Sequence[Mapping[str, object]],
    column: str,
    event_type: str,
    comparator: str,
    threshold: Optional[float],
    percentile: float,
) -> Tuple[Dict[int, List[Tuple[str, str, float]]], float]:
    values = [_as_float(row.get(column)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {}, math.nan
    used_threshold = float(np.nanpercentile(np.asarray(finite, dtype=float), percentile)) if threshold is None else float(threshold)
    out: Dict[int, List[Tuple[str, str, float]]] = {}
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        matched = value <= used_threshold if comparator == "<=" else value >= used_threshold
        if matched:
            out.setdefault(index, []).append((event_type, column, value))
    return out, used_threshold


def _combine_candidates(*candidate_maps: Dict[int, List[Tuple[str, str, float]]]) -> Dict[int, List[Tuple[str, str, float]]]:
    out: Dict[int, List[Tuple[str, str, float]]] = {}
    for candidate_map in candidate_maps:
        for index, records in candidate_map.items():
            out.setdefault(index, []).extend(records)
    return out


def detect_transition_events(
    feature_rows: Sequence[Mapping[str, object]],
    config: TransitionEventConfig = TransitionEventConfig(),
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Detect bridge transition events from a feature table."""

    config.validate()
    threshold_info: Dict[str, object] = {
        "nw_drop_threshold_used": math.nan,
        "dewet_rate_threshold_used": math.nan,
    }
    connectivity_map: Dict[int, List[Tuple[str, str, float]]] = {}
    if config.event_method in {"connectivity_loss", "hybrid"}:
        connectivity_map = _event_candidates_connectivity(feature_rows, config.connectivity_event)

    nw_map: Dict[int, List[Tuple[str, str, float]]] = {}
    if config.event_method in {"nw_drop", "hybrid"}:
        nw_map, threshold_info["nw_drop_threshold_used"] = _event_candidates_threshold(
            feature_rows,
            column="dNw_bridge_dt",
            event_type="nw_drop",
            comparator="<=",
            threshold=config.nw_drop_threshold,
            percentile=10.0,
        )

    dewet_map: Dict[int, List[Tuple[str, str, float]]] = {}
    if config.event_method in {"dewet_jump", "hybrid"}:
        dewet_map, threshold_info["dewet_rate_threshold_used"] = _event_candidates_threshold(
            feature_rows,
            column="ddewet_fraction_dt",
            event_type="dewet_jump",
            comparator=">=",
            threshold=config.dewet_rate_threshold,
            percentile=90.0,
        )

    if config.event_method == "connectivity_loss":
        candidate_map = connectivity_map
    elif config.event_method == "nw_drop":
        candidate_map = nw_map
    elif config.event_method == "dewet_jump":
        candidate_map = dewet_map
    else:
        candidate_map = _combine_candidates(connectivity_map, nw_map, dewet_map)

    candidate_indices = sorted(candidate_map)
    if config.dewet_fraction_threshold is not None:
        filtered = []
        for index in candidate_indices:
            value = _as_float(feature_rows[index].get("dewet_fraction"))
            if math.isfinite(value) and value >= float(config.dewet_fraction_threshold):
                filtered.append(index)
        candidate_indices = filtered

    selected: List[int] = []
    last_index: Optional[int] = None
    for index in candidate_indices:
        if last_index is None:
            selected.append(index)
            last_index = index
            continue
        previous_frame = _as_float(feature_rows[last_index].get("frame"))
        current_frame = _as_float(feature_rows[index].get("frame"))
        gap = current_frame - previous_frame if math.isfinite(previous_frame) and math.isfinite(current_frame) else index - last_index
        if gap >= float(config.min_event_separation):
            selected.append(index)
            last_index = index

    if not config.allow_partial_windows:
        selected = [
            index
            for index in selected
            if index - int(config.event_window_before) >= 0 and index + int(config.event_window_after) < len(feature_rows)
        ]

    events: List[Dict[str, object]] = []
    priority = ["connectivity_loss", "connectivity_gain", "connectivity_change", "nw_drop", "dewet_jump"]
    for index in selected:
        source = feature_rows[index]
        records = candidate_map.get(index, [])
        event_types = "+".join(sorted({record[0] for record in records}))
        trigger_metric = ""
        trigger_value = math.nan
        for preferred in priority:
            matched = [record for record in records if record[0] == preferred]
            if matched:
                trigger_metric = matched[0][1]
                trigger_value = matched[0][2]
                break
        event = {
            "event_id": len(events) + 1,
            "event_anchor_index": index,
            "event_frame": source.get("frame", math.nan),
            "event_time": source.get("time", math.nan),
            "event_type": event_types,
            "trigger_metric": trigger_metric,
            "trigger_value": trigger_value,
            "Nw_bridge": source.get("Nw_bridge", math.nan),
            "dewet_fraction": source.get("dewet_fraction", math.nan),
            "water_bridge_connected_flag": source.get("water_bridge_connected_flag", math.nan),
            "d3d_all": source.get("d3d_all", math.nan),
            "notes": "",
        }
        events.append(event)
    threshold_info["events_detected"] = len(events)
    return events, threshold_info


def extract_event_aligned_profiles(
    feature_rows: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    event_window_before: int,
    event_window_after: int,
    allow_partial_windows: bool = False,
) -> List[Dict[str, object]]:
    """Extract event-aligned feature rows."""

    profiles: List[Dict[str, object]] = []
    for event in events:
        anchor = int(_as_float(event.get("event_anchor_index")))
        left = anchor - int(event_window_before)
        right = anchor + int(event_window_after)
        if allow_partial_windows:
            left = max(left, 0)
            right = min(right, len(feature_rows) - 1)
        if left < 0 or right >= len(feature_rows):
            continue
        event_time = _as_float(event.get("event_time"))
        for index in range(left, right + 1):
            source = feature_rows[index]
            source_time = _as_float(source.get("time"))
            profile = {
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type", ""),
                "event_frame": event.get("event_frame", math.nan),
                "event_time": event.get("event_time", math.nan),
                "relative_index": index - anchor,
                "relative_time": source_time - event_time if math.isfinite(source_time) and math.isfinite(event_time) else math.nan,
                "frame": source.get("frame", math.nan),
                "time": source.get("time", math.nan),
            }
            for key, value in source.items():
                if key in {"frame", "time"}:
                    continue
                profile[key] = value
            profiles.append(profile)
    return profiles


def summarize_event_aligned_profiles(profiles: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Summarize numeric event-aligned features by relative index."""

    if not profiles:
        return []
    numeric_features = _numeric_columns(profiles, exclude=set(EVENT_ALIGNED_BASE_COLUMNS))
    relative_indices = sorted({int(_as_float(row.get("relative_index"))) for row in profiles if math.isfinite(_as_float(row.get("relative_index")))})
    rows: List[Dict[str, object]] = []
    for relative_index in relative_indices:
        chunk = [row for row in profiles if int(_as_float(row.get("relative_index"), math.inf)) == relative_index]
        rel_times = _finite_values(row.get("relative_time") for row in chunk)
        rel_time_mean = float(np.mean(rel_times)) if rel_times else math.nan
        for feature in numeric_features:
            values = _finite_values(row.get(feature) for row in chunk)
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            rows.append(
                {
                    "relative_index": relative_index,
                    "relative_time_mean": rel_time_mean,
                    "feature": feature,
                    "count": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if arr.size > 1 else math.nan,
                    "median": float(np.median(arr)),
                    "q25": float(np.quantile(arr, 0.25)),
                    "q75": float(np.quantile(arr, 0.75)),
                }
            )
    return sorted(rows, key=lambda item: (str(item["feature"]), int(item["relative_index"])))


def _pair_values_with_lag(predictor: Sequence[float], target: Sequence[float], lag: int) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for index, target_value in enumerate(target):
        source_index = index + int(lag)
        if source_index < 0 or source_index >= len(predictor):
            continue
        predictor_value = predictor[source_index]
        if math.isfinite(predictor_value) and math.isfinite(target_value):
            xs.append(float(predictor_value))
            ys.append(float(target_value))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def compute_feature_lag_correlation(
    feature_rows: Sequence[Mapping[str, object]],
    lag_window: int,
) -> List[Dict[str, object]]:
    """Compute Pearson correlations between numeric features and target metrics at integer lags."""

    target_candidates = ["dNw_bridge_dt", "ddewet_fraction_dt", "Nw_bridge", "dewet_fraction"]
    targets = [column for column in target_candidates if any(math.isfinite(_as_float(row.get(column))) for row in feature_rows)]
    if not targets:
        return []
    excluded = {"frame", "time", "source_row_index", *targets}
    features = _numeric_columns(feature_rows, exclude=excluded)
    rows: List[Dict[str, object]] = []
    for target in targets:
        target_values = [_as_float(row.get(target)) for row in feature_rows]
        for feature in features:
            feature_values = [_as_float(row.get(feature)) for row in feature_rows]
            for lag in range(-int(lag_window), int(lag_window) + 1):
                xs, ys = _pair_values_with_lag(feature_values, target_values, lag)
                if xs.size >= 3 and float(np.std(xs)) > 0.0 and float(np.std(ys)) > 0.0:
                    corr = float(np.corrcoef(xs, ys)[0, 1])
                else:
                    corr = math.nan
                rows.append(
                    {
                        "feature": feature,
                        "target": target,
                        "lag": lag,
                        "correlation": corr,
                        "n_pairs": int(xs.size),
                        "method": "pearson",
                    }
                )
    return rows


def compute_change_point_summary(
    feature_rows: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    window: int,
) -> List[Dict[str, object]]:
    """Compute pre/post feature shifts around event anchors."""

    if not events:
        return []
    numeric_features = _numeric_columns(feature_rows, exclude={"frame", "time", "source_row_index"})
    rows: List[Dict[str, object]] = []
    for event in events:
        anchor = int(_as_float(event.get("event_anchor_index")))
        pre_rows = feature_rows[max(0, anchor - int(window)) : anchor]
        post_rows = feature_rows[anchor + 1 : min(len(feature_rows), anchor + int(window) + 1)]
        if not pre_rows or not post_rows:
            continue
        for feature in numeric_features:
            pre = np.asarray(_finite_values(row.get(feature) for row in pre_rows), dtype=float)
            post = np.asarray(_finite_values(row.get(feature) for row in post_rows), dtype=float)
            if pre.size == 0 or post.size == 0:
                continue
            pre_mean = float(np.mean(pre))
            post_mean = float(np.mean(post))
            difference = post_mean - pre_mean
            pooled = math.nan
            if pre.size > 1 and post.size > 1:
                dof = (pre.size - 1) + (post.size - 1)
                pooled_var = ((pre.size - 1) * np.var(pre, ddof=1) + (post.size - 1) * np.var(post, ddof=1)) / dof
                if pooled_var > 0:
                    pooled = difference / math.sqrt(float(pooled_var))
            rows.append(
                {
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type", ""),
                    "feature": feature,
                    "pre_count": int(pre.size),
                    "post_count": int(post.size),
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "difference": difference,
                    "relative_difference": difference / abs(pre_mean) if abs(pre_mean) > 1e-14 else math.nan,
                    "effect_size_cohen_d": pooled,
                }
            )
    return rows


def analyze_transition_events(
    input_csv: Path,
    output_dir: Path,
    config: TransitionEventConfig = TransitionEventConfig(),
    frame_column: str = "frame",
    time_column: str = "time_ns",
    derivative_columns: Sequence[str] = DEFAULT_DERIVATIVE_COLUMNS,
) -> Dict[str, Path]:
    """Run event detection and write CSV outputs."""

    rows = load_feature_table(input_csv, frame_column=frame_column, time_column=time_column)
    feature_rows, derivative_basis, created_derivatives = add_derivative_features(rows, derivative_columns)
    events, threshold_info = detect_transition_events(feature_rows, config=config)
    profiles = extract_event_aligned_profiles(
        feature_rows,
        events,
        event_window_before=config.event_window_before,
        event_window_after=config.event_window_after,
        allow_partial_windows=config.allow_partial_windows,
    )
    aligned_summary = summarize_event_aligned_profiles(profiles)
    lag = compute_feature_lag_correlation(feature_rows, lag_window=config.lag_window)
    change = compute_change_point_summary(feature_rows, events, window=config.change_window)
    output_dir = Path(output_dir)
    outputs = {
        "feature_table": output_dir / "transition_feature_table.csv",
        "events": output_dir / "transition_events.csv",
        "event_aligned_profiles": output_dir / "event_aligned_profiles.csv",
        "event_aligned_summary": output_dir / "event_aligned_summary.csv",
        "feature_lag_correlation": output_dir / "feature_lag_correlation.csv",
        "feature_change_point_summary": output_dir / "feature_change_point_summary.csv",
        "state_statistics": output_dir / "state_statistics.csv",
    }
    _write_csv_rows(outputs["feature_table"], feature_rows)
    _write_csv_rows(outputs["events"], events, fieldnames=EVENT_OUTPUT_COLUMNS)
    _write_csv_rows(outputs["event_aligned_profiles"], profiles)
    _write_csv_rows(outputs["event_aligned_summary"], aligned_summary)
    _write_csv_rows(
        outputs["feature_lag_correlation"],
        lag,
        fieldnames=["feature", "target", "lag", "correlation", "n_pairs", "method"] if not lag else None,
    )
    _write_csv_rows(outputs["feature_change_point_summary"], change)
    stats = [
        {"metric": "input_csv", "value": str(input_csv)},
        {"metric": "total_feature_rows", "value": len(feature_rows)},
        {"metric": "derivative_basis", "value": derivative_basis},
        {"metric": "created_derivative_count", "value": len(created_derivatives)},
        {"metric": "created_derivatives", "value": ",".join(created_derivatives)},
        {"metric": "events_detected", "value": len(events)},
    ]
    stats.extend({"metric": key, "value": value} for key, value in threshold_info.items())
    _write_csv_rows(outputs["state_statistics"], stats, fieldnames=["metric", "value"])
    return outputs


def _parse_derivative_columns(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return DEFAULT_DERIVATIVE_COLUMNS
    out: List[str] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                out.append(item)
    return tuple(dict.fromkeys(out))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect transition events from a feature CSV")
    parser.add_argument("--input", type=Path, required=True, help="Input feature CSV, for example bridge_water_dewetting.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-column", default="frame")
    parser.add_argument("--time-column", default="time_ns")
    parser.add_argument("--derivative-column", action="append", help="Column to differentiate; may be repeated or comma separated")
    parser.add_argument("--event-method", choices=["connectivity_loss", "nw_drop", "dewet_jump", "hybrid"], default="hybrid")
    parser.add_argument("--connectivity-event", choices=["loss", "gain", "any"], default="loss")
    parser.add_argument("--nw-drop-threshold", type=float)
    parser.add_argument("--dewet-rate-threshold", type=float)
    parser.add_argument("--dewet-fraction-threshold", type=float)
    parser.add_argument("--min-event-separation", type=int, default=3)
    parser.add_argument("--event-window-before", type=int, default=20)
    parser.add_argument("--event-window-after", type=int, default=20)
    parser.add_argument("--allow-partial-windows", action="store_true")
    parser.add_argument("--lag-window", type=int, default=20)
    parser.add_argument("--change-window", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outputs = analyze_transition_events(
            input_csv=args.input,
            output_dir=args.output_dir,
            frame_column=args.frame_column,
            time_column=args.time_column,
            derivative_columns=_parse_derivative_columns(args.derivative_column),
            config=TransitionEventConfig(
                event_method=args.event_method,
                connectivity_event=args.connectivity_event,
                nw_drop_threshold=args.nw_drop_threshold,
                dewet_rate_threshold=args.dewet_rate_threshold,
                dewet_fraction_threshold=args.dewet_fraction_threshold,
                min_event_separation=args.min_event_separation,
                event_window_before=args.event_window_before,
                event_window_after=args.event_window_after,
                allow_partial_windows=args.allow_partial_windows,
                lag_window=args.lag_window,
                change_window=args.change_window,
            ),
        )
    except Exception as exc:
        print(f"Transition event analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
