"""Generic coupling summaries between predictor and target feature columns."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from molsimflow.postprocess.events import (
    extract_event_aligned_profiles,
    load_feature_table,
    summarize_event_aligned_profiles,
)


KNOWN_NON_PREDICTOR_COLUMNS = {
    "frame",
    "time",
    "time_ns",
    "timestep",
    "d3d_all",
    "bridge_cyl_env.sum",
    "bridge_cyl_env.mean",
    "Nw_bridge",
    "rho_bridge",
    "rho_bridge_per_A3",
    "Nw_expected",
    "DeltaN_dewet",
    "dewet_fraction",
    "largest_water_cluster_size_bridge",
    "water_bridge_connected_flag",
    "bridge_film_state",
    "barrier_top_flag",
    "event_id",
    "event_type",
    "event_frame",
    "event_time",
    "event_anchor_index",
}

DEFAULT_TARGET_COLUMNS = ("Nw_bridge", "dewet_fraction", "dNw_bridge_dt", "ddewet_fraction_dt")


@dataclass(frozen=True)
class CouplingConfig:
    """Settings for predictor-target coupling summaries."""

    lag_window: int = 20
    state_target: str = "dewet_fraction"
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    event_window_before: int = 20
    event_window_after: int = 20
    allow_partial_windows: bool = True

    def validate(self) -> None:
        if self.lag_window < 0:
            raise ValueError("lag_window must be non-negative")
        if not 0.0 <= self.low_quantile <= 1.0 or not 0.0 <= self.high_quantile <= 1.0:
            raise ValueError("state quantiles must be in [0, 1]")
        if self.high_quantile < self.low_quantile:
            raise ValueError("high_quantile must be >= low_quantile")


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
        "predictor",
        "target",
        "lag",
        "correlation",
        "slope",
        "n_pairs",
        "state_target",
        "state_group",
        "event_id",
        "relative_index",
        "relative_time",
        "feature",
        "count",
        "mean",
        "median",
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


def _finite_pairs(rows: Sequence[Mapping[str, object]], x_column: str, y_column: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = _as_float(row.get(x_column))
        y = _as_float(row.get(y_column))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 3 or ys.size < 3:
        return math.nan
    if float(np.std(xs)) <= 0.0 or float(np.std(ys)) <= 0.0:
        return math.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return math.nan
    variance = float(np.var(xs))
    if variance <= 0.0:
        return math.nan
    return float(np.cov(xs, ys, ddof=0)[0, 1] / variance)


def _numeric_columns(rows: Sequence[Mapping[str, object]]) -> List[str]:
    keys = sorted({key for row in rows for key in row.keys()})
    out: List[str] = []
    for key in keys:
        values = [_as_float(row.get(key)) for row in rows]
        if any(math.isfinite(value) for value in values):
            out.append(key)
    return out


def _split_columns(raw: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not raw:
        return ()
    out: List[str] = []
    for value in raw:
        for item in re.split(r"[;,]", str(value)):
            item = item.strip()
            if item:
                out.append(item)
    return tuple(dict.fromkeys(out))


def infer_predictor_columns(
    rows: Sequence[Mapping[str, object]],
    explicit: Sequence[str] = (),
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
) -> Tuple[str, ...]:
    """Infer numeric predictor columns if explicit columns are not supplied."""

    if explicit:
        return tuple(explicit)
    excluded = set(KNOWN_NON_PREDICTOR_COLUMNS)
    excluded.update(target_columns)
    excluded.update({"source_row_index"})
    predictors = []
    for column in _numeric_columns(rows):
        if column in excluded:
            continue
        if column.startswith("d") and column.endswith("_dt") and column[1:-3] in target_columns:
            continue
        predictors.append(column)
    return tuple(predictors)


def infer_target_columns(rows: Sequence[Mapping[str, object]], explicit: Sequence[str] = ()) -> Tuple[str, ...]:
    if explicit:
        return tuple(explicit)
    return tuple(column for column in DEFAULT_TARGET_COLUMNS if any(math.isfinite(_as_float(row.get(column))) for row in rows))


def compute_pairwise_coupling(
    rows: Sequence[Mapping[str, object]],
    predictors: Sequence[str],
    targets: Sequence[str],
) -> List[Dict[str, object]]:
    """Compute zero-lag predictor-target coupling statistics."""

    out: List[Dict[str, object]] = []
    for predictor in predictors:
        for target in targets:
            xs, ys = _finite_pairs(rows, predictor, target)
            out.append(
                {
                    "predictor": predictor,
                    "target": target,
                    "n_pairs": int(xs.size),
                    "correlation": _pearson(xs, ys),
                    "slope": _slope(xs, ys),
                    "predictor_mean": float(np.mean(xs)) if xs.size else math.nan,
                    "target_mean": float(np.mean(ys)) if ys.size else math.nan,
                    "predictor_std": float(np.std(xs, ddof=1)) if xs.size > 1 else math.nan,
                    "target_std": float(np.std(ys, ddof=1)) if ys.size > 1 else math.nan,
                }
            )
    return out


def _lag_pairs(values: Sequence[float], targets: Sequence[float], lag: int) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for index, target in enumerate(targets):
        source_index = index + int(lag)
        if source_index < 0 or source_index >= len(values):
            continue
        value = values[source_index]
        if math.isfinite(value) and math.isfinite(target):
            xs.append(value)
            ys.append(target)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def compute_lag_correlation(
    rows: Sequence[Mapping[str, object]],
    predictors: Sequence[str],
    targets: Sequence[str],
    lag_window: int,
) -> List[Dict[str, object]]:
    """Compute Pearson predictor-target correlations over integer lags."""

    out: List[Dict[str, object]] = []
    for predictor in predictors:
        predictor_values = [_as_float(row.get(predictor)) for row in rows]
        for target in targets:
            target_values = [_as_float(row.get(target)) for row in rows]
            for lag in range(-int(lag_window), int(lag_window) + 1):
                xs, ys = _lag_pairs(predictor_values, target_values, lag)
                out.append(
                    {
                        "predictor": predictor,
                        "target": target,
                        "lag": lag,
                        "correlation": _pearson(xs, ys),
                        "n_pairs": int(xs.size),
                        "method": "pearson",
                    }
                )
    return out


def compute_state_comparison(
    rows: Sequence[Mapping[str, object]],
    predictors: Sequence[str],
    state_target: str,
    low_quantile: float,
    high_quantile: float,
) -> List[Dict[str, object]]:
    """Compare predictor distributions between low/high target-state subsets."""

    target_values = np.asarray([_as_float(row.get(state_target)) for row in rows], dtype=float)
    finite = target_values[np.isfinite(target_values)]
    if finite.size == 0:
        return []
    low_threshold = float(np.quantile(finite, float(low_quantile)))
    high_threshold = float(np.quantile(finite, float(high_quantile)))
    groups = {
        "low": [row for row in rows if math.isfinite(_as_float(row.get(state_target))) and _as_float(row.get(state_target)) <= low_threshold],
        "high": [row for row in rows if math.isfinite(_as_float(row.get(state_target))) and _as_float(row.get(state_target)) >= high_threshold],
    }
    out: List[Dict[str, object]] = []
    for predictor in predictors:
        group_stats: Dict[str, Dict[str, float]] = {}
        for group_name, chunk in groups.items():
            values = np.asarray([_as_float(row.get(predictor)) for row in chunk], dtype=float)
            values = values[np.isfinite(values)]
            group_stats[group_name] = {
                "n": int(values.size),
                "mean": float(np.mean(values)) if values.size else math.nan,
                "median": float(np.median(values)) if values.size else math.nan,
                "std": float(np.std(values, ddof=1)) if values.size > 1 else math.nan,
            }
            out.append(
                {
                    "predictor": predictor,
                    "state_target": state_target,
                    "state_group": group_name,
                    "threshold": low_threshold if group_name == "low" else high_threshold,
                    "n_samples": group_stats[group_name]["n"],
                    "mean": group_stats[group_name]["mean"],
                    "median": group_stats[group_name]["median"],
                    "std": group_stats[group_name]["std"],
                }
            )
        low_mean = group_stats["low"]["mean"]
        high_mean = group_stats["high"]["mean"]
        out.append(
            {
                "predictor": predictor,
                "state_target": state_target,
                "state_group": "high_minus_low",
                "threshold": math.nan,
                "n_samples": min(group_stats["low"]["n"], group_stats["high"]["n"]),
                "mean": high_mean - low_mean if math.isfinite(high_mean) and math.isfinite(low_mean) else math.nan,
                "median": math.nan,
                "std": math.nan,
            }
        )
    return out


def load_event_rows(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        return []
    rows, _fields = _read_csv_rows(path)
    return rows


def analyze_coupling(
    feature_table: Path,
    output_dir: Path,
    predictor_columns: Sequence[str] = (),
    target_columns: Sequence[str] = (),
    transition_events: Optional[Path] = None,
    config: CouplingConfig = CouplingConfig(),
    frame_column: str = "frame",
    time_column: str = "time_ns",
) -> Dict[str, Path]:
    """Write coupling, lag, state, and optional event-aligned summaries."""

    config.validate()
    rows = load_feature_table(feature_table, frame_column=frame_column, time_column=time_column)
    predictors = infer_predictor_columns(rows, explicit=predictor_columns, target_columns=target_columns)
    targets = infer_target_columns(rows, explicit=target_columns)
    if not predictors:
        raise ValueError("No predictor columns found; pass --predictor-column")
    if not targets:
        raise ValueError("No target columns found; pass --target-column")

    coupling = compute_pairwise_coupling(rows, predictors=predictors, targets=targets)
    lag = compute_lag_correlation(rows, predictors=predictors, targets=targets, lag_window=config.lag_window)
    state = compute_state_comparison(
        rows,
        predictors=predictors,
        state_target=config.state_target,
        low_quantile=config.low_quantile,
        high_quantile=config.high_quantile,
    )
    events = load_event_rows(transition_events)
    event_profiles = []
    event_summary = []
    if events:
        event_profiles = extract_event_aligned_profiles(
            rows,
            events,
            event_window_before=config.event_window_before,
            event_window_after=config.event_window_after,
            allow_partial_windows=config.allow_partial_windows,
        )
        keep_features = set(predictors) | set(targets)
        event_profiles = [
            {key: value for key, value in row.items() if key in keep_features or key in {
                "event_id",
                "event_type",
                "event_frame",
                "event_time",
                "relative_index",
                "relative_time",
                "frame",
                "time",
            }}
            for row in event_profiles
        ]
        event_summary = [
            row for row in summarize_event_aligned_profiles(event_profiles) if str(row.get("feature", "")) in keep_features
        ]

    output_dir = Path(output_dir)
    outputs = {
        "feature_table": output_dir / "coupling_feature_table.csv",
        "coupling": output_dir / "ion_water_coupling.csv",
        "lag_correlation": output_dir / "ion_water_lag_correlation.csv",
        "state_comparison": output_dir / "ion_water_state_comparison.csv",
        "event_aligned_profiles": output_dir / "event_aligned_ion_water_profiles.csv",
        "event_aligned_summary": output_dir / "event_aligned_ion_water_summary.csv",
        "state_statistics": output_dir / "state_statistics.csv",
    }
    _write_csv_rows(outputs["feature_table"], rows)
    _write_csv_rows(outputs["coupling"], coupling)
    _write_csv_rows(outputs["lag_correlation"], lag)
    _write_csv_rows(outputs["state_comparison"], state)
    _write_csv_rows(outputs["event_aligned_profiles"], event_profiles)
    _write_csv_rows(outputs["event_aligned_summary"], event_summary)
    _write_csv_rows(
        outputs["state_statistics"],
        [
            {"metric": "input_feature_table", "value": str(feature_table)},
            {"metric": "transition_events", "value": str(transition_events) if transition_events else ""},
            {"metric": "predictor_columns", "value": ",".join(predictors)},
            {"metric": "target_columns", "value": ",".join(targets)},
            {"metric": "n_feature_rows", "value": len(rows)},
            {"metric": "n_coupling_rows", "value": len(coupling)},
            {"metric": "n_event_rows", "value": len(events)},
        ],
        fieldnames=["metric", "value"],
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute predictor-target ion/water coupling summaries")
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition-events", type=Path)
    parser.add_argument("--predictor-column", action="append", help="Predictor column; may be repeated or comma separated")
    parser.add_argument("--target-column", action="append", help="Target column; may be repeated or comma separated")
    parser.add_argument("--state-target", default="dewet_fraction")
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--lag-window", type=int, default=20)
    parser.add_argument("--event-window-before", type=int, default=20)
    parser.add_argument("--event-window-after", type=int, default=20)
    parser.add_argument("--no-partial-windows", dest="allow_partial_windows", action="store_false")
    parser.add_argument("--frame-column", default="frame")
    parser.add_argument("--time-column", default="time_ns")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        outputs = analyze_coupling(
            feature_table=args.feature_table,
            output_dir=args.output_dir,
            transition_events=args.transition_events,
            predictor_columns=_split_columns(args.predictor_column),
            target_columns=_split_columns(args.target_column),
            frame_column=args.frame_column,
            time_column=args.time_column,
            config=CouplingConfig(
                lag_window=args.lag_window,
                state_target=args.state_target,
                low_quantile=args.low_quantile,
                high_quantile=args.high_quantile,
                event_window_before=args.event_window_before,
                event_window_after=args.event_window_after,
                allow_partial_windows=args.allow_partial_windows,
            ),
        )
    except Exception as exc:
        print(f"Coupling analysis failed: {exc}")
        return 1
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
