"""Build cluster-level TPCL event-state tables from accepted tabular products.

The output contains only descriptors available before each event cluster, plus
explicit future response targets.  Paths and column routing are supplied by a
manifest so the module is independent of any one simulation layout.
"""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_CLUSTER_LEVEL_PRE_EVENT_STATE_TABLE_"
    "NOT_CAUSAL_OR_REPLICATE_LEVEL_EVIDENCE"
)

GLOBAL_FIELDS = {
    "global_mean_radius_A": ("frame_modes", "mean_radius_A"),
    "global_mode_2_amplitude_A": ("frame_modes", "mode_2_amplitude_A"),
    "global_mode_3_amplitude_A": ("frame_modes", "mode_3_amplitude_A"),
    "global_mode_4_amplitude_A": ("frame_modes", "mode_4_amplitude_A"),
    "global_unresolved_mode_rms_A": ("frame_modes", "unresolved_mode_rms_A"),
    "global_footprint_area_A2": ("geometry", "contact_contour_area_A2"),
    "global_footprint_circularity": ("geometry", "contact_contour_circularity"),
    "global_cap_angle_candidate_deg": (
        "geometry",
        "molecular_center_cap_angle_candidate_deg",
    ),
    "global_pressure_trace_bar": ("global_stress", "global_pressure_trace_bar"),
    "global_normal_minus_tangential_bar": (
        "global_stress",
        "global_normal_minus_tangential_bar",
    ),
    "global_shear_norm_bar": ("global_stress", "global_shear_norm_bar"),
}

LOCAL_FIELDS = (
    "nearest_site_distance_A",
    "local_ch3_fraction",
    "chemical_boundary_distance_proxy_A",
    "local_hydration_areal_density_A-2",
    "local_water_dipole_cos_z",
    "local_water_water_hbond_degree",
    "local_surface_water_hbond_per_h2o",
    "local_n2_min_distance_A",
)

RESPONSE_FIELDS = (
    "affected_arc_fraction",
    "event_size_residual_A2",
    "event_size_radius_A2",
    "mean_radius_change_A",
    "primary_residual_change_A",
    "delta_mode_2_amplitude_A",
    "delta_mode_3_amplitude_A",
    "delta_mode_4_amplitude_A",
    "delta_mode_5_amplitude_A",
    "delta_mode_6_amplitude_A",
    "delta_contact_contour_area_A2",
    "delta_contact_contour_perimeter_A",
    "delta_contact_contour_circularity",
    "delta_molecular_center_cap_angle_candidate_deg",
)


@dataclass(frozen=True)
class EventStateSource:
    """Explicit accepted paths for one case."""

    case_id: str
    propagation_summary: Path
    event_size_metrics: Path
    frame_modes: Path
    geometry: Path
    global_stress: Path
    event_environment: Path


@dataclass(frozen=True)
class EventStateConfig:
    """Frozen time and circular-distance bins."""

    block_ps: float = 200.0
    fast_window_ps: tuple[float, float] = (0.0, 5.0)
    slow_window_ps: tuple[float, float] = (5.0, 50.0)
    arc_bins: tuple[tuple[str, int, int], ...] = (
        ("near", 1, 4),
        ("mid", 4, 10),
        ("far", 10, 19),
    )
    pre_relative_frames: tuple[int, ...] = (-4, -3, -2, -1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: object, *, allow_nan: bool = False) -> float:
    number = float(value)
    if math.isfinite(number) or (allow_nan and math.isnan(number)):
        return number
    raise ValueError(f"expected finite number, got {value!r}")


def _index_unique(rows: Sequence[Mapping[str, str]], key: str, path: Path) -> dict[int, Mapping[str, str]]:
    indexed: dict[int, Mapping[str, str]] = {}
    for row in rows:
        value = round(_finite(row[key]))
        if value in indexed:
            raise ValueError(f"duplicate {key}={value} in {path}")
        indexed[value] = row
    return indexed


def _latest_at_or_before(
    steps: Sequence[int], indexed: Mapping[int, Mapping[str, str]], target: int
) -> tuple[Mapping[str, str], int]:
    position = bisect.bisect_right(steps, target) - 1
    if position < 0:
        raise ValueError(f"no state at or before step {target}")
    step = steps[position]
    return indexed[step], target - step


def _circular_distance(left: int, right: int, n_arcs: int) -> int:
    direct = abs(left - right)
    return min(direct, n_arcs - direct)


def read_sources(path: Path) -> list[EventStateSource]:
    """Read a path-explicit tab-separated source manifest."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    fields = {item.name for item in EventStateSource.__dataclass_fields__.values()}
    if not rows or fields.difference(rows[0]):
        raise ValueError(f"{path}: missing source rows or columns")
    sources = [
        EventStateSource(
            case_id=row["case_id"],
            propagation_summary=Path(row["propagation_summary"]),
            event_size_metrics=Path(row["event_size_metrics"]),
            frame_modes=Path(row["frame_modes"]),
            geometry=Path(row["geometry"]),
            global_stress=Path(row["global_stress"]),
            event_environment=Path(row["event_environment"]),
        )
        for row in rows
    ]
    case_ids = [source.case_id for source in sources]
    if any(not item for item in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be nonempty and unique")
    return sources


def _validate_config(config: EventStateConfig, n_arcs: int) -> None:
    if config.block_ps <= 0.0:
        raise ValueError("block_ps must be positive")
    for name, window in (("fast", config.fast_window_ps), ("slow", config.slow_window_ps)):
        if window[0] < 0.0 or window[1] <= window[0]:
            raise ValueError(f"invalid {name} window")
    expected_left = 1
    maximum_distance = n_arcs // 2
    for _, left, right in config.arc_bins:
        if left != expected_left or right <= left:
            raise ValueError("arc bins must be contiguous and start at distance one")
        expected_left = right
    if expected_left - 1 != maximum_distance:
        raise ValueError("arc bins must cover all positive circular distances")
    if not config.pre_relative_frames or any(value >= 0 for value in config.pre_relative_frames):
        raise ValueError("pre_relative_frames must be nonempty and strictly negative")


def _window_count(delta_ps: np.ndarray, cross_arc: np.ndarray, window: tuple[float, float]) -> int:
    return int(np.count_nonzero(cross_arc & (delta_ps > window[0]) & (delta_ps <= window[1])))


def _history_counts(
    delta_ps: np.ndarray,
    distances: np.ndarray,
    config: EventStateConfig,
    *,
    direction: str,
) -> dict[str, int]:
    if direction == "past":
        lag = -delta_ps
    elif direction == "future":
        lag = delta_ps
    else:
        raise ValueError(f"unknown direction {direction}")
    output = {}
    for window_name, window in (
        ("fast", config.fast_window_ps),
        ("slow", config.slow_window_ps),
    ):
        selected_time = (lag > window[0]) & (lag <= window[1])
        for bin_name, left, right in config.arc_bins:
            output[f"{direction}_history_{window_name}_{bin_name}_count"] = int(
                np.count_nonzero(selected_time & (distances >= left) & (distances < right))
            )
    return output


def _local_pre_state(
    rows: Sequence[Mapping[str, str]], event_id: int, config: EventStateConfig
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["sample_kind"] == "event"
        and round(_finite(row["event_id"])) == event_id
        and round(_finite(row["relative_frame"])) in config.pre_relative_frames
    ]
    if not selected:
        raise ValueError(f"event {event_id}: missing pre-event environment rows")
    output: dict[str, object] = {"pre_local_sample_count": len(selected)}
    for field in LOCAL_FIELDS:
        values = np.asarray([_finite(row[field], allow_nan=True) for row in selected], dtype=float)
        finite = values[np.isfinite(values)]
        output[f"pre_local_{field}"] = float(np.mean(finite)) if finite.size else math.nan
        output[f"pre_local_{field}_finite_count"] = int(finite.size)
    return output


def build_case_rows(source: EventStateSource, config: EventStateConfig) -> list[dict[str, object]]:
    """Build one pre-event row per accepted event cluster."""

    summary = json.loads(source.propagation_summary.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("case_id") != source.case_id:
        raise ValueError(f"{source.case_id}: propagation summary is not a matching PASS")
    n_arcs = int(summary["arc_count"])
    start_ps = _finite(summary["first_time_ns"]) * 1000.0
    stop_ps = _finite(summary["last_time_ns"]) * 1000.0
    _validate_config(config, n_arcs)

    clusters = _read_csv(source.event_size_metrics)
    if len(clusters) != int(summary["event_cluster_count"]):
        raise ValueError(f"{source.case_id}: cluster count mismatch")
    clusters.sort(key=lambda row: (_finite(row["transition_time_ns"]), int(row["cluster_id"])))
    times_ps = np.asarray([_finite(row["transition_time_ns"]) * 1000.0 for row in clusters])
    arcs = np.asarray([round(_finite(row["primary_arc_index"])) for row in clusters], dtype=int)
    if np.any(np.diff(times_ps) < 0.0) or np.any(times_ps < start_ps) or np.any(times_ps > stop_ps):
        raise ValueError(f"{source.case_id}: invalid cluster times")
    if np.any(arcs < 0) or np.any(arcs >= n_arcs):
        raise ValueError(f"{source.case_id}: invalid primary arc")

    frame_modes = _index_unique(_read_csv(source.frame_modes), "step", source.frame_modes)
    geometry = _index_unique(_read_csv(source.geometry), "step", source.geometry)
    stress = _index_unique(_read_csv(source.global_stress), "step", source.global_stress)
    stress_steps = sorted(stress)
    environment = _read_csv(source.event_environment)

    output = []
    for index, cluster in enumerate(clusters):
        step = round(_finite(cluster["transition_step"]))
        pre_step = round(_finite(cluster["pre_step"]))
        if pre_step >= step:
            raise ValueError(f"{source.case_id}/cluster {cluster['cluster_id']}: invalid pre_step")
        if pre_step not in frame_modes or pre_step not in geometry:
            raise ValueError(f"{source.case_id}/cluster {cluster['cluster_id']}: missing pre-state")
        stress_row, stress_lag_steps = _latest_at_or_before(stress_steps, stress, pre_step)
        delta = times_ps - times_ps[index]
        distances = np.asarray(
            [_circular_distance(int(arcs[index]), int(other), n_arcs) for other in arcs],
            dtype=int,
        )
        cross_arc = distances > 0
        block_index = min(
            math.floor((times_ps[index] - start_ps) / config.block_ps),
            math.ceil((stop_ps - start_ps) / config.block_ps) - 1,
        )
        rows_by_source = {
            "frame_modes": frame_modes[pre_step],
            "geometry": geometry[pre_step],
            "global_stress": stress_row,
        }
        row: dict[str, object] = {
            "case_id": source.case_id,
            "cluster_id": int(cluster["cluster_id"]),
            "primary_event_id": round(_finite(cluster["primary_event_id"])),
            "primary_arc_index": int(arcs[index]),
            "arc_count": n_arcs,
            "transition_step": step,
            "transition_time_ns": times_ps[index] / 1000.0,
            "pre_step": pre_step,
            "stress_alignment_lag_steps": stress_lag_steps,
            "time_block_200ps": block_index,
            "future_cross_arc_fast_count": _window_count(
                delta, cross_arc, config.fast_window_ps
            ),
            "future_cross_arc_slow_count": _window_count(
                delta, cross_arc, config.slow_window_ps
            ),
            "past_cross_arc_fast_count": _window_count(
                -delta, cross_arc, config.fast_window_ps
            ),
            "past_cross_arc_slow_count": _window_count(
                -delta, cross_arc, config.slow_window_ps
            ),
        }
        row.update(_history_counts(delta, distances, config, direction="past"))
        row.update(_history_counts(delta, distances, config, direction="future"))
        for output_name, (source_name, input_name) in GLOBAL_FIELDS.items():
            row[output_name] = _finite(rows_by_source[source_name][input_name])
        row.update(_local_pre_state(environment, int(row["primary_event_id"]), config))
        for field in RESPONSE_FIELDS:
            row[f"response_{field}"] = _finite(cluster[field], allow_nan=True)
        row["scientific_status"] = SCIENTIFIC_STATUS
        output.append(row)
    return output


def run_analysis(
    sources_path: Path,
    output_dir: Path,
    *,
    config: Optional[EventStateConfig] = None,
) -> dict[str, object]:
    """Build and write the combined event-state table."""

    config = config or EventStateConfig()
    sources = read_sources(sources_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    case_counts = {}
    for source in sources:
        case_rows = build_case_rows(source, config)
        rows.extend(case_rows)
        case_counts[source.case_id] = len(case_rows)
    _write_csv(output / "event_state_table.csv", rows)
    summary = {
        "status": "PASS",
        "case_count": len(sources),
        "row_count": len(rows),
        "case_counts": case_counts,
        "block_ps": config.block_ps,
        "fast_window_ps": config.fast_window_ps,
        "slow_window_ps": config.slow_window_ps,
        "arc_bins": config.arc_bins,
        "pre_relative_frames": config.pre_relative_frames,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        **summary,
        "sources_manifest": {"path": str(sources_path), "sha256": _sha256(sources_path)},
        "inputs": [
            {
                "case_id": source.case_id,
                **{
                    field: {"path": str(getattr(source, field)), "sha256": _sha256(getattr(source, field))}
                    for field in (
                        "propagation_summary",
                        "event_size_metrics",
                        "frame_modes",
                        "geometry",
                        "global_stress",
                        "event_environment",
                    )
                },
            }
            for source in sources
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "REPORT.md").write_text(
        "# TPCL cluster-level event-state table\n\n"
        "Each row is an accepted event cluster. Predictors use only the preceding "
        "0.5--2.0 ps environment and global state at or before the recorded pre-step. "
        "Future cross-arc counts and post-event response fields are explicit targets, "
        "not predictors. The table is retrospective single-trajectory evidence and "
        "does not establish causality or replicate-level uncertainty.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(args.sources, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
