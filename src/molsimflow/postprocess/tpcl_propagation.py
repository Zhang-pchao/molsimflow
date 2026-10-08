"""Quantify circular contact-line modes, event size, and event-pair propagation.

The module consumes explicit tabular products instead of project-specific
trajectories.  A row is one ``(step, arc_index)`` sample; admitted events are
selected by an explicit quality label.  Outputs are descriptive single-
trajectory diagnostics and never physical rate constants or causal tests.
"""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REQUIRED_ARC_COLUMNS = (
    "step",
    "time_ns",
    "arc_index",
    "local_radius_A",
    "local_residual_A",
    "localization_noise_A",
)
REQUIRED_EVENT_COLUMNS = (
    "event_id",
    "arc_index",
    "transition_step",
    "transition_time_ns",
    "end_step",
    "post_end_step",
    "quality_status",
)
DEFAULT_EVENT_STATUS = "operational_local_dwell_jump_candidate"


@dataclass(frozen=True)
class PropagationConfig:
    max_mode: int = 6
    cluster_window_ps: float = 0.5
    cluster_arc_distance: int = 1
    null_samples: int = 200
    random_seed: int = 20260904
    time_bin_edges_ps: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    event_status: str = DEFAULT_EVENT_STATUS


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader], tuple(reader.fieldnames)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected finite number, got {value!r}")
    return number


def _as_int(value: object) -> int:
    return round(_as_float(value))


def _require_columns(path: Path, fields: Iterable[str], required: Sequence[str]) -> None:
    missing = [name for name in required if name not in set(fields)]
    if missing:
        raise ValueError("{} missing required columns: {}".format(path, ", ".join(missing)))


def load_arc_field(path: Path) -> tuple[list[int], np.ndarray, dict[int, dict[str, np.ndarray]]]:
    """Load and validate a complete circular arc field keyed by step."""

    rows, fields = _read_csv(path)
    _require_columns(path, fields, REQUIRED_ARC_COLUMNS)
    grouped: dict[int, list[dict[str, str]]] = {}
    seen = set()
    for row in rows:
        step = _as_int(row["step"])
        arc = _as_int(row["arc_index"])
        key = (step, arc)
        if key in seen:
            raise ValueError(f"Duplicate arc sample {key} in {path}")
        seen.add(key)
        grouped.setdefault(step, []).append(row)
    if len(grouped) < 2:
        raise ValueError(f"Need at least two frames in {path}")

    steps = sorted(grouped)
    first_arcs = sorted(_as_int(row["arc_index"]) for row in grouped[steps[0]])
    if first_arcs != list(range(len(first_arcs))) or len(first_arcs) < 4:
        raise ValueError("arc_index must be contiguous from zero with at least four arcs")
    arc_indices = np.asarray(first_arcs, dtype=int)
    by_step: dict[int, dict[str, np.ndarray]] = {}
    last_time = -math.inf
    for step in steps:
        ordered = sorted(grouped[step], key=lambda row: _as_int(row["arc_index"]))
        arcs = [_as_int(row["arc_index"]) for row in ordered]
        if arcs != first_arcs:
            raise ValueError(f"Incomplete or inconsistent arcs at step {step}")
        times = np.asarray([_as_float(row["time_ns"]) for row in ordered])
        if not np.allclose(times, times[0], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Multiple times within step {step}")
        if times[0] <= last_time:
            raise ValueError(f"Non-increasing time at step {step}")
        last_time = float(times[0])
        by_step[step] = {
            "time_ns": np.asarray([times[0]]),
            "radius": np.asarray([_as_float(row["local_radius_A"]) for row in ordered]),
            "residual": np.asarray([_as_float(row["local_residual_A"]) for row in ordered]),
            "noise": np.asarray([_as_float(row["localization_noise_A"]) for row in ordered]),
        }
    return steps, arc_indices, by_step


def compute_frame_modes(
    steps: Sequence[int], arc_indices: np.ndarray, by_step: Mapping[int, Mapping[str, np.ndarray]], max_mode: int
) -> list[dict[str, object]]:
    """Return discrete Fourier amplitudes for every complete frame."""

    n_arcs = len(arc_indices)
    allowed_max = max(1, (n_arcs - 1) // 2)
    if max_mode < 1 or max_mode > allowed_max:
        raise ValueError(f"max_mode must be between 1 and {allowed_max} for {n_arcs} arcs")
    theta = 2.0 * math.pi * arc_indices / n_arcs
    output: list[dict[str, object]] = []
    for step in steps:
        radius = np.asarray(by_step[step]["radius"], dtype=float)
        mean_radius = float(np.mean(radius))
        reconstructed = np.full(n_arcs, mean_radius)
        item: dict[str, object] = {
            "step": step,
            "time_ns": float(by_step[step]["time_ns"][0]),
            "mean_radius_A": mean_radius,
        }
        for mode in range(1, max_mode + 1):
            cosine = float(2.0 * np.mean(radius * np.cos(mode * theta)))
            sine = float(2.0 * np.mean(radius * np.sin(mode * theta)))
            reconstructed += cosine * np.cos(mode * theta) + sine * np.sin(mode * theta)
            item[f"mode_{mode}_amplitude_A"] = math.hypot(cosine, sine)
            item[f"mode_{mode}_phase_deg"] = math.degrees(math.atan2(sine, cosine))
        item["unresolved_mode_rms_A"] = float(np.sqrt(np.mean((radius - reconstructed) ** 2)))
        output.append(item)
    return output


def load_events(path: Path, event_status: str) -> list[dict[str, object]]:
    rows, fields = _read_csv(path)
    _require_columns(path, fields, REQUIRED_EVENT_COLUMNS)
    admitted: list[dict[str, object]] = []
    for row in rows:
        if str(row["quality_status"]) != event_status:
            continue
        item: dict[str, object] = dict(row)
        for name in ("event_id", "arc_index", "transition_step", "end_step", "post_end_step"):
            item[name] = _as_int(row[name])
        item["transition_time_ns"] = _as_float(row["transition_time_ns"])
        for name in ("jump_distance_A", "localization_noise_A"):
            item[name] = _as_float(row[name]) if row.get(name, "") else math.nan
        admitted.append(item)
    admitted.sort(key=lambda row: (int(row["transition_step"]), int(row["arc_index"])))
    if len({int(row["event_id"]) for row in admitted}) != len(admitted):
        raise ValueError(f"Duplicate admitted event_id in {path}")
    return admitted


def _circular_arc_distance(left: int, right: int, n_arcs: int) -> int:
    direct = abs(left - right)
    return min(direct, n_arcs - direct)


def cluster_events(
    events: Sequence[Mapping[str, object]], n_arcs: int, window_ps: float, arc_distance: int
) -> list[list[Mapping[str, object]]]:
    """Cluster nearly simultaneous neighboring-arc event rows."""

    parent = list(range(len(events)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, event in enumerate(events):
        left_time_ps = float(event["transition_time_ns"]) * 1000.0
        for right in range(left + 1, len(events)):
            right_time_ps = float(events[right]["transition_time_ns"]) * 1000.0
            delta = right_time_ps - left_time_ps
            if delta > window_ps:
                break
            if _circular_arc_distance(
                int(event["arc_index"]), int(events[right]["arc_index"]), n_arcs
            ) <= arc_distance:
                union(left, right)
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for index, event in enumerate(events):
        grouped.setdefault(find(index), []).append(event)
    return sorted(grouped.values(), key=lambda group: min(float(row["transition_time_ns"]) for row in group))


def build_event_size_rows(
    clusters: Sequence[Sequence[Mapping[str, object]]],
    n_arcs: int,
    by_step: Mapping[int, Mapping[str, np.ndarray]],
    modes_by_step: Mapping[int, Mapping[str, object]],
    geometry_by_step: Optional[Mapping[int, Mapping[str, str]]] = None,
) -> list[dict[str, object]]:
    """Measure all-arc displacement and global response for each observed cluster."""

    rows: list[dict[str, object]] = []
    geometry_fields = (
        "contact_contour_area_A2",
        "contact_contour_perimeter_A",
        "contact_contour_equivalent_radius_A",
        "contact_contour_circularity",
        "molecular_center_cap_angle_candidate_deg",
        "shape_mode_1_amplitude_A",
        "shape_mode_2_amplitude_A",
        "shape_mode_3_amplitude_A",
    )
    for cluster_id, cluster in enumerate(clusters, start=1):
        primary = max(cluster, key=lambda row: float(row.get("jump_distance_A", 0.0)))
        pre_step, post_step = int(primary["end_step"]), int(primary["post_end_step"])
        if pre_step not in by_step or post_step not in by_step:
            continue
        pre = by_step[pre_step]
        post = by_step[post_step]
        delta_radius = np.asarray(post["radius"]) - np.asarray(pre["radius"])
        delta_residual = np.asarray(post["residual"]) - np.asarray(pre["residual"])
        threshold = float(primary.get("localization_noise_A", math.nan))
        if not math.isfinite(threshold):
            threshold = float(np.median(pre["noise"]))
        mean_radius = 0.5 * (float(np.mean(pre["radius"])) + float(np.mean(post["radius"])))
        arc_length = 2.0 * math.pi * mean_radius / n_arcs
        affected = np.abs(delta_residual) >= threshold
        item: dict[str, object] = {
            "cluster_id": cluster_id,
            "transition_time_ns": float(primary["transition_time_ns"]),
            "transition_step": int(primary["transition_step"]),
            "primary_event_id": int(primary["event_id"]),
            "primary_arc_index": int(primary["arc_index"]),
            "member_event_count": len(cluster),
            "member_arcs": ";".join(str(value) for value in sorted({int(row["arc_index"]) for row in cluster})),
            "pre_step": pre_step,
            "post_step": post_step,
            "affected_threshold_A": threshold,
            "affected_arc_count": int(np.count_nonzero(affected)),
            "affected_arc_fraction": float(np.mean(affected)),
            "affected_arc_length_A": float(np.count_nonzero(affected) * arc_length),
            "event_size_residual_A2": float(np.sum(np.abs(delta_residual)) * arc_length),
            "event_size_radius_A2": float(np.sum(np.abs(delta_radius)) * arc_length),
            "mean_radius_change_A": float(np.mean(delta_radius)),
            "primary_residual_change_A": float(delta_residual[int(primary["arc_index"])]),
        }
        for name, value in modes_by_step[post_step].items():
            if name.startswith("mode_") and name.endswith("_amplitude_A"):
                item[f"delta_{name}"] = float(value) - float(modes_by_step[pre_step][name])
        if geometry_by_step is not None and pre_step in geometry_by_step and post_step in geometry_by_step:
            for field in geometry_fields:
                before = geometry_by_step[pre_step].get(field, "")
                after = geometry_by_step[post_step].get(field, "")
                if before != "" and after != "":
                    item[f"delta_{field}"] = _as_float(after) - _as_float(before)
        rows.append(item)
    return rows


def load_geometry(path: Optional[Path]) -> Optional[dict[int, dict[str, str]]]:
    if path is None:
        return None
    rows, fields = _read_csv(path)
    _require_columns(path, fields, ("step",))
    output: dict[int, dict[str, str]] = {}
    for row in rows:
        step = _as_int(row["step"])
        if step in output:
            raise ValueError(f"Duplicate geometry step {step} in {path}")
        output[step] = row
    return output


def _arc_bin_edges(n_arcs: int) -> np.ndarray:
    upper = n_arcs // 2 + 1
    edges = sorted({0, 1, 2, 4, 8, upper})
    edges = [value for value in edges if value <= upper]
    if edges[-1] != upper:
        edges.append(upper)
    return np.asarray(edges, dtype=float)


def _event_pair_histogram(
    times_ps: np.ndarray,
    arcs: np.ndarray,
    n_arcs: int,
    arc_edges: np.ndarray,
    time_edges: np.ndarray,
) -> np.ndarray:
    delta_t = times_ps[None, :] - times_ps[:, None]
    direct = np.abs(arcs[None, :] - arcs[:, None])
    delta_arc = np.minimum(direct, n_arcs - direct)
    mask = (delta_t > 0.0) & (delta_t <= time_edges[-1])
    histogram, _, _ = np.histogram2d(delta_arc[mask], delta_t[mask], bins=(arc_edges, time_edges))
    return histogram


def build_pair_hazard_rows(
    events: Sequence[Mapping[str, object]], n_arcs: int, config: PropagationConfig
) -> list[dict[str, object]]:
    """Compare ordered event pairs with independent circular shifts per arc."""

    if len(events) < 2:
        return []
    times_ps = np.asarray([float(row["transition_time_ns"]) * 1000.0 for row in events])
    arcs = np.asarray([int(row["arc_index"]) for row in events], dtype=int)
    start, stop = float(np.min(times_ps)), float(np.max(times_ps))
    duration = stop - start
    if duration <= 0.0:
        raise ValueError("Admitted events do not span time")
    time_edges = np.asarray(config.time_bin_edges_ps, dtype=float)
    if len(time_edges) < 2 or time_edges[0] != 0.0 or np.any(np.diff(time_edges) <= 0.0):
        raise ValueError("time_bin_edges_ps must start at zero and increase")
    arc_edges = _arc_bin_edges(n_arcs)
    observed = _event_pair_histogram(times_ps, arcs, n_arcs, arc_edges, time_edges)
    rng = np.random.default_rng(config.random_seed)
    null = np.zeros((config.null_samples,) + observed.shape, dtype=float)
    for sample in range(config.null_samples):
        shifted = times_ps.copy()
        for arc in range(n_arcs):
            selected = arcs == arc
            if np.any(selected):
                offset = rng.uniform(0.0, duration)
                shifted[selected] = start + np.mod(times_ps[selected] - start + offset, duration)
        null[sample] = _event_pair_histogram(shifted, arcs, n_arcs, arc_edges, time_edges)
    null_mean = np.mean(null, axis=0)
    null_low = np.quantile(null, 0.025, axis=0)
    null_high = np.quantile(null, 0.975, axis=0)
    rows: list[dict[str, object]] = []
    for arc_bin in range(len(arc_edges) - 1):
        for time_bin in range(len(time_edges) - 1):
            expected = float(null_mean[arc_bin, time_bin])
            count = float(observed[arc_bin, time_bin])
            rows.append(
                {
                    "arc_distance_start_bins": int(arc_edges[arc_bin]),
                    "arc_distance_end_bins_exclusive": int(arc_edges[arc_bin + 1]),
                    "lag_start_ps": float(time_edges[time_bin]),
                    "lag_end_ps": float(time_edges[time_bin + 1]),
                    "observed_pairs": int(count),
                    "null_mean_pairs": expected,
                    "null_q025_pairs": float(null_low[arc_bin, time_bin]),
                    "null_q975_pairs": float(null_high[arc_bin, time_bin]),
                    "observed_to_null_ratio": count / expected if expected > 0.0 else math.nan,
                    "empirical_upper_p": float(
                        (1 + np.count_nonzero(null[:, arc_bin, time_bin] >= count))
                        / (config.null_samples + 1)
                    ),
                    "informative_null_count": expected >= 5.0,
                    "same_arc_null_preserves_intervals": int(arc_edges[arc_bin]) == 0,
                }
            )
    return rows


def analyze_tpcl_propagation(
    case_id: str,
    arc_path: Path,
    event_path: Path,
    output_dir: Path,
    geometry_path: Optional[Path] = None,
    config: Optional[PropagationConfig] = None,
) -> dict[str, Path]:
    """Run one case and write auditable tables, metadata, and a concise report."""

    if config is None:
        config = PropagationConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    steps, arcs, by_step = load_arc_field(Path(arc_path))
    modes = compute_frame_modes(steps, arcs, by_step, config.max_mode)
    modes_by_step = {int(row["step"]): row for row in modes}
    events = load_events(Path(event_path), config.event_status)
    clusters = cluster_events(events, len(arcs), config.cluster_window_ps, config.cluster_arc_distance)
    geometry = load_geometry(Path(geometry_path)) if geometry_path else None
    event_sizes = build_event_size_rows(clusters, len(arcs), by_step, modes_by_step, geometry)
    hazard = build_pair_hazard_rows(events, len(arcs), config)

    mode_fields = ["step", "time_ns", "mean_radius_A"]
    for mode in range(1, config.max_mode + 1):
        mode_fields.extend([f"mode_{mode}_amplitude_A", f"mode_{mode}_phase_deg"])
    mode_fields.append("unresolved_mode_rms_A")
    event_fields = list(event_sizes[0].keys()) if event_sizes else ["cluster_id"]
    hazard_fields = list(hazard[0].keys()) if hazard else ["observed_pairs"]
    paths = {
        "frame_modes": output_dir / "frame_modes.csv",
        "event_sizes": output_dir / "event_size_metrics.csv",
        "pair_hazard": output_dir / "event_pair_hazard.csv",
        "summary": output_dir / "summary.json",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "REPORT.md",
    }
    _write_csv(paths["frame_modes"], modes, mode_fields)
    _write_csv(paths["event_sizes"], event_sizes, event_fields)
    _write_csv(paths["pair_hazard"], hazard, hazard_fields)

    informative = [
        row
        for row in hazard
        if bool(row["informative_null_count"])
        and not bool(row["same_arc_null_preserves_intervals"])
        and math.isfinite(float(row["observed_to_null_ratio"]))
    ]
    strongest = max(informative, key=lambda row: float(row["observed_to_null_ratio"]), default=None)
    above_null = sum(
        int(row["observed_pairs"]) > float(row["null_q975_pairs"])
        for row in informative
    )
    summary = {
        "status": "PASS",
        "case_id": case_id,
        "frame_count": len(steps),
        "arc_count": len(arcs),
        "arc_row_count": len(steps) * len(arcs),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first_time_ns": float(by_step[steps[0]]["time_ns"][0]),
        "last_time_ns": float(by_step[steps[-1]]["time_ns"][0]),
        "admitted_event_rows": len(events),
        "event_cluster_count": len(clusters),
        "event_size_rows": len(event_sizes),
        "informative_cross_arc_hazard_bins": len(informative),
        "cross_arc_bins_above_null_q975": above_null,
        "strongest_informative_cross_arc_bin": strongest,
        "scientific_status": "SINGLE_TRAJECTORY_OPERATIONAL_EVENT_PROPAGATION_DIAGNOSTIC_NOT_CAUSAL_OR_AVALANCHE_EVIDENCE",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "case_id": case_id,
        "inputs": {
            "arc_kinematics": {"path": str(arc_path), "sha256": _sha256(Path(arc_path))},
            "events": {"path": str(event_path), "sha256": _sha256(Path(event_path))},
            "geometry": (
                {"path": str(geometry_path), "sha256": _sha256(Path(geometry_path))}
                if geometry_path
                else None
            ),
        },
        "config": {
            "max_mode": config.max_mode,
            "cluster_window_ps": config.cluster_window_ps,
            "cluster_arc_distance": config.cluster_arc_distance,
            "null_samples": config.null_samples,
            "random_seed": config.random_seed,
            "time_bin_edges_ps": list(config.time_bin_edges_ps),
            "event_status": config.event_status,
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    strongest_text = "No informative cross-arc bin"
    if strongest is not None:
        strongest_text = (
            "ratio={:.3g}, arc bins=[{}, {}), lag=({}, {}] ps".format(
                float(strongest["observed_to_null_ratio"]),
                strongest["arc_distance_start_bins"],
                strongest["arc_distance_end_bins_exclusive"],
                strongest["lag_start_ps"],
                strongest["lag_end_ps"],
            )
        )
    paths["report"].write_text(
        "\n".join(
            [
                f"# TPCL propagation diagnostic: {case_id}",
                "",
                f"- Frames/arcs: {len(steps)} / {len(arcs)}",
                f"- Admitted event rows/clusters: {len(events)} / {len(clusters)}",
                f"- Informative cross-arc bins above the circular-shift 97.5% bound: {above_null} / {len(informative)}",
                f"- Strongest informative bin: {strongest_text}",
                "",
                "The per-arc circular-shift null preserves each arc's internal event intervals. Same-arc bins are therefore diagnostic only. Cross-arc excess is a single-trajectory conditional association, not proof of triggering, propagation, depinning, or avalanche dynamics.",
                "",
            ]
        )
    )
    return paths


def _parse_edges(raw: str) -> tuple[float, ...]:
    return tuple(float(value) for value in raw.split(",") if value.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--arc-kinematics", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-mode", type=int, default=6)
    parser.add_argument("--cluster-window-ps", type=float, default=0.5)
    parser.add_argument("--cluster-arc-distance", type=int, default=1)
    parser.add_argument("--null-samples", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260904)
    parser.add_argument("--time-bin-edges-ps", default="0,0.5,1,2,5,10,20,50")
    parser.add_argument("--event-status", default=DEFAULT_EVENT_STATUS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = PropagationConfig(
        max_mode=args.max_mode,
        cluster_window_ps=args.cluster_window_ps,
        cluster_arc_distance=args.cluster_arc_distance,
        null_samples=args.null_samples,
        random_seed=args.random_seed,
        time_bin_edges_ps=_parse_edges(args.time_bin_edges_ps),
        event_status=args.event_status,
    )
    analyze_tpcl_propagation(
        case_id=args.case_id,
        arc_path=args.arc_kinematics,
        event_path=args.events,
        geometry_path=args.geometry,
        output_dir=args.output_dir,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
