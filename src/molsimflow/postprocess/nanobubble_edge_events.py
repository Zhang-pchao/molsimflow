"""Detect persistent nanobubble footprint-edge events against shared f0 baselines.

The detector uses only bubble geometry and pre-registered f0 fluctuation
thresholds.  Water and surface descriptors are joined after event detection,
so they cannot define the events they are later asked to explain.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.constant_force_oxygen import (
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)

EDGE_FIELDS = ("front_edge_unwrapped_A", "rear_edge_unwrapped_A")
REQUIRED_FIELDS = {
    "case_id", "branch_id", "direction", "step", "time_ps",
    "front_edge_unwrapped_A", "rear_edge_unwrapped_A",
    "bubble_drive_unwrapped_A", "bubble_height_A", "bubble_drive_width_A",
    "bubble_transverse_width_A", "bubble_drive_span_fraction",
    "bubble_transverse_span_fraction", "pbc_spanning_drive",
    "pbc_spanning_transverse", "film_water_oxygen_count",
    "front_water_oxygen_count", "front_oxygen_density_A3",
    "front_water_drive_velocity_mps", "front_flow_corrected_temperature_K",
    "rear_water_oxygen_count", "rear_oxygen_density_A3",
    "rear_water_drive_velocity_mps", "rear_flow_corrected_temperature_K",
}

CALIBRATION_FIELDS = (
    "case_id", "direction", "edge", "baseline_branch_id", "sample_interval_ps",
    "null_anchor_count", "f0_quantile", "f0_quantile_displacement_A",
    "minimum_displacement_A", "primary_threshold_A",
)
EVENT_FIELDS = (
    "event_id", "case_id", "branch_id", "direction", "edge", "process",
    "start_step", "anchor_step", "end_step", "start_time_ps",
    "anchor_time_ps", "end_time_ps", "periodic_image", "pre_position_A",
    "post_position_A", "displacement_A", "threshold_A", "score",
    "persistence_fraction", "validity_status", "ambiguity_reason",
    "pbc_spanning_drive", "pbc_spanning_transverse",
    "local_water_oxygen_count", "local_oxygen_density_A3",
    "local_water_drive_velocity_mps", "local_flow_corrected_temperature_K",
    "film_water_oxygen_count", "bubble_center_A", "bubble_height_A",
    "bubble_drive_width_A", "bubble_transverse_width_A",
    "footprint_width_change_A", "curvature_status",
)
MODE_FIELDS = (
    "case_id", "branch_id", "direction", "block_index", "start_time_ps",
    "end_time_ps", "frames", "front_displacement_A", "rear_displacement_A",
    "center_mode_displacement_A", "width_mode_displacement_A",
    "pbc_spanning_drive_fraction", "pbc_spanning_transverse_fraction",
    "valid_event_count", "ambiguous_event_count",
)
SENSITIVITY_FIELDS = (
    "case_id", "direction", "edge", "f0_quantile", "threshold_A",
    "candidate_event_count", "valid_event_count", "ambiguous_event_count",
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_FIELDS.difference(fields)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        rows = list(reader)
    if len(rows) < 5:
        raise ValueError(f"Too few frames in {path}")
    return rows


def _validate_series(rows: Sequence[Mapping[str, str]], *, label: str) -> float:
    steps = np.asarray([int(row["step"]) for row in rows], dtype=int)
    times = np.asarray([float(row["time_ps"]) for row in rows], dtype=float)
    if np.any(np.diff(steps) <= 0) or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{label}: frames are not strictly increasing")
    intervals = np.diff(times)
    cadence = float(np.median(intervals))
    if not np.allclose(intervals, cadence, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"{label}: irregular frame cadence")
    case_ids = {row["case_id"] for row in rows}
    directions = {row["direction"].lower() for row in rows}
    if len(case_ids) != 1 or len(directions) != 1:
        raise ValueError(f"{label}: mixed case or direction")
    return cadence


def rolling_median(values: Sequence[float], window_frames: int) -> np.ndarray:
    """Centered rolling median with edge windows truncated, not padded."""

    array = np.asarray(values, dtype=float)
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    if window_frames % 2 == 0:
        window_frames += 1
    half = window_frames // 2
    return np.asarray([
        np.median(array[max(0, index - half): min(len(array), index + half + 1)])
        for index in range(len(array))
    ])


def anchor_displacements(
    values: np.ndarray, pre_frames: int, post_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices, displacements, pre_levels, post_levels = [], [], [], []
    for index in range(pre_frames, len(values) - post_frames):
        pre = float(np.median(values[index - pre_frames:index]))
        post = float(np.median(values[index + 1:index + post_frames + 1]))
        indices.append(index); pre_levels.append(pre); post_levels.append(post)
        displacements.append(post - pre)
    return (
        np.asarray(indices, dtype=int), np.asarray(displacements, dtype=float),
        np.asarray(pre_levels, dtype=float), np.asarray(post_levels, dtype=float),
    )


def _event_process(edge: str, displacement: float) -> str:
    gas_advances = displacement > 0.0 if edge == "front" else displacement < 0.0
    return "gas_advance_liquid_retreat" if gas_advances else "gas_retreat_liquid_advance"


def detect_events(
    rows: Sequence[Mapping[str, str]],
    edge: str,
    threshold_A: float,
    *,
    cadence_ps: float,
    smoothing_ps: float,
    pre_window_ps: float,
    post_window_ps: float,
    persistence_ps: float,
    persistence_fraction: float,
    refractory_ps: float,
    drive_box_length_A: float,
) -> list[dict[str, object]]:
    field = f"{edge}_edge_unwrapped_A"
    smoothed = rolling_median(
        [float(row[field]) for row in rows],
        max(1, round(smoothing_ps / cadence_ps)),
    )
    pre_frames = max(1, round(pre_window_ps / cadence_ps))
    post_frames = max(1, round(post_window_ps / cadence_ps))
    persist_frames = max(1, round(persistence_ps / cadence_ps))
    refractory_frames = max(1, round(refractory_ps / cadence_ps))
    indices, changes, pre_levels, post_levels = anchor_displacements(
        smoothed, pre_frames, post_frames
    )
    candidates: list[tuple[int, float, float, float, float]] = []
    for index, change, pre, post in zip(indices, changes, pre_levels, post_levels):
        if abs(change) < threshold_A or index + persist_frames >= len(smoothed):
            continue
        sign = 1.0 if change > 0.0 else -1.0
        midpoint = pre + 0.5 * change
        persistence = float(np.mean(
            sign * (smoothed[index + 1:index + persist_frames + 1] - midpoint) >= 0.0
        ))
        if persistence >= persistence_fraction:
            candidates.append((int(index), float(change), float(pre), float(post), persistence))
    selected: list[tuple[int, float, float, float, float]] = []
    for candidate in sorted(candidates, key=lambda item: abs(item[1]), reverse=True):
        if all(abs(candidate[0] - item[0]) > refractory_frames for item in selected):
            selected.append(candidate)
    output: list[dict[str, object]] = []
    for index, change, pre, post, persistence in sorted(selected):
        start_index = index - pre_frames
        end_index = index + post_frames
        anchor = rows[index]
        window = rows[start_index:end_index + 1]
        drive_pbc = any(int(row["pbc_spanning_drive"]) for row in window)
        transverse_pbc = any(int(row["pbc_spanning_transverse"]) for row in window)
        prefix = edge
        water_count = int(anchor[f"{prefix}_water_oxygen_count"])
        density = float(anchor[f"{prefix}_oxygen_density_A3"])
        water_velocity = float(anchor[f"{prefix}_water_drive_velocity_mps"])
        temperature = float(anchor[f"{prefix}_flow_corrected_temperature_K"])
        reasons = []
        if drive_pbc:
            reasons.append("drive_spanning")
        if transverse_pbc:
            reasons.append("transverse_spanning")
        if water_count < 5 or not all(map(math.isfinite, (density, water_velocity, temperature))):
            reasons.append("local_water_underpopulated")
        width_change = (
            float(rows[end_index]["bubble_drive_width_A"])
            - float(rows[start_index]["bubble_drive_width_A"])
        )
        output.append({
            "case_id": anchor["case_id"], "branch_id": anchor["branch_id"],
            "direction": anchor["direction"], "edge": edge,
            "process": _event_process(edge, change),
            "start_step": rows[start_index]["step"], "anchor_step": anchor["step"],
            "end_step": rows[end_index]["step"],
            "start_time_ps": rows[start_index]["time_ps"],
            "anchor_time_ps": anchor["time_ps"], "end_time_ps": rows[end_index]["time_ps"],
            "periodic_image": math.floor(float(anchor[field]) / drive_box_length_A),
            "pre_position_A": pre, "post_position_A": post,
            "displacement_A": change, "threshold_A": threshold_A,
            "score": abs(change) / threshold_A, "persistence_fraction": persistence,
            "validity_status": "VALID" if not reasons else "AMBIGUOUS",
            "ambiguity_reason": ";".join(reasons),
            "pbc_spanning_drive": int(drive_pbc),
            "pbc_spanning_transverse": int(transverse_pbc),
            "local_water_oxygen_count": water_count,
            "local_oxygen_density_A3": density,
            "local_water_drive_velocity_mps": water_velocity,
            "local_flow_corrected_temperature_K": temperature,
            "film_water_oxygen_count": anchor["film_water_oxygen_count"],
            "bubble_center_A": anchor["bubble_drive_unwrapped_A"],
            "bubble_height_A": anchor["bubble_height_A"],
            "bubble_drive_width_A": anchor["bubble_drive_width_A"],
            "bubble_transverse_width_A": anchor["bubble_transverse_width_A"],
            "footprint_width_change_A": width_change,
            "curvature_status": "NOT_AVAILABLE_FROM_GLOBAL_EDGE_PROXY",
        })
    return output


def _mode_rows(
    rows: Sequence[Mapping[str, str]],
    events: Sequence[Mapping[str, object]],
    *,
    smoothing_ps: float,
    cadence_ps: float,
    block_ps: float,
) -> list[dict[str, object]]:
    front = rolling_median(
        [float(row["front_edge_unwrapped_A"]) for row in rows],
        max(1, round(smoothing_ps / cadence_ps)),
    )
    rear = rolling_median(
        [float(row["rear_edge_unwrapped_A"]) for row in rows],
        max(1, round(smoothing_ps / cadence_ps)),
    )
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[math.floor(float(row["time_ps"]) / block_ps)].append(index)
    output = []
    for block, indices in sorted(grouped.items()):
        first, last = indices[0], indices[-1]
        start, end = block * block_ps, (block + 1) * block_ps
        selected_events = [
            event for event in events
            if start <= float(event["anchor_time_ps"]) < end
        ]
        front_delta = float(front[last] - front[first])
        rear_delta = float(rear[last] - rear[first])
        output.append({
            "case_id": rows[0]["case_id"], "branch_id": rows[0]["branch_id"],
            "direction": rows[0]["direction"], "block_index": block,
            "start_time_ps": start, "end_time_ps": end, "frames": len(indices),
            "front_displacement_A": front_delta, "rear_displacement_A": rear_delta,
            "center_mode_displacement_A": 0.5 * (front_delta + rear_delta),
            "width_mode_displacement_A": 0.5 * (front_delta - rear_delta),
            "pbc_spanning_drive_fraction": float(np.mean([
                int(rows[index]["pbc_spanning_drive"]) for index in indices
            ])),
            "pbc_spanning_transverse_fraction": float(np.mean([
                int(rows[index]["pbc_spanning_transverse"]) for index in indices
            ])),
            "valid_event_count": sum(
                event["validity_status"] == "VALID" for event in selected_events
            ),
            "ambiguous_event_count": sum(
                event["validity_status"] != "VALID" for event in selected_events
            ),
        })
    return output


def _plot(
    series: Sequence[tuple[str, list[dict[str, str]], list[dict[str, object]]]],
    output: Path,
    smoothing_ps: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    preferred_colors = {
        "oh_only": "#4C78A8", "ch3_only": "#F58518",
        "mixed291": "#54A24B", "mixed275": "#E45756",
    }
    fallback_colors = plt.get_cmap("tab10").colors
    case_ids = list(dict.fromkeys(
        rows[0]["case_id"] for _, rows, _ in series if rows
    ))
    colors = {
        case_id: preferred_colors.get(
            case_id, fallback_colors[index % len(fallback_colors)]
        )
        for index, case_id in enumerate(case_ids)
    }
    figure, axes = plt.subplots(4, 2, figsize=(12.0, 10.0), sharex=False)
    for axis, (label, rows, events) in zip(axes.flat, series):
        cadence = _validate_series(rows, label=label)
        window = max(1, round(smoothing_ps / cadence))
        time = np.asarray([float(row["time_ps"]) for row in rows]) / 1000.0
        for edge, style in (("front", "-"), ("rear", "--")):
            values = rolling_median(
                [float(row[f"{edge}_edge_unwrapped_A"]) for row in rows], window
            )
            axis.plot(time, values, style, lw=1.0, color=colors[rows[0]["case_id"]], label=edge)
        for event in events:
            marker = "o" if event["validity_status"] == "VALID" else "x"
            y = event["post_position_A"]
            axis.scatter(float(event["anchor_time_ps"]) / 1000.0, y, marker=marker, s=24, color="black")
        axis.set_title(label.replace("_", " "))
        axis.set_xlabel("Time (ns)"); axis.set_ylabel("Edge coordinate (Å)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7)
    figure.tight_layout(); figure.savefig(output / "edge_event_overview.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    required_positive = (
        "smoothing_ps", "pre_window_ps", "post_window_ps", "persistence_ps",
        "persistence_fraction", "refractory_ps", "minimum_displacement_A", "block_ps",
    )
    if any(float(raw.get(key, 0.0)) <= 0.0 for key in required_positive):
        raise ValueError("event times, persistence, displacement, and block size must be positive")
    if not 0.0 < float(raw["f0_quantile"]) < 1.0:
        raise ValueError("f0_quantile must lie between zero and one")
    if not 0.0 < float(raw["persistence_fraction"]) <= 1.0:
        raise ValueError("persistence_fraction must lie in (0, 1]")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    base = contract_path.parent
    output.mkdir(parents=True)
    calibrations, all_events, all_modes, sensitivities = [], [], [], []
    plot_series = []
    input_rows = [{
        "role": "contract", "path": str(contract_path),
        "size_bytes": contract_path.stat().st_size, "sha256": sha256(contract_path),
    }]
    event_index = 1
    for item in cases:
        driven_path = resolve_path(item["driven_frames"], base)
        baseline_path = resolve_path(item["baseline_frames"], base)
        for role, path in (("driven_frames", driven_path), ("baseline_frames", baseline_path)):
            input_rows.append({
                "role": role, "path": str(path), "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        driven, baseline = _read_tsv(driven_path), _read_tsv(baseline_path)
        driven_cadence = _validate_series(driven, label=str(driven_path))
        baseline_cadence = _validate_series(baseline, label=str(baseline_path))
        if not math.isclose(driven_cadence, baseline_cadence, abs_tol=1.0e-8):
            raise ValueError("Driven and f0 cadence differ")
        if driven[0]["case_id"] != baseline[0]["case_id"]:
            raise ValueError("Driven and f0 case differ")
        if driven[0]["direction"].lower() != baseline[0]["direction"].lower():
            raise ValueError("Driven and f0 direction differ")
        for edge in ("front", "rear"):
            baseline_smoothed = rolling_median(
                [float(row[f"{edge}_edge_unwrapped_A"]) for row in baseline],
                max(1, round(float(raw["smoothing_ps"]) / baseline_cadence)),
            )
            _, null_changes, _, _ = anchor_displacements(
                baseline_smoothed,
                max(1, round(float(raw["pre_window_ps"]) / baseline_cadence)),
                max(1, round(float(raw["post_window_ps"]) / baseline_cadence)),
            )
            quantile_value = float(np.quantile(np.abs(null_changes), float(raw["f0_quantile"])))
            threshold = max(float(raw["minimum_displacement_A"]), quantile_value)
            calibrations.append({
                "case_id": driven[0]["case_id"], "direction": driven[0]["direction"],
                "edge": edge, "baseline_branch_id": baseline[0]["branch_id"],
                "sample_interval_ps": baseline_cadence, "null_anchor_count": len(null_changes),
                "f0_quantile": raw["f0_quantile"],
                "f0_quantile_displacement_A": quantile_value,
                "minimum_displacement_A": raw["minimum_displacement_A"],
                "primary_threshold_A": threshold,
            })
            events = detect_events(
                driven, edge, threshold,
                cadence_ps=driven_cadence,
                smoothing_ps=float(raw["smoothing_ps"]),
                pre_window_ps=float(raw["pre_window_ps"]),
                post_window_ps=float(raw["post_window_ps"]),
                persistence_ps=float(raw["persistence_ps"]),
                persistence_fraction=float(raw["persistence_fraction"]),
                refractory_ps=float(raw["refractory_ps"]),
                drive_box_length_A=float(item["drive_box_length_A"]),
            )
            for event in events:
                event["event_id"] = f"E{event_index:05d}"
                event_index += 1
            all_events.extend(events)
            for quantile in raw.get("sensitivity_quantiles", [0.99, 0.995, 0.999]):
                q_threshold = max(
                    float(raw["minimum_displacement_A"]),
                    float(np.quantile(np.abs(null_changes), float(quantile))),
                )
                selected = detect_events(
                    driven, edge, q_threshold,
                    cadence_ps=driven_cadence,
                    smoothing_ps=float(raw["smoothing_ps"]),
                    pre_window_ps=float(raw["pre_window_ps"]),
                    post_window_ps=float(raw["post_window_ps"]),
                    persistence_ps=float(raw["persistence_ps"]),
                    persistence_fraction=float(raw["persistence_fraction"]),
                    refractory_ps=float(raw["refractory_ps"]),
                    drive_box_length_A=float(item["drive_box_length_A"]),
                )
                sensitivities.append({
                    "case_id": driven[0]["case_id"], "direction": driven[0]["direction"],
                    "edge": edge, "f0_quantile": quantile, "threshold_A": q_threshold,
                    "candidate_event_count": len(selected),
                    "valid_event_count": sum(row["validity_status"] == "VALID" for row in selected),
                    "ambiguous_event_count": sum(row["validity_status"] != "VALID" for row in selected),
                })
        branch_events = [
            event for event in all_events
            if event["case_id"] == driven[0]["case_id"]
            and event["direction"] == driven[0]["direction"]
            and event["branch_id"] == driven[0]["branch_id"]
        ]
        all_modes.extend(_mode_rows(
            driven, branch_events, smoothing_ps=float(raw["smoothing_ps"]),
            cadence_ps=driven_cadence, block_ps=float(raw["block_ps"]),
        ))
        plot_series.append((
            f"{driven[0]['case_id']}-{driven[0]['direction'].upper()}", driven, branch_events
        ))
    write_tsv(output / "calibration.tsv", calibrations, CALIBRATION_FIELDS)
    write_tsv(output / "edge_events.tsv", all_events, EVENT_FIELDS)
    write_tsv(output / "response_modes.tsv", all_modes, MODE_FIELDS)
    write_tsv(output / "event_sensitivity.tsv", sensitivities, SENSITIVITY_FIELDS)
    write_tsv(output / "input_manifest.tsv", input_rows, ("role", "path", "size_bytes", "sha256"))
    if bool(raw.get("write_plots", True)):
        _plot(plot_series, output, float(raw["smoothing_ps"]))
    valid = sum(event["validity_status"] == "VALID" for event in all_events)
    ambiguous = len(all_events) - valid
    summary = {
        "status": "PASS", "branches": len(cases), "events": len(all_events),
        "valid_events": valid, "ambiguous_events": ambiguous,
        "event_definition": "geometry-only persistent front/rear edge displacement",
        "threshold_source": "case/axis/edge-specific shared-f0 displacement null",
        "claim_boundary": (
            "Events are global molecular footprint-edge proxies from one trajectory per branch; "
            "they are not force thresholds, rates, causality, or replicate uncertainty."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "EVENT-DEFINITION-AND-SENSITIVITY.md").write_text(
        "# Persistent footprint-edge events\n\n"
        "Events are defined only from smoothed front/rear footprint-edge coordinates. "
        "Each primary threshold is the larger of the frozen molecular-resolution floor "
        "and the specified quantile of the matching shared-f0 displacement distribution. "
        "A candidate must persist beyond the pre/post midpoint and is deduplicated by a "
        "frozen refractory interval. Water fields are joined only after detection.\n\n"
        "PBC-spanning and underpopulated-water windows remain in the table as AMBIGUOUS. "
        "The detector yields observer events, not depinning forces or independent rates.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary
