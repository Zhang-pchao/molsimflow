"""Audit raw, excess, occupancy-weighted, and power-coupled film transport."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.constant_force_events import stitch_motion_tables
from molsimflow.postprocess.constant_force_oxygen import (
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def occupancy_weighted_velocity(
    counts: Sequence[float], velocities_mps: Sequence[float]
) -> float:
    """Return molecule-sample weighted velocity while ignoring empty layers."""

    count = np.asarray(counts, dtype=float)
    velocity = np.asarray(velocities_mps, dtype=float)
    valid = (count > 0.0) & np.isfinite(velocity)
    if not np.any(valid):
        return math.nan
    return float(np.sum(count[valid] * velocity[valid]) / np.sum(count[valid]))


def layer_flux_closure(
    counts: Sequence[float], velocities_mps: Sequence[float], global_velocity_Aps: float
) -> dict[str, float]:
    """Close the sum of layer molecule velocities against the global oxygen COM."""

    count = np.asarray(counts, dtype=float)
    velocity = np.asarray(velocities_mps, dtype=float)
    valid = (count > 0.0) & np.isfinite(velocity)
    layer_sum = float(np.sum(count[valid] * velocity[valid] / 100.0))
    expected = float(np.sum(count[valid]) * global_velocity_Aps)
    return {
        "layer_velocity_sum_molecule_A_per_ps": layer_sum,
        "global_velocity_sum_molecule_A_per_ps": expected,
        "closure_residual_molecule_A_per_ps": layer_sum - expected,
        "relative_closure_residual": (
            (layer_sum - expected) / max(abs(layer_sum), abs(expected), 1.0e-12)
        ),
    }


def _block_id(time_ps: float, block_ps: float) -> int:
    if time_ps <= 0.0:
        return 0
    return int(math.floor((time_ps - 1.0e-10) / block_ps))


def _summarize_layers(
    rows: Sequence[Mapping[str, object]],
    baseline: Mapping[tuple[int, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["case_id"]),
                str(row["branch_id"]),
                str(row["direction"]),
                int(row["layer_index"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for (case_id, branch_id, direction, layer), selected in sorted(grouped.items()):
        axis = "x" if direction in {"none", "x"} else "y"
        count = np.asarray([float(row["count"]) for row in selected])
        raw = np.asarray([float(row[f"mean_v{axis}_mps"]) for row in selected])
        base_count = np.asarray(
            [float(baseline[(int(float(row["step"])), layer)]["count"]) for row in selected]
        )
        base = np.asarray(
            [float(baseline[(int(float(row["step"])), layer)][f"mean_v{axis}_mps"]) for row in selected]
        )
        raw_weighted = occupancy_weighted_velocity(count, raw)
        baseline_weighted = occupancy_weighted_velocity(base_count, base)
        output.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "layer_index": layer,
                "samples": len(selected),
                "occupied_fraction": float(np.mean(count > 0.0)),
                "mean_count": float(np.mean(count)),
                "molecule_samples": float(np.sum(count)),
                "raw_axis_velocity_mps": raw_weighted,
                "baseline_axis_velocity_mps": baseline_weighted,
                "excess_axis_velocity_mps": raw_weighted - baseline_weighted,
                "raw_velocity_negative_fraction": float(np.mean(raw[np.isfinite(raw)] < 0.0))
                if np.any(np.isfinite(raw))
                else math.nan,
                "baseline_velocity_negative_fraction": float(
                    np.mean(base[np.isfinite(base)] < 0.0)
                )
                if np.any(np.isfinite(base))
                else math.nan,
            }
        )
    return output


def _summarize_layer_blocks(
    rows: Sequence[Mapping[str, object]],
    baseline: Mapping[tuple[int, int], Mapping[str, object]],
    block_ps: float,
) -> list[dict[str, object]]:
    """Return occupancy-weighted layer velocities in fixed time blocks."""

    grouped: dict[tuple[str, str, str, int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["case_id"]),
                str(row["branch_id"]),
                str(row["direction"]),
                int(row["layer_index"]),
                _block_id(float(row["time_ps"]), block_ps),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for (case_id, branch_id, direction, layer, block), selected in sorted(grouped.items()):
        axis = "x" if direction in {"none", "x"} else "y"
        count = np.asarray([float(row["count"]) for row in selected])
        velocity = np.asarray([float(row[f"mean_v{axis}_mps"]) for row in selected])
        reference = [baseline[(int(row["step"]), layer)] for row in selected]
        baseline_count = np.asarray([float(row["count"]) for row in reference])
        baseline_velocity = np.asarray(
            [float(row[f"mean_v{axis}_mps"]) for row in reference]
        )
        raw_weighted = occupancy_weighted_velocity(count, velocity)
        baseline_weighted = occupancy_weighted_velocity(baseline_count, baseline_velocity)
        output.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "layer_index": layer,
                "block_index": block,
                "start_ps": block * block_ps,
                "end_ps": (block + 1) * block_ps,
                "samples": len(selected),
                "occupied_samples": int(np.sum(count > 0.0)),
                "mean_count": float(np.mean(count)),
                "molecule_samples": float(np.sum(count)),
                "raw_axis_velocity_mps": raw_weighted,
                "baseline_axis_velocity_mps": baseline_weighted,
                "excess_axis_velocity_mps": raw_weighted - baseline_weighted,
            }
        )
    return output


def _plot(
    summaries: Sequence[Mapping[str, object]],
    blocks: Sequence[Mapping[str, object]],
    layer_blocks: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    driven = [row for row in summaries if row["direction"] != "none" and row["mean_count"] >= 1.0]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for branch in sorted({str(row["branch_id"]) for row in driven}):
        selected = sorted(
            [row for row in driven if row["branch_id"] == branch],
            key=lambda row: int(row["layer_index"]),
        )
        layer = [int(row["layer_index"]) for row in selected]
        axes[0].plot(layer, [row["raw_axis_velocity_mps"] for row in selected], marker="o", label=branch)
        axes[1].plot(layer, [row["excess_axis_velocity_mps"] for row in selected], marker="o", label=branch)
    axes[0].set(title="Raw driven velocity", ylabel="Velocity (m/s)", xlabel="Layer index")
    axes[1].set(title="Driven minus F0", ylabel="Velocity response (m/s)", xlabel="Layer index")
    for axis in axes:
        axis.axhline(0.0, color="black", lw=0.7)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "layer_flux_audit.png", dpi=240)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    for branch in sorted({str(row["branch_id"]) for row in blocks if row["direction"] != "none"}):
        selected = [row for row in blocks if row["branch_id"] == branch]
        time = [0.001 * (float(row["start_ps"]) + float(row["end_ps"])) / 2.0 for row in selected]
        axis.plot(time, [row["total_excess_velocity_mps"] for row in selected], label=branch)
    axis.axhline(0.0, color="black", lw=0.7)
    axis.set(xlabel="Time (ns)", ylabel="Occupancy-weighted F - F0 (m/s)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "film_total_response_blocks.png", dpi=240)
    plt.close(figure)

    driven_layer_blocks = [
        row
        for row in layer_blocks
        if row["direction"] != "none" and float(row["mean_count"]) >= 1.0
    ]
    branches = sorted({str(row["branch_id"]) for row in driven_layer_blocks})
    figure, axes = plt.subplots(
        len(branches), 2, figsize=(11.0, 3.8 * len(branches)), squeeze=False, sharex=True
    )
    for branch_index, branch in enumerate(branches):
        branch_rows = [row for row in driven_layer_blocks if row["branch_id"] == branch]
        for layer in sorted({int(row["layer_index"]) for row in branch_rows}):
            selected = [row for row in branch_rows if int(row["layer_index"]) == layer]
            time = [
                0.001 * (float(row["start_ps"]) + float(row["end_ps"])) / 2.0
                for row in selected
            ]
            axes[branch_index, 0].plot(
                time,
                [row["raw_axis_velocity_mps"] for row in selected],
                label=f"layer {layer}",
            )
            axes[branch_index, 1].plot(
                time,
                [row["excess_axis_velocity_mps"] for row in selected],
                label=f"layer {layer}",
            )
        axes[branch_index, 0].set_title(f"{branch}: raw")
        axes[branch_index, 1].set_title(f"{branch}: driven minus F0")
        axes[branch_index, 0].set_ylabel("Velocity (m/s)")
        for axis in axes[branch_index]:
            axis.axhline(0.0, color="black", lw=0.7)
            axis.legend(frameon=False, ncol=3)
    for axis in axes[-1]:
        axis.set_xlabel("Time (ns)")
    figure.tight_layout()
    figure.savefig(output / "layer_response_blocks.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    """Run the layer-flux, raw-sign, closure, and drive-power audit."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    block_ps = float(raw.get("block_ps", 50.0))
    oxygen_count = int(raw.get("oxygen_count", 0))
    if block_ps <= 0.0 or oxygen_count <= 0:
        raise ValueError("block_ps and oxygen_count must be positive")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    base = contract_path.parent
    layer_path = resolve_path(raw["layer_timeseries"], base)
    layer_all = _read_tsv(layer_path)
    output.mkdir(parents=True)
    inputs = [
        {"path": str(contract_path), "size_bytes": contract_path.stat().st_size, "sha256": sha256(contract_path)},
        {"path": str(layer_path), "size_bytes": layer_path.stat().st_size, "sha256": sha256(layer_path)},
    ]
    case_lookup = {(str(entry["case_id"]), str(entry["branch_id"])): entry for entry in cases}
    selected_rows: list[dict[str, object]] = []
    motion_by_case: dict[tuple[str, str], tuple[dict[int, np.ndarray], dict[str, int]]] = {}
    for key, entry in case_lookup.items():
        direction = str(entry["direction"]).lower()
        if direction not in {"none", "x", "y"}:
            raise ValueError("direction must be none, x, or y")
        paths = [resolve_path(path, base) for path in entry["motion_tables"]]
        for path in paths:
            inputs.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
        columns, table = stitch_motion_tables(paths)
        index = {name: position for position, name in enumerate(columns)}
        required = {"TimeStep", "v_vdriveOx", "v_vdriveOy", "v_drivepower"}
        missing = required.difference(index)
        if missing:
            raise ValueError(f"Motion tables are missing {sorted(missing)}")
        motion_by_case[key] = (
            {int(round(row[index["TimeStep"]])): row for row in table},
            index,
        )
        selected_rows.extend(
            {
                **row,
                "step": int(float(row["step"])),
                "time_ps": float(row["time_ps"]),
                "layer_index": int(row["layer_index"]),
                "count": int(row["count"]),
            }
            for row in layer_all
            if (row["case_id"], row["branch_id"]) == key
        )
    if not selected_rows:
        raise ValueError("No matching layer rows")
    baseline_branch = next(
        str(entry["branch_id"]) for entry in cases if str(entry["direction"]).lower() == "none"
    )
    baseline = {
        (int(row["step"]), int(row["layer_index"])): row
        for row in selected_rows
        if row["branch_id"] == baseline_branch
    }
    for row in selected_rows:
        if (int(row["step"]), int(row["layer_index"])) not in baseline:
            raise ValueError("Layer rows are not aligned with the zero-force branch")
    summaries = _summarize_layers(selected_rows, baseline)
    layer_blocks = _summarize_layer_blocks(selected_rows, baseline, block_ps)

    frame_groups: dict[tuple[str, str, str, int, float], list[dict[str, object]]] = defaultdict(list)
    for row in selected_rows:
        frame_groups[
            (
                str(row["case_id"]),
                str(row["branch_id"]),
                str(row["direction"]),
                int(row["step"]),
                float(row["time_ps"]),
            )
        ].append(row)
    frame_rows: list[dict[str, object]] = []
    for (case_id, branch_id, direction, step, time_ps), layers in sorted(frame_groups.items()):
        axis = "x" if direction in {"none", "x"} else "y"
        velocity = [float(row[f"mean_v{axis}_mps"]) for row in layers]
        count = [float(row["count"]) for row in layers]
        motion, index = motion_by_case[(case_id, branch_id)]
        if step not in motion:
            raise ValueError(f"Motion table is missing layer step {step}")
        motion_row = motion[step]
        global_Aps = float(motion_row[index[f"v_vdriveO{axis}"]])
        closure = layer_flux_closure(count, velocity, global_Aps)
        reconstructed_velocity = (
            100.0 * closure["layer_velocity_sum_molecule_A_per_ps"] / oxygen_count
        )
        reference_layers = [baseline[(step, int(row["layer_index"]))] for row in layers]
        reference_velocity = occupancy_weighted_velocity(
            [float(row["count"]) for row in reference_layers],
            [float(row[f"mean_v{axis}_mps"]) for row in reference_layers],
        )
        force = float(case_lookup[(case_id, branch_id)].get("force_per_oxygen_eV_A", 0.0))
        reconstructed_power = force * closure["layer_velocity_sum_molecule_A_per_ps"]
        recorded_power = float(motion_row[index["v_drivepower"]])
        frame_rows.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "step": step,
                "time_ps": time_ps,
                "oxygen_count_in_layers": int(sum(count)),
                "raw_total_velocity_mps": reconstructed_velocity,
                "baseline_total_velocity_mps": reference_velocity,
                "excess_total_velocity_mps": reconstructed_velocity - reference_velocity,
                "global_motion_velocity_mps": 100.0 * global_Aps,
                **closure,
                "recorded_drive_power_eV_per_ps": recorded_power,
                "reconstructed_drive_power_eV_per_ps": reconstructed_power,
                "drive_power_residual_eV_per_ps": recorded_power - reconstructed_power,
            }
        )

    grouped_blocks: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in frame_rows:
        grouped_blocks[
            (
                str(row["case_id"]),
                str(row["branch_id"]),
                str(row["direction"]),
                _block_id(float(row["time_ps"]), block_ps),
            )
        ].append(row)
    blocks: list[dict[str, object]] = []
    for (case_id, branch_id, direction, block), selected in sorted(grouped_blocks.items()):
        blocks.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "direction": direction,
                "block_index": block,
                "start_ps": block * block_ps,
                "end_ps": (block + 1) * block_ps,
                "samples": len(selected),
                "raw_total_velocity_mps": float(np.mean([row["raw_total_velocity_mps"] for row in selected])),
                "baseline_total_velocity_mps": float(np.mean([row["baseline_total_velocity_mps"] for row in selected])),
                "total_excess_velocity_mps": float(np.mean([row["excess_total_velocity_mps"] for row in selected])),
                "mean_relative_flux_closure": float(np.mean([abs(row["relative_closure_residual"]) for row in selected])),
                "maximum_relative_flux_closure": float(max(abs(row["relative_closure_residual"]) for row in selected)),
                "mean_recorded_drive_power_eV_per_ps": float(np.mean([row["recorded_drive_power_eV_per_ps"] for row in selected])),
                "mean_reconstructed_drive_power_eV_per_ps": float(np.mean([row["reconstructed_drive_power_eV_per_ps"] for row in selected])),
            }
        )

    write_tsv(output / "layer_weighted_summary.tsv", summaries, tuple(summaries[0]))
    write_tsv(output / "layer_response_blocks_50ps.tsv", layer_blocks, tuple(layer_blocks[0]))
    write_tsv(output / "layer_flux_closure_timeseries.tsv", frame_rows, tuple(frame_rows[0]))
    write_tsv(output / "film_response_blocks_50ps.tsv", blocks, tuple(blocks[0]))
    write_tsv(output / "input_manifest.tsv", inputs, ("path", "size_bytes", "sha256"))
    if bool(raw.get("write_plots", True)):
        _plot(summaries, blocks, layer_blocks, output)
    maximum_flux_residual = max(abs(float(row["relative_closure_residual"])) for row in frame_rows)
    maximum_power_residual = max(abs(float(row["drive_power_residual_eV_per_ps"])) for row in frame_rows)
    summary = {
        "status": "PASS",
        "case_branches": len(cases),
        "layer_summary_rows": len(summaries),
        "layer_block_rows": len(layer_blocks),
        "frame_rows": len(frame_rows),
        "block_rows": len(blocks),
        "maximum_relative_flux_closure": maximum_flux_residual,
        "maximum_drive_power_residual_eV_per_ps": maximum_power_residual,
        "single_trajectory_descriptive_only": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Occupancy-weighted layer transport audit\n\n"
        "The audit reports raw driven velocity, time-aligned F0 velocity, and their difference. "
        "Layer fluxes are weighted by molecule count and closed against the global driven-oxygen "
        "center-of-mass velocity. Recorded drive power is compared with force times the reconstructed "
        "molecule-velocity sum. Sparse layers remain in tables but are excluded from the primary plot.\n\n"
        "Layer and total-film time blocks are single-trajectory diagnostics, not independent replicas.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary
