"""Audit constant-force transport response across velocity estimators."""

from __future__ import annotations

import argparse
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

AXES = ("x", "y")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _sign_match(left: float, right: float, tolerance: float = 1.0e-12) -> str:
    if abs(left) <= tolerance or abs(right) <= tolerance:
        return "ZERO_UNRESOLVED"
    return "SAME" if left * right > 0.0 else "OPPOSITE"


def time_weighted_mean(
    steps: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    timestep_fs: float,
) -> float:
    """Integrate sampled velocity over time with the trapezoidal rule."""

    step_array = np.asarray(steps, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if len(step_array) != len(value_array) or len(step_array) < 2:
        raise ValueError("steps and values must have the same length of at least two")
    if np.any(np.diff(step_array) <= 0.0):
        raise ValueError("steps must be strictly increasing")
    time_ps = (step_array - step_array[0]) * timestep_fs / 1000.0
    duration_ps = float(time_ps[-1])
    if duration_ps <= 0.0:
        raise ValueError("duration must be positive")
    integral = float(np.sum(0.5 * (value_array[1:] + value_array[:-1]) * np.diff(time_ps)))
    return integral / duration_ps


def response_comparison_rows(
    case_branches: Mapping[str, Mapping[str, str]],
    displacement_raw: Mapping[tuple[str, str], np.ndarray],
    high_frequency_raw: Mapping[tuple[str, str], np.ndarray],
    sparse_snapshot_raw: Mapping[tuple[str, str], np.ndarray],
) -> list[dict[str, object]]:
    """Compare F0-subtracted matrices from three raw-velocity estimators."""

    output: list[dict[str, object]] = []
    for case_id, branches in sorted(case_branches.items()):
        baseline_branch = next(
            branch_id for branch_id, direction in branches.items() if direction == "none"
        )
        for branch_id, direction in sorted(branches.items()):
            if direction == "none":
                continue
            for axis_index, component in enumerate(AXES):
                displacement = float(
                    displacement_raw[(case_id, branch_id)][axis_index]
                    - displacement_raw[(case_id, baseline_branch)][axis_index]
                )
                high_frequency = float(
                    high_frequency_raw[(case_id, branch_id)][axis_index]
                    - high_frequency_raw[(case_id, baseline_branch)][axis_index]
                )
                sparse_snapshot = float(
                    sparse_snapshot_raw[(case_id, branch_id)][axis_index]
                    - sparse_snapshot_raw[(case_id, baseline_branch)][axis_index]
                )
                output.append(
                    {
                        "case_id": case_id,
                        "branch_id": branch_id,
                        "drive_direction": direction,
                        "response_component": component,
                        "is_longitudinal": int(direction == component),
                        "displacement_response_mps": displacement,
                        "high_frequency_velocity_response_mps": high_frequency,
                        "sparse_snapshot_velocity_response_mps": sparse_snapshot,
                        "high_frequency_minus_displacement_mps": high_frequency
                        - displacement,
                        "sparse_minus_high_frequency_mps": sparse_snapshot
                        - high_frequency,
                        "high_frequency_displacement_sign_match": _sign_match(
                            high_frequency, displacement
                        ),
                        "sparse_high_frequency_sign_match": _sign_match(
                            sparse_snapshot, high_frequency
                        ),
                        "relative_high_frequency_displacement_difference": abs(
                            high_frequency - displacement
                        )
                        / max(abs(high_frequency), abs(displacement), 1.0e-12),
                        "evidence_limit": (
                            "single_trajectory_estimator_consistency_not_independent_uncertainty"
                        ),
                    }
                )
    return output


def response_block_rows(
    case_branches: Mapping[str, Mapping[str, str]],
    series: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    *,
    start_step: int,
    end_step: int,
    timestep_fs: float,
    block_sizes_ps: Sequence[float],
) -> list[dict[str, object]]:
    """Return time-integrated F0-subtracted response in fixed blocks."""

    full_window_ps = (end_step - start_step) * timestep_fs / 1000.0
    output: list[dict[str, object]] = []
    for case_id, branches in sorted(case_branches.items()):
        baseline_branch = next(
            branch_id for branch_id, direction in branches.items() if direction == "none"
        )
        baseline_steps, baseline_values = series[(case_id, baseline_branch)]
        for branch_id, direction in sorted(branches.items()):
            if direction == "none":
                continue
            steps, values = series[(case_id, branch_id)]
            if not np.array_equal(steps, baseline_steps):
                raise ValueError(f"{case_id}/{branch_id}: motion steps do not align with F0")
            elapsed_ps = (steps - start_step) * timestep_fs / 1000.0
            for block_ps in block_sizes_ps:
                if block_ps <= 0.0:
                    raise ValueError("block sizes must be positive")
                block_count = int(math.ceil(full_window_ps / block_ps - 1.0e-12))
                for block_index in range(block_count):
                    begin = block_index * block_ps
                    end = min((block_index + 1) * block_ps, full_window_ps)
                    mask = (elapsed_ps >= begin - 1.0e-12) & (elapsed_ps <= end + 1.0e-12)
                    selected_steps = steps[mask]
                    if len(selected_steps) < 2:
                        raise ValueError(
                            f"{case_id}/{branch_id}: fewer than two samples in block {block_index}"
                        )
                    for axis_index, component in enumerate(AXES):
                        raw = time_weighted_mean(
                            selected_steps, values[mask, axis_index], timestep_fs
                        )
                        baseline = time_weighted_mean(
                            selected_steps, baseline_values[mask, axis_index], timestep_fs
                        )
                        output.append(
                            {
                                "case_id": case_id,
                                "branch_id": branch_id,
                                "drive_direction": direction,
                                "response_component": component,
                                "is_longitudinal": int(direction == component),
                                "block_ps": block_ps,
                                "block_index": block_index,
                                "start_ps": begin,
                                "end_ps": end,
                                "duration_ps": end - begin,
                                "sample_count": len(selected_steps),
                                "raw_velocity_mps": raw,
                                "baseline_velocity_mps": baseline,
                                "response_velocity_mps": raw - baseline,
                            }
                        )
    return output


def response_block_stability(
    block_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    full_response = {
        (str(row["case_id"]), str(row["branch_id"]), str(row["response_component"])): float(
            row["high_frequency_velocity_response_mps"]
        )
        for row in comparison_rows
    }
    grouped: dict[tuple[str, str, str, str, float], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in block_rows:
        grouped[
            (
                str(row["case_id"]),
                str(row["branch_id"]),
                str(row["drive_direction"]),
                str(row["response_component"]),
                float(row["block_ps"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for (case_id, branch_id, direction, component, block_ps), selected in sorted(
        grouped.items()
    ):
        values = np.asarray([float(row["response_velocity_mps"]) for row in selected])
        duration = np.asarray([float(row["duration_ps"]) for row in selected])
        reference = full_response[(case_id, branch_id, component)]
        matches = [_sign_match(float(value), reference) for value in values]
        output.append(
            {
                "case_id": case_id,
                "branch_id": branch_id,
                "drive_direction": direction,
                "response_component": component,
                "is_longitudinal": int(direction == component),
                "block_ps": block_ps,
                "block_count": len(selected),
                "duration_weighted_mean_mps": float(np.average(values, weights=duration)),
                "block_standard_deviation_mps": float(np.std(values)),
                "positive_block_fraction": float(np.mean(values > 0.0)),
                "full_response_sign_match_fraction": float(
                    np.mean([match == "SAME" for match in matches])
                ),
                "minimum_block_response_mps": float(np.min(values)),
                "maximum_block_response_mps": float(np.max(values)),
                "evidence_limit": "time_blocks_are_descriptive_not_independent_samples",
            }
        )
    return output


def _load_displacement_raw(path: Path) -> dict[tuple[str, str], np.ndarray]:
    output: dict[tuple[str, str], np.ndarray] = {}
    for row in _read_tsv(path):
        case_id = row["case_id"]
        direction = row["drive_direction"]
        output[(case_id, "none")] = np.asarray(
            [float(row["baseline_vx_mps"]), float(row["baseline_vy_mps"])], dtype=float
        )
        output[(case_id, direction)] = np.asarray(
            [float(row["raw_vx_mps"]), float(row["raw_vy_mps"])], dtype=float
        )
    return output


def _motion_series(
    paths: Sequence[Path],
    *,
    start_step: int,
    end_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    columns, table = stitch_motion_tables(paths)
    index = {name: position for position, name in enumerate(columns)}
    required = {"TimeStep", "v_vdriveOx", "v_vdriveOy"}
    missing = required.difference(index)
    if missing:
        raise ValueError(f"motion tables are missing {sorted(missing)}")
    rows = [
        row
        for row in table
        if start_step <= int(round(row[index["TimeStep"]])) <= end_step
    ]
    steps = np.asarray([int(round(row[index["TimeStep"]])) for row in rows], dtype=int)
    values = np.asarray(
        [[row[index["v_vdriveOx"]], row[index["v_vdriveOy"]]] for row in rows],
        dtype=float,
    )
    if len(steps) < 2 or steps[0] != start_step or steps[-1] != end_step:
        raise ValueError("motion tables do not span the exact accepted window")
    return steps, values * 100.0


def _plot_matrices(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    cases = sorted({str(row["case_id"]) for row in rows})
    estimators = (
        ("displacement_response_mps", "Displacement"),
        ("high_frequency_velocity_response_mps", "0.1 ps velocity"),
        ("sparse_snapshot_velocity_response_mps", "10 ps snapshots"),
    )
    figure, axes = plt.subplots(len(cases), 3, figsize=(10.5, 3.5 * len(cases)), squeeze=False)
    for case_index, case_id in enumerate(cases):
        selected = [row for row in rows if row["case_id"] == case_id]
        values = [abs(float(row[key])) for row in selected for key, _ in estimators]
        limit = max(values, default=1.0e-12)
        for estimator_index, (key, label) in enumerate(estimators):
            matrix = np.zeros((2, 2), dtype=float)
            for row in selected:
                i = AXES.index(str(row["response_component"]))
                j = AXES.index(str(row["drive_direction"]))
                matrix[i, j] = float(row[key])
            axis = axes[case_index, estimator_index]
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
            for i in range(2):
                for j in range(2):
                    axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")
            axis.set_xticks([0, 1], ["Fx", "Fy"])
            axis.set_yticks([0, 1], ["Jx", "Jy"])
            axis.set_title(f"{case_id}: {label}")
            figure.colorbar(image, ax=axis, shrink=0.72, label="m/s")
    figure.tight_layout()
    figure.savefig(output / "estimator_response_matrices.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(contract.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    timestep_fs = float(contract["timestep_fs"])
    start_step = int(contract["start_step"])
    end_step = int(contract["end_step"])
    snapshot_stride_steps = int(contract["snapshot_stride_steps"])
    block_sizes_ps = tuple(float(value) for value in contract["block_sizes_ps"])
    if timestep_fs <= 0.0 or start_step >= end_step or snapshot_stride_steps <= 0:
        raise ValueError("invalid time-window contract")
    base = contract_path.parent
    displacement_path = resolve_path(contract["displacement_response_detail"], base)
    input_paths: set[Path] = {contract_path, displacement_path}
    case_branches: dict[str, dict[str, str]] = {}
    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for case in contract["cases"]:
        case_id = str(case["case_id"])
        case_branches[case_id] = {}
        for branch in case["branches"]:
            branch_id = str(branch["branch_id"])
            direction = str(branch["direction"]).lower()
            if direction not in {"none", "x", "y"}:
                raise ValueError("direction must be none, x, or y")
            paths = [resolve_path(path, base) for path in branch["motion_tables"]]
            input_paths.update(paths)
            case_branches[case_id][branch_id] = direction
            series[(case_id, branch_id)] = _motion_series(
                paths, start_step=start_step, end_step=end_step
            )
    displacement_by_direction = _load_displacement_raw(displacement_path)
    displacement_raw = {
        (case_id, branch_id): displacement_by_direction[(case_id, direction)]
        for case_id, branches in case_branches.items()
        for branch_id, direction in branches.items()
    }
    high_frequency_raw: dict[tuple[str, str], np.ndarray] = {}
    sparse_snapshot_raw: dict[tuple[str, str], np.ndarray] = {}
    branch_rows: list[dict[str, object]] = []
    for key, (steps, values) in sorted(series.items()):
        sparse_mask = (steps - start_step) % snapshot_stride_steps == 0
        sparse_steps = steps[sparse_mask]
        if len(sparse_steps) < 2:
            raise ValueError(f"{key}: sparse estimator has fewer than two samples")
        high = np.asarray(
            [time_weighted_mean(steps, values[:, index], timestep_fs) for index in range(2)]
        )
        sparse = np.asarray(
            [
                time_weighted_mean(sparse_steps, values[sparse_mask, index], timestep_fs)
                for index in range(2)
            ]
        )
        high_frequency_raw[key] = high
        sparse_snapshot_raw[key] = sparse
        direction = case_branches[key[0]][key[1]]
        for index, component in enumerate(AXES):
            branch_rows.append(
                {
                    "case_id": key[0],
                    "branch_id": key[1],
                    "direction": direction,
                    "velocity_component": component,
                    "high_frequency_velocity_mps": float(high[index]),
                    "sparse_snapshot_velocity_mps": float(sparse[index]),
                    "sparse_minus_high_frequency_mps": float(sparse[index] - high[index]),
                    "high_frequency_sample_count": len(steps),
                    "sparse_snapshot_sample_count": len(sparse_steps),
                }
            )
    comparisons = response_comparison_rows(
        case_branches,
        displacement_raw,
        high_frequency_raw,
        sparse_snapshot_raw,
    )
    blocks = response_block_rows(
        case_branches,
        series,
        start_step=start_step,
        end_step=end_step,
        timestep_fs=timestep_fs,
        block_sizes_ps=block_sizes_ps,
    )
    stability = response_block_stability(blocks, comparisons)
    output.mkdir(parents=True)
    write_tsv(output / "branch_raw_estimators.tsv", branch_rows, tuple(branch_rows[0]))
    write_tsv(output / "estimator_response_comparison.tsv", comparisons, tuple(comparisons[0]))
    write_tsv(output / "response_blocks.tsv", blocks, tuple(blocks[0]))
    write_tsv(output / "response_block_stability.tsv", stability, tuple(stability[0]))
    input_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(input_paths)
    ]
    write_tsv(output / "input_manifest.tsv", input_rows, ("path", "size_bytes", "sha256"))
    _plot_matrices(comparisons, output)
    longitudinal = [row for row in comparisons if int(row["is_longitudinal"]) == 1]
    high_sign_matches = sum(
        row["high_frequency_displacement_sign_match"] == "SAME" for row in comparisons
    )
    sparse_sign_matches = sum(
        row["sparse_high_frequency_sign_match"] == "SAME" for row in comparisons
    )
    summary = {
        "status": "PASS",
        "case_count": len(case_branches),
        "response_comparison_rows": len(comparisons),
        "response_block_rows": len(blocks),
        "response_block_stability_rows": len(stability),
        "high_frequency_displacement_sign_matches": high_sign_matches,
        "high_frequency_displacement_sign_total": len(comparisons),
        "longitudinal_high_frequency_displacement_sign_matches": sum(
            row["high_frequency_displacement_sign_match"] == "SAME" for row in longitudinal
        ),
        "longitudinal_high_frequency_displacement_sign_total": len(longitudinal),
        "sparse_high_frequency_sign_matches": sparse_sign_matches,
        "sparse_high_frequency_sign_total": len(comparisons),
        "maximum_absolute_high_frequency_displacement_difference_mps": max(
            abs(float(row["high_frequency_minus_displacement_mps"])) for row in comparisons
        ),
        "maximum_absolute_sparse_high_frequency_difference_mps": max(
            abs(float(row["sparse_minus_high_frequency_mps"])) for row in comparisons
        ),
        "sparse_snapshot_is_time_aliased_diagnostic": True,
        "single_trajectory_descriptive_only": True,
        "new_md_submitted": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Constant-force estimator-consistency audit\n\n"
        "Displacement integration and the complete 0.1 ps velocity series are treated as "
        "the two primary global-response estimators. The 10 ps instantaneous-velocity "
        "subsample is retained only to quantify temporal aliasing in layer-resolved "
        "snapshots.\n\n"
        f"The high-frequency and displacement estimators match signs in {high_sign_matches}/"
        f"{len(comparisons)} response components and "
        f"{summary['longitudinal_high_frequency_displacement_sign_matches']}/"
        f"{len(longitudinal)} longitudinal components. The sparse and high-frequency "
        f"velocity estimators match signs in {sparse_sign_matches}/{len(comparisons)} "
        "components.\n\n"
        "Time blocks are descriptive diagnostics from single trajectories. Estimator "
        "agreement does not supply independent-replica uncertainty or establish a "
        "layer-resolved mechanism.\n",
        encoding="utf-8",
    )
    write_output_hashes(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_contract(args.contract, args.output)
    print(args.output.resolve())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
