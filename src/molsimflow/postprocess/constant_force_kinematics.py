"""Analyze baseline-subtracted kinematics for constant-force interface trajectories."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.constant_force_events import (
    _resolve_path,
    _sha256,
    _write_tsv,
    stitch_motion_tables,
)


BLOCK_FIELDS = (
    "case_id",
    "branch_id",
    "direction",
    "block_index",
    "block_start_ps",
    "block_end_ps",
    "samples",
    "vx_mps",
    "vy_mps",
    "axis_velocity_mps",
    "baseline_axis_velocity_mps",
    "excess_axis_velocity_mps",
)

SUMMARY_FIELDS = (
    "case_id",
    "branch_id",
    "direction",
    "samples",
    "time_start_ps",
    "time_end_ps",
    "dx_final_A",
    "dy_final_A",
    "vx_full_mps",
    "vy_full_mps",
    "axis_velocity_full_mps",
    "baseline_axis_velocity_full_mps",
    "excess_axis_velocity_full_mps",
    "excess_block_mean_mps",
    "excess_block_sem_mps",
    "classification_threshold_mps",
    "positive_active_blocks",
    "negative_active_blocks",
    "response_class",
    "vx_acf_positive_tau_ps",
    "vy_acf_positive_tau_ps",
    "excess_axis_acf_positive_tau_ps",
)

VELOCITY_FIELDS = (
    "case_id",
    "branch_id",
    "direction",
    "time_ps",
    "vx_mps",
    "vy_mps",
    "axis_velocity_mps",
    "baseline_axis_velocity_mps",
    "excess_axis_velocity_mps",
)

ACF_FIELDS = (
    "case_id",
    "branch_id",
    "direction",
    "signal",
    "lag_ps",
    "acf",
)


def _float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _linear_velocity_mps(time_ps: np.ndarray, displacement_A: np.ndarray) -> float:
    """Return an ordinary-least-squares displacement slope in metres per second."""

    time = np.asarray(time_ps, dtype=float)
    displacement = np.asarray(displacement_A, dtype=float)
    finite = np.isfinite(time) & np.isfinite(displacement)
    if int(np.sum(finite)) < 2:
        return math.nan
    time = time[finite]
    displacement = displacement[finite]
    centered = time - float(np.mean(time))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return math.nan
    numerator = float(np.dot(centered, displacement - float(np.mean(displacement))))
    slope_A_per_ps = numerator / denominator
    return 100.0 * slope_A_per_ps


def _sample_velocity(
    time_ps: np.ndarray,
    values_A: np.ndarray,
    sample_ps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate displacement to a regular grid and return interval velocities."""

    start = math.ceil(float(time_ps[0]) / sample_ps) * sample_ps
    stop = math.floor(float(time_ps[-1]) / sample_ps) * sample_ps
    grid = np.arange(start, stop + 0.5 * sample_ps, sample_ps, dtype=float)
    if len(grid) < 2:
        raise ValueError("Trajectory is shorter than one velocity sampling interval")
    sampled = np.interp(grid, time_ps, values_A)
    midpoint = 0.5 * (grid[:-1] + grid[1:])
    velocity_mps = np.diff(sampled) / np.diff(grid) * 100.0
    return midpoint, velocity_mps


def normalized_autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Return a mean-subtracted, overlap-normalized autocorrelation."""

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return np.array([math.nan])
    centered = finite - float(np.mean(finite))
    variance = float(np.dot(centered, centered) / len(centered))
    maximum = min(int(max_lag), len(centered) - 1)
    if variance <= np.finfo(float).eps:
        result = np.full(maximum + 1, math.nan)
        result[0] = 1.0
        return result
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    overlap = np.arange(len(centered), 0, -1, dtype=float)
    correlation = correlation / overlap / variance
    return correlation[: maximum + 1]


def positive_acf_time_ps(acf: np.ndarray, sample_ps: float) -> float:
    """Integrate the positive ACF lobe with a trapezoidal endpoint at lag zero."""

    values = np.asarray(acf, dtype=float)
    if not len(values) or not math.isfinite(float(values[0])):
        return math.nan
    positive: list[float] = []
    for value in values[1:]:
        if not math.isfinite(float(value)) or value <= 0.0:
            break
        positive.append(float(value))
    return sample_ps * (0.5 + float(np.sum(positive)))


def classify_response(
    block_excess_velocity_mps: Sequence[float],
    full_excess_velocity_mps: float,
    *,
    minimum_effect_mps: float,
    sigma_multiplier: float,
) -> tuple[str, float, int, int, float, float]:
    """Classify a single-path response using paired time blocks.

    The returned uncertainty is a within-trajectory block SEM. It is not
    replicate uncertainty and is intentionally retained as a diagnostic only.
    """

    values = np.asarray(block_excess_velocity_mps, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "FLUCTUATION", minimum_effect_mps, 0, 0, math.nan, math.nan
    mean = float(np.mean(values))
    sem = (
        float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else math.nan
    )
    threshold = max(
        float(minimum_effect_mps),
        float(sigma_multiplier) * sem if math.isfinite(sem) else 0.0,
    )
    positive = int(np.sum(values > threshold))
    negative = int(np.sum(values < -threshold))
    required_sustained = max(1, int(math.ceil(0.75 * len(values))))
    if positive and negative:
        label = "REVERSAL"
    elif negative and not positive:
        label = "REVERSAL"
    elif positive >= required_sustained and full_excess_velocity_mps > threshold:
        label = "SUSTAINED"
    elif positive:
        label = "INTERMITTENT"
    else:
        label = "FLUCTUATION"
    return label, threshold, positive, negative, mean, sem


def _validate_contract(raw: Mapping[str, object]) -> None:
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("contract cases must be a non-empty list")
    for key in ("timestep_fs", "block_ps", "velocity_sample_ps", "acf_max_lag_ps"):
        if _float(raw.get(key)) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if _float(raw.get("minimum_effect_mps", 0.05)) < 0.0:
        raise ValueError("minimum_effect_mps must be non-negative")
    if _float(raw.get("classification_sigma_multiplier", 1.0)) < 0.0:
        raise ValueError("classification_sigma_multiplier must be non-negative")
    identities: list[tuple[str, str]] = []
    baselines: dict[str, int] = defaultdict(int)
    for entry in cases:
        if not isinstance(entry, dict):
            raise ValueError("Each case entry must be an object")
        for key in ("case_id", "branch_id", "direction", "motion_tables"):
            if key not in entry:
                raise ValueError(f"Case entry is missing {key}")
        direction = str(entry["direction"]).lower()
        if direction not in {"none", "x", "y"}:
            raise ValueError("direction must be one of none, x, or y")
        if not isinstance(entry["motion_tables"], list) or not entry["motion_tables"]:
            raise ValueError("motion_tables must be a non-empty list")
        case_id = str(entry["case_id"])
        identities.append((case_id, str(entry["branch_id"])))
        baselines[case_id] += int(direction == "none")
    if len(identities) != len(set(identities)):
        raise ValueError("case_id/branch_id pairs must be unique")
    bad = sorted(case_id for case_id, count in baselines.items() if count != 1)
    if bad:
        raise ValueError(f"Each case_id must have exactly one direction=none branch: {bad}")


def _load_branch(
    entry: Mapping[str, object],
    *,
    base: Path,
    time_origin_step: int,
    timestep_fs: float,
    columns: Mapping[str, str],
) -> dict[str, object]:
    paths = [_resolve_path(value, base) for value in entry["motion_tables"]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    names, table = stitch_motion_tables(
        paths,
        step_column=columns["step"],
        displacement_columns=(columns["x"], columns["y"]),
    )
    index = {name: position for position, name in enumerate(names)}
    missing = [columns[key] for key in ("step", "x", "y") if columns[key] not in index]
    if missing:
        raise ValueError(f"Missing motion columns {missing}")
    step = table[:, index[columns["step"]]]
    time_ps = (step - time_origin_step) * timestep_fs / 1000.0
    if np.any(np.diff(time_ps) <= 0.0):
        raise ValueError(
            f"Non-increasing stitched time for {entry['case_id']}/{entry['branch_id']}"
        )
    return {
        "case_id": str(entry["case_id"]),
        "branch_id": str(entry["branch_id"]),
        "direction": str(entry["direction"]).lower(),
        "paths": paths,
        "step": step,
        "time_ps": time_ps,
        "x_A": table[:, index[columns["x"]]],
        "y_A": table[:, index[columns["y"]]],
    }


def _acf_rows(
    branch: Mapping[str, object],
    signal: str,
    values: np.ndarray,
    *,
    sample_ps: float,
    max_lag_ps: float,
) -> tuple[list[dict[str, object]], float]:
    acf = normalized_autocorrelation(values, int(round(max_lag_ps / sample_ps)))
    rows = [
        {
            "case_id": branch["case_id"],
            "branch_id": branch["branch_id"],
            "direction": branch["direction"],
            "signal": signal,
            "lag_ps": lag * sample_ps,
            "acf": value,
        }
        for lag, value in enumerate(acf)
    ]
    return rows, positive_acf_time_ps(acf, sample_ps)


def _write_plots(
    output: Path,
    branch_series: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    acf_rows: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    case_ids = list(dict.fromkeys(str(item["case_id"]) for item in branch_series))
    summary_by_key = {
        (str(row["case_id"]), str(row["branch_id"])): row for row in summaries
    }
    figure, axes = plt.subplots(
        len(case_ids),
        2,
        figsize=(10.0, max(3.0, 2.7 * len(case_ids))),
        squeeze=False,
        sharex=True,
    )
    for row_index, case_id in enumerate(case_ids):
        for column_index, direction in enumerate(("x", "y")):
            axis = axes[row_index, column_index]
            selected = [
                item
                for item in branch_series
                if item["case_id"] == case_id and item["direction"] == direction
            ]
            for item in selected:
                summary = summary_by_key[(case_id, str(item["branch_id"]))]
                axis.plot(
                    np.asarray(item["time_ps"]) / 1000.0,
                    np.asarray(item["excess_axis_A"]),
                    lw=1.1,
                    label=f"{item['branch_id']} ({summary['response_class']})",
                )
            axis.axhline(0.0, color="0.5", lw=0.6)
            axis.set_title(f"{case_id} / {direction.upper()}", loc="left", fontsize=9)
            axis.set_ylabel("Force - F0 displacement (Å)")
            axis.grid(alpha=0.2)
            if selected:
                axis.legend(frameon=False, fontsize=7)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (ns)")
    figure.tight_layout()
    figure.savefig(output / "kinematics_overview.png", dpi=220)
    figure.savefig(output / "kinematics_overview.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(
        len(case_ids),
        2,
        figsize=(10.0, max(3.0, 2.7 * len(case_ids))),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row_index, case_id in enumerate(case_ids):
        for column_index, direction in enumerate(("x", "y")):
            axis = axes[row_index, column_index]
            selected_summaries = [
                row
                for row in summaries
                if row["case_id"] == case_id and row["direction"] == direction
            ]
            branch_ids = {str(row["branch_id"]) for row in selected_summaries}
            for branch_id in branch_ids:
                selected = [
                    row
                    for row in acf_rows
                    if row["case_id"] == case_id
                    and row["branch_id"] == branch_id
                    and row["signal"] == "excess_axis_velocity"
                ]
                if selected:
                    axis.plot(
                        [float(row["lag_ps"]) for row in selected],
                        [float(row["acf"]) for row in selected],
                        lw=1.0,
                        label=branch_id,
                    )
            axis.axhline(0.0, color="0.5", lw=0.6)
            axis.set_title(f"{case_id} / {direction.upper()}", loc="left", fontsize=9)
            axis.set_ylabel("Velocity ACF")
            axis.grid(alpha=0.2)
            if branch_ids:
                axis.legend(frameon=False, fontsize=7)
    for axis in axes[-1, :]:
        axis.set_xlabel("Lag (ps)")
    figure.tight_layout()
    figure.savefig(output / "velocity_acf_overview.png", dpi=220)
    figure.savefig(output / "velocity_acf_overview.pdf")
    plt.close(figure)


def run_contract(contract_path: Path, output_dir: Path) -> dict[str, object]:
    """Run a contract-driven four-interface kinematics analysis."""

    contract_path = Path(contract_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(raw)
    base = contract_path.parent
    time_origin_step = int(raw.get("time_origin_step", 0))
    timestep_fs = _float(raw["timestep_fs"])
    block_ps = _float(raw["block_ps"])
    velocity_sample_ps = _float(raw["velocity_sample_ps"])
    acf_max_lag_ps = _float(raw["acf_max_lag_ps"])
    minimum_effect_mps = _float(raw.get("minimum_effect_mps", 0.05))
    sigma_multiplier = _float(raw.get("classification_sigma_multiplier", 1.0))
    columns = {
        "step": "TimeStep",
        "x": "v_dxrel",
        "y": "v_dyrel",
        **dict(raw.get("motion_columns", {})),
    }
    branches = [
        _load_branch(
            entry,
            base=base,
            time_origin_step=time_origin_step,
            timestep_fs=timestep_fs,
            columns=columns,
        )
        for entry in raw["cases"]
    ]
    baseline_by_case = {
        str(branch["case_id"]): branch
        for branch in branches
        if branch["direction"] == "none"
    }
    block_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    velocity_rows: list[dict[str, object]] = []
    acf_rows: list[dict[str, object]] = []
    branch_series: list[dict[str, object]] = []
    input_paths: set[Path] = {contract_path}

    for branch in branches:
        input_paths.update(branch["paths"])
        time_ps = np.asarray(branch["time_ps"], dtype=float)
        x_A = np.asarray(branch["x_A"], dtype=float)
        y_A = np.asarray(branch["y_A"], dtype=float)
        baseline = baseline_by_case[str(branch["case_id"])]
        baseline_time = np.asarray(baseline["time_ps"], dtype=float)
        baseline_x = np.interp(time_ps, baseline_time, np.asarray(baseline["x_A"], dtype=float))
        baseline_y = np.interp(time_ps, baseline_time, np.asarray(baseline["y_A"], dtype=float))
        direction = str(branch["direction"])
        if direction == "x":
            axis_A, baseline_axis_A = x_A, baseline_x
        elif direction == "y":
            axis_A, baseline_axis_A = y_A, baseline_y
        else:
            axis_A = baseline_axis_A = np.full_like(time_ps, math.nan)
        excess_axis_A = axis_A - baseline_axis_A
        branch_series.append(
            {
                **branch,
                "excess_axis_A": excess_axis_A,
            }
        )

        midpoint, vx = _sample_velocity(time_ps, x_A, velocity_sample_ps)
        midpoint_y, vy = _sample_velocity(time_ps, y_A, velocity_sample_ps)
        if not np.array_equal(midpoint, midpoint_y):
            raise RuntimeError("X/Y velocity grids differ")
        if direction in {"x", "y"}:
            _, axis_velocity = _sample_velocity(time_ps, axis_A, velocity_sample_ps)
            _, baseline_axis_velocity = _sample_velocity(
                time_ps, baseline_axis_A, velocity_sample_ps
            )
            excess_axis_velocity = axis_velocity - baseline_axis_velocity
        else:
            axis_velocity = baseline_axis_velocity = excess_axis_velocity = np.full_like(
                midpoint, math.nan
            )
        velocity_rows.extend(
            {
                "case_id": branch["case_id"],
                "branch_id": branch["branch_id"],
                "direction": direction,
                "time_ps": time_value,
                "vx_mps": vx_value,
                "vy_mps": vy_value,
                "axis_velocity_mps": axis_value,
                "baseline_axis_velocity_mps": baseline_value,
                "excess_axis_velocity_mps": excess_value,
            }
            for time_value, vx_value, vy_value, axis_value, baseline_value, excess_value in zip(
                midpoint,
                vx,
                vy,
                axis_velocity,
                baseline_axis_velocity,
                excess_axis_velocity,
            )
        )
        vx_acf, vx_tau = _acf_rows(
            branch,
            "vx",
            vx,
            sample_ps=velocity_sample_ps,
            max_lag_ps=acf_max_lag_ps,
        )
        vy_acf, vy_tau = _acf_rows(
            branch,
            "vy",
            vy,
            sample_ps=velocity_sample_ps,
            max_lag_ps=acf_max_lag_ps,
        )
        acf_rows.extend(vx_acf)
        acf_rows.extend(vy_acf)
        if direction in {"x", "y"}:
            excess_acf, excess_tau = _acf_rows(
                branch,
                "excess_axis_velocity",
                excess_axis_velocity,
                sample_ps=velocity_sample_ps,
                max_lag_ps=acf_max_lag_ps,
            )
            acf_rows.extend(excess_acf)
        else:
            excess_tau = math.nan

        block_start = math.floor(float(time_ps[0]) / block_ps) * block_ps
        final_time = float(time_ps[-1])
        branch_blocks: list[dict[str, object]] = []
        block_index = 0
        while block_start < final_time - 1.0e-9:
            block_end = min(block_start + block_ps, final_time)
            include_end = math.isclose(block_end, final_time)
            mask = (time_ps >= block_start - 1.0e-9) & (
                time_ps <= block_end + 1.0e-9 if include_end else time_ps < block_end - 1.0e-9
            )
            if int(np.sum(mask)) >= 2:
                vx_block = _linear_velocity_mps(time_ps[mask], x_A[mask])
                vy_block = _linear_velocity_mps(time_ps[mask], y_A[mask])
                if direction in {"x", "y"}:
                    axis_block = _linear_velocity_mps(time_ps[mask], axis_A[mask])
                    baseline_block = _linear_velocity_mps(
                        time_ps[mask], baseline_axis_A[mask]
                    )
                    excess_block = _linear_velocity_mps(
                        time_ps[mask], excess_axis_A[mask]
                    )
                else:
                    axis_block = baseline_block = excess_block = math.nan
                row = {
                    "case_id": branch["case_id"],
                    "branch_id": branch["branch_id"],
                    "direction": direction,
                    "block_index": block_index,
                    "block_start_ps": block_start,
                    "block_end_ps": block_end,
                    "samples": int(np.sum(mask)),
                    "vx_mps": vx_block,
                    "vy_mps": vy_block,
                    "axis_velocity_mps": axis_block,
                    "baseline_axis_velocity_mps": baseline_block,
                    "excess_axis_velocity_mps": excess_block,
                }
                block_rows.append(row)
                branch_blocks.append(row)
            block_index += 1
            block_start += block_ps

        vx_full = _linear_velocity_mps(time_ps, x_A)
        vy_full = _linear_velocity_mps(time_ps, y_A)
        if direction in {"x", "y"}:
            axis_full = _linear_velocity_mps(time_ps, axis_A)
            baseline_full = _linear_velocity_mps(time_ps, baseline_axis_A)
            excess_full = _linear_velocity_mps(time_ps, excess_axis_A)
            classification = classify_response(
                [float(row["excess_axis_velocity_mps"]) for row in branch_blocks],
                excess_full,
                minimum_effect_mps=minimum_effect_mps,
                sigma_multiplier=sigma_multiplier,
            )
            response_class, threshold, positive, negative, block_mean, block_sem = classification
        else:
            axis_full = baseline_full = excess_full = math.nan
            block_mean = block_sem = threshold = math.nan
            positive = negative = 0
            response_class = "CONTROL"
        summary_rows.append(
            {
                "case_id": branch["case_id"],
                "branch_id": branch["branch_id"],
                "direction": direction,
                "samples": len(time_ps),
                "time_start_ps": float(time_ps[0]),
                "time_end_ps": float(time_ps[-1]),
                "dx_final_A": float(x_A[-1] - x_A[0]),
                "dy_final_A": float(y_A[-1] - y_A[0]),
                "vx_full_mps": vx_full,
                "vy_full_mps": vy_full,
                "axis_velocity_full_mps": axis_full,
                "baseline_axis_velocity_full_mps": baseline_full,
                "excess_axis_velocity_full_mps": excess_full,
                "excess_block_mean_mps": block_mean,
                "excess_block_sem_mps": block_sem,
                "classification_threshold_mps": threshold,
                "positive_active_blocks": positive,
                "negative_active_blocks": negative,
                "response_class": response_class,
                "vx_acf_positive_tau_ps": vx_tau,
                "vy_acf_positive_tau_ps": vy_tau,
                "excess_axis_acf_positive_tau_ps": excess_tau,
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    _write_tsv(output / "block_velocity.tsv", block_rows, fieldnames=BLOCK_FIELDS)
    _write_tsv(output / "velocity_timeseries.tsv", velocity_rows, fieldnames=VELOCITY_FIELDS)
    _write_tsv(output / "velocity_acf.tsv", acf_rows, fieldnames=ACF_FIELDS)
    _write_tsv(output / "branch_summary.tsv", summary_rows, fieldnames=SUMMARY_FIELDS)
    manifest_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(input_paths)
    ]
    _write_tsv(
        output / "input_manifest.tsv",
        manifest_rows,
        fieldnames=("path", "size_bytes", "sha256"),
    )
    if bool(raw.get("write_plots", True)):
        _write_plots(output, branch_series, summary_rows, acf_rows)

    report_lines = [
        "# Constant-force kinematics report",
        "",
        "| Case | Branch | Direction | excess velocity (m/s) | block SEM (m/s) "
        "| class | ACF positive time (ps) |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in summary_rows:
        if row["direction"] == "none":
            excess = sem = tau = ""
        else:
            excess = f"{float(row['excess_axis_velocity_full_mps']):.4f}"
            sem = f"{float(row['excess_block_sem_mps']):.4f}"
            tau = f"{float(row['excess_axis_acf_positive_tau_ps']):.2f}"
        report_lines.append(
            f"| {row['case_id']} | {row['branch_id']} | {str(row['direction']).upper()} | "
            f"{excess} | {sem} | {row['response_class']} | {tau} |"
        )
    report_lines.extend(
        [
            "",
            "The paired block SEM and velocity autocorrelation are within-trajectory diagnostics. ",
            "They are not independent-replicate uncertainty or an equilibrium "
            "transport coefficient.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    summary = {
        "status": "PASS",
        "case_branches": len(branches),
        "cases": len(baseline_by_case),
        "block_rows": len(block_rows),
        "velocity_rows": len(velocity_rows),
        "classification_counts": dict(
            sorted(
                (label, sum(row["response_class"] == label for row in summary_rows))
                for label in sorted({str(row["response_class"]) for row in summary_rows})
            )
        ),
        "block_uncertainty_is_within_trajectory_not_replicate_uncertainty": True,
        "velocity_acf_is_diagnostic_not_an_equilibrium_transport_coefficient": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
