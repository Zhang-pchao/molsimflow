"""Bounded robustness audit for the high-cadence TPCL observables.

The audit extends the synthetic detector challenge to slower and larger
responses and repeats the spatial/network analysis under four one-at-a-time
region definitions.  It consumes existing trajectories only.  Its scientific
gate is deliberately conservative: parameter stability in one parent history
cannot establish a causal depinning mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.tpcl_force_step_analysis import discover_runs, select_kinematic_events
from molsimflow.postprocess.tpcl_observable_qualification import (
    MOTION_FIELDS,
    NETWORK_FIELDS,
    _read_tsv,
    _sha256,
    _write_tsv,
    build_detector_trace,
    extract_high_cadence_regions,
    paired_region_summary,
    summarize_regions,
)


REGION_CONFIGURATIONS = (
    {
        "configuration": "primary",
        "contact_height_A": 5.0,
        "edge_tail_fraction": 0.10,
        "oo_cutoff_A": 3.5,
        "source": "accepted_stage_a",
    },
    {
        "configuration": "contact_height_4A",
        "contact_height_A": 4.0,
        "edge_tail_fraction": 0.10,
        "oo_cutoff_A": 3.5,
        "source": "recomputed",
    },
    {
        "configuration": "contact_height_6A",
        "contact_height_A": 6.0,
        "edge_tail_fraction": 0.10,
        "oo_cutoff_A": 3.5,
        "source": "recomputed",
    },
    {
        "configuration": "tail_fraction_0p20",
        "contact_height_A": 5.0,
        "edge_tail_fraction": 0.20,
        "oo_cutoff_A": 3.5,
        "source": "recomputed",
    },
    {
        "configuration": "oo_cutoff_3p2A",
        "contact_height_A": 5.0,
        "edge_tail_fraction": 0.10,
        "oo_cutoff_A": 3.2,
        "source": "recomputed",
    },
)

PAIRED_METRICS = (
    "paired_delta_persistent_mean_relative_slip_rate_A_per_ps",
    "paired_delta_mean_water_hbond_count",
    "paired_delta_mean_surface_hbond_count",
    "paired_delta_mean_sioh_surface_hbond_count",
    "paired_delta_water_hbond_formed_persistent_rate_per_ps",
    "paired_delta_water_hbond_broken_persistent_rate_per_ps",
    "paired_delta_water_hbond_formed_membership_rate_per_ps",
    "paired_delta_water_hbond_broken_membership_rate_per_ps",
    "paired_delta_surface_hbond_formed_persistent_rate_per_ps",
    "paired_delta_surface_hbond_broken_persistent_rate_per_ps",
)


def _sign(value: float, tolerance: float = 1.0e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def extended_detector_grid(
    reference_analysis: Path,
    *,
    amplitudes: Sequence[float] = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0),
    durations: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0),
    centers: Sequence[float] = (10.0, 35.0, 65.0, 90.0),
    modes: Sequence[str] = ("coherent", "leading_only", "trailing_only", "opposing", "retreat"),
    cases: Sequence[str] = ("ch3_only", "mixed291"),
    axes: Sequence[str] = ("x", "y"),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Challenge the accepted detector over a bounded extended grid."""

    trials: list[dict[str, object]] = []
    for case_id in cases:
        base_rows = _read_tsv(Path(reference_analysis) / "02_kinematics" / f"{case_id}__f0_shared.tsv")
        for axis in axes:
            for amplitude in amplitudes:
                for duration in durations:
                    for center in centers:
                        for mode in modes:
                            trace = build_detector_trace(
                                base_rows,
                                axis,
                                amplitude_A=float(amplitude),
                                duration_ps=float(duration),
                                center_ps=float(center),
                                mode=mode,
                            )
                            events, _, diagnostic = select_kinematic_events(trace)
                            tolerance = max(1.0, float(duration))
                            matched = [
                                event
                                for event in events
                                if abs(float(event["peak_time_ps"]) - float(center)) <= tolerance
                            ]
                            expected = mode in {"coherent", "leading_only", "trailing_only"} and amplitude > 0
                            trials.append(
                                {
                                    "case_id": case_id,
                                    "axis": axis,
                                    "mode": mode,
                                    "amplitude_A": float(amplitude),
                                    "duration_ps": float(duration),
                                    "center_ps": float(center),
                                    "expected_positive_center_advance": expected,
                                    "recovered": bool(matched),
                                    "matched_peak_time_ps": float(matched[0]["peak_time_ps"]) if matched else math.nan,
                                    "detected_event_count": len(events),
                                    "false_positive_count": len(events) - len(matched),
                                    "event_threshold_A_per_ps": float(diagnostic["event_threshold_A_per_ps"]),
                                }
                            )
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in trials:
        key = (
            row["case_id"],
            row["axis"],
            row["mode"],
            row["amplitude_A"],
            row["duration_ps"],
        )
        grouped[key].append(row)
    envelope: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        recovery = float(np.mean([bool(row["recovered"]) for row in values]))
        expected = bool(values[0]["expected_positive_center_advance"])
        envelope.append(
            {
                "case_id": key[0],
                "axis": key[1],
                "mode": key[2],
                "amplitude_A": key[3],
                "duration_ps": key[4],
                "trials": len(values),
                "expected_positive_center_advance": expected,
                "recovery_fraction": recovery,
                "mean_false_positive_count": float(
                    np.mean([int(row["false_positive_count"]) for row in values])
                ),
                "maximum_threshold_A_per_ps": max(
                    float(row["event_threshold_A_per_ps"]) for row in values
                ),
                "qualified_at_75pct_recovery": expected and recovery >= 0.75,
            }
        )
    return trials, envelope


def minimum_qualified_amplitudes(
    envelope: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in envelope:
        if bool(row["expected_positive_center_advance"]):
            key = (
                str(row["case_id"]),
                str(row["axis"]),
                str(row["mode"]),
                float(row["duration_ps"]),
            )
            grouped[key].append(row)
    result: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        qualified = [
            float(row["amplitude_A"])
            for row in rows
            if bool(row["qualified_at_75pct_recovery"])
        ]
        result.append(
            {
                "case_id": key[0],
                "axis": key[1],
                "mode": key[2],
                "duration_ps": key[3],
                "minimum_qualified_amplitude_A": min(qualified) if qualified else math.nan,
                "qualified": bool(qualified),
            }
        )
    return result


def membership_fraction_rows(
    configuration: str,
    summaries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in summaries:
        branch = str(row["branch_id"])
        axis = str(row["axis"])
        if branch not in {"f0_shared", f"f8e-5_{axis}"}:
            continue
        persistent = sum(
            float(row[field])
            for field in (
                "water_hbond_formed_persistent_rate_per_ps",
                "water_hbond_broken_persistent_rate_per_ps",
            )
        )
        membership = sum(
            float(row[field])
            for field in (
                "water_hbond_formed_membership_rate_per_ps",
                "water_hbond_broken_membership_rate_per_ps",
            )
        )
        total = persistent + membership
        output.append(
            {
                "configuration": configuration,
                "case_id": row["case_id"],
                "branch_id": branch,
                "axis": axis,
                "edge": row["edge"],
                "persistent_turnover_rate_per_ps": persistent,
                "membership_turnover_rate_per_ps": membership,
                "membership_fraction_of_apparent_turnover": membership / total if total > 0 else math.nan,
            }
        )
    return output


def aggregate_paired_sensitivity(
    paired_by_configuration: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    long_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for configuration, rows in paired_by_configuration.items():
        for row in rows:
            for metric in PAIRED_METRICS:
                value = float(row[metric])
                item = {
                    "configuration": configuration,
                    "case_id": row["case_id"],
                    "axis": row["axis"],
                    "edge": row["edge"],
                    "metric": metric,
                    "value": value,
                    "sign": _sign(value),
                }
                long_rows.append(item)
                grouped[
                    (str(row["case_id"]), str(row["axis"]), str(row["edge"]), metric)
                ].append(item)
    summary: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        values = [float(row["value"]) for row in rows]
        signs = [str(row["sign"]) for row in rows]
        primary = next(float(row["value"]) for row in rows if row["configuration"] == "primary")
        summary.append(
            {
                "case_id": key[0],
                "axis": key[1],
                "edge": key[2],
                "metric": key[3],
                "configuration_count": len(rows),
                "primary_value": primary,
                "minimum_value": min(values),
                "maximum_value": max(values),
                "positive_count": signs.count("positive"),
                "negative_count": signs.count("negative"),
                "zero_count": signs.count("zero"),
                "same_nonzero_sign_across_configurations": len(set(signs)) == 1 and signs[0] != "zero",
            }
        )
    return long_rows, summary


def paired_block_slip(
    configuration: str,
    rows: Sequence[Mapping[str, object]],
    *,
    window_ps: float = 20.0,
    block_ps: float = 5.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compare forced and F0 persistent-member slip in fixed non-overlapping blocks."""

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["case_id"]),
                str(row["branch_id"]),
                str(row["axis"]),
                str(row["edge"]),
            )
        ].append(row)
    block_rows: list[dict[str, object]] = []
    for key, forced_rows in sorted(grouped.items()):
        case_id, branch_id, axis, edge = key
        if branch_id != f"f8e-5_{axis}":
            continue
        baseline_rows = grouped[(case_id, "f0_shared", axis, edge)]
        for block_index, lower in enumerate(np.arange(0.0, window_ps, block_ps)):
            upper = min(float(lower + block_ps), window_ps)

            def rate(sample: Sequence[Mapping[str, object]]) -> float:
                values = [
                    float(row["persistent_mean_relative_delta_A"])
                    for row in sample
                    if lower < float(row["time_ps"]) <= upper
                    and math.isfinite(float(row["persistent_mean_relative_delta_A"]))
                ]
                return float(np.sum(values)) / (upper - lower)

            forced_rate = rate(forced_rows)
            baseline_rate = rate(baseline_rows)
            delta = forced_rate - baseline_rate
            block_rows.append(
                {
                    "configuration": configuration,
                    "case_id": case_id,
                    "axis": axis,
                    "edge": edge,
                    "block_index": block_index,
                    "start_ps": float(lower),
                    "end_ps": upper,
                    "forced_rate_A_per_ps": forced_rate,
                    "f0_rate_A_per_ps": baseline_rate,
                    "paired_delta_rate_A_per_ps": delta,
                    "sign": _sign(delta),
                }
            )
    grouped_blocks: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in block_rows:
        grouped_blocks[
            (
                str(row["configuration"]),
                str(row["case_id"]),
                str(row["axis"]),
                str(row["edge"]),
            )
        ].append(row)
    summary: list[dict[str, object]] = []
    for key, sample in sorted(grouped_blocks.items()):
        signs = [str(row["sign"]) for row in sample]
        summary.append(
            {
                "configuration": key[0],
                "case_id": key[1],
                "axis": key[2],
                "edge": key[3],
                "block_count": len(sample),
                "positive_blocks": signs.count("positive"),
                "negative_blocks": signs.count("negative"),
                "zero_blocks": signs.count("zero"),
                "same_nonzero_sign_all_blocks": len(set(signs)) == 1 and signs[0] != "zero",
            }
        )
    return block_rows, summary


def _plot_results(
    output: Path,
    detector_envelope: Sequence[Mapping[str, object]],
    sensitivity_long: Sequence[Mapping[str, object]],
    membership_rows: Sequence[Mapping[str, object]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_paths: list[Path] = []
    cases_axes = (("ch3_only", "x"), ("ch3_only", "y"), ("mixed291", "x"), ("mixed291", "y"))
    durations = sorted({float(row["duration_ps"]) for row in detector_envelope})
    amplitudes = sorted(
        {float(row["amplitude_A"]) for row in detector_envelope if float(row["amplitude_A"]) > 0}
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), sharex=True, sharey=True)
    image = None
    for panel, (case_id, axis_name) in zip(axes.flat, cases_axes):
        lookup = {
            (float(row["duration_ps"]), float(row["amplitude_A"])): float(row["recovery_fraction"])
            for row in detector_envelope
            if row["case_id"] == case_id and row["axis"] == axis_name and row["mode"] == "coherent"
            and float(row["amplitude_A"]) > 0
        }
        matrix = np.asarray([[lookup[(duration, amplitude)] for duration in durations] for amplitude in amplitudes])
        image = panel.imshow(matrix, origin="lower", vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
        panel.set_title(f"{case_id} / {axis_name.upper()}")
        panel.set_xticks(range(len(durations)), [f"{value:g}" for value in durations], rotation=45)
        panel.set_yticks(range(len(amplitudes)), [f"{value:g}" for value in amplitudes])
        panel.set_xlabel("Duration (ps)")
        panel.set_ylabel("Amplitude (A)")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Recovery fraction")
    figure.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.93, wspace=0.25, hspace=0.30)
    path = output / "05_figures" / "extended_coherent_detector_recovery.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figure_paths.append(path)

    configurations = [str(item["configuration"]) for item in REGION_CONFIGURATIONS]
    slip = [
        row
        for row in sensitivity_long
        if row["metric"] == "paired_delta_persistent_mean_relative_slip_rate_A_per_ps"
    ]
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    keys = sorted({(str(row["case_id"]), str(row["axis"]), str(row["edge"])) for row in slip})
    for key in keys:
        lookup = {
            str(row["configuration"]): float(row["value"])
            for row in slip
            if (str(row["case_id"]), str(row["axis"]), str(row["edge"])) == key
        }
        axis.plot(range(len(configurations)), [lookup[name] for name in configurations], marker="o", label="/".join(key))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(configurations)), configurations, rotation=25, ha="right")
    axis.set_ylabel("Forced - F0 persistent slip rate (A/ps)")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    path = output / "05_figures" / "persistent_slip_region_sensitivity.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figure_paths.append(path)

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    data = [
        [
            float(row["membership_fraction_of_apparent_turnover"])
            for row in membership_rows
            if row["configuration"] == configuration
        ]
        for configuration in configurations
    ]
    axis.boxplot(data, labels=configurations, showmeans=True)
    axis.set_ylabel("Membership fraction of apparent water-pair turnover")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    path = output / "05_figures" / "membership_fraction_region_sensitivity.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figure_paths.append(path)
    return figure_paths


def analyze_robustness(
    package_root: Path,
    reference_analysis: Path,
    primary_analysis: Path,
    output_dir: Path,
    *,
    window_ps: float = 20.0,
    block_ps: float = 5.0,
) -> dict[str, object]:
    root = Path(package_root).resolve()
    reference = Path(reference_analysis).resolve()
    primary = Path(primary_analysis).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"immutable output already exists: {output}")
    for name in (
        "00_contract",
        "01_inputs",
        "02_detector_extension",
        "03_region_sensitivity",
        "04_robustness",
        "05_figures",
        "06_review",
        "07_validation",
    ):
        (output / name).mkdir(parents=True, exist_ok=False)

    contract = {
        "stage": "Stage A2: Existing-trajectory observable robustness",
        "coordinate_window_ps": window_ps,
        "block_ps": block_ps,
        "detector_amplitudes_A": [0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0],
        "detector_durations_ps": [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0],
        "region_configurations": list(REGION_CONFIGURATIONS),
        "design": "one-at-a-time sensitivity; no full factorial search",
        "trajectory_scope": "existing six high-cadence trajectories only",
        "independence_warning": "region definitions and time blocks do not create independent contact histories",
        "scientific_boundary": "robustness can reject fragile observables but cannot establish causality",
    }
    (output / "00_contract" / "ROBUSTNESS-CONTRACT.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    input_paths = [
        root / "04_jobs" / "SUBMISSION.tsv",
        primary / "08_validation" / "VALIDATION.json",
        primary / "08_validation" / "OUTPUT-SHA256SUMS",
        primary / "02_motion" / "region_summary.tsv",
        primary / "02_motion" / "paired_region_summary.tsv",
        reference / "02_kinematics" / "ch3_only__f0_shared.tsv",
        reference / "02_kinematics" / "mixed291__f0_shared.tsv",
    ]
    input_paths.extend(sorted((primary / "02_motion").glob("*__*.tsv")))
    input_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in dict.fromkeys(input_paths)
    ]
    _write_tsv(output / "01_inputs" / "INPUT-MANIFEST.tsv", input_rows)

    detector_trials, detector_envelope = extended_detector_grid(reference)
    detector_minimum = minimum_qualified_amplitudes(detector_envelope)
    _write_tsv(output / "02_detector_extension" / "injection_recovery_trials.tsv", detector_trials)
    _write_tsv(output / "02_detector_extension" / "detector_operating_envelope.tsv", detector_envelope)
    _write_tsv(output / "02_detector_extension" / "minimum_qualified_amplitude.tsv", detector_minimum)

    runs = discover_runs(root)
    summaries_by_configuration: dict[str, list[dict[str, object]]] = {}
    paired_by_configuration: dict[str, list[dict[str, object]]] = {}
    raw_by_configuration: dict[str, list[Mapping[str, object]]] = {}

    primary_summaries = _read_tsv(primary / "02_motion" / "region_summary.tsv")
    primary_paired = _read_tsv(primary / "02_motion" / "paired_region_summary.tsv")
    primary_raw: list[Mapping[str, object]] = []
    for path in sorted((primary / "02_motion").glob("*__*.tsv")):
        if path.name in {"region_summary.tsv", "paired_region_summary.tsv"}:
            continue
        primary_raw.extend(_read_tsv(path))
    summaries_by_configuration["primary"] = list(primary_summaries)
    paired_by_configuration["primary"] = list(primary_paired)
    raw_by_configuration["primary"] = primary_raw
    primary_dir = output / "03_region_sensitivity" / "primary"
    primary_dir.mkdir()
    _write_tsv(primary_dir / "region_summary.tsv", primary_summaries)
    _write_tsv(primary_dir / "paired_region_summary.tsv", primary_paired)
    (primary_dir / "SOURCE.txt").write_text(str(primary) + "\n", encoding="utf-8")

    for configuration in REGION_CONFIGURATIONS[1:]:
        name = str(configuration["configuration"])
        target = output / "03_region_sensitivity" / name
        (target / "02_motion").mkdir(parents=True)
        (target / "03_network").mkdir()
        all_rows: list[Mapping[str, object]] = []
        all_summaries: list[dict[str, object]] = []
        for run in runs:
            rows, lifetimes = extract_high_cadence_regions(
                run,
                window_ps=window_ps,
                contact_height_A=float(configuration["contact_height_A"]),
                edge_tail_fraction=float(configuration["edge_tail_fraction"]),
                oo_cutoff_A=float(configuration["oo_cutoff_A"]),
            )
            filename = f"{run.case_id}__{run.branch_id}.tsv"
            _write_tsv(target / "02_motion" / filename, rows, MOTION_FIELDS)
            _write_tsv(target / "03_network" / filename, rows, NETWORK_FIELDS)
            _write_tsv(target / "03_network" / filename.replace(".tsv", "__lifetimes.tsv"), lifetimes)
            all_rows.extend(rows)
            all_summaries.extend(summarize_regions(rows, lifetimes))
        paired = paired_region_summary(all_summaries)
        _write_tsv(target / "region_summary.tsv", all_summaries)
        _write_tsv(target / "paired_region_summary.tsv", paired)
        summaries_by_configuration[name] = all_summaries
        paired_by_configuration[name] = paired
        raw_by_configuration[name] = all_rows

    sensitivity_long, sensitivity_summary = aggregate_paired_sensitivity(paired_by_configuration)
    membership_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    block_summary: list[dict[str, object]] = []
    for configuration in [str(item["configuration"]) for item in REGION_CONFIGURATIONS]:
        membership_rows.extend(
            membership_fraction_rows(configuration, summaries_by_configuration[configuration])
        )
        blocks, blocks_summary = paired_block_slip(
            configuration,
            raw_by_configuration[configuration],
            window_ps=window_ps,
            block_ps=block_ps,
        )
        block_rows.extend(blocks)
        block_summary.extend(blocks_summary)
    _write_tsv(output / "04_robustness" / "paired_metric_sensitivity_long.tsv", sensitivity_long)
    _write_tsv(output / "04_robustness" / "paired_metric_sign_robustness.tsv", sensitivity_summary)
    _write_tsv(output / "04_robustness" / "membership_fraction_sensitivity.tsv", membership_rows)
    _write_tsv(output / "04_robustness" / "paired_block_slip.tsv", block_rows)
    _write_tsv(output / "04_robustness" / "paired_block_sign_summary.tsv", block_summary)

    figures = _plot_results(output, detector_envelope, sensitivity_long, membership_rows)
    baseline_trials = [row for row in detector_trials if float(row["amplitude_A"]) == 0.0]
    positive_cells = [row for row in detector_envelope if bool(row["expected_positive_center_advance"])]
    qualified_cells = [row for row in positive_cells if bool(row["qualified_at_75pct_recovery"])]
    slow_cells = [row for row in positive_cells if float(row["duration_ps"]) >= 2.0]
    slow_qualified = [row for row in slow_cells if bool(row["qualified_at_75pct_recovery"])]
    slip_summary = [
        row
        for row in sensitivity_summary
        if row["metric"] == "paired_delta_persistent_mean_relative_slip_rate_A_per_ps"
    ]
    stable_slip_keys = sum(bool(row["same_nonzero_sign_across_configurations"]) for row in slip_summary)
    stable_blocks = sum(bool(row["same_nonzero_sign_all_blocks"]) for row in block_summary)
    fractions = [float(row["membership_fraction_of_apparent_turnover"]) for row in membership_rows]
    validation = {
        "status": "PASS",
        "run_count": len(runs),
        "configuration_count": len(REGION_CONFIGURATIONS),
        "detector_trial_count": len(detector_trials),
        "detector_positive_operating_cells": len(positive_cells),
        "detector_qualified_operating_cells": len(qualified_cells),
        "detector_slow_positive_operating_cells": len(slow_cells),
        "detector_slow_qualified_operating_cells": len(slow_qualified),
        "baseline_false_positive_trial_fraction": float(
            np.mean([int(row["detected_event_count"]) > 0 for row in baseline_trials])
        ),
        "persistent_slip_keys_same_sign_across_all_configurations": stable_slip_keys,
        "persistent_slip_key_count": len(slip_summary),
        "configuration_key_pairs_with_same_sign_all_time_blocks": stable_blocks,
        "configuration_key_pair_count": len(block_summary),
        "membership_fraction_minimum": min(fractions),
        "membership_fraction_maximum": max(fractions),
        "figure_count": len(figures),
        "method_gate": "PASS_BOUNDED_ONE_AT_A_TIME_SENSITIVITY",
        "estimator_gate": "PASS_EXTENDED_OPERATING_ENVELOPE_MEASURED_NOT_UNIVERSAL",
        "scientific_gate": "MECHANISM_NOT_ESTABLISHED_SINGLE_HISTORY",
        "next_action": "REQUIRES_NEW_MD_FOR_CAUSAL_CLOSURE",
    }
    (output / "07_validation" / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Stage A2 robustness review",
        "",
        f"- Pipeline: `{validation['status']}`",
        f"- Method gate: `{validation['method_gate']}`",
        f"- Estimator gate: `{validation['estimator_gate']}`",
        f"- Scientific gate: `{validation['scientific_gate']}`",
        f"- Next action: `{validation['next_action']}`",
        f"- Qualified detector cells: `{len(qualified_cells)}/{len(positive_cells)}`.",
        f"- Qualified slow cells (2-10 ps): `{len(slow_qualified)}/{len(slow_cells)}`.",
        f"- Persistent-slip keys with one sign across all region definitions: `{stable_slip_keys}/{len(slip_summary)}`.",
        f"- Configuration/key pairs with one sign across all time blocks: `{stable_blocks}/{len(block_summary)}`.",
        f"- Membership contribution range: `{min(fractions):.3f}-{max(fractions):.3f}`.",
        "",
        "The detector extension separates a sharp-event operating envelope from slow distributed motion. Non-recovery remains an estimator limitation, not proof that motion is absent.",
        "",
        "The one-at-a-time region audit tests whether the principal signs survive reasonable contact-height, edge-width, and O-O cutoff choices. Even a stable sign in this audit is repeated measurement of the same parent history, not an independent causal validation.",
        "",
        "This package exhausts the pre-registered robustness analyses available from the present high-cadence trajectories. A positive water-network depinning mechanism requires new matched force-reversal trajectories across independent contact histories.",
    ]
    (output / "06_review" / "STAGE-A2-ROBUSTNESS-REVIEW.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    hashed = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "OUTPUT-SHA256SUMS":
            hashed.append(f"{_sha256(path)}  {path.relative_to(output)}")
    (output / "07_validation" / "OUTPUT-SHA256SUMS").write_text(
        "\n".join(hashed) + "\n", encoding="utf-8"
    )
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
    parser.add_argument("--primary-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-ps", type=float, default=20.0)
    parser.add_argument("--block-ps", type=float, default=5.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.window_ps <= 0 or args.block_ps <= 0 or args.window_ps % args.block_ps != 0:
        raise ValueError("window and block must be positive and form complete blocks")
    result = analyze_robustness(
        args.package_root,
        args.reference_analysis,
        args.primary_analysis,
        args.output_dir,
        window_ps=args.window_ps,
        block_ps=args.block_ps,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
