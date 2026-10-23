"""Optional water trajectory diagnostics; no case-specific inputs or defaults."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .alignment import time_values_to_fs, time_window_mask
from .estimators import alternate_estimator, diagnostic_gap_metrics
from .io import dump_frames, field, read_plumed, read_thermo
from .observables import soft_voronoi_occupancies
from .reference import run_reference
from .reporting import estimator_plot_labels, sampling_protocol_label, write_csv, write_manifest
from .types import AlignedPaths, JointResults, PreparedInputs, WeightedPaths
from .units import EV_TO_KCAL_MOL


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def reconstruction_within_tolerance(error: float, tolerance: float | None) -> bool:
    return tolerance is None or float(error) <= float(tolerance)


def ring_polymer_spread(
    paths: Sequence[Path], selected_steps: Sequence[int], type_labels: Mapping[int, str]
) -> List[Dict[str, float]]:
    wanted = {int(step) for step in selected_steps}
    require(bool(wanted), "no requested trajectory steps")
    first_step = min(wanted)
    last_step = max(wanted)
    generators = [dump_frames(path) for path in paths]
    output: List[Dict[str, float]] = []
    while True:
        frames = []
        for generator in generators:
            try:
                frames.append(next(generator))
            except StopIteration:
                frames = []
                break
        if not frames:
            break
        steps = {frame[0] for frame in frames}
        require(len(steps) == 1, "trajectory timestep mismatch")
        step = frames[0][0]
        require(
            all(np.array_equal(frame[2], frames[0][2]) for frame in frames), "atom order mismatch"
        )
        require(
            all(np.array_equal(frame[3], frames[0][3]) for frame in frames), "atom type mismatch"
        )
        if step < int(first_step):
            continue
        if step > int(last_step):
            break
        if step not in wanted:
            continue
        box = frames[0][1][:, 1] - frames[0][1][:, 0]
        positions = np.asarray([frame[4] for frame in frames])
        reference = positions[0]
        delta = positions - reference[None, :, :]
        delta -= box[None, None, :] * np.rint(delta / box[None, None, :])
        unwrapped = reference[None, :, :] + delta
        centroid = np.mean(unwrapped, axis=0)
        rg2 = np.mean(np.sum((unwrapped - centroid[None, :, :]) ** 2, axis=2), axis=0)
        row: Dict[str, float] = {
            "step": float(step),
            "rg_all_A": float(math.sqrt(float(np.mean(rg2)))),
            "rg_p95_A": float(math.sqrt(float(np.percentile(rg2, 95.0)))),
            "rg_max_A": float(math.sqrt(float(np.max(rg2)))),
        }
        types = frames[0][3]
        for atom_type, label in sorted(type_labels.items()):
            mask = types == int(atom_type)
            require(bool(np.any(mask)), f"atom type absent from trajectory: {atom_type}")
            row[f"rg_{label}_A"] = float(math.sqrt(float(np.mean(rg2[mask]))))
        output.append(row)
    observed = {int(row["step"]) for row in output}
    require(observed == wanted, "selected trajectory steps missing")
    return output


def ionization_candidate_details(
    paths: Sequence[Path],
    selected_steps: np.ndarray,
    time_ps: np.ndarray,
    centroid_values: np.ndarray,
    bead_values: np.ndarray,
    type_labels: Mapping[int, str],
    *,
    kappa: float,
    reference: float,
    minimum_score: float,
    reconstruction_tolerance: float | None,
    sampling_representation: str = "centroid",
) -> List[Dict[str, object]]:
    require(bead_values.shape == (len(selected_steps), len(paths)), "candidate bead shape mismatch")
    center_types = [key for key, value in type_labels.items() if value == "O"]
    assigned_types = [key for key, value in type_labels.items() if value == "H"]
    require(len(center_types) == 1 and len(assigned_types) == 1, "unique H/O type labels required")
    step_to_index = {int(step): index for index, step in enumerate(selected_steps)}
    wanted = {
        int(selected_steps[index])
        for index in range(len(selected_steps))
        if centroid_values[index] >= minimum_score or np.any(bead_values[index] >= minimum_score)
    }
    generators = [dump_frames(path) for path in paths]
    output: List[Dict[str, object]] = []
    while wanted:
        frames = []
        for generator in generators:
            try:
                frames.append(next(generator))
            except StopIteration:
                frames = []
                break
        if not frames:
            break
        steps = {frame[0] for frame in frames}
        require(len(steps) == 1, "candidate trajectory timestep mismatch")
        step = int(frames[0][0])
        if step not in wanted:
            continue
        index = step_to_index[step]
        require(
            all(np.array_equal(frame[2], frames[0][2]) for frame in frames),
            "candidate atom order mismatch",
        )
        require(
            all(np.array_equal(frame[3], frames[0][3]) for frame in frames),
            "candidate atom type mismatch",
        )
        ids = frames[0][2]
        types = frames[0][3]
        box = frames[0][1][:, 1] - frames[0][1][:, 0]
        positions = np.asarray([frame[4] for frame in frames])
        reference_positions = positions[0]
        delta = positions - reference_positions[None, :, :]
        delta -= box[None, None, :] * np.rint(delta / box[None, None, :])
        centroid_positions = reference_positions + np.mean(delta, axis=0)
        representations = []
        if sampling_representation == "centroid":
            representations.append(
                (sampling_representation, 0, centroid_values[index], centroid_positions)
            )
        representations.extend(
            (f"bead_{bead + 1}", bead + 1, bead_values[index, bead], positions[bead])
            for bead in range(len(paths))
        )
        center_ids = ids[types == int(center_types[0])]
        for label, bead, recorded, current_positions in representations:
            if float(recorded) < float(minimum_score):
                continue
            occupancies, hard_counts = soft_voronoi_occupancies(
                current_positions, types, box, center_types[0], assigned_types[0], kappa
            )
            score = float(np.sum((occupancies - float(reference)) ** 2))
            error = abs(score - float(recorded))
            require(
                reconstruction_within_tolerance(error, reconstruction_tolerance),
                f"ionization reconstruction mismatch at step {step} {label}",
            )
            minimum = int(np.argmin(occupancies))
            maximum = int(np.argmax(occupancies))
            output.append(
                {
                    "time_ps": float(time_ps[index]),
                    "step": step,
                    "representation": label,
                    "bead": bead,
                    "recorded_score": float(recorded),
                    "reconstructed_score": score,
                    "absolute_reconstruction_error": error,
                    "minimum_soft_occupancy": float(occupancies[minimum]),
                    "minimum_soft_occupancy_O_id": int(center_ids[minimum]),
                    "maximum_soft_occupancy": float(occupancies[maximum]),
                    "maximum_soft_occupancy_O_id": int(center_ids[maximum]),
                    "minimum_hard_H_count": int(np.min(hard_counts)),
                    "maximum_hard_H_count": int(np.max(hard_counts)),
                    "hard_undercoordinated_centers": int(np.count_nonzero(hard_counts <= 1)),
                    "hard_overcoordinated_centers": int(np.count_nonzero(hard_counts >= 3)),
                    "hard_pair_like": int(np.any(hard_counts <= 1) and np.any(hard_counts >= 3)),
                }
            )
        wanted.remove(step)
    require(not wanted, "candidate trajectory frames missing")
    return output


def threshold_run_rows(
    time_ps: np.ndarray,
    series: Mapping[str, np.ndarray],
    thresholds: Sequence[float],
) -> List[Dict[str, object]]:
    stride_ps = float(np.median(np.diff(time_ps)))
    rows: List[Dict[str, object]] = []
    for label, values in series.items():
        values = np.asarray(values, dtype=float)
        for threshold in thresholds:
            mask = values >= float(threshold)
            runs: List[Tuple[int, int]] = []
            start: int | None = None
            for index, active in enumerate(mask):
                if active and start is None:
                    start = index
                if start is not None and (not active or index == len(mask) - 1):
                    stop = index if active and index == len(mask) - 1 else index - 1
                    runs.append((start, stop))
                    start = None
            longest = max((stop - start + 1 for start, stop in runs), default=0)
            rows.append(
                {
                    "representation": label,
                    "threshold": float(threshold),
                    "frames_at_or_above": int(np.count_nonzero(mask)),
                    "fraction_frames": float(np.mean(mask)),
                    "contiguous_runs": len(runs),
                    "longest_run_frames": longest,
                    "longest_run_ps": longest * stride_ps,
                }
            )
    return rows


def finalize_water(
    contract: Mapping[str, object],
    output: Path,
    profile: str,
    inputs: PreparedInputs,
    aligned: AlignedPaths,
    weighted: WeightedPaths,
    joint: JointResults,
    derived_coordinate_summary,
    filtered_colvar: Path,
    _plotter,
):
    source = inputs.source
    selection = inputs.selection
    reweight = inputs.reweight
    bias_mode = inputs.bias_mode
    primary_estimator = inputs.primary_estimator
    kernel_time_scale_to_fs = inputs.kernel_time_scale_to_fs
    run_root = inputs.run_root
    restart_policy = aligned.restart_policy
    centroid_restart_duplicates = aligned.centroid_restart_duplicates
    bead_restart_duplicates = aligned.bead_restart_duplicates
    first_time_fs = aligned.first_time_fs
    last_time_fs = aligned.last_time_fs
    selected = aligned.selected
    selected_time_ps = aligned.selected_time_ps
    selected_steps = aligned.selected_steps
    cv_names = aligned.cv_names
    sampling_cv_names = aligned.sampling_cv_names
    bead_cv_names = aligned.bead_cv_names
    sampling_label = aligned.sampling_label
    sampling_slug = aligned.sampling_slug
    cv_labels = aligned.cv_labels
    centroid = aligned.centroid
    beads = aligned.beads
    diagnostic_spec = aligned.diagnostic_spec
    diagnostic_sampling = aligned.diagnostic_sampling
    diagnostic_beads = aligned.diagnostic_beads
    diagnostic_mean_error = aligned.diagnostic_mean_error
    weight_kind = weighted.weight_kind
    bias_column = weighted.bias_column
    bias_ev = weighted.bias_ev
    opes_bias_ev = weighted.opes_bias_ev
    extra_bias_columns = weighted.extra_bias_columns
    rct_ev = weighted.rct_ev
    zed = weighted.zed
    neff = weighted.neff
    nker = weighted.nker
    shared_diagnostic_max_deltas = weighted.shared_diagnostic_max_deltas
    kbt_ev = weighted.kbt_ev
    raw_log_weights = weighted.raw_log_weights
    weights = weighted.weights
    quasi_static_declared = weighted.quasi_static_declared
    x_grid = joint.x_grid
    y_grid = joint.y_grid
    variants = joint.variants
    primary_name = joint.primary_name
    primary_supports = joint.primary_supports
    primary_support = joint.primary_support
    primary = joint.primary
    primary_kcal = joint.primary_kcal
    supports_by_variant = joint.supports_by_variant
    kernels_fields, kernels = read_plumed(run_root / source["kernels"])
    kernel_time_fs = time_values_to_fs(
        field(kernels, kernels_fields, "time"), kernel_time_scale_to_fs
    )
    kernel_mask = time_window_mask(kernel_time_fs, first_time_fs, last_time_fs)
    require(bool(np.any(kernel_mask)), "no KERNELS rows in selected time window")
    kernel_time_ps = kernel_time_fs[kernel_mask] / 1000.0
    sigma_x = field(kernels, kernels_fields, f"sigma_{sampling_cv_names[0]}")[kernel_mask]
    sigma_y = field(kernels, kernels_fields, f"sigma_{sampling_cv_names[1]}")[kernel_mask]

    thermo_maps = [read_thermo(run_root / value) for value in source["thermo_logs"]]
    require(
        all(all(int(step) in rows for step in selected_steps) for rows in thermo_maps),
        "thermo steps missing",
    )
    temperatures = np.asarray(
        [[rows[int(step)]["Temp"] for rows in thermo_maps] for step in selected_steps]
    )
    potential = np.asarray(
        [[rows[int(step)]["PotEng"] for rows in thermo_maps] for step in selected_steps]
    )
    spring_per_bead = np.asarray(
        [[rows[int(step)]["f_fpimd[2]"] for rows in thermo_maps] for step in selected_steps]
    )
    global_values = {
        name: np.asarray([thermo_maps[0][int(step)][name] for step in selected_steps])
        for name in ("f_fpimd[4]", "f_fpimd[5]", "f_fpimd[6]", "f_fpimd[7]")
    }
    global_values["spring_per_bead"] = spring_per_bead
    write_csv(
        output / "tables" / "pimd-thermo-long.csv",
        (
            {
                "time_ps": selected_time_ps[frame],
                "bead": bead + 1,
                "temperature_raw_K": temperatures[frame, bead],
                "temperature_scaled_by_P_K": temperatures[frame, bead] / beads.shape[1],
                "potential_energy_eV": potential[frame, bead],
                "kinetic_energy_eV": thermo_maps[bead][int(selected_steps[frame])]["f_fpimd[1]"],
                "spring_energy_eV": spring_per_bead[frame, bead],
                "primitive_ke_estimator_eV": global_values["f_fpimd[5]"][frame],
                "virial_ke_estimator_eV": global_values["f_fpimd[6]"][frame],
                "centroid_virial_ke_estimator_eV": global_values["f_fpimd[7]"][frame],
            }
            for frame in range(len(selected_steps))
            for bead in range(beads.shape[1])
        ),
        [
            "time_ps",
            "bead",
            "temperature_raw_K",
            "temperature_scaled_by_P_K",
            "potential_energy_eV",
            "kinetic_energy_eV",
            "spring_energy_eV",
            "primitive_ke_estimator_eV",
            "virial_ke_estimator_eV",
            "centroid_virial_ke_estimator_eV",
        ],
    )

    trajectory_paths = [run_root / value for value in source["trajectories"]]
    type_labels = {int(key): str(value) for key, value in contract["system"]["type_labels"].items()}
    spread_rows = ring_polymer_spread(
        trajectory_paths,
        selected_steps,
        type_labels,
    )
    require(len(spread_rows) == len(selected_steps), "ring-polymer spread frame mismatch")
    spread_fields = list(spread_rows[0])
    write_csv(output / "tables" / "ring-polymer-spread.csv", spread_rows, spread_fields)

    ionization_component = cv_names.index("ionization")
    ionization_config = contract.get("ionization_diagnostic", {})
    ionization_thresholds = tuple(
        float(value) for value in ionization_config.get("thresholds", [0.5, 1.0, 1.5, 2.0])
    )
    ideal_pair_score = float(ionization_config.get("ideal_localized_pair_score", 2.0))
    reconstruction_tolerance_value = ionization_config.get("reconstruction_tolerance", 1e-9)
    reconstruction_tolerance = (
        None if reconstruction_tolerance_value is None else float(reconstruction_tolerance_value)
    )
    candidate_rows = ionization_candidate_details(
        trajectory_paths,
        selected_steps,
        selected_time_ps,
        centroid[:, ionization_component],
        beads[:, :, ionization_component],
        type_labels,
        kappa=float(ionization_config.get("kappa", 5.0)),
        reference=float(ionization_config.get("reference_occupancy", 2.0)),
        minimum_score=float(ionization_config.get("candidate_minimum_score", 0.25)),
        reconstruction_tolerance=reconstruction_tolerance,
        sampling_representation=sampling_slug,
    )
    candidate_fields = [
        "time_ps",
        "step",
        "representation",
        "bead",
        "recorded_score",
        "reconstructed_score",
        "absolute_reconstruction_error",
        "minimum_soft_occupancy",
        "minimum_soft_occupancy_O_id",
        "maximum_soft_occupancy",
        "maximum_soft_occupancy_O_id",
        "minimum_hard_H_count",
        "maximum_hard_H_count",
        "hard_undercoordinated_centers",
        "hard_overcoordinated_centers",
        "hard_pair_like",
    ]
    write_csv(output / "tables" / "ionization-candidates.csv", candidate_rows, candidate_fields)
    ionization_series = {sampling_slug: centroid[:, ionization_component]}
    ionization_series.update(
        {f"bead_{bead + 1}": beads[:, bead, ionization_component] for bead in range(beads.shape[1])}
    )
    ionization_series["maximum_bead"] = np.max(beads[:, :, ionization_component], axis=1)
    threshold_rows = threshold_run_rows(selected_time_ps, ionization_series, ionization_thresholds)
    write_csv(
        output / "qc" / "ionization-threshold-summary.csv",
        threshold_rows,
        [
            "representation",
            "threshold",
            "frames_at_or_above",
            "fraction_frames",
            "contiguous_runs",
            "longest_run_frames",
            "longest_run_ps",
        ],
    )

    primary_supports = supports_by_variant[primary_name]
    _plotter.plot_fes2d(
        output,
        x_grid,
        y_grid,
        primary_kcal,
        primary_supports,
        float(reweight["plot_max_kcal_mol"]),
        cv_labels,
        sampling_label=sampling_label,
        bias_mode=bias_mode,
        protocol_label=sampling_protocol_label(reweight),
        bead_count=beads.shape[1],
    )
    _plotter.plot_fes_differences(
        output,
        x_grid,
        y_grid,
        primary_kcal,
        primary_supports,
        float(reweight["difference_max_kcal_mol"]),
        cv_labels,
        sampling_label=sampling_label,
        bias_mode=bias_mode,
    )
    zoom = contract["plots"].get("fes_zoom")
    if zoom is not None:
        _plotter.plot_fes2d(
            output,
            x_grid,
            y_grid,
            primary_kcal,
            primary_supports,
            float(reweight["plot_max_kcal_mol"]),
            cv_labels,
            zoom=zoom,
            suffix="-sampled-region",
            sampling_label=sampling_label,
            bias_mode=bias_mode,
            protocol_label=sampling_protocol_label(reweight),
            bead_count=beads.shape[1],
        )
        _plotter.plot_fes_differences(
            output,
            x_grid,
            y_grid,
            primary_kcal,
            primary_supports,
            float(reweight["difference_max_kcal_mol"]),
            cv_labels,
            zoom=zoom,
            suffix="-sampled-region",
            sampling_label=sampling_label,
            bias_mode=bias_mode,
        )
    _plotter.plot_bead_cv_bias(
        output,
        selected_time_ps,
        centroid,
        beads,
        opes_bias_ev * EV_TO_KCAL_MOL,
        cv_labels,
        int(contract["plots"].get("scatter_stride", 5)),
        sampling_label=sampling_label,
        sampling_slug=sampling_slug,
        protocol_label=sampling_protocol_label(reweight),
    )
    _plotter.plot_cv_spread(output, selected_time_ps, beads, cv_labels)
    if diagnostic_spec is not None:
        require(
            diagnostic_sampling is not None and diagnostic_beads is not None,
            "diagnostic CV values absent",
        )
        _plotter.plot_diagnostic_cv_bias(
            output,
            selected_time_ps,
            diagnostic_sampling,
            diagnostic_beads,
            opes_bias_ev * EV_TO_KCAL_MOL,
            str(diagnostic_spec["label"]),
            sampling_label,
            int(contract["plots"].get("scatter_stride", 5)),
            protocol_label=sampling_protocol_label(reweight),
        )
    _plotter.plot_opes(
        output,
        selected_time_ps,
        opes_bias_ev * EV_TO_KCAL_MOL,
        rct_ev * EV_TO_KCAL_MOL,
        zed,
        neff,
        nker,
        raw_log_weights,
        kernel_time_ps,
        sigma_x,
        sigma_y,
        weights_are_log=True,
    )
    _plotter.plot_thermo(
        output,
        selected_time_ps,
        temperatures,
        potential,
        global_values,
        beads.shape[1],
        float(reweight["temperature_K"]),
    )
    _plotter.plot_ring_spread(output, spread_rows, float(selection["timestep_fs"]))
    _plotter.plot_ionization_events(
        output,
        selected_time_ps,
        centroid[:, ionization_component],
        beads[:, :, ionization_component],
        opes_bias_ev * EV_TO_KCAL_MOL,
        ionization_thresholds,
        ideal_pair_score,
        sampling_label=sampling_label,
        protocol_label=sampling_protocol_label(reweight),
    )

    reference_metrics = None
    reference_config = contract.get("reference")
    if reference_config is not None:
        require(isinstance(reference_config, Mapping), "reference must be an object")
        reference_metrics = run_reference(
            reference_config,
            filtered_colvar,
            output,
            x_grid,
            y_grid,
            primary_kcal["centroid"],
            primary_supports["centroid"],
            kbt_ev,
            variants[primary_name],
            sampling_cv_names,
            bias_column=str(bias_column),
        )
    raw_gap = (primary["raw_free_energy_mean"] - primary["raw_probability_mean"]) * EV_TO_KCAL_MOL
    diagnostic_gap = primary_kcal["free_energy_mean"] - primary_kcal["probability_mean"]
    centroid_threshold_rows = {
        float(row["threshold"]): row
        for row in threshold_rows
        if row["representation"] == sampling_slug
    }
    require(0.5 in centroid_threshold_rows, "ionization thresholds must include 0.5")
    bead_candidate_rows = [row for row in candidate_rows if int(row["bead"]) > 0]
    centroid_candidate_rows = [row for row in candidate_rows if int(row["bead"]) == 0]
    ionization_classification = (
        f"NO_SUSTAINED_{sampling_slug.upper().replace('-', '_')}_ION_PAIR; BEAD_LOCAL_PROTON_TRANSFER_LIKE_CONFIGURATIONS_PRESENT"
        if centroid_threshold_rows[0.5]["frames_at_or_above"] == 0
        and any(int(row["hard_pair_like"]) for row in bead_candidate_rows)
        else (
            f"{sampling_slug.upper().replace('-', '_')}_IONIZATION_EXCURSIONS_PRESENT; "
            "BEAD_LOCAL_PROTON_TRANSFER_LIKE_CONFIGURATIONS_PRESENT; "
            "SUSTAINED_PHYSICAL_ION_PAIR_NOT_ESTABLISHED"
            if centroid_threshold_rows[0.5]["frames_at_or_above"] > 0
            and any(int(row["hard_pair_like"]) for row in bead_candidate_rows)
            else "REQUIRES_MANUAL_INTERPRETATION"
        )
    )
    maximum_reconstruction_error = max(
        (float(row["absolute_reconstruction_error"]) for row in candidate_rows),
        default=0.0,
    )
    summary: Dict[str, object] = {
        "schema_version": 3,
        "status": "PASS",
        "source_job": source["job_id"],
        "analysis_profile": profile,
        "sampling_representation": {
            "label": sampling_label,
            "slug": sampling_slug,
            "bias_mode": bias_mode,
            "sampling_observable_id": source.get("sampling_observable_id"),
            "logical_cv_names": list(cv_names),
            "sampling_cv_columns": list(sampling_cv_names),
            "bead_cv_columns": list(bead_cv_names),
            "sampling_cv_source": (
                "mean_of_aligned_bead_cv"
                if bias_mode == "bead_density_shared"
                else "sampling_colvar"
            ),
        },
        "selection": {
            "first_time_ps": float(selected_time_ps[0]),
            "last_time_ps": float(selected_time_ps[-1]),
            "frames": int(len(selected)),
            "beads": int(beads.shape[1]),
            "stride_ps": float(np.median(np.diff(selected_time_ps))),
        },
        "reweighting": {
            "weight_kind": weight_kind,
            "primary_bias_column": reweight.get("bias_column"),
            "extra_bias_columns": list(reweight.get("extra_bias_columns", [])),
            "formula": (
                "precomputed log frame weight"
                if weight_kind == "precomputed"
                else "normalized exp(total_bias_energy/kBT)"
            ),
            "bias_columns": ([bias_column] if bias_column is not None else []) + extra_bias_columns,
            "total_bias_energy": (
                "mean_b bead_local_bias_energy"
                if bias_mode == "bead_density_shared"
                else "sampling_bias_energy"
            ),
            "quasi_static_declared": (
                quasi_static_declared if weight_kind == "quasi_static_opes" else None
            ),
            "rct_used": False,
            "energy_unit": "eV",
            "target_ensemble": (
                "defined by the supplied frame weights"
                if weight_kind == "precomputed"
                else "declared bias terms removed; undeclared potentials retained"
            ),
            "temperature_K": float(reweight["temperature_K"]),
            "kbt_eV": kbt_ev,
            "ess": 1.0 / float(np.sum(weights**2)),
            "ess_fraction": 1.0 / float(np.sum(weights**2)) / len(weights),
            "maximum_normalized_weight": float(np.max(weights)),
            "bias_range_kcal_mol": [
                float(np.min(bias_ev) * EV_TO_KCAL_MOL),
                float(np.max(bias_ev) * EV_TO_KCAL_MOL),
            ],
            "shared_diagnostic_max_deltas": shared_diagnostic_max_deltas,
        },
        "restart_alignment": {
            "duplicate_policy": restart_policy,
            "sampling_rows_removed": centroid_restart_duplicates,
            "bead_rows_removed": bead_restart_duplicates,
        },
        "fes": {
            "primary_estimator": primary_estimator,
            "alternate_estimator": alternate_estimator(primary_estimator),
            "target_observable": "bead_density_fes",
            "target_observable_id": reweight.get("target_observable_id"),
            "zero_reference": f"{primary_estimator}_minimum",
            "unit": "kcal/mol",
            "primary_support_points": int(np.count_nonzero(primary_support)),
            "probability_mean_label": estimator_plot_labels(bias_mode)["probability_mean"],
            "free_energy_mean_label": estimator_plot_labels(bias_mode)["free_energy_mean"],
            "primary_bandwidth": list(variants[primary_name]),
            "primary_bandwidth_name": primary_name,
            "common_support_points": int(np.count_nonzero(primary_supports["estimator_common"])),
            "estimator_pair_support_points": int(
                np.count_nonzero(primary_supports["estimator_common"])
            ),
            "sampling_support_points": int(np.count_nonzero(primary_supports["centroid"])),
            "probability_mean_support_points": int(
                np.count_nonzero(primary_supports["probability_mean"])
            ),
            "free_energy_mean_support_points": int(
                np.count_nonzero(primary_supports["free_energy_mean"])
            ),
            **diagnostic_gap_metrics(diagnostic_gap, primary_supports["estimator_common"]),
            "minimum_raw_jensen_gap_kcal_mol": float(np.min(raw_gap)),
        },
        "pimd": {
            "mean_scaled_temperature_K": float(np.mean(temperatures) / beads.shape[1]),
            "sd_scaled_temperature_K": float(np.std(temperatures / beads.shape[1])),
            "mean_centroid_virial_ke_eV": float(np.mean(global_values["f_fpimd[7]"])),
            "sd_centroid_virial_ke_eV": float(np.std(global_values["f_fpimd[7]"])),
            "mean_ring_spread_H_A": float(
                np.mean([row.get("rg_H_A", math.nan) for row in spread_rows])
            ),
            "mean_ring_spread_O_A": float(
                np.mean([row.get("rg_O_A", math.nan) for row in spread_rows])
            ),
        },
        "opes": {
            "final_zed": float(zed[-1]),
            "zed_range": [float(np.min(zed)), float(np.max(zed))],
            "final_neff": float(neff[-1]),
            "neff_range": [float(np.min(neff)), float(np.max(neff))],
            "final_nker": float(nker[-1]),
            "nker_range": [float(np.min(nker)), float(np.max(nker))],
            "kernel_rows_in_window": int(len(kernel_time_ps)),
            "kernel_time_window_ps": [
                float(kernel_time_ps[0]),
                float(kernel_time_ps[-1]),
            ],
            "final_sigma": [float(sigma_x[-1]), float(sigma_y[-1])],
        },
        "ionization_diagnostic": {
            "classification": ionization_classification,
            "definition": "sum over oxygen centers of squared soft occupancy defects relative to two H",
            "ideal_localized_H3O_OH_pair_score": ideal_pair_score,
            "centroid_maximum_score": float(np.max(centroid[:, ionization_component])),
            "sampling_maximum_score": float(np.max(centroid[:, ionization_component])),
            "maximum_bead_score": float(np.max(beads[:, :, ionization_component])),
            "centroid_frames_at_or_above_0_5": int(
                centroid_threshold_rows[0.5]["frames_at_or_above"]
            ),
            "sampling_frames_at_or_above_0_5": int(
                centroid_threshold_rows[0.5]["frames_at_or_above"]
            ),
            "centroid_candidate_hard_pair_like_frames": int(
                sum(int(row["hard_pair_like"]) for row in centroid_candidate_rows)
            ),
            "bead_candidate_hard_pair_like_rows": int(
                sum(int(row["hard_pair_like"]) for row in bead_candidate_rows)
            ),
            "maximum_reconstruction_error": maximum_reconstruction_error,
            "reconstruction_tolerance": reconstruction_tolerance,
            "reconstruction_comparison": (
                "MEASURE_ONLY" if reconstruction_tolerance is None else "ENFORCED"
            ),
            "reconstruction_within_tolerance": (
                None
                if reconstruction_tolerance is None
                else reconstruction_within_tolerance(
                    maximum_reconstruction_error, reconstruction_tolerance
                )
            ),
            "interpretation_boundary": "PIMD beads are imaginary-time configurations, not independent real-time trajectories",
        },
        "reference_crosscheck": reference_metrics,
        "gates": {
            "artifact_output": "PASS",
            "deterministic_numerical": "PASS",
            "nlist_numerical": contract["claims"].get("nlist_numerical", "NOT_APPLICABLE"),
            "postprocessing_plumbing": "PASS",
            "physical": "NOT_ASSESSED",
            "scientific_fes_convergence": "NOT_ASSESSED",
        },
    }
    if derived_coordinate_summary is not None:
        summary["derived_coordinate"] = derived_coordinate_summary
    if diagnostic_spec is not None:
        require(
            diagnostic_sampling is not None
            and diagnostic_beads is not None
            and diagnostic_mean_error is not None,
            "diagnostic CV summary absent",
        )
        summary["diagnostic_cv"] = {
            "name": str(diagnostic_spec["name"]),
            "label": str(diagnostic_spec["label"]),
            "sampling_column": str(diagnostic_spec["sampling_column"]),
            "bead_column": str(diagnostic_spec["bead_column"]),
            "sampling_range": [
                float(np.min(diagnostic_sampling)),
                float(np.max(diagnostic_sampling)),
            ],
            "bead_range": [
                float(np.min(diagnostic_beads)),
                float(np.max(diagnostic_beads)),
            ],
            "maximum_printed_mean_error": diagnostic_mean_error,
            "fes_assigned": False,
            "figure": "figures/iondistance-time-bias.png",
        }
    if reference_metrics is not None:
        require(
            float(reference_metrics["max_abs_difference_kcal_mol"])
            <= float(reference_config["max_abs_difference_kcal_mol"]),
            "reference FES mismatch",
        )
    require(float(np.min(raw_gap)) >= -1e-10, "Probability/free-energy mean Jensen relation failed")
    (output / "qc" / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "provenance" / "analysis-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        f"# PIMD {sampling_label.lower()}-biased post-processing",
        "",
        "Status: `PASS`",
        f"Source job: `{source['job_id']}`",
        f"Window: `{selected_time_ps[0]:.1f}--{selected_time_ps[-1]:.1f} ps` (`{len(selected)}` frames)",
        f"Reweighting ESS: `{summary['reweighting']['ess']:.2f}` (`{100.0 * summary['reweighting']['ess_fraction']:.1f}%`)",
        "Probability/free-energy mean diagnostic RMS gap (None = insufficient support): "
        f"`{summary['fes']['probability_free_energy_mean_rmse_common_support_kcal_mol']} kcal/mol`",
        f"Mean scaled bead temperature: `{summary['pimd']['mean_scaled_temperature_K']:.2f} K`",
        f"Mean H/O ring-polymer spread: `{summary['pimd']['mean_ring_spread_H_A']:.4f}` / `{summary['pimd']['mean_ring_spread_O_A']:.4f} A`",
        f"Ionization diagnostic: `{summary['ionization_diagnostic']['classification']}`",
        f"{sampling_label} / maximum-bead ionization score: `{summary['ionization_diagnostic']['sampling_maximum_score']:.4f}` / `{summary['ionization_diagnostic']['maximum_bead_score']:.4f}` (ideal localized pair about `{ideal_pair_score:g}`)",
        f"Maximum exact-reconstruction discrepancy on candidate rows: `{summary['ionization_diagnostic']['maximum_reconstruction_error']:.6g}` (`{summary['ionization_diagnostic']['reconstruction_comparison']}`)",
    ]
    if derived_coordinate_summary is not None:
        report.append(
            "Derived-coordinate FES: "
            f"`{derived_coordinate_summary['target']}` from "
            f"`{derived_coordinate_summary['source']}` with density Jacobian; "
            "printed transform PASS and sampling support unchanged."
        )
    if diagnostic_spec is not None:
        report.append(
            "Direct diagnostic CV: "
            f"`{diagnostic_spec['name']}` uses printed sampling/local columns; "
            "no independent FES bandwidth was assigned."
        )
    report.extend(
        [
            "",
            "All free energies are reported in kcal/mol. The selected analysis window is defined by the contract. `opes.rct` is retained only as a diagnostic and is not part of the weights.",
            "",
            "This is an engineering and post-processing assessment. Physical interpretation and scientific/FES convergence remain NOT_ASSESSED.",
        ]
    )
    (output / "analysis-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_manifest(output)
    return summary
