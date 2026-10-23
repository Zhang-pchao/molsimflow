"""PIMD path-bias reweighting and engineering diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from molsimflow.postprocess.pimd.units import KB_EV_PER_K, EV_TO_KCAL_MOL, KCAL_TO_KJ_MOL
from molsimflow.postprocess.pimd.plotting import FES_COLORS
from molsimflow.postprocess.pimd.reporting import FES_TABLE_FIELDS
from molsimflow.io.hashes import sha256_file
from molsimflow.postprocess.pimd.weights import (
    cumulative_weight_diagnostics,
)
from molsimflow.postprocess.pimd.alignment import (
    time_window_mask,
    time_scale_to_fs,
    time_values_to_fs,
    aligned_time_indices,
)
from molsimflow.postprocess.pimd.io import (
    verify_manifest_inputs,
    read_plumed,
    field,
    read_thermo,
    dump_frames,
)
from molsimflow.postprocess.pimd.kde import (
    logsumexp,
    _validate_kde_weights,
    weighted_log_kde_1d,
    weighted_log_kde_2d,
)
from molsimflow.postprocess.pimd.estimators import (
    alternate_estimator,
    estimator_support_masks,
    compute_surfaces,
    compute_marginals,
    surface_difference_metrics,
    optional_surface_difference_metrics,
    diagnostic_gap_metrics,
)
from molsimflow.postprocess.pimd.uncertainty import (
    quantum_kde_block_jackknife,
)
from molsimflow.postprocess.pimd.reporting import (
    fes_table_columns,
    estimator_plot_labels,
    default_cv_label,
    sampling_protocol_label,
    portable_artifact_path,
    write_csv,
    write_filtered_colvar,
    write_kde_uncertainty,
    write_manifest,
)
from molsimflow.postprocess.pimd.observables import (
    _validate_logdistance_parameters,
    piecewise_logdistance,
    inverse_piecewise_logdistance,
    piecewise_logdistance_jacobian,
    transform_piecewise_logdistance_fes,
    validate_piecewise_logdistance_printed,
    piecewise_derived_coordinate_spec,
    soft_voronoi_occupancies,
)
from molsimflow.postprocess.pimd.water_ionization import (
    reconstruction_within_tolerance,
    ring_polymer_spread,
    ionization_candidate_details,
    threshold_run_rows,
)
from molsimflow.postprocess.pimd.reference import (
    adapt_reference_source,
    run_reference,
)
from molsimflow.postprocess.pimd import plotting as _plotting
from molsimflow.postprocess.pimd.plotting import (
    _matplotlib,
    save_figure,
    scatter_norm,
)


from molsimflow.postprocess.pimd_fes import (
    bead_density_estimators,
    frame_log_weights,
    normalized_log_weights as _normalized_log_weights,
    restart_unique_indices,
    total_bias_energy,
    validate_bias_mode,
    validate_primary_estimator,
)


ANALYSIS_PROFILES = {"core", "water_ionization_opes"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def analysis_profile(contract: Mapping[str, object]) -> str:
    """Return the generic or case-specific diagnostics profile."""
    profile = str(contract.get("analysis_profile", "water_ionization_opes"))
    require(profile in ANALYSIS_PROFILES, f"unsupported analysis profile: {profile}")
    return profile


def sha256(path: Path) -> str:
    """Compatibility entry point for file-byte hashing."""
    return sha256_file(path)


def artifact_basename(config: Mapping[str, object], key: str, default: str) -> str:
    """Return a configured output basename without allowing path traversal."""
    name = str(config.get(key, default))
    require(bool(name) and Path(name).name == name, f"{key} must be a basename")
    return name


def cv_column_names(
    config: Mapping[str, object], key: str, logical_names: Sequence[str]
) -> Tuple[str, ...]:
    """Resolve representation-specific PLUMED columns for logical CV names."""
    names = tuple(str(value) for value in config.get(key, logical_names))
    require(len(names) == len(logical_names), f"{key} must match cv_names")
    return names


def diagnostic_cv_spec(config: Mapping[str, object] | None) -> Dict[str, object] | None:
    """Validate one directly printed diagnostic CV without defining a FES."""
    if config is None:
        return None
    require(isinstance(config, Mapping), "diagnostic_cv must be an object")
    result = {
        "name": str(config.get("name", "")),
        "sampling_column": str(config.get("sampling_column", "")),
        "bead_column": str(config.get("bead_column", "")),
        "label": str(config.get("label", config.get("name", ""))),
        "mean_tolerance": float(config.get("mean_tolerance", 1e-12)),
    }
    for key in ("name", "sampling_column", "bead_column"):
        value = str(result[key])
        require(bool(value) and Path(value).name == value, f"invalid diagnostic_cv {key}")
    require(float(result["mean_tolerance"]) >= 0.0, "negative diagnostic_cv tolerance")
    return result


def normalized_log_weights(raw: Sequence[float]) -> np.ndarray:
    """Normalize frame weights while preserving the workflow keyword API."""
    return _normalized_log_weights(raw)


def finalize_core_1d(
    *,
    contract: Mapping[str, object],
    output: Path,
    source: Mapping[str, object],
    reweight: Mapping[str, object],
    cv_name: str,
    cv_label: str,
    sampling_label: str,
    sampling_slug: str,
    bias_mode: str,
    weight_kind: str,
    selected_time_ps: np.ndarray,
    selected_steps: np.ndarray,
    sampling: np.ndarray,
    beads: np.ndarray,
    raw_log_weights: np.ndarray,
    bias_ev: np.ndarray | None,
    kbt_ev: float,
    sampling_restart_duplicates: int,
    bead_restart_duplicates: Sequence[int],
) -> Dict[str, object]:
    """Write the generic one-CV report without OPES or material diagnostics."""
    require("reference" not in contract, "legacy reference cross-check requires two CVs")
    primary_estimator = validate_primary_estimator(
        str(reweight.get("primary_estimator", "probability_mean"))
    )
    alternate = alternate_estimator(primary_estimator)
    grid = np.linspace(*reweight["grid"][cv_name])
    variants = {
        str(name): tuple(float(value) for value in values)
        for name, values in reweight["bandwidth_variants"].items()
    }
    require(all(len(value) == 1 for value in variants.values()), "1D bandwidths need one value")
    primary_name = str(reweight["primary_bandwidth"])
    require(primary_name in variants, "primary bandwidth absent")
    require(all(name and Path(name).name == name and name not in {".", ".."}
                for name in variants), "bandwidth variant names must be basenames")
    threshold = math.log(float(reweight["relative_density_support"]))
    curves_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    supports_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    sensitivity_rows: List[Dict[str, object]] = []
    for variant, bandwidth in variants.items():
        curves = compute_marginals(
            beads,
            sampling,
            raw_log_weights,
            grid,
            bandwidth[0],
            0,
            kbt_ev,
            primary_estimator=primary_estimator,
        )
        supports = estimator_support_masks(
            curves["log_centroid"],
            curves["log_probability_mean"],
            curves["log_beads"],
            threshold,
        )
        curves_by_variant[variant] = curves
        supports_by_variant[variant] = supports
        sensitivity_rows.append(
            {
                "variant": variant,
                "sigma": bandwidth[0],
                "primary_estimator": primary_estimator,
                "primary_support_points": int(np.count_nonzero(supports[primary_estimator])),
                "probability_mean_support_points": int(np.count_nonzero(supports["probability_mean"])),
                "free_energy_mean_support_points": int(np.count_nonzero(supports["free_energy_mean"])),
                "common_support_points": int(np.count_nonzero(supports["common"])),
            }
        )

    primary = curves_by_variant[primary_name]
    primary_supports = supports_by_variant[primary_name]
    primary_support = primary_supports[primary_estimator]
    estimator_common_support = primary_supports["estimator_common"]
    require(
        np.count_nonzero(primary_support) >= 2,
        f"empty {primary_estimator} support",
    )
    primary_kcal = {
        key: primary[key] * EV_TO_KCAL_MOL for key in ("centroid", "probability_mean", "free_energy_mean")
    }
    for row in sensitivity_rows:
        variant = str(row["variant"])
        comparison_support = (
            primary_supports[primary_estimator]
            & supports_by_variant[variant][primary_estimator]
        )
        count, rmse, maximum = optional_surface_difference_metrics(
            primary_kcal[primary_estimator],
            curves_by_variant[variant][primary_estimator] * EV_TO_KCAL_MOL,
            comparison_support,
        )
        row["comparison_support_points"] = count
        row["primary_rmse_vs_selected_bandwidth_kcal_mol"] = rmse
        row["primary_max_abs_vs_selected_bandwidth_kcal_mol"] = maximum
    write_csv(
        output / "qc" / "bandwidth-sensitivity.csv",
        sensitivity_rows,
        [
            "variant",
            "sigma",
            "primary_estimator",
            "primary_support_points",
            "probability_mean_support_points",
            "free_energy_mean_support_points",
            "common_support_points",
            "comparison_support_points",
            "primary_rmse_vs_selected_bandwidth_kcal_mol",
            "primary_max_abs_vs_selected_bandwidth_kcal_mol",
        ],
    )
    write_csv(
        output / "fes1d" / f"{cv_name}.csv",
        (
            {
                cv_name: value,
                **fes_table_columns(
                    primary_kcal, primary_supports, primary_estimator, index
                ),
            }
            for index, value in enumerate(grid)
        ),
        [
            cv_name,
            *FES_TABLE_FIELDS,
        ],
    )

    weights = np.exp(normalized_log_weights(raw_log_weights))
    frame_rows = []
    for frame, time_ps in enumerate(selected_time_ps):
        row: Dict[str, object] = {
            "time_ps": time_ps,
            "step": int(selected_steps[frame]),
            f"{sampling_slug}_{cv_name}": sampling[frame, 0],
            f"bead_mean_{cv_name}": float(np.mean(beads[frame, :, 0])),
            f"bead_std_{cv_name}": float(np.std(beads[frame, :, 0])),
            "log_frame_weight": raw_log_weights[frame],
            "normalized_weight": weights[frame],
        }
        if bias_ev is not None:
            row["bias_eV"] = bias_ev[frame]
            row["bias_kcal_mol"] = bias_ev[frame] * EV_TO_KCAL_MOL
        frame_rows.append(row)
    write_csv(output / "tables" / "frame-series.csv", frame_rows, list(frame_rows[0]))
    bead_fields = ["time_ps", "bead", cv_name]
    write_csv(
        output / "tables" / "bead-cv-long.csv",
        (
            {
                "time_ps": selected_time_ps[frame],
                "bead": bead + 1,
                cv_name: beads[frame, bead, 0],
            }
            for frame in range(beads.shape[0])
            for bead in range(beads.shape[1])
        ),
        bead_fields,
    )

    block_count = int(reweight["blocks"])
    require(1 <= block_count <= len(selected_time_ps), "invalid block count")
    block_rows = []
    for block, indices in enumerate(np.array_split(np.arange(len(selected_time_ps)), block_count)):
        current = compute_marginals(
            beads[indices],
            sampling[indices],
            raw_log_weights[indices],
            grid,
            variants[primary_name][0],
            0,
            kbt_ev,
            primary_estimator=primary_estimator,
        )
        current_supports = estimator_support_masks(
            current["log_centroid"],
            current["log_probability_mean"],
            current["log_beads"],
            threshold,
        )
        comparison_support = (
            primary_supports[primary_estimator]
            & current_supports[primary_estimator]
        )
        count, rmse, maximum = optional_surface_difference_metrics(
            primary_kcal[primary_estimator],
            current[primary_estimator] * EV_TO_KCAL_MOL,
            comparison_support,
        )
        block_weights = np.exp(normalized_log_weights(raw_log_weights[indices]))
        block_rows.append(
            {
                "block": block + 1,
                "first_time_ps": selected_time_ps[indices[0]],
                "last_time_ps": selected_time_ps[indices[-1]],
                "frames": len(indices),
                "ess": 1.0 / float(np.sum(block_weights**2)),
                "max_weight": float(np.max(block_weights)),
                "primary_estimator": primary_estimator,
                "comparison_support_points": count,
                "primary_rmse_vs_full_kcal_mol": rmse,
                "primary_max_abs_vs_full_kcal_mol": maximum,
            }
        )
    write_csv(
        output / "blocks" / "block-diagnostics.csv",
        block_rows,
        list(block_rows[0]),
    )

    plot_fes1d(
        output,
        cv_name,
        grid,
        primary_kcal,
        primary_supports,
        float(reweight["plot_max_kcal_mol"]),
        cv_label,
        sampling_label=sampling_label,
        bias_mode=bias_mode,
    )
    plot_cv_time_series(
        output, selected_time_ps, sampling, beads, [cv_label], sampling_label
    )
    plot_cv_spread(output, selected_time_ps, beads, [cv_label])

    labels = estimator_plot_labels(bias_mode)
    raw_gap = (
        np.mean(-kbt_ev * primary["log_beads"], axis=0)
        + kbt_ev * primary["log_probability_mean"]
    ) * EV_TO_KCAL_MOL
    diagnostic_gap = primary_kcal["free_energy_mean"] - primary_kcal["probability_mean"]
    summary: Dict[str, object] = {
        "schema_version": 3,
        "status": "PASS",
        "analysis_profile": "core",
        "source_job": source.get("job_id"),
        "sampling_representation": {
            "label": sampling_label,
            "slug": sampling_slug,
            "bias_mode": bias_mode,
            "sampling_observable_id": source.get("sampling_observable_id"),
            "logical_cv_names": [cv_name],
        },
        "selection": {
            "first_time_ps": float(selected_time_ps[0]),
            "last_time_ps": float(selected_time_ps[-1]),
            "frames": int(len(selected_time_ps)),
            "beads": int(beads.shape[1]),
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
            "protocol_label": sampling_protocol_label(reweight),
            "rct_used": False,
            "energy_unit": "eV",
            "target_ensemble": (
                "defined by the supplied frame weights" if weight_kind == "precomputed"
                else "declared bias terms removed; undeclared potentials retained"
            ),
            "temperature_K": float(reweight["temperature_K"]),
            "kbt_eV": kbt_ev,
            "ess": 1.0 / float(np.sum(weights**2)),
            "ess_fraction": 1.0 / float(np.sum(weights**2)) / len(weights),
            "maximum_normalized_weight": float(np.max(weights)),
            "bias_range_kcal_mol": (
                None
                if bias_ev is None
                else [
                    float(np.min(bias_ev) * EV_TO_KCAL_MOL),
                    float(np.max(bias_ev) * EV_TO_KCAL_MOL),
                ]
            ),
        },
        "restart_alignment": {
            "duplicate_policy": str(source.get("restart_duplicate_policy", "keep_first")),
            "sampling_rows_removed": sampling_restart_duplicates,
            "bead_rows_removed": list(bead_restart_duplicates),
        },
        "fes": {
            "primary_estimator": primary_estimator,
            "alternate_estimator": alternate,
            "target_observable": "bead_density_fes",
            "target_observable_id": reweight.get("target_observable_id"),
            "zero_reference": f"{primary_estimator}_minimum",
            "dimensions": 1,
            "unit": "kcal/mol",
            "primary_support_points": int(np.count_nonzero(primary_support)),
            "primary_bandwidth": list(variants[primary_name]),
            "primary_bandwidth_name": primary_name,
            "probability_mean_label": labels["probability_mean"],
            "free_energy_mean_label": labels["free_energy_mean"],
            "common_support_points": int(np.count_nonzero(estimator_common_support)),
            "estimator_pair_support_points": int(
                np.count_nonzero(estimator_common_support)
            ),
            **diagnostic_gap_metrics(diagnostic_gap, estimator_common_support),
            "minimum_raw_jensen_gap_kcal_mol": float(np.min(raw_gap)),
        },
        "reference_crosscheck": None,
        "gates": {
            "artifact_output": "PASS",
            "deterministic_numerical": "PASS",
            "postprocessing_plumbing": "PASS",
            "physical": "NOT_ASSESSED",
            "scientific_fes_convergence": "NOT_ASSESSED",
        },
    }
    require(float(np.min(raw_gap)) >= -1e-10, "bead density Jensen relation failed")
    (output / "qc" / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "provenance" / "analysis-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# PIMD path-bias post-processing",
        "",
        "Status: `PASS`",
        "Analysis profile: `core`",
        f"CV: `{cv_name}`; beads: `{beads.shape[1]}`; frames: `{len(selected_time_ps)}`",
        f"Weight provider: `{weight_kind}`; ESS: `{summary['reweighting']['ess']:.2f}`",
        f"Primary estimator: `{labels[primary_estimator]}`",
        f"Alternate estimator: `{labels[alternate]}`",
        "",
        "The core profile does not require OPES kernels, PIMD thermo logs, atom trajectories, water-ionization diagnostics, or an external reference driver.",
        "",
        "This is an engineering and post-processing assessment. Physical interpretation and scientific/FES convergence remain NOT_ASSESSED.",
    ]
    (output / "analysis-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_manifest(output)
    return summary


def finalize_core_2d(
    *,
    contract: Mapping[str, object],
    output: Path,
    source: Mapping[str, object],
    reweight: Mapping[str, object],
    cv_names: Sequence[str],
    cv_labels: Sequence[str],
    sampling_label: str,
    sampling_slug: str,
    bias_mode: str,
    weight_kind: str,
    selected_time_ps: np.ndarray,
    sampling: np.ndarray,
    beads: np.ndarray,
    raw_log_weights: np.ndarray,
    bias_ev: np.ndarray | None,
    kbt_ev: float,
    primary_name: str,
    primary: Mapping[str, np.ndarray],
    primary_kcal: Mapping[str, np.ndarray],
    primary_supports: Mapping[str, np.ndarray],
    variants: Mapping[str, Sequence[float]],
    filtered_colvar: Path,
    sampling_restart_duplicates: int,
    bead_restart_duplicates: Sequence[int],
) -> Dict[str, object]:
    """Write a generic two-CV report without material-specific diagnostics."""
    primary_estimator = validate_primary_estimator(
        str(reweight.get("primary_estimator", "probability_mean"))
    )
    alternate = alternate_estimator(primary_estimator)
    primary_support = primary_supports[primary_estimator]
    estimator_common_support = primary_supports["estimator_common"]
    protocol_label = sampling_protocol_label(reweight)
    plot_fes2d(
        output,
        np.linspace(*reweight["grid"][cv_names[0]]),
        np.linspace(*reweight["grid"][cv_names[1]]),
        primary_kcal,
        primary_supports,
        float(reweight["plot_max_kcal_mol"]),
        cv_labels,
        sampling_label=sampling_label,
        bias_mode=bias_mode,
        protocol_label=protocol_label,
        bead_count=beads.shape[1],
    )
    plot_fes_differences(
        output,
        np.linspace(*reweight["grid"][cv_names[0]]),
        np.linspace(*reweight["grid"][cv_names[1]]),
        primary_kcal,
        primary_supports,
        float(reweight["difference_max_kcal_mol"]),
        cv_labels,
        sampling_label=sampling_label,
        bias_mode=bias_mode,
    )
    plot_cv_time_series(
        output, selected_time_ps, sampling, beads, cv_labels, sampling_label
    )
    plot_cv_spread(output, selected_time_ps, beads, cv_labels)

    reference_metrics = None
    reference_config = contract.get("reference")
    if reference_config is not None:
        require(isinstance(reference_config, Mapping), "reference must be an object")
        require(
            bias_ev is not None and weight_kind != "precomputed",
            "legacy reference cross-check requires bias-energy weights",
        )
        reference_metrics = run_reference(
            reference_config,
            filtered_colvar,
            output,
            np.linspace(*reweight["grid"][cv_names[0]]),
            np.linspace(*reweight["grid"][cv_names[1]]),
            primary_kcal["centroid"],
            primary_supports["centroid"],
            kbt_ev,
            variants[primary_name],
            cv_names,
            bias_column=str(reweight["bias_column"]),
        )
        require(
            float(reference_metrics["max_abs_difference_kcal_mol"])
            <= float(reference_config["max_abs_difference_kcal_mol"]),
            "reference FES mismatch",
        )

    weights = np.exp(normalized_log_weights(raw_log_weights))
    labels = estimator_plot_labels(bias_mode)
    raw_gap = (primary["raw_free_energy_mean"] - primary["raw_probability_mean"]) * EV_TO_KCAL_MOL
    diagnostic_gap = primary_kcal["free_energy_mean"] - primary_kcal["probability_mean"]
    require(float(np.min(raw_gap)) >= -1e-10, "bead density Jensen relation failed")
    summary: Dict[str, object] = {
        "schema_version": 3,
        "status": "PASS",
        "analysis_profile": "core",
        "source_job": source.get("job_id"),
        "sampling_representation": {
            "label": sampling_label,
            "slug": sampling_slug,
            "bias_mode": bias_mode,
            "sampling_observable_id": source.get("sampling_observable_id"),
            "logical_cv_names": list(cv_names),
        },
        "selection": {
            "first_time_ps": float(selected_time_ps[0]),
            "last_time_ps": float(selected_time_ps[-1]),
            "frames": int(len(selected_time_ps)),
            "beads": int(beads.shape[1]),
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
            "protocol_label": protocol_label,
            "rct_used": False,
            "energy_unit": "eV",
            "target_ensemble": (
                "defined by the supplied frame weights" if weight_kind == "precomputed"
                else "declared bias terms removed; undeclared potentials retained"
            ),
            "temperature_K": float(reweight["temperature_K"]),
            "kbt_eV": kbt_ev,
            "ess": 1.0 / float(np.sum(weights**2)),
            "ess_fraction": 1.0 / float(np.sum(weights**2)) / len(weights),
            "maximum_normalized_weight": float(np.max(weights)),
            "bias_range_kcal_mol": (
                None
                if bias_ev is None
                else [
                    float(np.min(bias_ev) * EV_TO_KCAL_MOL),
                    float(np.max(bias_ev) * EV_TO_KCAL_MOL),
                ]
            ),
        },
        "restart_alignment": {
            "duplicate_policy": str(source.get("restart_duplicate_policy", "keep_first")),
            "sampling_rows_removed": sampling_restart_duplicates,
            "bead_rows_removed": list(bead_restart_duplicates),
        },
        "fes": {
            "primary_estimator": primary_estimator,
            "alternate_estimator": alternate,
            "target_observable": "bead_density_fes",
            "target_observable_id": reweight.get("target_observable_id"),
            "zero_reference": f"{primary_estimator}_minimum",
            "dimensions": 2,
            "unit": "kcal/mol",
            "primary_support_points": int(np.count_nonzero(primary_support)),
            "primary_bandwidth": list(variants[primary_name]),
            "primary_bandwidth_name": primary_name,
            "probability_mean_label": labels["probability_mean"],
            "free_energy_mean_label": labels["free_energy_mean"],
            "common_support_points": int(np.count_nonzero(estimator_common_support)),
            "estimator_pair_support_points": int(
                np.count_nonzero(estimator_common_support)
            ),
            **diagnostic_gap_metrics(diagnostic_gap, estimator_common_support),
            "minimum_raw_jensen_gap_kcal_mol": float(np.min(raw_gap)),
        },
        "reference_crosscheck": reference_metrics,
        "gates": {
            "artifact_output": "PASS",
            "deterministic_numerical": "PASS",
            "postprocessing_plumbing": "PASS",
            "physical": "NOT_ASSESSED",
            "scientific_fes_convergence": "NOT_ASSESSED",
        },
    }
    (output / "qc" / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "provenance" / "analysis-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reference_line = (
        "External FES_from_Reweighting.py cross-check: `PASS`."
        if reference_metrics is not None
        else "External reference cross-check: `NOT_REQUESTED`; the in-package log-space estimator is authoritative."
    )
    report = [
        "# PIMD path-bias post-processing",
        "",
        "Status: `PASS`",
        "Analysis profile: `core`",
        f"CVs: `{', '.join(cv_names)}`; beads: `{beads.shape[1]}`; frames: `{len(selected_time_ps)}`",
        f"Weight provider: `{weight_kind}`; ESS: `{summary['reweighting']['ess']:.2f}`",
        f"Primary estimator: `{labels[primary_estimator]}`",
        f"Alternate estimator: `{labels[alternate]}`",
        reference_line,
        "",
        "The core profile does not require OPES kernels, PIMD thermo logs, atom trajectories, or water-ionization diagnostics.",
        "",
        "This is an engineering and post-processing assessment. Physical interpretation and scientific/FES convergence remain NOT_ASSESSED.",
    ]
    (output / "analysis-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_manifest(output)
    return summary


def analyze(contract_path: Path, output: Path) -> Dict[str, object]:
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    profile = analysis_profile(contract)
    output = Path(output)
    require(not output.exists(), f"output exists: {output}")
    for name in ("inputs", "tables", "fes1d", "fes2d", "blocks", "figures", "qc", "provenance"):
        (output / name).mkdir(parents=True, exist_ok=False)

    source = contract["source"]
    selection = contract["selection"]
    reweight = contract["reweight"]
    # Resolve names before mode-specific admission, preserving the input contract
    # and sampling labels verbatim in provenance and output filenames.
    sampling_mode = source.get("sampling_slug", "centroid")
    inferred_bias_mode = {
        "centroid": "centroid_coord",
        "bead_density": "bead_density_shared",
    }.get(sampling_mode, sampling_mode)
    bias_mode = validate_bias_mode(str(reweight.get("bias_mode", inferred_bias_mode)))
    if bias_mode == "contracted_bead_mean":
        from molsimflow.postprocess.path_contraction import validate_contraction_metadata

        metadata = reweight.get("path_contraction")
        validate_contraction_metadata(metadata)
        require(reweight.get("weight_kind") == "fixed_bias",
                "contracted bead mean currently requires stationary frozen-bias records")
        require(bool(reweight.get("bias_column")), "contracted bead mean requires total bias energy")
        require(reweight.get("bias_mode") is not None
                and validate_bias_mode(str(source.get("sampling_slug"))) == bias_mode
                and bool(source.get("sampling_label")), "explicit contracted-bead-mean labels are required")
        require(bool(reweight.get("bead_cv_names")), "declare the original real-bead CV columns")
        (output / "qc" / "path-contraction.json").write_text(
            json.dumps({"status": "METADATA_VALIDATED", **metadata}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    probability_spec = None
    probability_modes = {"bead_probability_mixture", "centroid_probability_mixture"}
    if bias_mode in probability_modes:
        probability_spec = reweight.get("probability_mixture")
        require(profile == "core" and len(reweight["cv_names"]) == 1,
                "probability mixtures currently require the one-dimensional core profile")
        require(reweight.get("weight_kind") == "fixed_bias",
                "probability mixtures require stationary frozen-bias records")
        require(isinstance(probability_spec, dict), "probability_mixture manifest specification is required")
        require(bool(reweight.get("bias_column")) and not reweight.get("extra_bias_columns"),
                "probability mixtures require one complete-path total-bias column")
        require(reweight.get("bias_mode") is not None
                and validate_bias_mode(str(source.get("sampling_slug"))) == bias_mode
                and bool(source.get("sampling_label")), "explicit probability-mixture labels are required")
    conditional_spec = None
    if bias_mode == "centroid_conditioned":
        conditional_spec = reweight.get("conditional_path")
        require(profile == "core" and len(reweight["cv_names"]) == 1,
                "centroid_conditioned currently requires the one-dimensional core profile")
        require(reweight.get("weight_kind") == "fixed_bias",
                "centroid_conditioned requires a stationary frozen-bias record")
        require(isinstance(conditional_spec, dict) and conditional_spec.get("frozen") is True,
                "conditional_path must declare a frozen model")
        require(bool(reweight.get("bias_column")) and not reweight.get("extra_bias_columns"),
                "centroid_conditioned requires one total-bias column without extra biases")
        require(source.get("sampling_slug") == "centroid_conditioned"
                and bool(source.get("sampling_label")), "explicit conditional-path labels are required")
    primary_estimator = validate_primary_estimator(
        str(reweight.get("primary_estimator", "probability_mean"))
    )
    require(not (contract.get("reference") is not None
                 and reweight.get("weight_kind") == "precomputed"),
            "legacy reference cross-check requires bias-energy weights")
    if profile == "water_ionization_opes":
        require(bool(reweight.get("bias_column")), "water-ionization diagnostics require bias_column")
    for config, key in ((source, "sampling_observable_id"), (reweight, "target_observable_id")):
        value = config.get(key)
        require(value is None or (isinstance(value, str) and bool(value.strip())),
                f"{key} must be a nonempty string")
    sampling_time_scale_to_fs = time_scale_to_fs(
        source.get("sampling_time_scale_to_fs", 1.0),
        "sampling_time_scale_to_fs",
    )
    bead_time_scale_to_fs = time_scale_to_fs(
        source.get("bead_time_scale_to_fs", 1.0),
        "bead_time_scale_to_fs",
    )
    kernel_time_scale_to_fs = time_scale_to_fs(
        source.get("kernel_time_scale_to_fs", sampling_time_scale_to_fs),
        "kernel_time_scale_to_fs",
    )
    bead_offset_fs = float(source.get("bead_time_offset_fs", 0.0))
    require(np.isfinite(bead_offset_fs), "invalid bead_time_offset_fs")
    run_root = Path(source["run_root"])
    require(run_root.is_dir(), f"run root missing: {run_root}")
    raw_manifest = run_root / source["raw_manifest"]
    require(sha256(raw_manifest) == source["raw_manifest_sha256"], "raw manifest hash mismatch")

    sampling_colvar = source.get("sampling_colvar")
    if sampling_colvar is None:
        sampling_colvar = source["centroid_colvar"]
    centroid_path = run_root / str(sampling_colvar)
    bead_paths = [run_root / value for value in source["bead_colvars"]]
    require(bool(bead_paths), "bead_colvars must contain a complete path")
    if "expected_beads" in source:
        expected_beads = source["expected_beads"]
        require(isinstance(expected_beads, int) and not isinstance(expected_beads, bool)
                and expected_beads > 0, "expected_beads must be a positive integer")
        require(len(bead_paths) == expected_beads, "bead input count differs from expected_beads")
    if profile == "water_ionization_opes":
        for key in ("thermo_logs", "trajectories"):
            require(len(source[key]) == len(bead_paths), f"{key} must contain one file per bead")
    # Resolve filesystem identity, including symlinks and hard links, without
    # rejecting distinct files whose bead observations happen to be identical.
    bead_stats = [path.stat() for path in bead_paths]
    require(
        len({(stat.st_dev, stat.st_ino) for stat in bead_stats}) == len(bead_paths),
        "duplicate bead input file",
    )
    consumed_paths = [centroid_path, *bead_paths]
    conditional_model = None
    if conditional_spec is not None:
        from molsimflow.postprocess.conditional_path import load_model
        model_path = run_root / conditional_spec["model_file"]
        conditional_model = load_model(model_path, expected_sha256=conditional_spec["model_sha256"])
        consumed_paths.append(model_path)
    probability_model = None
    if probability_spec is not None:
        from molsimflow.postprocess.probability_mixture import load_manifest
        model_path = run_root / probability_spec["manifest_file"]
        probability_model = load_manifest(model_path, expected_sha256=probability_spec["manifest_sha256"])
        require(probability_model["mode"] == bias_mode, "probability mixture mode mismatch")
        require(probability_model["expected_beads"] == len(bead_paths), "probability mixture bead count mismatch")
        require(probability_model["energy_unit"] == "eV", "probability mixture field must use eV")
        consumed_paths.append(model_path)
        field_keys = ["field"]
        if probability_model["mode"] == "centroid_probability_mixture":
            field_keys = (["field"] if probability_model["coupling"] > 0 else []) + ["centroid_field"]
        for key in field_keys:
            input_path = run_root / probability_spec[key + "_file"]
            require(sha256(input_path) == probability_model[key + "_sha256"],
                    "probability mixture frozen field hash mismatch")
            consumed_paths.append(input_path)
    if profile == "water_ionization_opes":
        consumed_paths.append(run_root / source["kernels"])
        consumed_paths.extend(run_root / name for name in source["thermo_logs"])
        consumed_paths.extend(run_root / name for name in source["trajectories"])
    verified_inputs = verify_manifest_inputs(raw_manifest, run_root, consumed_paths)
    (output / "provenance" / "verified-inputs.json").write_text(
        json.dumps({"scope": "consumed inputs", "sha256": verified_inputs}, indent=2)
        + "\n", encoding="utf-8",
    )
    fields, centroid_data = read_plumed(centroid_path)
    bead_tables = [read_plumed(path) for path in bead_paths]
    require(
        all(table_fields == fields[:4] or "time" in table_fields for table_fields, _ in bead_tables),
        "invalid bead table headers",
    )
    restart_policy = str(source.get("restart_duplicate_policy", "keep_first"))
    centroid_keep, centroid_restart_duplicates = restart_unique_indices(
        field(centroid_data, fields, "time"), policy=restart_policy
    )
    centroid_data = centroid_data[centroid_keep]
    deduplicated_bead_tables = []
    bead_restart_duplicates = []
    for bead_fields, bead_data in bead_tables:
        bead_keep, duplicate_count = restart_unique_indices(
            field(bead_data, bead_fields, "time"), policy=restart_policy
        )
        deduplicated_bead_tables.append((bead_fields, bead_data[bead_keep]))
        bead_restart_duplicates.append(duplicate_count)
    bead_tables = deduplicated_bead_tables
    time_fs = time_values_to_fs(field(centroid_data, fields, "time"), sampling_time_scale_to_fs)
    first_time_fs = float(selection["first_time_ps"]) * 1000.0
    last_time_fs = float(selection["last_time_ps"]) * 1000.0
    all_steps = np.rint(time_fs / float(selection["timestep_fs"])).astype(int)
    selected_mask = (time_fs >= first_time_fs - 1e-9) & (time_fs <= last_time_fs + 1e-9)
    sample_stride_steps = int(selection.get("sample_stride_steps", 1))
    require(sample_stride_steps > 0, "sample_stride_steps must be positive")
    sample_phase_step = int(selection.get("sample_phase_step", 0))
    selected_mask &= (all_steps - sample_phase_step) % sample_stride_steps == 0
    selected = np.flatnonzero(selected_mask)
    require(selected.size == int(selection["expected_frames"]), "selected frame count mismatch")
    selected_time_ps = time_fs[selected] / 1000.0
    selected_steps = np.rint(
        time_fs[selected] / float(selection["timestep_fs"])
    ).astype(int)
    cv_names = tuple(reweight["cv_names"])
    require(
        len(cv_names) in ({1, 2} if profile == "core" else {2}),
        "core analysis supports one or two CVs; water-ionization diagnostics require two",
    )
    sampling_cv_names = cv_column_names(reweight, "sampling_cv_names", cv_names)
    bead_cv_names = cv_column_names(reweight, "bead_cv_names", cv_names)
    sampling_label = str(source.get("sampling_label", "Centroid"))
    sampling_slug = str(source.get("sampling_slug", "centroid"))
    require(bool(sampling_slug) and Path(sampling_slug).name == sampling_slug, "invalid sampling_slug")
    derived_spec = piecewise_derived_coordinate_spec(
        contract.get("derived_coordinate"), cv_names
    )
    require(
        derived_spec is None
        or (
            profile == "water_ionization_opes"
            and len(cv_names) == 2
            and bias_mode != "bead_density_shared"
        ),
        "derived_coordinate is available only in the two-CV water-ionization profile",
    )
    configured_labels = contract["plots"].get("cv_labels", {})
    cv_labels = tuple(
        str(configured_labels.get(name, default_cv_label(name))) for name in cv_names
    )
    centroid = np.column_stack([field(centroid_data, fields, name)[selected] for name in sampling_cv_names])
    bead_arrays = []
    bead_selections = []
    for bead_fields, bead_data in bead_tables:
        bead_time_fs = time_values_to_fs(
            field(bead_data, bead_fields, "time"), bead_time_scale_to_fs
        )
        bead_selected = aligned_time_indices(
            bead_time_fs + bead_offset_fs, time_fs[selected]
        )
        bead_selections.append(bead_selected)
        bead_arrays.append(
            np.column_stack(
                [
                    field(bead_data, bead_fields, name)[bead_selected]
                    for name in bead_cv_names
                ]
            )
        )
    beads = np.stack(bead_arrays, axis=1)
    if bias_mode == "bead_density_shared":
        centroid = np.mean(beads, axis=1)

    def selected_bead_field(column: str) -> np.ndarray:
        return np.stack(
            [
                field(bead_data, bead_fields, column)[bead_selected]
                for (bead_fields, bead_data), bead_selected in zip(
                    bead_tables, bead_selections
                )
            ],
            axis=1,
        )

    diagnostic_spec = diagnostic_cv_spec(contract.get("diagnostic_cv"))
    diagnostic_sampling = None
    diagnostic_beads = None
    diagnostic_mean_error = None
    if diagnostic_spec is not None:
        diagnostic_beads = selected_bead_field(str(diagnostic_spec["bead_column"]))
        if bias_mode == "bead_density_shared":
            diagnostic_sampling = np.mean(diagnostic_beads, axis=1)
            diagnostic_mean_error = 0.0
        else:
            diagnostic_sampling = field(
                centroid_data, fields, str(diagnostic_spec["sampling_column"])
            )[selected]
            diagnostic_mean_error = float(
                np.max(np.abs(diagnostic_sampling - np.mean(diagnostic_beads, axis=1)))
            )
            require(
                diagnostic_mean_error <= float(diagnostic_spec["mean_tolerance"]),
                "printed diagnostic bead mean mismatch",
            )

    derived_centroid = None
    derived_beads = None
    derived_validation = None
    if derived_spec is not None:
        source_name = str(derived_spec["source"])
        printed_column = str(derived_spec["printed_column"])
        sampling_printed_column = str(
            derived_spec.get("sampling_printed_column", printed_column)
        )
        bead_printed_column = str(derived_spec.get("bead_printed_column", printed_column))
        source_component = cv_names.index(source_name)
        derived_centroid = field(centroid_data, fields, sampling_printed_column)[selected]
        derived_beads = np.stack(
            [
                field(bead_data, bead_fields, bead_printed_column)[bead_selected]
                for (bead_fields, bead_data), bead_selected in zip(
                    bead_tables, bead_selections
                )
            ],
            axis=1,
        )
        transform_options = {
            "switch": float(derived_spec["switch"]),
            "offset": float(derived_spec["offset"]),
            "linear_shift": float(derived_spec["linear_shift"]),
            "log_scale": float(derived_spec["log_scale"]),
            "log_reference": float(derived_spec["log_reference"]),
        }
        centroid_validation = validate_piecewise_logdistance_printed(
            centroid[:, source_component],
            derived_centroid,
            tolerance=float(derived_spec["printed_transform_tolerance"]),
            **transform_options,
        )
        bead_validations = [
            validate_piecewise_logdistance_printed(
                beads[:, bead, source_component],
                derived_beads[:, bead],
                tolerance=float(derived_spec["printed_transform_tolerance"]),
                **transform_options,
            )
            for bead in range(beads.shape[1])
        ]
        derived_validation = {
            "centroid": centroid_validation,
            "bead_maximum_absolute_error": max(
                value["maximum_absolute_error"] for value in bead_validations
            ),
            "all_maximum_absolute_error": max(
                [centroid_validation["maximum_absolute_error"]]
                + [value["maximum_absolute_error"] for value in bead_validations]
            ),
        }

    weight_kind = str(reweight.get("weight_kind", "quasi_static_opes"))
    bias_column_value = reweight.get("bias_column")
    bias_column = None if bias_column_value is None else str(bias_column_value)
    bias_ev = None
    shared_diagnostic_max_deltas: Dict[str, float] = {}
    if bias_column is not None:
        if bias_mode == "bead_density_shared":
            bias_ev = total_bias_energy(
                bias_mode,
                bead_bias_energies=selected_bead_field(bias_column),
            )
        else:
            bias_ev = total_bias_energy(
                bias_mode,
                sampling_bias_energy=field(centroid_data, fields, bias_column)[selected],
            )
    opes_bias_ev = bias_ev
    extra_bias_columns = reweight.get("extra_bias_columns", [])
    require(isinstance(extra_bias_columns, list), "extra_bias_columns must be a list")
    require(all(isinstance(name, str) and name for name in extra_bias_columns),
            "invalid extra bias column")
    require(len(set(extra_bias_columns)) == len(extra_bias_columns), "duplicate extra bias column")
    require(bias_column not in extra_bias_columns, "primary bias repeated in extra_bias_columns")
    require(not extra_bias_columns or bias_ev is not None, "extra biases require bias_column")
    for name in extra_bias_columns:
        if bias_mode == "bead_density_shared":
            extra = total_bias_energy(bias_mode, bead_bias_energies=selected_bead_field(name))
        else:
            extra = total_bias_energy(
                bias_mode, sampling_bias_energy=field(centroid_data, fields, name)[selected])
        bias_ev = bias_ev + extra
    require(
        profile == "core" or bias_ev is not None,
        "water-ionization diagnostics require bias_column",
    )

    rct_ev = zed = neff = nker = None
    if profile == "water_ionization_opes":
        diagnostic_columns = {
            "rct": str(reweight["rct_column"]),
            "zed": str(reweight.get("zed_column", "opes.zed")),
            "neff": str(reweight.get("neff_column", "opes.neff")),
            "nker": str(reweight.get("nker_column", "opes.nker")),
        }
        if bias_mode == "bead_density_shared":
            diagnostics = {}
            tolerance = float(reweight.get("shared_diagnostic_tolerance", 1e-9))
            require(tolerance >= 0.0, "shared diagnostic tolerance must be nonnegative")
            for name, column in diagnostic_columns.items():
                values = selected_bead_field(column)
                delta = float(np.max(np.max(values, axis=1) - np.min(values, axis=1)))
                require(delta <= tolerance, f"shared OPES diagnostic mismatch: {column}")
                shared_diagnostic_max_deltas[column] = delta
                diagnostics[name] = np.mean(values, axis=1)
            rct_ev, zed, neff, nker = (
                diagnostics["rct"],
                diagnostics["zed"],
                diagnostics["neff"],
                diagnostics["nker"],
            )
        else:
            rct_ev = field(centroid_data, fields, diagnostic_columns["rct"])[selected]
            zed = field(centroid_data, fields, diagnostic_columns["zed"])[selected]
            neff = field(centroid_data, fields, diagnostic_columns["neff"])[selected]
            nker = field(centroid_data, fields, diagnostic_columns["nker"])[selected]
    kbt_ev = float(reweight["kbt_eV"])
    temperature = float(reweight["temperature_K"])
    require(np.isfinite(temperature) and temperature > 0.0, "temperature_K must be finite and positive")
    require(np.isfinite(kbt_ev) and kbt_ev > 0.0, "kbt_eV must be finite and positive")
    require(reweight.get("energy_unit", "eV") == "eV", "energy_unit must be eV; convert energy columns before analysis")
    expected_kbt = KB_EV_PER_K * temperature
    require(abs(kbt_ev - expected_kbt) <= 1e-12, "kBT/temperature mismatch")
    if conditional_spec is not None:
        from molsimflow.postprocess.conditional_path import audit_record
        audit = audit_record(
            conditional_model, centroid[:, 0],
            selected_bead_field(conditional_spec["bead_region_column"]),
            field(centroid_data, fields, conditional_spec["log_normalizer_column"])[selected],
            field(centroid_data, fields, conditional_spec["centroid_bias_column"])[selected],
            bias_ev, coupling=conditional_spec["coupling"], kbt=kbt_ev,
            energy_atol=conditional_spec["energy_atol_eV"],
            normalizer_atol=conditional_spec["log_normalizer_atol"],
        )
        audit["model_sha256"] = conditional_spec["model_sha256"]
        (output / "qc" / "conditional-path.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if probability_spec is not None:
        from molsimflow.postprocess.probability_mixture import audit_record
        require(abs(probability_model["kbt"] - kbt_ev) <= 1e-12,
                "probability mixture kBT mismatch")
        centroid_bias = None
        if probability_model["mode"] == "centroid_probability_mixture":
            centroid_bias = field(centroid_data, fields, probability_spec["centroid_bias_column"])[selected]
        audit = audit_record(
            probability_model,
            None if probability_model["mode"] == "centroid_probability_mixture"
            and probability_model["coupling"] == 0
            else selected_bead_field(probability_spec["bead_bias_column"]),
            bias_ev, field_sha256=probability_model["field_sha256"],
            centroid_bias=centroid_bias, energy_atol=probability_spec["energy_atol_eV"],
        )
        audit["manifest_sha256"] = probability_spec["manifest_sha256"]
        (output / "qc" / "probability-mixture.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    quasi_static_declared = reweight.get("quasi_static") is True
    if weight_kind == "precomputed":
        require(not extra_bias_columns, "precomputed weights cannot also use extra bias columns")
        log_weight_column = str(reweight["log_weight_column"])
        raw_log_weights = frame_log_weights(
            weight_kind,
            precomputed=field(centroid_data, fields, log_weight_column)[selected],
        )
    else:
        require(bias_ev is not None, f"{weight_kind} requires bias_column")
        raw_log_weights = frame_log_weights(
            weight_kind,
            bias_energy=bias_ev,
            kbt=kbt_ev,
            quasi_static=quasi_static_declared,
        )
    log_weights = normalized_log_weights(raw_log_weights)
    weights = np.exp(log_weights)
    require(abs(float(np.sum(weights)) - 1.0) <= 1e-12, "weight normalization failed")

    write_kde_uncertainty(output, reweight, cv_names, beads, raw_log_weights, kbt_ev)

    if len(cv_names) == 1:
        return finalize_core_1d(
            contract=contract,
            output=output,
            source=source,
            reweight=reweight,
            cv_name=str(cv_names[0]),
            cv_label=str(cv_labels[0]),
            sampling_label=sampling_label,
            sampling_slug=sampling_slug,
            bias_mode=bias_mode,
            weight_kind=weight_kind,
            selected_time_ps=selected_time_ps,
            selected_steps=selected_steps,
            sampling=centroid,
            beads=beads,
            raw_log_weights=raw_log_weights,
            bias_ev=bias_ev,
            kbt_ev=kbt_ev,
            sampling_restart_duplicates=centroid_restart_duplicates,
            bead_restart_duplicates=bead_restart_duplicates,
        )

    x_grid = np.linspace(*reweight["grid"][cv_names[0]])
    y_grid = np.linspace(*reweight["grid"][cv_names[1]])
    variants = {name: tuple(values) for name, values in reweight["bandwidth_variants"].items()}
    primary_name = str(reweight["primary_bandwidth"])
    require(primary_name in variants, "primary bandwidth absent")
    require(all(name and Path(name).name == name and name not in {".", ".."}
                for name in variants), "bandwidth variant names must be basenames")
    surfaces_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    supports_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    sensitivity_rows: List[Dict[str, object]] = []
    for variant, bandwidth in variants.items():
        surfaces = compute_surfaces(
            beads,
            centroid,
            raw_log_weights,
            x_grid,
            y_grid,
            bandwidth,
            kbt_ev,
            primary_estimator=primary_estimator,
        )
        surfaces_by_variant[variant] = surfaces
        log_threshold = math.log(float(reweight["relative_density_support"]))
        supports = estimator_support_masks(
            surfaces["log_centroid"],
            surfaces["log_probability_mean"],
            surfaces["log_beads"],
            log_threshold,
        )
        supports_by_variant[variant] = supports
        kcal = {key: surfaces[key] * EV_TO_KCAL_MOL for key in ("centroid", "probability_mean", "free_energy_mean")}
        rows = []
        for iy, y_value in enumerate(y_grid):
            for ix, x_value in enumerate(x_grid):
                rows.append(
                    {
                        cv_names[0]: x_value,
                        cv_names[1]: y_value,
                        **fes_table_columns(
                            kcal, supports, primary_estimator, (iy, ix)
                        ),
                    }
                )
        write_csv(
            output / "fes2d" / f"{variant}.csv", rows,
            [cv_names[0], cv_names[1], *FES_TABLE_FIELDS],
        )
        sensitivity_rows.append(
            {
                "variant": variant,
                "sigma_x": bandwidth[0],
                "sigma_y": bandwidth[1],
                "primary_estimator": primary_estimator,
                "primary_support_points": int(np.count_nonzero(supports[primary_estimator])),
                "probability_mean_support_points": int(np.count_nonzero(supports["probability_mean"])),
                "free_energy_mean_support_points": int(np.count_nonzero(supports["free_energy_mean"])),
                "common_support_points": int(np.count_nonzero(supports["common"])),
            }
        )
    primary_supports = supports_by_variant[primary_name]
    primary_support = primary_supports[primary_estimator]
    require(
        np.count_nonzero(primary_support) >= 2,
        f"empty {primary_estimator} support",
    )
    primary = surfaces_by_variant[primary_name]
    primary_kcal = {key: primary[key] * EV_TO_KCAL_MOL for key in ("centroid", "probability_mean", "free_energy_mean")}
    for row in sensitivity_rows:
        variant = str(row["variant"])
        current = surfaces_by_variant[variant][primary_estimator] * EV_TO_KCAL_MOL
        comparison_support = (
            primary_supports[primary_estimator]
            & supports_by_variant[variant][primary_estimator]
        )
        try:
            count, rmse, maximum = optional_surface_difference_metrics(
                primary_kcal[primary_estimator], current, comparison_support
            )
        except ValueError as exc:
            raise ValueError(f"bandwidth comparison failed for {variant}: {exc}") from exc
        row["comparison_support_points"] = count
        row["primary_rmse_vs_selected_bandwidth_kcal_mol"] = rmse
        row["primary_max_abs_vs_selected_bandwidth_kcal_mol"] = maximum
    write_csv(
        output / "qc" / "bandwidth-sensitivity.csv", sensitivity_rows,
        [
            "variant", "sigma_x", "sigma_y", "primary_estimator",
            "primary_support_points", "probability_mean_support_points",
            "free_energy_mean_support_points", "common_support_points",
            "comparison_support_points",
            "primary_rmse_vs_selected_bandwidth_kcal_mol",
            "primary_max_abs_vs_selected_bandwidth_kcal_mol",
        ],
    )

    marginal_tables: Dict[str, Dict[str, np.ndarray]] = {}
    for component, name in enumerate(cv_names):
        grid = x_grid if component == 0 else y_grid
        curves = compute_marginals(
            beads,
            centroid,
            raw_log_weights,
            grid,
            variants[primary_name][component],
            component,
            kbt_ev,
            primary_estimator=primary_estimator,
        )
        curves_kcal = {key: values * EV_TO_KCAL_MOL for key, values in curves.items()}
        log_threshold = math.log(float(reweight["relative_density_support"]))
        supports = estimator_support_masks(
            curves["log_centroid"],
            curves["log_probability_mean"],
            curves["log_beads"],
            log_threshold,
        )
        marginal_tables[name] = {
            **{key: curves_kcal[key] for key in ("centroid", "probability_mean", "free_energy_mean")},
            **{f"{key}_support": value for key, value in supports.items()},
        }
        write_csv(
            output / "fes1d" / f"{name}.csv",
            (
                {
                    name: value,
                    **fes_table_columns(
                        curves_kcal, supports, primary_estimator, index
                    ),
                }
                for index, value in enumerate(grid)
            ),
            [name, *FES_TABLE_FIELDS],
        )
        plot_fes1d(
            output, name, grid, curves_kcal, supports,
            float(reweight["plot_max_kcal_mol"]), cv_labels[component],
            sampling_label=sampling_label,
            bias_mode=bias_mode,
        )

    derived_coordinate_summary = None
    if derived_spec is not None:
        require(derived_validation is not None, "derived-coordinate validation absent")
        source_name = str(derived_spec["source"])
        target_name = str(derived_spec["target"])
        source_component = cv_names.index(source_name)
        source_grid = x_grid if source_component == 0 else y_grid
        transform_options = {
            "switch": float(derived_spec["switch"]),
            "offset": float(derived_spec["offset"]),
            "linear_shift": float(derived_spec["linear_shift"]),
            "log_scale": float(derived_spec["log_scale"]),
            "log_reference": float(derived_spec["log_reference"]),
        }
        source_marginals = marginal_tables[source_name]
        derived_curves_kcal: Dict[str, np.ndarray] = {}
        target_grid = None
        density_jacobian = None
        for key in ("centroid", "probability_mean", "free_energy_mean"):
            current_grid, current_jacobian, current_curve = (
                transform_piecewise_logdistance_fes(
                    source_grid,
                    source_marginals[key],
                    kbt_ev * EV_TO_KCAL_MOL,
                    zero_minimum=False,
                    **transform_options,
                )
            )
            if target_grid is None:
                target_grid = current_grid
                density_jacobian = current_jacobian
            else:
                require(np.array_equal(current_grid, target_grid), "derived grids differ")
                require(
                    np.array_equal(current_jacobian, density_jacobian),
                    "derived Jacobians differ",
                )
            derived_curves_kcal[key] = current_curve
        require(target_grid is not None and density_jacobian is not None, "derived grid absent")
        primary_zero = np.min(
            derived_curves_kcal[primary_estimator][
                source_marginals[f"{primary_estimator}_support"]
            ]
        )
        for key in ("probability_mean", "free_energy_mean"):
            derived_curves_kcal[key] -= primary_zero
        derived_curves_kcal["centroid"] -= np.min(
            derived_curves_kcal["centroid"][source_marginals["centroid_support"]])
        derived_marginal_supports = {
            key: np.asarray(source_marginals[f"{key}_support"], dtype=bool).copy()
            for key in (
                "centroid",
                "probability_mean",
                "free_energy_mean",
                "common",
                "estimator_common",
                "all_common",
                "sampling_probability_common",
                "sampling_free_energy_common",
            )
        }
        require(
            all(
                np.array_equal(
                    derived_marginal_supports[key],
                    source_marginals[f"{key}_support"],
                )
                for key in derived_marginal_supports
            ),
            "derived marginal support changed",
        )
        jacobian_name = f"d{source_name}_d{target_name}"
        write_csv(
            output / "fes1d" / f"{target_name}.csv",
            (
                {
                    target_name: target_grid[index],
                    source_name: source_grid[index],
                    jacobian_name: density_jacobian[index],
                    **fes_table_columns(
                        derived_curves_kcal,
                        derived_marginal_supports,
                        primary_estimator,
                        index,
                    ),
                }
                for index in range(len(target_grid))
            ),
            [
                target_name,
                source_name,
                jacobian_name,
                *FES_TABLE_FIELDS,
            ],
        )
        plot_fes1d(
            output,
            target_name,
            target_grid,
            derived_curves_kcal,
            derived_marginal_supports,
            float(reweight["plot_max_kcal_mol"]),
            str(derived_spec["label"]),
            sampling_label=sampling_label,
            bias_mode=bias_mode,
        )

        surface_axis = 1 if source_component == 0 else 0
        derived_surfaces_kcal: Dict[str, np.ndarray] = {}
        for key in ("centroid", "probability_mean", "free_energy_mean"):
            current_grid, current_jacobian, current_surface = (
                transform_piecewise_logdistance_fes(
                    source_grid,
                    primary_kcal[key],
                    kbt_ev * EV_TO_KCAL_MOL,
                    axis=surface_axis,
                    zero_minimum=False,
                    **transform_options,
                )
            )
            require(np.array_equal(current_grid, target_grid), "derived 1D/2D grids differ")
            require(
                np.array_equal(current_jacobian, density_jacobian),
                "derived 1D/2D Jacobians differ",
            )
            derived_surfaces_kcal[key] = current_surface
        primary_zero = np.min(
            derived_surfaces_kcal[primary_estimator][
                primary_supports[primary_estimator]
            ]
        )
        for key in ("probability_mean", "free_energy_mean"):
            derived_surfaces_kcal[key] -= primary_zero
        derived_surfaces_kcal["centroid"] -= np.min(
            derived_surfaces_kcal["centroid"][primary_supports["centroid"]])
        derived_supports = {
            key: np.asarray(primary_supports[key], dtype=bool).copy()
            for key in (
                "centroid",
                "probability_mean",
                "free_energy_mean",
                "common",
                "estimator_common",
                "all_common",
                "sampling_probability_common",
                "sampling_free_energy_common",
            )
        }
        require(
            all(
                np.array_equal(derived_supports[key], primary_supports[key])
                for key in derived_supports
            ),
            "derived 2D support changed",
        )
        derived_names = list(cv_names)
        derived_names[source_component] = target_name
        derived_labels = list(cv_labels)
        derived_labels[source_component] = str(derived_spec["label"])
        derived_x_grid = target_grid if source_component == 0 else x_grid
        derived_y_grid = y_grid if source_component == 0 else target_grid
        derived_rows = []
        for iy, y_value in enumerate(derived_y_grid):
            for ix, x_value in enumerate(derived_x_grid):
                source_index = ix if source_component == 0 else iy
                derived_rows.append(
                    {
                        derived_names[0]: x_value,
                        derived_names[1]: y_value,
                        source_name: source_grid[source_index],
                        jacobian_name: density_jacobian[source_index],
                        **fes_table_columns(
                            derived_surfaces_kcal,
                            derived_supports,
                            primary_estimator,
                            (iy, ix),
                        ),
                    }
                )
        derived_surface_name = f"{primary_name}-{'-'.join(derived_names)}"
        write_csv(
            output / "fes2d" / f"{derived_surface_name}.csv",
            derived_rows,
            [
                derived_names[0],
                derived_names[1],
                source_name,
                jacobian_name,
                *FES_TABLE_FIELDS,
            ],
        )
        plot_fes2d(
            output,
            derived_x_grid,
            derived_y_grid,
            derived_surfaces_kcal,
            derived_supports,
            float(reweight["plot_max_kcal_mol"]),
            derived_labels,
            suffix=f"-{derived_surface_name}",
            sampling_label=sampling_label,
            bias_mode=bias_mode,
            protocol_label=sampling_protocol_label(reweight),
            bead_count=beads.shape[1],
        )
        source_zoom = contract["plots"].get("fes_zoom")
        derived_zoom = None
        if source_zoom is not None:
            derived_zoom = [list(bounds) for bounds in source_zoom]
            derived_zoom[source_component] = list(
                inverse_piecewise_logdistance(
                    source_zoom[source_component], **transform_options
                )
            )
            plot_fes2d(
                output,
                derived_x_grid,
                derived_y_grid,
                derived_surfaces_kcal,
                derived_supports,
                float(reweight["plot_max_kcal_mol"]),
                derived_labels,
                zoom=derived_zoom,
                suffix=f"-{derived_surface_name}-sampled-region",
                sampling_label=sampling_label,
                bias_mode=bias_mode,
                protocol_label=sampling_protocol_label(reweight),
                bead_count=beads.shape[1],
            )
        derived_coordinate_summary = {
            "kind": str(derived_spec["kind"]),
            "source": source_name,
            "target": target_name,
            "printed_column": str(derived_spec["printed_column"]),
            "printed_transform_tolerance": float(
                derived_spec["printed_transform_tolerance"]
            ),
            "printed_transform_validation": derived_validation,
            "density_jacobian": jacobian_name,
            "sampling_support_unchanged": True,
            "weighting_unchanged": True,
            "independent_cv": False,
            "fes1d": f"fes1d/{target_name}.csv",
            "fes2d": f"fes2d/{derived_surface_name}.csv",
            "figure": f"figures/fes2d-comparison-{derived_surface_name}.png",
            "sampled_region_figure": (
                f"figures/fes2d-comparison-{derived_surface_name}-sampled-region.png"
                if derived_zoom is not None
                else None
            ),
        }

    block_rows = []
    for block, indices in enumerate(np.array_split(np.arange(len(selected)), int(reweight["blocks"]))):
        block_surface = compute_surfaces(
            beads[indices], centroid[indices], raw_log_weights[indices], x_grid, y_grid,
            variants[primary_name], kbt_ev, primary_estimator=primary_estimator,
        )
        block_weights = block_surface["weights"]
        block_supports = estimator_support_masks(
            block_surface["log_centroid"],
            block_surface["log_probability_mean"],
            block_surface["log_beads"],
            math.log(float(reweight["relative_density_support"])),
        )
        block_comparison_support = (
            primary_supports[primary_estimator]
            & block_supports[primary_estimator]
        )
        comparison_count, block_rmse, block_maximum = optional_surface_difference_metrics(
            primary[primary_estimator] * EV_TO_KCAL_MOL,
            block_surface[primary_estimator] * EV_TO_KCAL_MOL,
            block_comparison_support,
        )
        block_rows.append(
            {
                "block": block + 1,
                "first_time_ps": selected_time_ps[indices[0]],
                "last_time_ps": selected_time_ps[indices[-1]],
                "frames": len(indices),
                "ess": 1.0 / float(np.sum(block_weights**2)),
                "max_weight": float(np.max(block_weights)),
                "primary_estimator": primary_estimator,
                "comparison_support_points": comparison_count,
                "primary_rmse_vs_full_kcal_mol": block_rmse,
                "primary_max_abs_vs_full_kcal_mol": block_maximum,
            }
        )
    write_csv(
        output / "blocks" / "block-diagnostics.csv", block_rows,
        [
            "block", "first_time_ps", "last_time_ps", "frames", "ess",
            "max_weight", "primary_estimator", "comparison_support_points",
            "primary_rmse_vs_full_kcal_mol",
            "primary_max_abs_vs_full_kcal_mol",
        ],
    )

    frame_rows = []
    for local, source_index in enumerate(selected):
        row = {
            "time_ps": selected_time_ps[local],
            "step": int(round(time_fs[source_index] / float(selection["timestep_fs"]))),
            "log_frame_weight": raw_log_weights[local],
            "normalized_weight": weights[local],
        }
        if bias_ev is not None:
            row["bias_eV"] = bias_ev[local]
            row["bias_kcal_mol"] = bias_ev[local] * EV_TO_KCAL_MOL
        if rct_ev is not None and zed is not None and neff is not None and nker is not None:
            row.update(
                {
                    "rct_eV": rct_ev[local],
                    "rct_kcal_mol": rct_ev[local] * EV_TO_KCAL_MOL,
                    "opes_zed": zed[local],
                    "opes_neff": neff[local],
                    "opes_nker": nker[local],
                }
            )
        for component, name in enumerate(cv_names):
            row[f"{sampling_slug}_{name}"] = centroid[local, component]
            row[f"bead_mean_{name}"] = float(np.mean(beads[local, :, component]))
            row[f"bead_std_{name}"] = float(np.std(beads[local, :, component]))
        if derived_spec is not None:
            require(
                derived_centroid is not None and derived_beads is not None,
                "derived-coordinate values absent",
            )
            target_name = str(derived_spec["target"])
            row[f"{sampling_slug}_{target_name}"] = derived_centroid[local]
            row[f"bead_mean_{target_name}"] = float(
                np.mean(derived_beads[local, :])
            )
            row[f"bead_std_{target_name}"] = float(
                np.std(derived_beads[local, :])
            )
        if diagnostic_spec is not None:
            require(
                diagnostic_sampling is not None and diagnostic_beads is not None,
                "diagnostic CV values absent",
            )
            diagnostic_name = str(diagnostic_spec["name"])
            row[f"{sampling_slug}_{diagnostic_name}"] = diagnostic_sampling[local]
            row[f"bead_mean_{diagnostic_name}"] = float(
                np.mean(diagnostic_beads[local, :])
            )
            row[f"bead_std_{diagnostic_name}"] = float(
                np.std(diagnostic_beads[local, :])
            )
        frame_rows.append(row)
    frame_fields = list(frame_rows[0])
    write_csv(output / "tables" / "frame-series.csv", frame_rows, frame_fields)
    bead_cv_fields = ["time_ps", "bead"]
    if bias_ev is not None:
        bead_cv_fields.append("bias_kcal_mol")
    bead_cv_fields.extend([cv_names[0], cv_names[1]])
    if derived_spec is not None:
        bead_cv_fields.append(str(derived_spec["target"]))
    if diagnostic_spec is not None:
        bead_cv_fields.append(str(diagnostic_spec["name"]))
    write_csv(
        output / "tables" / "bead-cv-long.csv",
        (
            {
                "time_ps": selected_time_ps[frame],
                "bead": bead + 1,
                **(
                    {"bias_kcal_mol": bias_ev[frame] * EV_TO_KCAL_MOL}
                    if bias_ev is not None
                    else {}
                ),
                cv_names[0]: beads[frame, bead, 0],
                cv_names[1]: beads[frame, bead, 1],
                **(
                    {
                        str(derived_spec["target"]): derived_beads[frame, bead]
                    }
                    if derived_spec is not None and derived_beads is not None
                    else {}
                ),
                **(
                    {
                        str(diagnostic_spec["name"]): diagnostic_beads[frame, bead]
                    }
                    if diagnostic_spec is not None and diagnostic_beads is not None
                    else {}
                ),
            }
            for frame in range(beads.shape[0])
            for bead in range(beads.shape[1])
        ),
        bead_cv_fields,
    )
    filtered_colvar = output / "inputs" / artifact_basename(
        selection, "filtered_colvar_name", "COLVAR.after-50ps"
    )
    filtered_data = centroid_data
    if bias_mode == "bead_density_shared":
        filtered_data = centroid_data.copy()
        for component, column in enumerate(sampling_cv_names):
            filtered_data[selected, tuple(fields).index(column)] = centroid[:, component]
        if bias_column is not None and bias_ev is not None:
            filtered_data[selected, tuple(fields).index(bias_column)] = bias_ev
    if extra_bias_columns:
        filtered_data = filtered_data.copy()
        filtered_data[selected, tuple(fields).index(bias_column)] = bias_ev
    write_filtered_colvar(filtered_colvar, fields, filtered_data, selected)

    if profile == "core":
        return finalize_core_2d(
            contract=contract,
            output=output,
            source=source,
            reweight=reweight,
            cv_names=cv_names,
            cv_labels=cv_labels,
            sampling_label=sampling_label,
            sampling_slug=sampling_slug,
            bias_mode=bias_mode,
            weight_kind=weight_kind,
            selected_time_ps=selected_time_ps,
            sampling=centroid,
            beads=beads,
            raw_log_weights=raw_log_weights,
            bias_ev=bias_ev,
            kbt_ev=kbt_ev,
            primary_name=primary_name,
            primary=primary,
            primary_kcal=primary_kcal,
            primary_supports=primary_supports,
            variants=variants,
            filtered_colvar=filtered_colvar,
            sampling_restart_duplicates=centroid_restart_duplicates,
            bead_restart_duplicates=bead_restart_duplicates,
        )

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
    require(all(all(int(step) in rows for step in selected_steps) for rows in thermo_maps), "thermo steps missing")
    temperatures = np.asarray([[rows[int(step)]["Temp"] for rows in thermo_maps] for step in selected_steps])
    potential = np.asarray([[rows[int(step)]["PotEng"] for rows in thermo_maps] for step in selected_steps])
    spring_per_bead = np.asarray([[rows[int(step)]["f_fpimd[2]"] for rows in thermo_maps] for step in selected_steps])
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
        ["time_ps", "bead", "temperature_raw_K", "temperature_scaled_by_P_K", "potential_energy_eV", "kinetic_energy_eV", "spring_energy_eV", "primitive_ke_estimator_eV", "virial_ke_estimator_eV", "centroid_virial_ke_estimator_eV"],
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
    reconstruction_tolerance_value = ionization_config.get(
        "reconstruction_tolerance", 1e-9
    )
    reconstruction_tolerance = (
        None
        if reconstruction_tolerance_value is None
        else float(reconstruction_tolerance_value)
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
        "time_ps", "step", "representation", "bead", "recorded_score",
        "reconstructed_score", "absolute_reconstruction_error",
        "minimum_soft_occupancy", "minimum_soft_occupancy_O_id",
        "maximum_soft_occupancy", "maximum_soft_occupancy_O_id",
        "minimum_hard_H_count", "maximum_hard_H_count",
        "hard_undercoordinated_centers", "hard_overcoordinated_centers",
        "hard_pair_like",
    ]
    write_csv(
        output / "tables" / "ionization-candidates.csv", candidate_rows, candidate_fields
    )
    ionization_series = {sampling_slug: centroid[:, ionization_component]}
    ionization_series.update(
        {
            f"bead_{bead + 1}": beads[:, bead, ionization_component]
            for bead in range(beads.shape[1])
        }
    )
    ionization_series["maximum_bead"] = np.max(beads[:, :, ionization_component], axis=1)
    threshold_rows = threshold_run_rows(
        selected_time_ps, ionization_series, ionization_thresholds
    )
    write_csv(
        output / "qc" / "ionization-threshold-summary.csv",
        threshold_rows,
        [
            "representation", "threshold", "frames_at_or_above", "fraction_frames",
            "contiguous_runs", "longest_run_frames", "longest_run_ps",
        ],
    )

    primary_supports = supports_by_variant[primary_name]
    plot_fes2d(
        output, x_grid, y_grid, primary_kcal, primary_supports,
        float(reweight["plot_max_kcal_mol"]), cv_labels,
        sampling_label=sampling_label,
        bias_mode=bias_mode,
        protocol_label=sampling_protocol_label(reweight),
        bead_count=beads.shape[1],
    )
    plot_fes_differences(
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
        plot_fes2d(
            output, x_grid, y_grid, primary_kcal, primary_supports,
            float(reweight["plot_max_kcal_mol"]), cv_labels, zoom=zoom, suffix="-sampled-region",
            sampling_label=sampling_label,
            bias_mode=bias_mode,
            protocol_label=sampling_protocol_label(reweight),
            bead_count=beads.shape[1],
        )
        plot_fes_differences(
            output, x_grid, y_grid, primary_kcal, primary_supports,
            float(reweight["difference_max_kcal_mol"]), cv_labels, zoom=zoom, suffix="-sampled-region",
            sampling_label=sampling_label,
            bias_mode=bias_mode,
        )
    plot_bead_cv_bias(
        output, selected_time_ps, centroid, beads, opes_bias_ev * EV_TO_KCAL_MOL, cv_labels,
        int(contract["plots"].get("scatter_stride", 5)),
        sampling_label=sampling_label,
        sampling_slug=sampling_slug,
        protocol_label=sampling_protocol_label(reweight),
    )
    plot_cv_spread(output, selected_time_ps, beads, cv_labels)
    if diagnostic_spec is not None:
        require(
            diagnostic_sampling is not None and diagnostic_beads is not None,
            "diagnostic CV values absent",
        )
        plot_diagnostic_cv_bias(
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
    plot_opes(
        output, selected_time_ps, opes_bias_ev * EV_TO_KCAL_MOL, rct_ev * EV_TO_KCAL_MOL,
        zed, neff, nker, raw_log_weights,
        kernel_time_ps, sigma_x, sigma_y, weights_are_log=True,
    )
    plot_thermo(
        output, selected_time_ps, temperatures, potential, global_values, beads.shape[1],
        float(reweight["temperature_K"]),
    )
    plot_ring_spread(output, spread_rows, float(selection["timestep_fs"]))
    plot_ionization_events(
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
    diagnostic_gap = (primary_kcal["free_energy_mean"] - primary_kcal["probability_mean"])
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
                "precomputed log frame weight" if weight_kind == "precomputed"
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
                "defined by the supplied frame weights" if weight_kind == "precomputed"
                else "declared bias terms removed; undeclared potentials retained"
            ),
            "temperature_K": float(reweight["temperature_K"]),
            "kbt_eV": kbt_ev,
            "ess": 1.0 / float(np.sum(weights**2)),
            "ess_fraction": 1.0 / float(np.sum(weights**2)) / len(weights),
            "maximum_normalized_weight": float(np.max(weights)),
            "bias_range_kcal_mol": [float(np.min(bias_ev) * EV_TO_KCAL_MOL), float(np.max(bias_ev) * EV_TO_KCAL_MOL)],
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
            "probability_mean_label": estimator_plot_labels(bias_mode)[
                "probability_mean"
            ],
            "free_energy_mean_label": estimator_plot_labels(bias_mode)["free_energy_mean"],
            "primary_bandwidth": list(variants[primary_name]),
            "primary_bandwidth_name": primary_name,
            "common_support_points": int(
                np.count_nonzero(primary_supports["estimator_common"])
            ),
            "estimator_pair_support_points": int(
                np.count_nonzero(primary_supports["estimator_common"])
            ),
            "sampling_support_points": int(np.count_nonzero(primary_supports["centroid"])),
            "probability_mean_support_points": int(np.count_nonzero(primary_supports["probability_mean"])),
            "free_energy_mean_support_points": int(np.count_nonzero(primary_supports["free_energy_mean"])),
            **diagnostic_gap_metrics(
                diagnostic_gap, primary_supports["estimator_common"]
            ),
            "minimum_raw_jensen_gap_kcal_mol": float(np.min(raw_gap)),
        },
        "pimd": {
            "mean_scaled_temperature_K": float(np.mean(temperatures) / beads.shape[1]),
            "sd_scaled_temperature_K": float(np.std(temperatures / beads.shape[1])),
            "mean_centroid_virial_ke_eV": float(np.mean(global_values["f_fpimd[7]"])),
            "sd_centroid_virial_ke_eV": float(np.std(global_values["f_fpimd[7]"])),
            "mean_ring_spread_H_A": float(np.mean([row.get("rg_H_A", math.nan) for row in spread_rows])),
            "mean_ring_spread_O_A": float(np.mean([row.get("rg_O_A", math.nan) for row in spread_rows])),
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
            "nlist_numerical": contract["claims"].get(
                "nlist_numerical", "NOT_APPLICABLE"
            ),
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
    (output / "qc" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "provenance" / "analysis-contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def get_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reweight centroid-, bead-mean-, or bead-density-biased PIMD"
    )
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = get_args(argv)
    try:
        summary = analyze(args.contract, args.output)
    except Exception as exc:
        print(f"PIMD reweighting failed: {exc}")
        return 1
    print(f"PIMD_REWEIGHT_{summary['status']}")
    return 0


def plot_fes2d(
    output: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces_kcal: Mapping[str, np.ndarray],
    supports: Mapping[str, np.ndarray],
    max_kcal: float,
    cv_labels: Sequence[str],
    *,
    zoom: Sequence[Sequence[float]] | None = None,
    suffix: str = "",
    sampling_label: str = "Centroid",
    bias_mode: str = "centroid_coord",
    protocol_label: str = "bias",
    bead_count: int | None = None,
) -> None:
    return _plotting.plot_fes2d(output, x_grid, y_grid, surfaces_kcal, supports, max_kcal, cv_labels, zoom=zoom, suffix=suffix, sampling_label=sampling_label, bias_mode=bias_mode, protocol_label=protocol_label, bead_count=bead_count, _save_figure=save_figure)


def plot_fes_differences(
    output: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces_kcal: Mapping[str, np.ndarray],
    supports: Mapping[str, np.ndarray],
    max_abs_kcal: float,
    cv_labels: Sequence[str],
    *,
    zoom: Sequence[Sequence[float]] | None = None,
    suffix: str = "",
    sampling_label: str = "Centroid",
    bias_mode: str = "centroid_coord",
) -> None:
    return _plotting.plot_fes_differences(output, x_grid, y_grid, surfaces_kcal, supports, max_abs_kcal, cv_labels, zoom=zoom, suffix=suffix, sampling_label=sampling_label, bias_mode=bias_mode, _save_figure=save_figure)


def plot_fes1d(
    output: Path,
    name: str,
    grid: np.ndarray,
    curves_kcal: Mapping[str, np.ndarray],
    supports: Mapping[str, np.ndarray],
    max_kcal: float,
    cv_label: str,
    *,
    sampling_label: str = "Centroid",
    bias_mode: str = "centroid_coord",
) -> None:
    return _plotting.plot_fes1d(output, name, grid, curves_kcal, supports, max_kcal, cv_label, sampling_label=sampling_label, bias_mode=bias_mode, _save_figure=save_figure)


def plot_bead_cv_bias(
    output: Path,
    time_ps: np.ndarray,
    centroid: np.ndarray,
    beads: np.ndarray,
    bias_kcal: np.ndarray,
    cv_labels: Sequence[str],
    plot_stride: int,
    *,
    sampling_label: str = "Centroid",
    sampling_slug: str = "centroid",
    protocol_label: str = "bias",
) -> None:
    return _plotting.plot_bead_cv_bias(output, time_ps, centroid, beads, bias_kcal, cv_labels, plot_stride, sampling_label=sampling_label, sampling_slug=sampling_slug, protocol_label=protocol_label, _save_figure=save_figure)


def plot_cv_spread(output: Path, time_ps: np.ndarray, beads: np.ndarray, names: Sequence[str]) -> None:
    return _plotting.plot_cv_spread(output, time_ps, beads, names, _save_figure=save_figure)


def plot_cv_time_series(
    output: Path,
    time_ps: np.ndarray,
    sampling: np.ndarray,
    beads: np.ndarray,
    cv_labels: Sequence[str],
    sampling_label: str,
) -> None:
    return _plotting.plot_cv_time_series(output, time_ps, sampling, beads, cv_labels, sampling_label, _save_figure=save_figure)


def plot_diagnostic_cv_bias(
    output: Path,
    time_ps: np.ndarray,
    sampling: np.ndarray,
    beads: np.ndarray,
    bias_kcal: np.ndarray,
    label: str,
    sampling_label: str,
    plot_stride: int,
    protocol_label: str = "bias",
) -> None:
    return _plotting.plot_diagnostic_cv_bias(output, time_ps, sampling, beads, bias_kcal, label, sampling_label, plot_stride, protocol_label, _save_figure=save_figure)


def plot_opes(
    output: Path,
    time_ps: np.ndarray,
    bias_kcal: np.ndarray,
    rct_kcal: np.ndarray,
    zed: np.ndarray,
    neff: np.ndarray,
    nker: np.ndarray,
    weights: np.ndarray,
    kernel_time_ps: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    *,
    weights_are_log: bool = False,
) -> None:
    return _plotting.plot_opes(output, time_ps, bias_kcal, rct_kcal, zed, neff, nker, weights, kernel_time_ps, sigma_x, sigma_y, weights_are_log=weights_are_log, _save_figure=save_figure)


def plot_thermo(
    output: Path,
    time_ps: np.ndarray,
    temperatures: np.ndarray,
    potential: np.ndarray,
    global_values: Mapping[str, np.ndarray],
    beads: int,
    target_temperature: float,
) -> None:
    return _plotting.plot_thermo(output, time_ps, temperatures, potential, global_values, beads, target_temperature, _save_figure=save_figure)


def plot_ring_spread(output: Path, rows: Sequence[Mapping[str, float]], timestep_fs: float) -> None:
    return _plotting.plot_ring_spread(output, rows, timestep_fs, _save_figure=save_figure)


def plot_ionization_events(
    output: Path,
    time_ps: np.ndarray,
    centroid: np.ndarray,
    beads: np.ndarray,
    bias_kcal: np.ndarray,
    thresholds: Sequence[float],
    ideal_pair_score: float,
    *,
    sampling_label: str = "Centroid",
    protocol_label: str = "bias",
) -> None:
    return _plotting.plot_ionization_events(output, time_ps, centroid, beads, bias_kcal, thresholds, ideal_pair_score, sampling_label=sampling_label, protocol_label=protocol_label, _save_figure=save_figure)


if __name__ == "__main__":
    raise SystemExit(run())
