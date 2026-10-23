"""PIMD analysis orchestration with explicit stage boundaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .estimators import (
    alternate_estimator,
    compute_marginals,
    compute_surfaces,
    diagnostic_gap_metrics,
    estimator_support_masks,
    optional_surface_difference_metrics,
)
from .reference import run_reference
from .reporting import (
    FES_TABLE_FIELDS,
    estimator_plot_labels,
    fes_table_columns,
    sampling_protocol_label,
    write_csv,
    write_manifest,
)
from .types import (
    AlignedPaths,
    JointResults,
    PreparedInputs,
    WeightedPaths,
    validate_primary_estimator,
)
from .units import EV_TO_KCAL_MOL
from .weights import normalized_log_weights


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def build_joint_outputs(
    contract: Mapping[str, object],
    output: Path,
    profile: str,
    inputs: PreparedInputs,
    aligned: AlignedPaths,
    weighted: WeightedPaths,
    _plotter,
) -> JointResults:
    reweight = inputs.reweight
    bias_mode = inputs.bias_mode
    primary_estimator = inputs.primary_estimator
    cv_names = aligned.cv_names
    sampling_label = aligned.sampling_label
    cv_labels = aligned.cv_labels
    centroid = aligned.centroid
    beads = aligned.beads
    kbt_ev = weighted.kbt_ev
    raw_log_weights = weighted.raw_log_weights
    x_grid = np.linspace(*reweight["grid"][cv_names[0]])
    y_grid = np.linspace(*reweight["grid"][cv_names[1]])
    variants = {name: tuple(values) for name, values in reweight["bandwidth_variants"].items()}
    primary_name = str(reweight["primary_bandwidth"])
    require(primary_name in variants, "primary bandwidth absent")
    require(
        all(name and Path(name).name == name and name not in {".", ".."} for name in variants),
        "bandwidth variant names must be basenames",
    )
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
        kcal = {
            key: surfaces[key] * EV_TO_KCAL_MOL
            for key in ("centroid", "probability_mean", "free_energy_mean")
        }
        rows = []
        for iy, y_value in enumerate(y_grid):
            for ix, x_value in enumerate(x_grid):
                rows.append(
                    {
                        cv_names[0]: x_value,
                        cv_names[1]: y_value,
                        **fes_table_columns(kcal, supports, primary_estimator, (iy, ix)),
                    }
                )
        write_csv(
            output / "fes2d" / f"{variant}.csv",
            rows,
            [cv_names[0], cv_names[1], *FES_TABLE_FIELDS],
        )
        sensitivity_rows.append(
            {
                "variant": variant,
                "sigma_x": bandwidth[0],
                "sigma_y": bandwidth[1],
                "primary_estimator": primary_estimator,
                "primary_support_points": int(np.count_nonzero(supports[primary_estimator])),
                "probability_mean_support_points": int(
                    np.count_nonzero(supports["probability_mean"])
                ),
                "free_energy_mean_support_points": int(
                    np.count_nonzero(supports["free_energy_mean"])
                ),
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
    primary_kcal = {
        key: primary[key] * EV_TO_KCAL_MOL
        for key in ("centroid", "probability_mean", "free_energy_mean")
    }
    for row in sensitivity_rows:
        variant = str(row["variant"])
        current = surfaces_by_variant[variant][primary_estimator] * EV_TO_KCAL_MOL
        comparison_support = (
            primary_supports[primary_estimator] & supports_by_variant[variant][primary_estimator]
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
        output / "qc" / "bandwidth-sensitivity.csv",
        sensitivity_rows,
        [
            "variant",
            "sigma_x",
            "sigma_y",
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
            **{
                key: curves_kcal[key]
                for key in ("centroid", "probability_mean", "free_energy_mean")
            },
            **{f"{key}_support": value for key, value in supports.items()},
        }
        write_csv(
            output / "fes1d" / f"{name}.csv",
            (
                {
                    name: value,
                    **fes_table_columns(curves_kcal, supports, primary_estimator, index),
                }
                for index, value in enumerate(grid)
            ),
            [name, *FES_TABLE_FIELDS],
        )
        _plotter.plot_fes1d(
            output,
            name,
            grid,
            curves_kcal,
            supports,
            float(reweight["plot_max_kcal_mol"]),
            cv_labels[component],
            sampling_label=sampling_label,
            bias_mode=bias_mode,
        )
    return JointResults(
        x_grid=x_grid,
        y_grid=y_grid,
        variants=variants,
        primary_name=primary_name,
        primary_supports=primary_supports,
        primary_support=primary_support,
        primary=primary,
        primary_kcal=primary_kcal,
        marginal_tables=marginal_tables,
        supports_by_variant=supports_by_variant,
    )


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
    _plotter,
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
    require(
        all(name and Path(name).name == name and name not in {".", ".."} for name in variants),
        "bandwidth variant names must be basenames",
    )
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
                "probability_mean_support_points": int(
                    np.count_nonzero(supports["probability_mean"])
                ),
                "free_energy_mean_support_points": int(
                    np.count_nonzero(supports["free_energy_mean"])
                ),
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
        key: primary[key] * EV_TO_KCAL_MOL
        for key in ("centroid", "probability_mean", "free_energy_mean")
    }
    for row in sensitivity_rows:
        variant = str(row["variant"])
        comparison_support = (
            primary_supports[primary_estimator] & supports_by_variant[variant][primary_estimator]
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
                **fes_table_columns(primary_kcal, primary_supports, primary_estimator, index),
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
            primary_supports[primary_estimator] & current_supports[primary_estimator]
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

    _plotter.plot_fes1d(
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
    _plotter.plot_cv_time_series(
        output, selected_time_ps, sampling, beads, [cv_label], sampling_label
    )
    _plotter.plot_cv_spread(output, selected_time_ps, beads, [cv_label])

    labels = estimator_plot_labels(bias_mode)
    raw_gap = (
        np.mean(-kbt_ev * primary["log_beads"], axis=0) + kbt_ev * primary["log_probability_mean"]
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
                "defined by the supplied frame weights"
                if weight_kind == "precomputed"
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
            "estimator_pair_support_points": int(np.count_nonzero(estimator_common_support)),
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
    _plotter,
) -> Dict[str, object]:
    """Write a generic two-CV report without material-specific diagnostics."""
    primary_estimator = validate_primary_estimator(
        str(reweight.get("primary_estimator", "probability_mean"))
    )
    alternate = alternate_estimator(primary_estimator)
    primary_support = primary_supports[primary_estimator]
    estimator_common_support = primary_supports["estimator_common"]
    protocol_label = sampling_protocol_label(reweight)
    _plotter.plot_fes2d(
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
    _plotter.plot_fes_differences(
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
    _plotter.plot_cv_time_series(
        output, selected_time_ps, sampling, beads, cv_labels, sampling_label
    )
    _plotter.plot_cv_spread(output, selected_time_ps, beads, cv_labels)

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
                "defined by the supplied frame weights"
                if weight_kind == "precomputed"
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
            "estimator_pair_support_points": int(np.count_nonzero(estimator_common_support)),
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
