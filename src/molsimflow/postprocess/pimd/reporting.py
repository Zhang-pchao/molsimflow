"""PIMD table, label, manifest and uncertainty output compatibility."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from molsimflow.io.hashes import sha256_file as sha256

from .contract import artifact_basename
from .estimators import (
    compute_surfaces,
    estimator_support_masks,
    optional_surface_difference_metrics,
)
from .types import (
    AlignedPaths,
    JointResults,
    PreparedInputs,
    WeightedPaths,
    validate_bias_mode,
    validate_primary_estimator,
)
from .uncertainty import quantum_kde_block_jackknife
from .units import EV_TO_KCAL_MOL, KCAL_TO_KJ_MOL

FES_TABLE_FIELDS = (
    "sampling_support",
    "probability_mean_support",
    "free_energy_mean_support",
    "common_support",
    "estimator_common_support",
    "primary_support",
    "F_sampling_kcal_mol",
    "F_quantum_probability_mean_kcal_mol",
    "F_quantum_free_energy_mean_kcal_mol",
    "F_primary_kcal_mol",
    # Historical public name retained as a compatibility alias.
    "F_bead_free_energy_mean_diagnostic_kcal_mol",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def fes_table_columns(
    surfaces_kcal: Mapping[str, np.ndarray],
    supports: Mapping[str, np.ndarray],
    primary_estimator: str,
    index: int | Tuple[int, ...],
) -> Dict[str, float | int]:
    """Return schema-3 FES columns plus the historical diagnostic alias."""

    primary = validate_primary_estimator(primary_estimator)
    return {
        "sampling_support": int(supports["centroid"][index]),
        "probability_mean_support": int(supports["probability_mean"][index]),
        "free_energy_mean_support": int(supports["free_energy_mean"][index]),
        "common_support": int(supports["estimator_common"][index]),
        "estimator_common_support": int(supports["estimator_common"][index]),
        "primary_support": int(supports[primary][index]),
        "F_sampling_kcal_mol": float(surfaces_kcal["centroid"][index]),
        "F_quantum_probability_mean_kcal_mol": float(surfaces_kcal["probability_mean"][index]),
        "F_quantum_free_energy_mean_kcal_mol": float(surfaces_kcal["free_energy_mean"][index]),
        "F_primary_kcal_mol": float(surfaces_kcal[primary][index]),
        "F_bead_free_energy_mean_diagnostic_kcal_mol": float(
            surfaces_kcal["free_energy_mean"][index]
        ),
    }


def estimator_plot_labels(bias_mode: str) -> Dict[str, str]:
    """Name both documented bead estimators without assigning policy priority."""
    validate_bias_mode(bias_mode)
    return {
        "probability_mean": "Probability-mean bead FES",
        "free_energy_mean": "Free-energy-mean bead FES",
    }


def default_cv_label(name: str) -> str:
    """Return publication-readable labels for the maintained water-ionization CVs."""
    return {
        "ionization": r"Ionization state, $s_a$ (dimensionless)",
        "iondistance": r"Ion separation, $s_t$ ($\AA$)",
        "logdistance": r"Log ion-separation coordinate, $s'_t$ (dimensionless)",
    }.get(str(name), str(name))


def sampling_protocol_label(reweight: Mapping[str, object]) -> str:
    """Return an explicit plot label for the declared weighting protocol."""
    if "protocol_label" in reweight:
        label = str(reweight["protocol_label"])
        require(bool(label), "protocol_label must not be empty")
        return label
    return {
        "fixed_bias": "fixed bias",
        "quasi_static_opes": "quasi-static OPES",
        "precomputed": "precomputed weights",
    }[str(reweight.get("weight_kind", "quasi_static_opes"))]


def portable_artifact_path(path: Path | str, output: Path) -> str:
    """Represent paths inside a movable output tree without staging prefixes."""
    value = Path(path)
    if not value.is_absolute():
        return str(path)
    try:
        relative = value.relative_to(output)
    except ValueError:
        return str(path)
    return "{output}/" + relative.as_posix()


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write CSV while adding kJ/mol peers for legacy kcal/mol energy columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expanded = []
    kj_columns = {}
    for name in fieldnames:
        expanded.append(name)
        if name.endswith("_kcal_mol"):
            target = name.replace("_kcal_mol", "_kJ_mol")
            expanded.append(target)
            kj_columns[name] = target
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=expanded)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for legacy, target in kj_columns.items():
                row[target] = float(row[legacy]) * KCAL_TO_KJ_MOL
            writer.writerow(row)


def write_filtered_colvar(
    path: Path, fields: Sequence[str], data: np.ndarray, selected: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#! FIELDS " + " ".join(fields) + "\n")
        for row in data[selected]:
            handle.write(" ".join(f"{float(value):.16e}" for value in row) + "\n")


def write_kde_uncertainty(
    output: Path,
    reweight: Mapping[str, object],
    cv_names: Sequence[str],
    beads: np.ndarray,
    raw_log_weights: np.ndarray,
    kbt: float,
) -> None:
    """Write optional whole-frame sampling errors without changing block diagnostics."""
    if "uncertainty" not in reweight:
        return
    config = reweight["uncertainty"]
    require(
        isinstance(config, dict) and set(config) == {"block_frames", "reference_grid_index"},
        "invalid uncertainty configuration",
    )
    grids = [np.linspace(*reweight["grid"][name]) for name in cv_names]
    bandwidth = reweight["bandwidth_variants"][reweight["primary_bandwidth"]]
    result = quantum_kde_block_jackknife(
        beads,
        raw_log_weights,
        grids,
        bandwidth,
        kbt=kbt,
        block_frames=config["block_frames"],
        reference_grid_index=config["reference_grid_index"],
        relative_density_support=float(reweight["relative_density_support"]),
        primary_estimator=str(reweight.get("primary_estimator", "probability_mean")),
    )
    points = np.column_stack([axis.ravel() for axis in np.meshgrid(*grids)])
    fields = list(cv_names) + ["delta_F_eV", "standard_error_eV", "support"]
    rows = []
    for n, point in enumerate(points):
        row = dict(zip(cv_names, point))
        row.update(
            delta_F_eV=result["free_energy_difference"].ravel()[n],
            standard_error_eV=result["standard_error"].ravel()[n],
            support=int(result["support"].ravel()[n]),
        )
        rows.append(row)
    write_csv(output / "blocks" / "quantum-fes-uncertainty.csv", rows, fields)
    metadata = {
        "estimator": str(reweight.get("primary_estimator", "probability_mean")),
        "method": "delete_one_frame_block_jackknife",
        "blocks": result["blocks"],
        "block_frames": result["block_frames"],
        "reference_grid_index": config["reference_grid_index"],
        "reference_coordinates": [
            float(axis[i]) for axis, i in zip(grids, config["reference_grid_index"])
        ],
        "bandwidth": list(bandwidth),
        "relative_density_support": float(reweight["relative_density_support"]),
        "units": "eV",
        "scope": "Fixed-bandwidth sampling standard error; not smoothing bias or a confidence interval.",
    }
    (output / "blocks" / "quantum-fes-uncertainty.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def write_manifest(output: Path) -> None:
    records = []
    for path in sorted((item for item in output.rglob("*") if item.is_file())):
        if path.name == "OUTPUT-SHA256SUMS":
            continue
        records.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "provenance" / "OUTPUT-SHA256SUMS").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )


def write_block_diagnostics(
    contract: Mapping[str, object],
    output: Path,
    profile: str,
    inputs: PreparedInputs,
    aligned: AlignedPaths,
    weighted: WeightedPaths,
    joint: JointResults,
):
    reweight = inputs.reweight
    primary_estimator = inputs.primary_estimator
    selected = aligned.selected
    selected_time_ps = aligned.selected_time_ps
    centroid = aligned.centroid
    beads = aligned.beads
    kbt_ev = weighted.kbt_ev
    raw_log_weights = weighted.raw_log_weights
    x_grid = joint.x_grid
    y_grid = joint.y_grid
    variants = joint.variants
    primary_name = joint.primary_name
    primary_supports = joint.primary_supports
    primary = joint.primary
    block_rows = []
    for block, indices in enumerate(
        np.array_split(np.arange(len(selected)), int(reweight["blocks"]))
    ):
        block_surface = compute_surfaces(
            beads[indices],
            centroid[indices],
            raw_log_weights[indices],
            x_grid,
            y_grid,
            variants[primary_name],
            kbt_ev,
            primary_estimator=primary_estimator,
        )
        block_weights = block_surface["weights"]
        block_supports = estimator_support_masks(
            block_surface["log_centroid"],
            block_surface["log_probability_mean"],
            block_surface["log_beads"],
            math.log(float(reweight["relative_density_support"])),
        )
        block_comparison_support = (
            primary_supports[primary_estimator] & block_supports[primary_estimator]
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
        output / "blocks" / "block-diagnostics.csv",
        block_rows,
        [
            "block",
            "first_time_ps",
            "last_time_ps",
            "frames",
            "ess",
            "max_weight",
            "primary_estimator",
            "comparison_support_points",
            "primary_rmse_vs_full_kcal_mol",
            "primary_max_abs_vs_full_kcal_mol",
        ],
    )


def write_frame_tables(
    contract: Mapping[str, object],
    output: Path,
    profile: str,
    inputs: PreparedInputs,
    aligned: AlignedPaths,
    weighted: WeightedPaths,
):
    selection = inputs.selection
    bias_mode = inputs.bias_mode
    fields = inputs.fields
    centroid_data = aligned.centroid_data
    time_fs = aligned.time_fs
    selected = aligned.selected
    selected_time_ps = aligned.selected_time_ps
    cv_names = aligned.cv_names
    sampling_cv_names = aligned.sampling_cv_names
    sampling_slug = aligned.sampling_slug
    derived_spec = aligned.derived_spec
    centroid = aligned.centroid
    beads = aligned.beads
    diagnostic_spec = aligned.diagnostic_spec
    diagnostic_sampling = aligned.diagnostic_sampling
    diagnostic_beads = aligned.diagnostic_beads
    derived_centroid = aligned.derived_centroid
    derived_beads = aligned.derived_beads
    bias_column = weighted.bias_column
    bias_ev = weighted.bias_ev
    extra_bias_columns = weighted.extra_bias_columns
    rct_ev = weighted.rct_ev
    zed = weighted.zed
    neff = weighted.neff
    nker = weighted.nker
    raw_log_weights = weighted.raw_log_weights
    weights = weighted.weights
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
            row[f"bead_mean_{target_name}"] = float(np.mean(derived_beads[local, :]))
            row[f"bead_std_{target_name}"] = float(np.std(derived_beads[local, :]))
        if diagnostic_spec is not None:
            require(
                diagnostic_sampling is not None and diagnostic_beads is not None,
                "diagnostic CV values absent",
            )
            diagnostic_name = str(diagnostic_spec["name"])
            row[f"{sampling_slug}_{diagnostic_name}"] = diagnostic_sampling[local]
            row[f"bead_mean_{diagnostic_name}"] = float(np.mean(diagnostic_beads[local, :]))
            row[f"bead_std_{diagnostic_name}"] = float(np.std(diagnostic_beads[local, :]))
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
                    {str(derived_spec["target"]): derived_beads[frame, bead]}
                    if derived_spec is not None and derived_beads is not None
                    else {}
                ),
                **(
                    {str(diagnostic_spec["name"]): diagnostic_beads[frame, bead]}
                    if diagnostic_spec is not None and diagnostic_beads is not None
                    else {}
                ),
            }
            for frame in range(beads.shape[0])
            for bead in range(beads.shape[1])
        ),
        bead_cv_fields,
    )
    filtered_colvar = (
        output
        / "inputs"
        / artifact_basename(selection, "filtered_colvar_name", "COLVAR.after-50ps")
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
    return filtered_colvar


def write_unsmoothed_histogram(output, reweight, cv_names, beads, raw_log_weights, kbt_ev):
    """Write an opt-in direct-count probability-mean companion with explicit bins."""
    from .estimators import weighted_unsmoothed_path_histogram

    config = reweight["histogram"]
    require(
        isinstance(config, dict) and set(config) == {"edges"}, "histogram requires explicit edges"
    )
    require(set(config["edges"]) == set(cv_names), "histogram edges must match CV names")
    edges = [np.asarray(config["edges"][name], float) for name in cv_names]
    result = weighted_unsmoothed_path_histogram(beads, raw_log_weights, edges, kbt_ev)
    centers = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    points = np.column_stack([axis.ravel() for axis in np.meshgrid(*centers)])
    fields = list(cv_names) + [
        "probability_mean_kcal_mol",
        "support",
        "weight_mass",
        "observed_beads",
        "observed_complete_frames",
    ]
    rows = []
    for index, point in enumerate(points):
        row = dict(zip(cv_names, point))
        row["probability_mean_kcal_mol"] = result["fes_kcal_mol"].ravel()[index]
        for key in fields[len(cv_names) + 1 :]:
            row[key] = result[key].ravel()[index]
        rows.append(row)
    write_csv(output / "tables" / "unsmoothed-path-histogram.csv", rows, fields)
    (output / "qc" / "unsmoothed-path-histogram.json").write_text(
        json.dumps(
            {
                "estimator": "probability_mean",
                "smoothing": "none",
                "pseudocount": 0,
                "edges": config["edges"],
                "out_of_grid_weight_mass": result["out_of_grid_weight_mass"],
                "normalization": "All complete paths, including observations outside the grid",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
