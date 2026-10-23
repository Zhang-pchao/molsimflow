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

from types import SimpleNamespace
from molsimflow.postprocess.pimd import pipeline as _pipeline, core_reporting as _core_reporting
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
from molsimflow.postprocess.pimd.contract import (
    analysis_profile,
    artifact_basename,
    cv_column_names,
    diagnostic_cv_spec,
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


from molsimflow.postprocess.pimd.contract import ANALYSIS_PROFILES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    """Compatibility entry point for file-byte hashing."""
    return sha256_file(path)


def normalized_log_weights(raw: Sequence[float]) -> np.ndarray:
    """Normalize frame weights while preserving the workflow keyword API."""
    return _normalized_log_weights(raw)


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
    return _plotting.plot_fes2d(
        output,
        x_grid,
        y_grid,
        surfaces_kcal,
        supports,
        max_kcal,
        cv_labels,
        zoom=zoom,
        suffix=suffix,
        sampling_label=sampling_label,
        bias_mode=bias_mode,
        protocol_label=protocol_label,
        bead_count=bead_count,
        _save_figure=save_figure,
    )


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
    return _plotting.plot_fes_differences(
        output,
        x_grid,
        y_grid,
        surfaces_kcal,
        supports,
        max_abs_kcal,
        cv_labels,
        zoom=zoom,
        suffix=suffix,
        sampling_label=sampling_label,
        bias_mode=bias_mode,
        _save_figure=save_figure,
    )


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
    return _plotting.plot_fes1d(
        output,
        name,
        grid,
        curves_kcal,
        supports,
        max_kcal,
        cv_label,
        sampling_label=sampling_label,
        bias_mode=bias_mode,
        _save_figure=save_figure,
    )


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
    return _plotting.plot_bead_cv_bias(
        output,
        time_ps,
        centroid,
        beads,
        bias_kcal,
        cv_labels,
        plot_stride,
        sampling_label=sampling_label,
        sampling_slug=sampling_slug,
        protocol_label=protocol_label,
        _save_figure=save_figure,
    )


def plot_cv_spread(
    output: Path, time_ps: np.ndarray, beads: np.ndarray, names: Sequence[str]
) -> None:
    return _plotting.plot_cv_spread(output, time_ps, beads, names, _save_figure=save_figure)


def plot_cv_time_series(
    output: Path,
    time_ps: np.ndarray,
    sampling: np.ndarray,
    beads: np.ndarray,
    cv_labels: Sequence[str],
    sampling_label: str,
) -> None:
    return _plotting.plot_cv_time_series(
        output, time_ps, sampling, beads, cv_labels, sampling_label, _save_figure=save_figure
    )


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
    return _plotting.plot_diagnostic_cv_bias(
        output,
        time_ps,
        sampling,
        beads,
        bias_kcal,
        label,
        sampling_label,
        plot_stride,
        protocol_label,
        _save_figure=save_figure,
    )


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
    return _plotting.plot_opes(
        output,
        time_ps,
        bias_kcal,
        rct_kcal,
        zed,
        neff,
        nker,
        weights,
        kernel_time_ps,
        sigma_x,
        sigma_y,
        weights_are_log=weights_are_log,
        _save_figure=save_figure,
    )


def plot_thermo(
    output: Path,
    time_ps: np.ndarray,
    temperatures: np.ndarray,
    potential: np.ndarray,
    global_values: Mapping[str, np.ndarray],
    beads: int,
    target_temperature: float,
) -> None:
    return _plotting.plot_thermo(
        output,
        time_ps,
        temperatures,
        potential,
        global_values,
        beads,
        target_temperature,
        _save_figure=save_figure,
    )


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
    return _plotting.plot_ionization_events(
        output,
        time_ps,
        centroid,
        beads,
        bias_kcal,
        thresholds,
        ideal_pair_score,
        sampling_label=sampling_label,
        protocol_label=protocol_label,
        _save_figure=save_figure,
    )


def _compat_plotter():
    return SimpleNamespace(save_figure=save_figure,
        plot_fes2d=plot_fes2d,
        plot_fes_differences=plot_fes_differences,
        plot_fes1d=plot_fes1d,
        plot_bead_cv_bias=plot_bead_cv_bias,
        plot_cv_spread=plot_cv_spread,
        plot_cv_time_series=plot_cv_time_series,
        plot_diagnostic_cv_bias=plot_diagnostic_cv_bias,
        plot_opes=plot_opes,
        plot_thermo=plot_thermo,
        plot_ring_spread=plot_ring_spread,
        plot_ionization_events=plot_ionization_events,
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
) -> Dict[str, object]:
    return _core_reporting.finalize_core_1d(
        contract=contract,
        output=output,
        source=source,
        reweight=reweight,
        cv_name=cv_name,
        cv_label=cv_label,
        sampling_label=sampling_label,
        sampling_slug=sampling_slug,
        bias_mode=bias_mode,
        weight_kind=weight_kind,
        selected_time_ps=selected_time_ps,
        selected_steps=selected_steps,
        sampling=sampling,
        beads=beads,
        raw_log_weights=raw_log_weights,
        bias_ev=bias_ev,
        kbt_ev=kbt_ev,
        sampling_restart_duplicates=sampling_restart_duplicates,
        bead_restart_duplicates=bead_restart_duplicates,
        _plotter=_compat_plotter(),
    )


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
    return _core_reporting.finalize_core_2d(
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
        sampling=sampling,
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
        sampling_restart_duplicates=sampling_restart_duplicates,
        bead_restart_duplicates=bead_restart_duplicates,
        _plotter=_compat_plotter(),
    )


def analyze(contract_path: Path, output: Path) -> Dict[str, object]:
    return _pipeline.analyze(contract_path, output, _plotter=_compat_plotter())


if __name__ == "__main__":
    raise SystemExit(run())
