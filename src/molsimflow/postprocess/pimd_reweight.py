"""PIMD path-bias reweighting and engineering diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from molsimflow.postprocess.pimd_fes import (
    frame_log_weights,
    normalized_log_weights as _normalized_log_weights,
    restart_unique_indices,
    total_bias_energy,
    validate_bias_mode,
)


KB_EV_PER_K = 8.617333262145e-5
EV_TO_KCAL_MOL = 23.06054783061903
ANALYSIS_PROFILES = {"core", "water_ionization_opes"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def analysis_profile(contract: Mapping[str, object]) -> str:
    """Return the generic or case-specific diagnostics profile."""
    profile = str(contract.get("analysis_profile", "water_ionization_opes"))
    require(profile in ANALYSIS_PROFILES, f"unsupported analysis profile: {profile}")
    return profile


def estimator_plot_labels(bias_mode: str) -> Dict[str, str]:
    """Return method-aware plot labels without leaking internal estimator keys."""
    mode = validate_bias_mode(bias_mode)
    suffixes = (
        (" (Lamaire Eq. 8)", " (Lamaire Eq. 10)")
        if mode == "centroid_coord"
        else ("", "")
    )
    return {
        "probability_mean": f"Quantum FES{suffixes[0]}",
        "logmean": f"Bead-logmean diagnostic{suffixes[1]}",
    }


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def adapt_reference_source(source: str) -> str:
    """Apply the minimal non-square-grid fix to the legacy reference driver."""
    original = "x,y=np.meshgrid(grid_cv_x,grid_cv_y)"
    replacement = "x,y=np.meshgrid(grid_cv_x,grid_cv_y,indexing='ij')"
    require(source.count(original) == 1, "unexpected reference meshgrid implementation")
    return source.replace(original, replacement)


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    if axis is None:
        return float(result.squeeze())
    return np.squeeze(result, axis=axis)


def normalized_log_weights(raw: Sequence[float]) -> np.ndarray:
    """Normalize frame weights while preserving the workflow keyword API."""
    return _normalized_log_weights(raw)


def cumulative_weight_diagnostics(
    weights: Sequence[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cumulative Kish ESS fraction and maximum weight share."""
    values = np.asarray(weights, dtype=float)
    require(values.ndim == 1 and values.size > 0, "empty weights")
    require(np.isfinite(values).all() and np.all(values > 0.0), "invalid weights")
    total = np.cumsum(values)
    ess = total**2 / np.cumsum(values**2)
    count = np.arange(1, values.size + 1, dtype=float)
    return ess / count, np.maximum.accumulate(values) / total


def time_window_mask(
    values: Sequence[float] | np.ndarray,
    first: float,
    last: float,
    *,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Select a finite closed time interval with an explicit tolerance."""
    times = np.asarray(values, dtype=float)
    require(times.ndim == 1 and times.size > 0, "invalid time axis")
    require(np.isfinite(times).all(), "non-finite time axis")
    require(float(first) <= float(last), "inverted time window")
    require(float(tolerance) >= 0.0, "negative time-window tolerance")
    return (times >= float(first) - float(tolerance)) & (
        times <= float(last) + float(tolerance)
    )


def aligned_time_indices(
    source: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Match each target to one distinct source frame within a finite tolerance.

    Missing or ambiguous matches raise instead of interpolating or reusing data.
    """
    source_times = np.asarray(source, dtype=float)
    target_times = np.asarray(target, dtype=float)
    require(source_times.ndim == target_times.ndim == 1, "invalid time-grid rank")
    require(source_times.size > 0 and target_times.size > 0, "empty time grid")
    require(np.isfinite(source_times).all(), "non-finite source time grid")
    require(np.isfinite(target_times).all(), "non-finite target time grid")
    require(np.all(np.diff(source_times) > 0.0), "source time grid is not strictly increasing")
    require(np.all(np.diff(target_times) > 0.0), "target time grid is not strictly increasing")
    tolerance = float(tolerance)
    require(np.isfinite(tolerance) and tolerance >= 0.0, "invalid time-grid tolerance")
    # Find the entire closed tolerance interval, not just its right neighbour.
    indices = np.searchsorted(source_times, target_times - tolerance, side="left")
    stops = np.searchsorted(source_times, target_times + tolerance, side="right")
    matches = stops - indices
    require(np.all(matches > 0), "target time is absent from source grid")
    require(np.all(matches == 1), "ambiguous time-grid match within tolerance")
    require(np.all(np.diff(indices) > 0), "target times map to the same source frame")
    return indices


def surface_difference_metrics(
    reference: np.ndarray, current: np.ndarray, support: np.ndarray
) -> Tuple[int, float, float]:
    difference = np.asarray(current, dtype=float) - np.asarray(reference, dtype=float)
    mask = np.asarray(support, dtype=bool)
    require(difference.shape == mask.shape, "surface/support shape mismatch")
    count = int(np.count_nonzero(mask))
    require(count >= 2, "insufficient surface-comparison support")
    selected = difference[mask]
    return count, float(np.sqrt(np.mean(selected**2))), float(np.max(np.abs(selected)))


def reconstruction_within_tolerance(
    error: float, tolerance: float | None
) -> bool:
    return tolerance is None or float(error) <= float(tolerance)


def piecewise_logdistance(
    iondistance: Sequence[float] | np.ndarray,
    *,
    switch: float = 1.0,
    offset: float = 0.03,
    linear_shift: float = 0.9704412,
) -> np.ndarray:
    """Evaluate the Reactive Voronoi piecewise log-distance coordinate."""
    values = np.asarray(iondistance, dtype=float)
    require(np.isfinite(values).all(), "non-finite iondistance")
    require(np.all(values + float(offset) > 0.0), "iondistance is outside transform domain")
    return np.where(
        values < float(switch),
        np.log(values + float(offset)),
        values - float(linear_shift),
    )


def inverse_piecewise_logdistance(
    logdistance: Sequence[float] | np.ndarray,
    *,
    switch: float = 1.0,
    offset: float = 0.03,
    linear_shift: float = 0.9704412,
) -> np.ndarray:
    """Invert the continuous Reactive Voronoi piecewise log-distance map."""
    values = np.asarray(logdistance, dtype=float)
    require(np.isfinite(values).all(), "non-finite logdistance")
    boundary = 0.5 * (
        math.log(float(switch) + float(offset))
        + float(switch)
        - float(linear_shift)
    )
    return np.where(
        values < boundary,
        np.exp(values) - float(offset),
        values + float(linear_shift),
    )


def piecewise_logdistance_jacobian(
    iondistance: Sequence[float] | np.ndarray,
    *,
    switch: float = 1.0,
    offset: float = 0.03,
) -> np.ndarray:
    """Return the absolute d(logdistance)/d(iondistance) density Jacobian."""
    values = np.asarray(iondistance, dtype=float)
    require(np.isfinite(values).all(), "non-finite iondistance")
    require(np.all(values + float(offset) > 0.0), "iondistance is outside Jacobian domain")
    return np.where(values < float(switch), 1.0 / (values + float(offset)), 1.0)


def transform_piecewise_logdistance_fes(
    logdistance_grid: Sequence[float] | np.ndarray,
    free_energy: np.ndarray,
    kbt: float,
    *,
    axis: int = -1,
    switch: float = 1.0,
    offset: float = 0.03,
    linear_shift: float = 0.9704412,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a logdistance FES axis to iondistance with its Jacobian."""
    source_grid = np.asarray(logdistance_grid, dtype=float)
    values = np.asarray(free_energy, dtype=float)
    require(source_grid.ndim == 1 and source_grid.size > 1, "invalid logdistance grid")
    require(np.all(np.diff(source_grid) > 0.0), "logdistance grid is not strictly increasing")
    require(values.ndim > 0, "invalid free-energy array")
    normalized_axis = int(axis) % values.ndim
    require(
        values.shape[normalized_axis] == source_grid.size,
        "FES/logdistance grid shape mismatch",
    )
    require(np.isfinite(values).all(), "non-finite free energy")
    require(float(kbt) > 0.0, "kBT must be positive")
    iondistance = inverse_piecewise_logdistance(
        source_grid,
        switch=switch,
        offset=offset,
        linear_shift=linear_shift,
    )
    jacobian = piecewise_logdistance_jacobian(
        iondistance, switch=switch, offset=offset
    )
    correction_shape = [1] * values.ndim
    correction_shape[normalized_axis] = source_grid.size
    transformed = values - float(kbt) * np.log(jacobian).reshape(correction_shape)
    transformed -= float(np.min(transformed))
    return iondistance, jacobian, transformed


def validate_piecewise_logdistance_printed(
    logdistance: Sequence[float] | np.ndarray,
    iondistance: Sequence[float] | np.ndarray,
    *,
    tolerance: float,
    switch: float = 1.0,
    offset: float = 0.03,
    linear_shift: float = 0.9704412,
) -> Dict[str, float]:
    """Fail closed unless printed source and derived coordinates agree."""
    source = np.asarray(logdistance, dtype=float)
    printed = np.asarray(iondistance, dtype=float)
    require(source.shape == printed.shape and source.size > 0, "printed transform shape mismatch")
    require(np.isfinite(source).all(), "non-finite printed logdistance")
    require(np.isfinite(printed).all(), "non-finite printed iondistance")
    require(float(tolerance) >= 0.0, "negative printed-transform tolerance")
    forward_error = float(
        np.max(
            np.abs(
                piecewise_logdistance(
                    printed,
                    switch=switch,
                    offset=offset,
                    linear_shift=linear_shift,
                )
                - source
            )
        )
    )
    inverse_error = float(
        np.max(
            np.abs(
                inverse_piecewise_logdistance(
                    source,
                    switch=switch,
                    offset=offset,
                    linear_shift=linear_shift,
                )
                - printed
            )
        )
    )
    require(
        forward_error <= float(tolerance),
        "printed logdistance/iondistance transform mismatch",
    )
    return {
        "forward_maximum_absolute_error": forward_error,
        "inverse_maximum_absolute_error": inverse_error,
        "maximum_absolute_error": forward_error,
    }


def piecewise_derived_coordinate_spec(
    config: Mapping[str, object] | None,
    cv_names: Sequence[str],
) -> Dict[str, object] | None:
    """Validate the optional deterministic derived-coordinate contract."""
    if config is None:
        return None
    require(isinstance(config, Mapping), "derived_coordinate must be an object")
    require(config.get("kind") == "piecewise_logdistance", "unsupported derived coordinate")
    source = str(config.get("source", ""))
    target = str(config.get("target", ""))
    printed_column = str(config.get("printed_column", ""))
    sampling_printed_column = str(
        config.get("sampling_printed_column", printed_column)
    )
    bead_printed_column = str(config.get("bead_printed_column", printed_column))
    require(source in cv_names, "derived-coordinate source is not a biased CV")
    require(target not in cv_names, "derived coordinate must not be an independent biased CV")
    for label, value in (
        ("target", target),
        ("printed_column", printed_column),
        ("sampling_printed_column", sampling_printed_column),
        ("bead_printed_column", bead_printed_column),
    ):
        require(bool(value) and Path(value).name == value, f"invalid derived-coordinate {label}")
    switch = float(config.get("switch", 1.0))
    offset = float(config.get("offset", 0.03))
    linear_shift = float(config.get("linear_shift", 0.9704412))
    tolerance = float(config.get("printed_transform_tolerance", 1e-12))
    require(switch + offset > 0.0, "invalid piecewise-logdistance domain")
    require(tolerance >= 0.0, "negative printed-transform tolerance")
    return {
        "kind": "piecewise_logdistance",
        "source": source,
        "target": target,
        "printed_column": printed_column,
        "sampling_printed_column": sampling_printed_column,
        "bead_printed_column": bead_printed_column,
        "switch": switch,
        "offset": offset,
        "linear_shift": linear_shift,
        "printed_transform_tolerance": tolerance,
        "label": str(config.get("label", target)),
    }


def weighted_log_kde_1d(
    samples: np.ndarray, log_weights: np.ndarray, grid: np.ndarray, bandwidth: float
) -> np.ndarray:
    require(float(bandwidth) > 0.0, "bandwidth must be positive")
    exponent = log_weights[None, :] - 0.5 * (
        (grid[:, None] - np.asarray(samples, dtype=float)[None, :]) / float(bandwidth)
    ) ** 2
    return np.asarray(logsumexp(exponent, axis=1)) - math.log(
        float(bandwidth) * math.sqrt(2.0 * math.pi)
    )


def weighted_log_kde_2d(
    samples: np.ndarray,
    log_weights: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth: Sequence[float],
    *,
    chunk_points: int = 128,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    require(samples.ndim == 2 and samples.shape[1] == 2, "2D sample shape mismatch")
    hx, hy = (float(value) for value in bandwidth)
    require(hx > 0.0 and hy > 0.0, "bandwidths must be positive")
    xx, yy = np.meshgrid(x_grid, y_grid)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    output = np.empty(points.shape[0], dtype=float)
    for start in range(0, points.shape[0], int(chunk_points)):
        stop = min(points.shape[0], start + int(chunk_points))
        current = points[start:stop]
        exponent = (
            log_weights[None, :]
            - 0.5 * ((current[:, 0, None] - samples[None, :, 0]) / hx) ** 2
            - 0.5 * ((current[:, 1, None] - samples[None, :, 1]) / hy) ** 2
        )
        output[start:stop] = np.asarray(logsumexp(exponent, axis=1))
    output -= math.log(2.0 * math.pi * hx * hy)
    return output.reshape(len(y_grid), len(x_grid))


def compute_surfaces(
    beads: np.ndarray,
    centroid: np.ndarray,
    raw_log_weights: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth: Sequence[float],
    kbt_ev: float,
) -> Dict[str, np.ndarray]:
    beads = np.asarray(beads, dtype=float)
    centroid = np.asarray(centroid, dtype=float)
    require(beads.ndim == 3 and beads.shape[2] == 2, "beads must be frame x bead x 2")
    require(centroid.shape == (beads.shape[0], 2), "centroid/bead shape mismatch")
    require(np.isfinite(beads).all() and np.isfinite(centroid).all(), "non-finite CV values")
    log_weights = normalized_log_weights(raw_log_weights)
    log_beads = np.asarray(
        [
            weighted_log_kde_2d(beads[:, bead, :], log_weights, x_grid, y_grid, bandwidth)
            for bead in range(beads.shape[1])
        ]
    )
    log_centroid = weighted_log_kde_2d(
        centroid, log_weights, x_grid, y_grid, bandwidth
    )
    log_eq8 = np.asarray(logsumexp(log_beads, axis=0)) - math.log(beads.shape[1])
    raw_centroid = -float(kbt_ev) * log_centroid
    raw_eq8 = -float(kbt_ev) * log_eq8
    raw_per_bead = -float(kbt_ev) * log_beads
    raw_eq10 = np.mean(raw_per_bead, axis=0)
    return {
        "log_weights": log_weights,
        "weights": np.exp(log_weights),
        "log_centroid": log_centroid,
        "log_beads": log_beads,
        "log_eq8": log_eq8,
        "raw_centroid": raw_centroid,
        "raw_eq8": raw_eq8,
        "raw_eq10": raw_eq10,
        "centroid": raw_centroid - np.min(raw_centroid),
        "eq8": raw_eq8 - np.min(raw_eq8),
        "eq10": raw_eq10 - np.min(raw_eq10),
    }


def compute_marginals(
    beads: np.ndarray,
    centroid: np.ndarray,
    raw_log_weights: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
    component: int,
    kbt_ev: float,
) -> Dict[str, np.ndarray]:
    log_weights = normalized_log_weights(raw_log_weights)
    log_beads = np.asarray(
        [
            weighted_log_kde_1d(beads[:, bead, component], log_weights, grid, bandwidth)
            for bead in range(beads.shape[1])
        ]
    )
    log_centroid = weighted_log_kde_1d(
        centroid[:, component], log_weights, grid, bandwidth
    )
    centroid_f = -float(kbt_ev) * log_centroid
    log_eq8 = np.asarray(logsumexp(log_beads, axis=0)) - math.log(beads.shape[1])
    eq8_f = -float(kbt_ev) * log_eq8
    eq10_f = np.mean(-float(kbt_ev) * log_beads, axis=0)
    return {
        "centroid": centroid_f - np.min(centroid_f),
        "eq8": eq8_f - np.min(eq8_f),
        "eq10": eq10_f - np.min(eq10_f),
        "log_centroid": log_centroid,
        "log_eq8": log_eq8,
        "log_beads": log_beads,
    }


def read_plumed(path: Path) -> Tuple[Tuple[str, ...], np.ndarray]:
    fields: Tuple[str, ...] | None = None
    rows: List[List[float]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#! FIELDS"):
                candidate = tuple(line.split()[2:])
                if fields is None:
                    fields = candidate
                else:
                    require(fields == candidate, f"FIELDS changed in {path}")
            elif line.startswith("#") or not line.strip():
                continue
            else:
                require(fields is not None, f"missing FIELDS header: {path}")
                values = [float(value) for value in line.split()]
                require(len(values) == len(fields), f"column mismatch: {path}")
                rows.append(values)
    require(fields is not None and rows, f"empty PLUMED table: {path}")
    data = np.asarray(rows, dtype=float)
    require(np.isfinite(data).all(), f"non-finite PLUMED table: {path}")
    return fields, data


def field(data: np.ndarray, fields: Sequence[str], name: str) -> np.ndarray:
    try:
        index = tuple(fields).index(name)
    except ValueError as exc:
        raise ValueError(f"column not found: {name}") from exc
    return data[:, index]


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_filtered_colvar(
    path: Path, fields: Sequence[str], data: np.ndarray, selected: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#! FIELDS " + " ".join(fields) + "\n")
        for row in data[selected]:
            handle.write(" ".join(f"{float(value):.16e}" for value in row) + "\n")


def read_thermo(path: Path) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    header: List[str] | None = None
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if tokens and tokens[0] == "Step" and "Time" in tokens:
                header = tokens
                continue
            if header is None or len(tokens) != len(header):
                continue
            try:
                values = [float(value) for value in tokens]
            except ValueError:
                continue
            row = dict(zip(header, values))
            step = int(round(row["Step"]))
            rows[step] = row
    require(rows, f"no thermo rows: {path}")
    return rows


def dump_frames(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                return
            require(marker.startswith("ITEM: TIMESTEP"), f"bad dump marker: {path}")
            step = int(handle.readline().strip())
            require(handle.readline().startswith("ITEM: NUMBER OF ATOMS"), f"bad atom count: {path}")
            atoms = int(handle.readline().strip())
            require(handle.readline().startswith("ITEM: BOX BOUNDS"), f"bad box header: {path}")
            bounds = np.asarray(
                [[float(value) for value in handle.readline().split()[:2]] for _ in range(3)]
            )
            atom_header = handle.readline().split()[2:]
            required = [atom_header.index(name) for name in ("id", "type", "x", "y", "z")]
            values = np.empty((atoms, len(atom_header)), dtype=float)
            for atom in range(atoms):
                tokens = handle.readline().split()
                require(len(tokens) == len(atom_header), f"bad atom row: {path}")
                values[atom] = [float(value) for value in tokens]
            ids = values[:, required[0]].astype(int)
            order = np.argsort(ids)
            types = values[order, required[1]].astype(int)
            positions = values[order][:, required[2:5]]
            yield step, bounds, ids[order], types, positions


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
        require(all(np.array_equal(frame[2], frames[0][2]) for frame in frames), "atom order mismatch")
        require(all(np.array_equal(frame[3], frames[0][3]) for frame in frames), "atom type mismatch")
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


def soft_voronoi_occupancies(
    positions: np.ndarray,
    types: np.ndarray,
    box: np.ndarray,
    center_type: int,
    assigned_type: int,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    types = np.asarray(types, dtype=int)
    box = np.asarray(box, dtype=float)
    require(positions.ndim == 2 and positions.shape[1] == 3, "position shape mismatch")
    require(types.shape == (positions.shape[0],), "type shape mismatch")
    require(box.shape == (3,) and np.all(box > 0.0), "invalid periodic box")
    require(float(kappa) > 0.0, "kappa must be positive")
    centers = positions[types == int(center_type)]
    assigned = positions[types == int(assigned_type)]
    require(len(centers) > 0 and len(assigned) > 0, "center or assigned atoms missing")
    delta = assigned[None, :, :] - centers[:, None, :]
    delta -= box[None, None, :] * np.rint(delta / box[None, None, :])
    distances = np.linalg.norm(delta, axis=2)
    shifted = distances - np.min(distances, axis=0, keepdims=True)
    weights = np.exp(-float(kappa) * shifted)
    weights /= np.sum(weights, axis=0, keepdims=True)
    occupancies = np.sum(weights, axis=1)
    nearest_counts = np.bincount(
        np.argmin(distances, axis=0), minlength=len(centers)
    )
    return occupancies, nearest_counts


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
        if centroid_values[index] >= minimum_score
        or np.any(bead_values[index] >= minimum_score)
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
        require(all(np.array_equal(frame[2], frames[0][2]) for frame in frames), "candidate atom order mismatch")
        require(all(np.array_equal(frame[3], frames[0][3]) for frame in frames), "candidate atom type mismatch")
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
                    "hard_pair_like": int(
                        np.any(hard_counts <= 1) and np.any(hard_counts >= 3)
                    ),
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


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    return plt, TwoSlopeNorm


def save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")


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
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.2), sharex=True, sharey=True, constrained_layout=True
    )
    estimator_labels = estimator_plot_labels(bias_mode)
    labels = (
        ("centroid", sampling_label),
        ("eq8", estimator_labels["probability_mean"]),
        ("eq10", estimator_labels["logmean"]),
    )
    image = None
    for axis, (name, title) in zip(axes, labels):
        values = np.where(supports[name], surfaces_kcal[name], np.nan)
        image = axis.pcolormesh(x_grid, y_grid, values, shading="auto", cmap="viridis", vmin=0.0, vmax=max_kcal)
        axis.set_title(title)
        axis.set_xlabel(cv_labels[0])
    axes[0].set_ylabel(cv_labels[1])
    if zoom is not None:
        for axis in axes:
            axis.set_xlim(*zoom[0])
            axis.set_ylim(*zoom[1])
    fig.colorbar(image, ax=axes, label="Free energy (kcal/mol)", pad=0.02, shrink=0.9)
    bead_prefix = f"P={bead_count} " if bead_count is not None else ""
    fig.suptitle(
        f"{bead_prefix}{sampling_label.lower()} sampling with {protocol_label}: "
        "reweighted 2D free-energy surfaces"
    )
    save_figure(fig, output / "figures" / f"fes2d-comparison{suffix}")
    plt.close(fig)


def plot_fes_differences(
    output: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces_kcal: Mapping[str, np.ndarray],
    support: np.ndarray,
    max_abs_kcal: float,
    cv_labels: Sequence[str],
    *,
    zoom: Sequence[Sequence[float]] | None = None,
    suffix: str = "",
    sampling_label: str = "Centroid",
    bias_mode: str = "centroid_coord",
) -> None:
    plt, TwoSlopeNorm = _matplotlib()
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.2), sharex=True, sharey=True, constrained_layout=True
    )
    estimator_labels = estimator_plot_labels(bias_mode)
    probability_label = estimator_labels["probability_mean"]
    logmean_label = estimator_labels["logmean"]
    panels = (
        (surfaces_kcal["eq8"] - surfaces_kcal["centroid"], f"{probability_label} - {sampling_label}"),
        (surfaces_kcal["eq10"] - surfaces_kcal["centroid"], f"{logmean_label} - {sampling_label}"),
        (surfaces_kcal["eq10"] - surfaces_kcal["eq8"], f"{logmean_label} - {probability_label}"),
    )
    image = None
    norm = TwoSlopeNorm(vmin=-max_abs_kcal, vcenter=0.0, vmax=max_abs_kcal)
    for axis, (values, title) in zip(axes, panels):
        image = axis.pcolormesh(
            x_grid, y_grid, np.where(support, values, np.nan), shading="auto", cmap="coolwarm", norm=norm
        )
        axis.set_title(title)
        axis.set_xlabel(cv_labels[0])
    axes[0].set_ylabel(cv_labels[1])
    if zoom is not None:
        for axis in axes:
            axis.set_xlim(*zoom[0])
            axis.set_ylim(*zoom[1])
    fig.colorbar(
        image, ax=axes, label="Free-energy difference (kcal/mol)", pad=0.02, shrink=0.9
    )
    fig.suptitle("Reweighted 2D free-energy differences on common support")
    save_figure(fig, output / "figures" / f"fes2d-differences{suffix}")
    plt.close(fig)


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
    plt, _ = _matplotlib()
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    estimator_labels = estimator_plot_labels(bias_mode)
    styles = {
        "centroid": (sampling_label, "#d97706", "-"),
        "eq8": (estimator_labels["probability_mean"], "#2563eb", "--"),
        "eq10": (estimator_labels["logmean"], "#b91c1c", "-"),
    }
    for key in ("centroid", "eq8", "eq10"):
        label, color, linestyle = styles[key]
        axis.plot(
            grid,
            np.where(supports[key], curves_kcal[key], np.nan),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )
    axis.set_xlabel(cv_label)
    axis.set_ylabel("Free energy (kcal/mol)")
    axis.set_ylim(0.0, max_kcal)
    axis.set_title(f"Reweighted 1D free energy: {name}")
    axis.legend(frameon=False)
    axis.grid(color="#d1d5db", linewidth=0.6, alpha=0.7)
    save_figure(fig, output / "figures" / f"fes1d-{name}")
    plt.close(fig)


def scatter_norm(values: np.ndarray):
    _, TwoSlopeNorm = _matplotlib()
    low = float(np.min(values))
    high = float(np.max(values))
    if low < 0.0 < high:
        return TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
    return None


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
    plt, _ = _matplotlib()
    stride = max(1, int(plot_stride))
    norm = scatter_norm(bias_kcal)
    fig, axes = plt.subplots(
        beads.shape[1], 2, figsize=(12.5, 2.6 * beads.shape[1]), sharex=True, sharey="col",
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for bead in range(beads.shape[1]):
        for component, name in enumerate(cv_labels):
            axis = axes[bead, component]
            image = axis.scatter(
                time_ps[::stride], beads[::stride, bead, component], c=bias_kcal[::stride],
                s=5, cmap="coolwarm", norm=norm, rasterized=True
            )
            axis.set_ylabel(f"Bead {bead + 1}\n{name}")
            axis.grid(color="#e5e7eb", linewidth=0.4)
    for axis in axes[-1]:
        axis.set_xlabel("Time (ps)")
    fig.colorbar(
        image, ax=axes, label=f"{sampling_label} {protocol_label} (kcal/mol)", pad=0.01, shrink=0.9
    )
    fig.suptitle(
        f"{beads.shape[1]}-bead CV trajectories colored by the "
        f"{sampling_label.lower()} {protocol_label}"
    )
    save_figure(fig, output / "figures" / "bead-cv-time-bias")
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 1, figsize=(12.5, 6.8), sharex=True, constrained_layout=True
    )
    image = None
    for component, (axis, name) in enumerate(zip(axes, cv_labels)):
        image = axis.scatter(
            time_ps[::stride],
            centroid[::stride, component],
            c=bias_kcal[::stride],
            s=7,
            cmap="coolwarm",
            norm=norm,
            rasterized=True,
        )
        axis.set_ylabel(name)
        axis.grid(color="#e5e7eb", linewidth=0.4)
    axes[-1].set_xlabel("Time (ps)")
    fig.colorbar(
        image, ax=axes, label=f"{sampling_label} {protocol_label} (kcal/mol)", pad=0.01,
        shrink=0.9,
    )
    fig.suptitle(
        f"{sampling_label} CV trajectories colored by the applied {protocol_label}"
    )
    save_figure(fig, output / "figures" / f"{sampling_slug}-cv-time-bias")
    plt.close(fig)

    columns = min(2, beads.shape[1])
    rows = math.ceil(beads.shape[1] / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.25 * columns, 4.25 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for bead, axis in enumerate(axes.ravel()[: beads.shape[1]]):
        image = axis.scatter(
            beads[::stride, bead, 0], beads[::stride, bead, 1], c=bias_kcal[::stride],
            s=6, cmap="coolwarm", norm=norm, rasterized=True
        )
        axis.set_title(f"Bead {bead + 1}")
        axis.set_xlabel(cv_labels[0])
        axis.set_ylabel(cv_labels[1])
    for axis in axes.ravel()[beads.shape[1] :]:
        axis.remove()
    fig.colorbar(
        image, ax=axes, label=f"{sampling_label} {protocol_label} (kcal/mol)", pad=0.01, shrink=0.9
    )
    fig.suptitle(
        f"{beads.shape[1]}-bead CV sampling colored by the shared "
        f"{sampling_label.lower()} {protocol_label}"
    )
    save_figure(fig, output / "figures" / "bead-cv2d-bias")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    image = axis.scatter(
        centroid[::stride, 0], centroid[::stride, 1], c=bias_kcal[::stride],
        s=8, cmap="coolwarm", norm=norm, rasterized=True,
    )
    axis.set_xlabel(cv_labels[0])
    axis.set_ylabel(cv_labels[1])
    axis.set_title(
        f"{sampling_label} CV sampling colored by the applied {protocol_label}"
    )
    axis.grid(color="#e5e7eb", linewidth=0.4)
    fig.colorbar(
        image,
        ax=axis,
        label=f"{sampling_label} {protocol_label} (kcal/mol)",
        pad=0.02,
    )
    save_figure(fig, output / "figures" / f"{sampling_slug}-cv2d-bias")
    plt.close(fig)


def plot_cv_spread(output: Path, time_ps: np.ndarray, beads: np.ndarray, names: Sequence[str]) -> None:
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(
        len(names), 1, figsize=(9.5, 3.4 * len(names)), sharex=True, squeeze=False
    )
    for component, (axis, name) in enumerate(zip(axes[:, 0], names)):
        axis.plot(time_ps, np.std(beads[:, :, component], axis=1), color="#2563eb", linewidth=0.8)
        axis.set_ylabel(f"Std. across beads\n{name}")
        axis.grid(color="#e5e7eb", linewidth=0.5)
    axes[-1, 0].set_xlabel("Time (ps)")
    fig.suptitle(f"Instantaneous spread of the {beads.shape[1]} bead CVs")
    save_figure(fig, output / "figures" / "bead-cv-spread")
    plt.close(fig)


def plot_cv_time_series(
    output: Path,
    time_ps: np.ndarray,
    sampling: np.ndarray,
    beads: np.ndarray,
    cv_labels: Sequence[str],
    sampling_label: str,
) -> None:
    """Plot arbitrary one- or two-dimensional CV trajectories without case assumptions."""
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(
        len(cv_labels),
        1,
        figsize=(10.0, 3.6 * len(cv_labels)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for component, (axis, label) in enumerate(zip(axes[:, 0], cv_labels)):
        for bead in range(beads.shape[1]):
            axis.plot(
                time_ps,
                beads[:, bead, component],
                linewidth=0.55,
                alpha=0.6,
                label=f"Bead {bead + 1}",
            )
        axis.plot(
            time_ps,
            sampling[:, component],
            color="#111827",
            linewidth=1.2,
            label=sampling_label,
        )
        axis.set_ylabel(label)
        axis.grid(color="#e5e7eb", linewidth=0.5)
    axes[0, 0].legend(frameon=False, ncol=min(beads.shape[1] + 1, 5), fontsize=8)
    axes[-1, 0].set_xlabel("Time (ps)")
    fig.suptitle("Sampling and bead-local CV trajectories")
    save_figure(fig, output / "figures" / "cv-time-series")
    plt.close(fig)


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
    plt, _ = _matplotlib()
    stride = max(1, int(plot_stride))
    norm = scatter_norm(bias_kcal)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, constrained_layout=True)
    image = axes[0].scatter(
        time_ps[::stride], sampling[::stride], c=bias_kcal[::stride],
        s=7, cmap="coolwarm", norm=norm, rasterized=True,
    )
    axes[0].set_ylabel(label)
    axes[0].set_title(f"{sampling_label} diagnostic CV")
    for bead in range(beads.shape[1]):
        axes[1].scatter(
            time_ps[::stride], beads[::stride, bead], c=bias_kcal[::stride],
            s=4, cmap="coolwarm", norm=norm, alpha=0.45, rasterized=True,
        )
    axes[1].set_ylabel(f"Bead-local {label}")
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_title(f"{beads.shape[1]} bead-local diagnostic CVs")
    for axis in axes:
        axis.grid(color="#e5e7eb", linewidth=0.4)
    fig.colorbar(
        image, ax=axes, label=f"{sampling_label} {protocol_label} (kcal/mol)",
        pad=0.01, shrink=0.9,
    )
    fig.suptitle("Directly printed diagnostic CV; no independent FES bandwidth assigned")
    save_figure(fig, output / "figures" / "iondistance-time-bias")
    plt.close(fig)


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
) -> None:
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=False)
    stride = max(1, int(math.ceil(len(time_ps) / 4000)))
    axes[0, 0].scatter(
        time_ps[::stride],
        bias_kcal[::stride],
        label="bias samples",
        color="#d97706",
        s=4,
        alpha=0.25,
        edgecolors="none",
        rasterized=True,
    )
    axes[0, 0].plot(time_ps, rct_kcal, label="rct (diagnostic only)", color="#4b5563", linewidth=0.9)
    axes[0, 0].set_ylabel("Energy (kcal/mol)")
    axes[0, 0].set_xlabel("Time (ps)")
    axes[0, 0].set_title("Bias samples and reweighting offset")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(
        time_ps,
        zed,
        label="opes.zed",
        color="#059669",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0, 1].step(
        time_ps,
        nker,
        label="opes.nker",
        color="#d97706",
        where="post",
        linewidth=0.9,
    )
    axes[0, 1].set_yscale("log")
    twin = axes[0, 1].twinx()
    twin.plot(time_ps, neff, label="opes.neff", color="#2563eb", linewidth=0.9)
    axes[0, 1].set_ylabel("OPES zed / kernel count (log scale)")
    twin.set_ylabel("OPES neff")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[0, 1].legend(
        handles + twin_handles,
        labels + twin_labels,
        frameon=False,
        loc="best",
    )
    axes[0, 1].set_xlabel("Time (ps)")
    axes[0, 1].set_title("Adaptive OPES state")
    ess_fraction, maximum_share = cumulative_weight_diagnostics(weights)
    axes[1, 0].plot(
        time_ps,
        ess_fraction,
        label="cumulative Kish ESS fraction",
        color="#7c3aed",
        linewidth=1.0,
    )
    weight_axis = axes[1, 0].twinx()
    weight_axis.plot(
        time_ps,
        maximum_share,
        label="maximum cumulative weight share",
        color="#dc2626",
        linewidth=0.9,
    )
    weight_axis.set_yscale("log")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylabel("Cumulative Kish ESS fraction")
    weight_axis.set_ylabel("Maximum weight share (log scale)")
    axes[1, 0].set_xlabel("Time (ps)")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    weight_handles, weight_labels = weight_axis.get_legend_handles_labels()
    axes[1, 0].legend(
        handles + weight_handles,
        labels + weight_labels,
        frameon=False,
        loc="best",
    )
    axes[1, 0].set_title("Cumulative reweighting quality")
    axes[1, 1].plot(kernel_time_ps, sigma_x, label="sigma_logdistance", color="#2563eb")
    axes[1, 1].plot(kernel_time_ps, sigma_y, label="sigma_ionization", color="#d97706")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("OPES kernel sigma")
    axes[1, 1].set_xlabel("Time (ps)")
    axes[1, 1].set_title("Adaptive kernel widths")
    axes[1, 1].legend(frameon=False)
    for axis in axes.ravel():
        axis.grid(color="#e5e7eb", linewidth=0.5)
    fig.suptitle("OPES bias, reweighting, and adaptive-kernel diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output / "figures" / "opes-diagnostics")
    plt.close(fig)


def plot_thermo(
    output: Path,
    time_ps: np.ndarray,
    temperatures: np.ndarray,
    potential: np.ndarray,
    global_values: Mapping[str, np.ndarray],
    beads: int,
    target_temperature: float,
) -> None:
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), sharex=True)
    for bead in range(beads):
        axes[0, 0].plot(time_ps, temperatures[:, bead] / beads, linewidth=0.55, alpha=0.55)
    axes[0, 0].plot(time_ps, np.mean(temperatures, axis=1) / beads, color="#111827", linewidth=1.0, label="bead mean / P")
    axes[0, 0].axhline(target_temperature, color="#b91c1c", linestyle="--", label="target")
    axes[0, 0].set_ylabel("Scaled kinetic temperature (K)")
    axes[0, 0].legend(frameon=False)
    mean_potential = np.mean(potential, axis=1) * EV_TO_KCAL_MOL
    axes[0, 1].plot(time_ps, mean_potential - np.mean(mean_potential), color="#2563eb", linewidth=0.8)
    axes[0, 1].set_ylabel("Mean bead potential - mean (kcal/mol)")
    axes[1, 0].plot(time_ps, global_values["f_fpimd[5]"] * EV_TO_KCAL_MOL, label="primitive", color="#d97706", linewidth=0.8)
    axes[1, 0].plot(time_ps, global_values["f_fpimd[7]"] * EV_TO_KCAL_MOL, label="centroid virial", color="#2563eb", linewidth=0.8)
    axes[1, 0].set_ylabel("Kinetic-energy estimator (kcal/mol)")
    axes[1, 0].legend(frameon=False)
    spring = np.sum(global_values["spring_per_bead"], axis=1) * EV_TO_KCAL_MOL
    axes[1, 1].plot(time_ps, spring, color="#7c3aed", linewidth=0.8)
    axes[1, 1].set_ylabel("Total ring-polymer spring energy (kcal/mol)")
    for axis in axes[-1]:
        axis.set_xlabel("Time (ps)")
    for axis in axes.ravel():
        axis.grid(color="#e5e7eb", linewidth=0.5)
    fig.suptitle("PIMD thermostat and energy-estimator diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output / "figures" / "pimd-thermo-energy")
    plt.close(fig)


def plot_ring_spread(output: Path, rows: Sequence[Mapping[str, float]], timestep_fs: float) -> None:
    plt, _ = _matplotlib()
    time_ps = np.asarray([row["step"] * timestep_fs / 1000.0 for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    for name, color in (("rg_H_A", "#d97706"), ("rg_O_A", "#2563eb"), ("rg_all_A", "#111827")):
        if name in rows[0]:
            axes[0].plot(time_ps, [row[name] for row in rows], label=name.replace("rg_", "").replace("_A", ""), color=color, linewidth=0.8)
    axes[0].set_ylabel("Ring-polymer RMS spread (A)")
    axes[0].legend(frameon=False)
    axes[1].plot(time_ps, [row["rg_p95_A"] for row in rows], label="95th percentile", color="#7c3aed", linewidth=0.8)
    axes[1].plot(time_ps, [row["rg_max_A"] for row in rows], label="maximum atom", color="#b91c1c", linewidth=0.6, alpha=0.8)
    axes[1].set_ylabel("Atomic bead spread (A)")
    axes[1].set_xlabel("Time (ps)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(color="#e5e7eb", linewidth=0.5)
    fig.suptitle("Ring-polymer spatial spread")
    save_figure(fig, output / "figures" / "ring-polymer-spread")
    plt.close(fig)


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
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    bead_colors = ("#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8")
    for bead, color in enumerate(bead_colors):
        axes[0].plot(
            time_ps, beads[:, bead], color=color, linewidth=0.45, alpha=0.55,
            label=f"Bead {bead + 1}",
        )
    axes[0].plot(time_ps, centroid, color="#111827", linewidth=1.0, label=sampling_label)
    axes[0].axhline(
        float(ideal_pair_score), color="#b45309", linestyle="--", linewidth=1.4,
        label=f"Ideal localized ion-pair score ({ideal_pair_score:g})",
    )
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Ionization defect score")
    axes[0].set_title(f"{sampling_label} and bead ionization scores")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].grid(color="#e5e7eb", linewidth=0.5)

    stride = 5
    maximum_bead = np.max(beads, axis=1)
    image = axes[1].scatter(
        centroid[::stride], maximum_bead[::stride], c=bias_kcal[::stride],
        s=9, cmap="coolwarm", norm=scatter_norm(bias_kcal), rasterized=True,
    )
    upper = max(float(ideal_pair_score), float(np.max(maximum_bead))) * 1.04
    axes[1].plot([0.0, upper], [0.0, upper], color="#6b7280", linestyle=":", linewidth=1.0)
    for threshold in thresholds:
        axes[1].axvline(float(threshold), color="#9ca3af", linewidth=0.5, alpha=0.5)
        axes[1].axhline(float(threshold), color="#9ca3af", linewidth=0.5, alpha=0.5)
    axes[1].set_xlim(0.0, upper)
    axes[1].set_ylim(0.0, upper)
    axes[1].set_xlabel(f"{sampling_label} ionization score")
    axes[1].set_ylabel("Maximum bead ionization score")
    axes[1].set_title(f"Bead-local excursions versus {sampling_label.lower()} response")
    axes[1].grid(color="#e5e7eb", linewidth=0.5)
    fig.colorbar(
        image,
        ax=axes[1],
        label=f"{sampling_label} {protocol_label} (kcal/mol)",
        pad=0.02,
    )
    fig.suptitle("Water ionization diagnostic; an ideal localized H3O+/OH- pair scores about 2")
    save_figure(fig, output / "figures" / "ionization-event-diagnostics")
    plt.close(fig)


def run_reference(
    config: Mapping[str, object],
    filtered_colvar: Path,
    output: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centroid_kcal: np.ndarray,
    support: np.ndarray,
    kbt_ev: float,
    bandwidth: Sequence[float],
    cv_columns: Sequence[str] = ("logdistance", "ionization"),
    bias_column: str = "opes.bias",
) -> Dict[str, object]:
    driver = Path(str(config["driver"]))
    require(driver.is_file(), f"reference driver missing: {driver}")
    driver_source = driver.read_text(encoding="utf-8")
    adapted_driver = output / "qc" / "FES_from_Reweighting.non-square-grid-compatible.py"
    adapted_driver.write_text(adapt_reference_source(driver_source), encoding="utf-8")

    def command_for(reference: Path, blocks: int) -> List[str]:
        return [
            sys.executable,
            str(adapted_driver),
            "--outfile",
            str(reference),
            "--colvar",
            str(filtered_colvar),
            "--sigma",
            ",".join(str(float(value)) for value in bandwidth),
            "--kt",
            str(float(kbt_ev)),
            "--cv",
            ",".join(str(value) for value in cv_columns),
            "--bias",
            str(bias_column),
            f"--min={float(x_grid[0])},{float(y_grid[0])}",
            f"--max={float(x_grid[-1])},{float(y_grid[-1])}",
            f"--bin={len(x_grid) - 1},{len(y_grid) - 1}",
            "--fmt=%.16e",
            "--blocks",
            str(int(blocks)),
        ]

    def execute(command: Sequence[str], log_path: Path) -> None:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        require(completed.returncode == 0, f"reference driver failed; see {log_path}")

    def load_surface(reference: Path) -> np.ndarray:
        values = np.loadtxt(reference, comments="#")
        require(values.shape[0] == len(x_grid) * len(y_grid), "reference grid size mismatch")
        surface = values[:, 2].reshape(len(x_grid), len(y_grid)).T * EV_TO_KCAL_MOL
        return surface - np.nanmin(surface)

    exact_reference = output / "qc" / "fes-from-reweighting-centroid-exact-eV.dat"
    exact_command = command_for(exact_reference, 1)
    execute(exact_command, output / "qc" / "fes-from-reweighting-exact.log")
    exact_kcal = load_surface(exact_reference)
    exact_difference = np.abs(exact_kcal - centroid_kcal)

    requested_blocks = int(config.get("blocks", 10))
    block_reference = output / "qc" / f"fes-from-reweighting-centroid-blocks{requested_blocks}-eV.dat"
    block_command = command_for(block_reference, requested_blocks)
    execute(block_command, output / "qc" / "fes-from-reweighting-blocks.log")
    block_kcal = load_surface(block_reference)
    block_difference = np.abs(block_kcal - exact_kcal)
    portable_command = [portable_artifact_path(value, output) for value in exact_command]
    portable_block_command = [portable_artifact_path(value, output) for value in block_command]
    return {
        "original_driver": str(driver),
        "original_driver_sha256": sha256(driver),
        "adapted_driver": portable_artifact_path(adapted_driver, output),
        "adapted_driver_sha256": sha256(adapted_driver),
        "compatibility_patch": "np.meshgrid(..., indexing='ij') for non-square 2D grids",
        "bin_semantics": "contract point counts converted to legacy driver interval counts",
        "command": portable_command,
        "max_abs_difference_kcal_mol": float(np.max(exact_difference[support])),
        "rmse_difference_kcal_mol": float(np.sqrt(np.mean(exact_difference[support] ** 2))),
        "reference_file": portable_artifact_path(exact_reference, output),
        "blocks": requested_blocks,
        "block_command": portable_block_command,
        "block_reference_file": portable_artifact_path(block_reference, output),
        "block_vs_exact_max_abs_kcal_mol": float(np.max(block_difference[support])),
        "block_vs_exact_rmse_kcal_mol": float(np.sqrt(np.mean(block_difference[support] ** 2))),
    }


def write_manifest(output: Path) -> None:
    records = []
    for path in sorted((item for item in output.rglob("*") if item.is_file())):
        if path.name == "OUTPUT-SHA256SUMS":
            continue
        records.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "provenance" / "OUTPUT-SHA256SUMS").write_text("\n".join(records) + "\n", encoding="utf-8")


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
    grid = np.linspace(*reweight["grid"][cv_name])
    variants = {
        str(name): tuple(float(value) for value in values)
        for name, values in reweight["bandwidth_variants"].items()
    }
    require(all(len(value) == 1 for value in variants.values()), "1D bandwidths need one value")
    primary_name = str(reweight["primary_bandwidth"])
    require(primary_name in variants, "primary bandwidth absent")
    threshold = math.log(float(reweight["relative_density_support"]))
    curves_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    supports_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    sensitivity_rows: List[Dict[str, object]] = []
    for variant, bandwidth in variants.items():
        curves = compute_marginals(
            beads, sampling, raw_log_weights, grid, bandwidth[0], 0, kbt_ev
        )
        relative_beads = curves["log_beads"] - np.max(
            curves["log_beads"], axis=1, keepdims=True
        )
        supports = {
            "centroid": curves["log_centroid"] - np.max(curves["log_centroid"])
            >= threshold,
            "eq8": curves["log_eq8"] - np.max(curves["log_eq8"]) >= threshold,
            "eq10": np.all(relative_beads >= threshold, axis=0),
        }
        supports["common"] = supports["centroid"] & supports["eq10"]
        curves_by_variant[variant] = curves
        supports_by_variant[variant] = supports
        sensitivity_rows.append(
            {
                "variant": variant,
                "sigma": bandwidth[0],
                "eq10_support_points": int(np.count_nonzero(supports["eq10"])),
                "common_support_points": int(np.count_nonzero(supports["common"])),
            }
        )

    primary = curves_by_variant[primary_name]
    primary_supports = supports_by_variant[primary_name]
    primary_support = primary_supports["common"]
    require(np.count_nonzero(primary_support) >= 2, "empty common support")
    primary_kcal = {
        key: primary[key] * EV_TO_KCAL_MOL for key in ("centroid", "eq8", "eq10")
    }
    for row in sensitivity_rows:
        variant = str(row["variant"])
        comparison_support = (
            primary_supports["eq10"] & supports_by_variant[variant]["eq10"]
        )
        count, rmse, maximum = surface_difference_metrics(
            primary_kcal["eq10"],
            curves_by_variant[variant]["eq10"] * EV_TO_KCAL_MOL,
            comparison_support,
        )
        row["comparison_support_points"] = count
        row["eq10_rmse_vs_primary_kcal_mol"] = rmse
        row["eq10_max_abs_vs_primary_kcal_mol"] = maximum
    write_csv(
        output / "qc" / "bandwidth-sensitivity.csv",
        sensitivity_rows,
        [
            "variant",
            "sigma",
            "eq10_support_points",
            "common_support_points",
            "comparison_support_points",
            "eq10_rmse_vs_primary_kcal_mol",
            "eq10_max_abs_vs_primary_kcal_mol",
        ],
    )
    write_csv(
        output / "fes1d" / f"{cv_name}.csv",
        (
            {
                cv_name: value,
                "sampling_support": int(primary_supports["centroid"][index]),
                "probability_mean_support": int(primary_supports["eq8"][index]),
                "logmean_support": int(primary_supports["eq10"][index]),
                "common_support": int(primary_support[index]),
                "F_sampling_kcal_mol": primary_kcal["centroid"][index],
                "F_quantum_probability_mean_kcal_mol": primary_kcal["eq8"][index],
                "F_bead_logmean_diagnostic_kcal_mol": primary_kcal["eq10"][index],
            }
            for index, value in enumerate(grid)
        ),
        [
            cv_name,
            "sampling_support",
            "probability_mean_support",
            "logmean_support",
            "common_support",
            "F_sampling_kcal_mol",
            "F_quantum_probability_mean_kcal_mol",
            "F_bead_logmean_diagnostic_kcal_mol",
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
        )
        current_relative = current["log_beads"] - np.max(
            current["log_beads"], axis=1, keepdims=True
        )
        comparison_support = primary_supports["eq10"] & np.all(
            current_relative >= threshold, axis=0
        )
        count, rmse, maximum = surface_difference_metrics(
            primary_kcal["eq10"],
            current["eq10"] * EV_TO_KCAL_MOL,
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
                "comparison_support_points": count,
                "logmean_rmse_vs_full_kcal_mol": rmse,
                "logmean_max_abs_vs_full_kcal_mol": maximum,
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
        + kbt_ev * primary["log_eq8"]
    ) * EV_TO_KCAL_MOL
    eq_gap = primary_kcal["eq10"] - primary_kcal["eq8"]
    summary: Dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "analysis_profile": "core",
        "source_job": source.get("job_id"),
        "sampling_representation": {
            "label": sampling_label,
            "slug": sampling_slug,
            "bias_mode": bias_mode,
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
            "formula": (
                "precomputed log frame weight"
                if weight_kind == "precomputed"
                else "normalized exp(total_bias_energy/kBT)"
            ),
            "protocol_label": sampling_protocol_label(reweight),
            "rct_used": False,
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
            "dimensions": 1,
            "unit": "kcal/mol",
            "primary_bandwidth": list(variants[primary_name]),
            "probability_mean_label": labels["probability_mean"],
            "logmean_label": labels["logmean"],
            "common_support_points": int(np.count_nonzero(primary_support)),
            "probability_logmean_rmse_common_support_kcal_mol": float(
                np.sqrt(np.mean(eq_gap[primary_support] ** 2))
            ),
            "probability_logmean_max_abs_common_support_kcal_mol": float(
                np.max(np.abs(eq_gap[primary_support]))
            ),
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
    require(float(np.min(raw_gap)) >= -1e-10, "probability/logmean Jensen relation failed")
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
        f"Primary estimator: `{labels['probability_mean']}`",
        f"Finite-sampling diagnostic: `{labels['logmean']}`",
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
    primary_support: np.ndarray,
    primary_supports: Mapping[str, np.ndarray],
    variants: Mapping[str, Sequence[float]],
    filtered_colvar: Path,
    sampling_restart_duplicates: int,
    bead_restart_duplicates: Sequence[int],
) -> Dict[str, object]:
    """Write a generic two-CV report without material-specific diagnostics."""
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
        primary_support,
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
            primary_support,
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
    raw_gap = (primary["raw_eq10"] - primary["raw_eq8"]) * EV_TO_KCAL_MOL
    eq_gap = primary_kcal["eq10"] - primary_kcal["eq8"]
    require(float(np.min(raw_gap)) >= -1e-10, "probability/logmean Jensen relation failed")
    summary: Dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "analysis_profile": "core",
        "source_job": source.get("job_id"),
        "sampling_representation": {
            "label": sampling_label,
            "slug": sampling_slug,
            "bias_mode": bias_mode,
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
            "formula": (
                "precomputed log frame weight"
                if weight_kind == "precomputed"
                else "normalized exp(total_bias_energy/kBT)"
            ),
            "protocol_label": protocol_label,
            "rct_used": False,
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
            "dimensions": 2,
            "unit": "kcal/mol",
            "primary_bandwidth": list(variants[primary_name]),
            "probability_mean_label": labels["probability_mean"],
            "logmean_label": labels["logmean"],
            "common_support_points": int(np.count_nonzero(primary_support)),
            "probability_logmean_rmse_common_support_kcal_mol": float(
                np.sqrt(np.mean(eq_gap[primary_support] ** 2))
            ),
            "probability_logmean_max_abs_common_support_kcal_mol": float(
                np.max(np.abs(eq_gap[primary_support]))
            ),
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
        f"Primary estimator: `{labels['probability_mean']}`",
        f"Finite-sampling diagnostic: `{labels['logmean']}`",
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
    run_root = Path(source["run_root"])
    require(run_root.is_dir(), f"run root missing: {run_root}")
    raw_manifest = run_root / source["raw_manifest"]
    require(sha256(raw_manifest) == source["raw_manifest_sha256"], "raw manifest hash mismatch")

    sampling_colvar = source.get("sampling_colvar")
    if sampling_colvar is None:
        sampling_colvar = source["centroid_colvar"]
    centroid_path = run_root / str(sampling_colvar)
    bead_paths = [run_root / value for value in source["bead_colvars"]]
    # Resolve filesystem identity, including symlinks and hard links, without
    # rejecting distinct files whose bead observations happen to be identical.
    bead_stats = [path.stat() for path in bead_paths]
    require(
        len({(stat.st_dev, stat.st_ino) for stat in bead_stats}) == len(bead_paths),
        "duplicate bead input file",
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
    time_fs = field(centroid_data, fields, "time")
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
    inferred_bias_mode = {
        "centroid": "centroid_coord",
        "bead_mean": "bead_mean",
        "bead_density": "bead_density_shared",
    }.get(sampling_slug, sampling_slug)
    bias_mode = validate_bias_mode(str(reweight.get("bias_mode", inferred_bias_mode)))
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
    cv_labels = tuple(str(configured_labels.get(name, name)) for name in cv_names)
    centroid = np.column_stack([field(centroid_data, fields, name)[selected] for name in sampling_cv_names])
    bead_arrays = []
    bead_selections = []
    for bead_fields, bead_data in bead_tables:
        bead_time = field(bead_data, bead_fields, "time")
        bead_offset_fs = float(source.get("bead_time_offset_fs", 0.0))
        bead_selected = aligned_time_indices(
            bead_time + bead_offset_fs, time_fs[selected]
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
    expected_kbt = KB_EV_PER_K * float(reweight["temperature_K"])
    require(abs(kbt_ev - expected_kbt) <= 1e-12, "kBT/temperature mismatch")
    quasi_static_declared = reweight.get("quasi_static") is True
    if weight_kind == "precomputed":
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
    surfaces_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    supports_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    sensitivity_rows: List[Dict[str, object]] = []
    primary_support = None
    for variant, bandwidth in variants.items():
        surfaces = compute_surfaces(beads, centroid, raw_log_weights, x_grid, y_grid, bandwidth, kbt_ev)
        surfaces_by_variant[variant] = surfaces
        log_threshold = math.log(float(reweight["relative_density_support"]))
        relative_beads = surfaces["log_beads"] - np.max(
            surfaces["log_beads"], axis=(1, 2), keepdims=True
        )
        supports = {
            "centroid": surfaces["log_centroid"] - np.max(surfaces["log_centroid"])
            >= log_threshold,
            "eq8": surfaces["log_eq8"] - np.max(surfaces["log_eq8"]) >= log_threshold,
            "eq10": np.all(relative_beads >= log_threshold, axis=0),
        }
        supports["common"] = supports["centroid"] & supports["eq10"]
        supports_by_variant[variant] = supports
        if variant == primary_name:
            primary_support = supports["common"]
        kcal = {key: surfaces[key] * EV_TO_KCAL_MOL for key in ("centroid", "eq8", "eq10")}
        rows = []
        for iy, y_value in enumerate(y_grid):
            for ix, x_value in enumerate(x_grid):
                rows.append(
                    {
                        cv_names[0]: x_value,
                        cv_names[1]: y_value,
                        "centroid_support": int(supports["centroid"][iy, ix]),
                        "eq8_support": int(supports["eq8"][iy, ix]),
                        "eq10_support": int(supports["eq10"][iy, ix]),
                        "common_support": int(supports["common"][iy, ix]),
                        "F_centroid_kcal_mol": kcal["centroid"][iy, ix],
                        "F_eq8_kcal_mol": kcal["eq8"][iy, ix],
                        "F_eq10_kcal_mol": kcal["eq10"][iy, ix],
                    }
                )
        write_csv(
            output / "fes2d" / f"{variant}.csv", rows,
            [cv_names[0], cv_names[1], "centroid_support", "eq8_support", "eq10_support", "common_support", "F_centroid_kcal_mol", "F_eq8_kcal_mol", "F_eq10_kcal_mol"],
        )
        sensitivity_rows.append(
            {
                "variant": variant,
                "sigma_x": bandwidth[0],
                "sigma_y": bandwidth[1],
                "eq10_support_points": int(np.count_nonzero(supports["eq10"])),
                "common_support_points": int(np.count_nonzero(supports["common"])),
            }
        )
    require(primary_support is not None and np.count_nonzero(primary_support) >= 2, "empty common support")
    primary = surfaces_by_variant[primary_name]
    primary_kcal = {key: primary[key] * EV_TO_KCAL_MOL for key in ("centroid", "eq8", "eq10")}
    for row in sensitivity_rows:
        variant = str(row["variant"])
        current = surfaces_by_variant[variant]["eq10"] * EV_TO_KCAL_MOL
        comparison_support = (
            supports_by_variant[primary_name]["eq10"] & supports_by_variant[variant]["eq10"]
        )
        try:
            count, rmse, maximum = surface_difference_metrics(
                primary_kcal["eq10"], current, comparison_support
            )
        except ValueError as exc:
            raise ValueError(f"bandwidth comparison failed for {variant}: {exc}") from exc
        row["comparison_support_points"] = count
        row["eq10_rmse_vs_primary_kcal_mol"] = rmse
        row["eq10_max_abs_vs_primary_kcal_mol"] = maximum
    write_csv(
        output / "qc" / "bandwidth-sensitivity.csv", sensitivity_rows,
        ["variant", "sigma_x", "sigma_y", "eq10_support_points", "common_support_points", "comparison_support_points", "eq10_rmse_vs_primary_kcal_mol", "eq10_max_abs_vs_primary_kcal_mol"],
    )

    marginal_tables: Dict[str, Dict[str, np.ndarray]] = {}
    for component, name in enumerate(cv_names):
        grid = x_grid if component == 0 else y_grid
        curves = compute_marginals(
            beads, centroid, raw_log_weights, grid, variants[primary_name][component], component, kbt_ev
        )
        curves_kcal = {key: values * EV_TO_KCAL_MOL for key, values in curves.items()}
        log_threshold = math.log(float(reweight["relative_density_support"]))
        relative_beads = curves["log_beads"] - np.max(
            curves["log_beads"], axis=1, keepdims=True
        )
        supports = {
            "centroid": curves["log_centroid"] - np.max(curves["log_centroid"])
            >= log_threshold,
            "eq8": curves["log_eq8"] - np.max(curves["log_eq8"]) >= log_threshold,
            "eq10": np.all(relative_beads >= log_threshold, axis=0),
        }
        supports["common"] = supports["centroid"] & supports["eq10"]
        marginal_tables[name] = {
            **{key: curves_kcal[key] for key in ("centroid", "eq8", "eq10")},
            **{f"{key}_support": value for key, value in supports.items()},
        }
        write_csv(
            output / "fes1d" / f"{name}.csv",
            (
                {
                    name: value,
                    "centroid_support": int(supports["centroid"][index]),
                    "eq8_support": int(supports["eq8"][index]),
                    "eq10_support": int(supports["eq10"][index]),
                    "common_support": int(supports["common"][index]),
                    "F_centroid_kcal_mol": curves_kcal["centroid"][index],
                    "F_eq8_kcal_mol": curves_kcal["eq8"][index],
                    "F_eq10_kcal_mol": curves_kcal["eq10"][index],
                }
                for index, value in enumerate(grid)
            ),
            [name, "centroid_support", "eq8_support", "eq10_support", "common_support", "F_centroid_kcal_mol", "F_eq8_kcal_mol", "F_eq10_kcal_mol"],
        )
        plot_fes1d(
            output, name, grid, curves_kcal, supports,
            float(reweight["plot_max_kcal_mol"]), cv_labels[component],
            sampling_label=sampling_label,
            bias_mode=bias_mode,
        )

    primary_supports = supports_by_variant[primary_name]
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
        }
        source_marginals = marginal_tables[source_name]
        derived_curves_kcal: Dict[str, np.ndarray] = {}
        target_grid = None
        density_jacobian = None
        for key in ("centroid", "eq8", "eq10"):
            current_grid, current_jacobian, current_curve = (
                transform_piecewise_logdistance_fes(
                    source_grid,
                    source_marginals[key],
                    kbt_ev * EV_TO_KCAL_MOL,
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
        derived_marginal_supports = {
            key: np.asarray(source_marginals[f"{key}_support"], dtype=bool).copy()
            for key in ("centroid", "eq8", "eq10", "common")
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
                    "centroid_support": int(derived_marginal_supports["centroid"][index]),
                    "eq8_support": int(derived_marginal_supports["eq8"][index]),
                    "eq10_support": int(derived_marginal_supports["eq10"][index]),
                    "common_support": int(derived_marginal_supports["common"][index]),
                    "F_centroid_kcal_mol": derived_curves_kcal["centroid"][index],
                    "F_eq8_kcal_mol": derived_curves_kcal["eq8"][index],
                    "F_eq10_kcal_mol": derived_curves_kcal["eq10"][index],
                }
                for index in range(len(target_grid))
            ),
            [
                target_name,
                source_name,
                jacobian_name,
                "centroid_support",
                "eq8_support",
                "eq10_support",
                "common_support",
                "F_centroid_kcal_mol",
                "F_eq8_kcal_mol",
                "F_eq10_kcal_mol",
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
        for key in ("centroid", "eq8", "eq10"):
            current_grid, current_jacobian, current_surface = (
                transform_piecewise_logdistance_fes(
                    source_grid,
                    primary_kcal[key],
                    kbt_ev * EV_TO_KCAL_MOL,
                    axis=surface_axis,
                    **transform_options,
                )
            )
            require(np.array_equal(current_grid, target_grid), "derived 1D/2D grids differ")
            require(
                np.array_equal(current_jacobian, density_jacobian),
                "derived 1D/2D Jacobians differ",
            )
            derived_surfaces_kcal[key] = current_surface
        derived_supports = {
            key: np.asarray(primary_supports[key], dtype=bool).copy()
            for key in ("centroid", "eq8", "eq10", "common")
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
                        "centroid_support": int(derived_supports["centroid"][iy, ix]),
                        "eq8_support": int(derived_supports["eq8"][iy, ix]),
                        "eq10_support": int(derived_supports["eq10"][iy, ix]),
                        "common_support": int(derived_supports["common"][iy, ix]),
                        "F_centroid_kcal_mol": derived_surfaces_kcal["centroid"][iy, ix],
                        "F_eq8_kcal_mol": derived_surfaces_kcal["eq8"][iy, ix],
                        "F_eq10_kcal_mol": derived_surfaces_kcal["eq10"][iy, ix],
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
                "centroid_support",
                "eq8_support",
                "eq10_support",
                "common_support",
                "F_centroid_kcal_mol",
                "F_eq8_kcal_mol",
                "F_eq10_kcal_mol",
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
            variants[primary_name], kbt_ev,
        )
        block_weights = block_surface["weights"]
        block_relative_beads = block_surface["log_beads"] - np.max(
            block_surface["log_beads"], axis=(1, 2), keepdims=True
        )
        block_eq10_support = np.all(
            block_relative_beads
            >= math.log(float(reweight["relative_density_support"])),
            axis=0,
        )
        block_comparison_support = (
            supports_by_variant[primary_name]["eq10"] & block_eq10_support
        )
        comparison_count, block_rmse, block_maximum = surface_difference_metrics(
            primary["eq10"] * EV_TO_KCAL_MOL,
            block_surface["eq10"] * EV_TO_KCAL_MOL,
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
                "comparison_support_points": comparison_count,
                "eq10_rmse_vs_full_kcal_mol": block_rmse,
                "eq10_max_abs_vs_full_kcal_mol": block_maximum,
            }
        )
    write_csv(
        output / "blocks" / "block-diagnostics.csv", block_rows,
        ["block", "first_time_ps", "last_time_ps", "frames", "ess", "max_weight", "comparison_support_points", "eq10_rmse_vs_full_kcal_mol", "eq10_max_abs_vs_full_kcal_mol"],
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
            primary_support=primary_support,
            primary_supports=primary_supports,
            variants=variants,
            filtered_colvar=filtered_colvar,
            sampling_restart_duplicates=centroid_restart_duplicates,
            bead_restart_duplicates=bead_restart_duplicates,
        )

    kernels_fields, kernels = read_plumed(run_root / source["kernels"])
    kernel_time_fs = field(kernels, kernels_fields, "time")
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
        primary_support,
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
            output, x_grid, y_grid, primary_kcal, primary_support,
            float(reweight["difference_max_kcal_mol"]), cv_labels, zoom=zoom, suffix="-sampled-region",
            sampling_label=sampling_label,
            bias_mode=bias_mode,
        )
    plot_bead_cv_bias(
        output, selected_time_ps, centroid, beads, bias_ev * EV_TO_KCAL_MOL, cv_labels,
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
            bias_ev * EV_TO_KCAL_MOL,
            str(diagnostic_spec["label"]),
            sampling_label,
            int(contract["plots"].get("scatter_stride", 5)),
            protocol_label=sampling_protocol_label(reweight),
        )
    plot_opes(
        output, selected_time_ps, bias_ev * EV_TO_KCAL_MOL, rct_ev * EV_TO_KCAL_MOL,
        zed, neff, nker, weights,
        kernel_time_ps, sigma_x, sigma_y,
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
        bias_ev * EV_TO_KCAL_MOL,
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
            primary_support,
            kbt_ev,
            variants[primary_name],
            sampling_cv_names,
            bias_column=str(bias_column),
        )
    raw_gap = (primary["raw_eq10"] - primary["raw_eq8"]) * EV_TO_KCAL_MOL
    eq_gap = (primary_kcal["eq10"] - primary_kcal["eq8"])
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
        "schema_version": 1,
        "status": "PASS",
        "source_job": source["job_id"],
        "analysis_profile": profile,
        "sampling_representation": {
            "label": sampling_label,
            "slug": sampling_slug,
            "bias_mode": bias_mode,
            "logical_cv_names": list(cv_names),
            "sampling_cv_columns": list(sampling_cv_names),
            "bead_cv_columns": list(bead_cv_names),
            "sampling_cv_source": (
                "mean_of_aligned_bead_cv"
                if bias_mode == "bead_density_shared"
                else "sampling_colvar"
            ),
            "legacy_internal_fes_key": "centroid",
        },
        "selection": {
            "first_time_ps": float(selected_time_ps[0]),
            "last_time_ps": float(selected_time_ps[-1]),
            "frames": int(len(selected)),
            "stride_ps": float(np.median(np.diff(selected_time_ps))),
        },
        "reweighting": {
            "weight_kind": weight_kind,
            "formula": "normalized exp(total_bias_energy/kBT)",
            "total_bias_energy": (
                "mean_b bead_local_bias_energy"
                if bias_mode == "bead_density_shared"
                else "sampling_bias_energy"
            ),
            "quasi_static_declared": (
                quasi_static_declared if weight_kind == "quasi_static_opes" else None
            ),
            "rct_used": False,
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
            "unit": "kcal/mol",
            "probability_mean_label": estimator_plot_labels(bias_mode)[
                "probability_mean"
            ],
            "logmean_label": estimator_plot_labels(bias_mode)["logmean"],
            "primary_bandwidth": list(variants[primary_name]),
            "common_support_points": int(np.count_nonzero(primary_support)),
            "centroid_support_points": int(np.count_nonzero(primary_supports["centroid"])),
            "eq8_support_points": int(np.count_nonzero(primary_supports["eq8"])),
            "eq10_support_points": int(np.count_nonzero(primary_supports["eq10"])),
            "eq8_eq10_rmse_common_support_kcal_mol": float(np.sqrt(np.mean(eq_gap[primary_support] ** 2))),
            "eq8_eq10_max_abs_common_support_kcal_mol": float(np.max(np.abs(eq_gap[primary_support]))),
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
    require(float(np.min(raw_gap)) >= -1e-10, "Eq.8/Eq.10 Jensen relation failed")
    (output / "qc" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "provenance" / "analysis-contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        f"# PIMD {sampling_label.lower()}-biased post-processing",
        "",
        "Status: `PASS`",
        f"Source job: `{source['job_id']}`",
        f"Window: `{selected_time_ps[0]:.1f}--{selected_time_ps[-1]:.1f} ps` (`{len(selected)}` frames)",
        f"Reweighting ESS: `{summary['reweighting']['ess']:.2f}` (`{100.0 * summary['reweighting']['ess_fraction']:.1f}%`)",
        "Probability-mean/logmean common-support RMS gap: "
        f"`{summary['fes']['eq8_eq10_rmse_common_support_kcal_mol']:.4f} kcal/mol`",
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


if __name__ == "__main__":
    raise SystemExit(run())
