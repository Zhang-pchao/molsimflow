"""CV transforms and geometric observables, independent of FES estimators."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def _validate_logdistance_parameters(
    switch: float,
    offset: float,
    linear_shift: float,
    log_scale: float,
    log_reference: float,
) -> None:
    """Reject noninvertible joins; allow the archived eight-digit shift rounding."""
    parameters = np.asarray([switch, offset, linear_shift, log_scale, log_reference])
    require(np.isfinite(parameters).all(), "non-finite piecewise-logdistance parameter")
    require(switch + offset > 0.0, "invalid piecewise-logdistance domain")
    require(log_scale > 0.0 and log_reference > 0.0, "invalid logarithm scale/reference")
    left = log_scale * math.log((switch + offset) / log_reference)
    right = switch - linear_shift
    require(abs(left - right) <= 5e-8, "discontinuous piecewise-logdistance join")


def piecewise_logdistance(
    iondistance: Sequence[float] | np.ndarray,
    *,
    switch: float = 1.0,
    offset: float = 0.03,
    linear_shift: float = 0.9704412,
    log_scale: float = 1.0,
    log_reference: float = 1.0,
) -> np.ndarray:
    """Evaluate the Reactive Voronoi piecewise log-distance coordinate."""
    _validate_logdistance_parameters(switch, offset, linear_shift, log_scale, log_reference)
    values = np.asarray(iondistance, dtype=float)
    require(np.isfinite(values).all(), "non-finite iondistance")
    require(np.all(values + float(offset) > 0.0), "iondistance is outside transform domain")
    return np.where(
        values < float(switch),
        float(log_scale) * np.log((values + float(offset)) / float(log_reference)),
        values - float(linear_shift),
    )


def inverse_piecewise_logdistance(
    logdistance: Sequence[float] | np.ndarray,
    *,
    switch: float = 1.0,
    offset: float = 0.03,
    linear_shift: float = 0.9704412,
    log_scale: float = 1.0,
    log_reference: float = 1.0,
) -> np.ndarray:
    """Invert the continuous Reactive Voronoi piecewise log-distance map."""
    _validate_logdistance_parameters(switch, offset, linear_shift, log_scale, log_reference)
    values = np.asarray(logdistance, dtype=float)
    require(np.isfinite(values).all(), "non-finite logdistance")
    # Prefer the exact linear branch at the switch. Archived shifts are rounded
    # by a few nanounits; midpoint classification misinverts the switch itself.
    boundary = float(switch) - float(linear_shift)
    return np.where(
        values < boundary,
        float(log_reference) * np.exp(np.minimum(values, boundary) / float(log_scale))
        - float(offset),
        values + float(linear_shift),
    )


def piecewise_logdistance_jacobian(
    iondistance: Sequence[float] | np.ndarray,
    *,
    switch: float = 1.0,
    offset: float = 0.03,
    log_scale: float = 1.0,
    log_reference: float = 1.0,
    linear_shift: float = 0.9704412,
) -> np.ndarray:
    """Return the absolute d(logdistance)/d(iondistance) density Jacobian."""
    _validate_logdistance_parameters(switch, offset, linear_shift, log_scale, log_reference)
    values = np.asarray(iondistance, dtype=float)
    require(np.isfinite(values).all(), "non-finite iondistance")
    require(np.all(values + float(offset) > 0.0), "iondistance is outside Jacobian domain")
    return np.where(values < float(switch), float(log_scale) / (values + float(offset)), 1.0)


def transform_piecewise_logdistance_fes(
    logdistance_grid: Sequence[float] | np.ndarray,
    free_energy: np.ndarray,
    kbt: float,
    *,
    axis: int = -1,
    switch: float = 1.0,
    offset: float = 0.03,
    linear_shift: float = 0.9704412,
    log_scale: float = 1.0,
    log_reference: float = 1.0,
    zero_minimum: bool = True,
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
    require(np.isfinite(kbt) and float(kbt) > 0.0, "kBT must be positive")
    iondistance = inverse_piecewise_logdistance(
        source_grid,
        switch=switch,
        offset=offset,
        linear_shift=linear_shift,
        log_scale=log_scale,
        log_reference=log_reference,
    )
    jacobian = piecewise_logdistance_jacobian(
        iondistance,
        switch=switch,
        offset=offset,
        linear_shift=linear_shift,
        log_scale=log_scale,
        log_reference=log_reference,
    )
    correction_shape = [1] * values.ndim
    correction_shape[normalized_axis] = source_grid.size
    transformed = values - float(kbt) * np.log(jacobian).reshape(correction_shape)
    if zero_minimum:
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
    log_scale: float = 1.0,
    log_reference: float = 1.0,
) -> Dict[str, float]:
    """Fail closed unless printed source and derived coordinates agree."""
    source = np.asarray(logdistance, dtype=float)
    printed = np.asarray(iondistance, dtype=float)
    require(source.shape == printed.shape and source.size > 0, "printed transform shape mismatch")
    require(np.isfinite(source).all(), "non-finite printed logdistance")
    require(np.isfinite(printed).all(), "non-finite printed iondistance")
    require(
        np.isfinite(tolerance) and float(tolerance) >= 0.0, "invalid printed-transform tolerance"
    )
    forward_error = float(
        np.max(
            np.abs(
                piecewise_logdistance(
                    printed,
                    switch=switch,
                    offset=offset,
                    linear_shift=linear_shift,
                    log_scale=log_scale,
                    log_reference=log_reference,
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
                    log_scale=log_scale,
                    log_reference=log_reference,
                )
                - printed
            )
        )
    )
    require(
        max(forward_error, inverse_error) <= float(tolerance),
        "printed logdistance/iondistance transform mismatch",
    )
    return {
        "forward_maximum_absolute_error": forward_error,
        "inverse_maximum_absolute_error": inverse_error,
        "maximum_absolute_error": max(forward_error, inverse_error),
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
    sampling_printed_column = str(config.get("sampling_printed_column", printed_column))
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
    log_scale = float(config.get("log_scale", 1.0))
    log_reference = float(config.get("log_reference", 1.0))
    tolerance = float(config.get("printed_transform_tolerance", 1e-12))
    _validate_logdistance_parameters(switch, offset, linear_shift, log_scale, log_reference)
    require(switch + offset > 0.0, "invalid piecewise-logdistance domain")
    require(np.isfinite(tolerance) and tolerance >= 0.0, "invalid printed-transform tolerance")
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
        "log_scale": log_scale,
        "log_reference": log_reference,
        "printed_transform_tolerance": tolerance,
        "label": str(config.get("label", target)),
    }


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
    nearest_counts = np.bincount(np.argmin(distances, axis=0), minlength=len(centers))
    return occupancies, nearest_counts
