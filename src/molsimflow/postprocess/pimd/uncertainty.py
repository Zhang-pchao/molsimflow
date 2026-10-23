"""Delete whole frame blocks; beads are not independent uncertainty units."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .estimators import bead_density_estimators, quantum_fes_1d
from .kde import logsumexp, weighted_log_kde_1d, weighted_log_kde_2d
from .types import (
    validate_primary_estimator,
)
from .weights import normalized_log_weights


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def quantum_fes_block_jackknife_1d(
    bead_cv: Sequence[Sequence[float]] | np.ndarray,
    log_frame_weights: Sequence[float] | np.ndarray,
    bin_edges: Sequence[float] | np.ndarray,
    *,
    kbt: float,
    block_size: int,
    reference_bin: int,
    primary_estimator: str = "probability_mean",
) -> dict[str, object]:
    """Estimate standard errors of FES differences using contiguous frame blocks.

    All beads of a frame stay together. Equal blocks must cover all frames;
    the caller must choose blocks long enough to accommodate correlation.
    This does not estimate autocorrelation times or assert block independence.
    Unsupported bins receive NaN uncertainty, never zero uncertainty.
    """
    values = np.asarray(bead_cv, dtype=float)
    weights = np.asarray(log_frame_weights, dtype=float)
    primary = validate_primary_estimator(primary_estimator)
    full = quantum_fes_1d(values, weights, bin_edges, kbt=kbt, primary_estimator=primary)
    _require(
        isinstance(block_size, (int, np.integer))
        and not isinstance(block_size, bool)
        and block_size > 0,
        "block_size must be a positive integer",
    )
    block_size = int(block_size)
    frames = len(values)
    _require(frames % block_size == 0, "equal blocks must cover all frames")
    blocks = frames // block_size
    _require(blocks >= 2, "jackknife requires at least two blocks")
    bins = len(full[primary])
    _require(
        isinstance(reference_bin, (int, np.integer))
        and not isinstance(reference_bin, bool)
        and 0 <= reference_bin < bins,
        "invalid reference_bin",
    )
    support_key = (
        "probability_support" if primary == "probability_mean" else "free_energy_mean_support"
    )
    _require(full[support_key][reference_bin], "reference bin has no support")
    estimates = []
    for block in range(blocks):
        keep = np.ones(frames, dtype=bool)
        keep[block * block_size : (block + 1) * block_size] = False
        # Normalize afresh so deleting a dominant-weight block remains stable.
        partial = quantum_fes_1d(
            values[keep],
            weights[keep],
            bin_edges,
            kbt=kbt,
            primary_estimator=primary,
        )
        _require(
            partial[support_key][reference_bin],
            "reference bin loses support after block deletion",
        )
        curve = partial[primary]
        estimates.append(curve - curve[reference_bin])
    leave_one_out = np.asarray(estimates)
    support = np.all(np.isfinite(leave_one_out), axis=0)
    standard_error = np.full(bins, np.nan)
    supported = leave_one_out[:, support]
    standard_error[support] = np.sqrt(
        (blocks - 1) / blocks * np.sum((supported - np.mean(supported, axis=0)) ** 2, axis=0)
    )
    curve = full[primary]
    return {
        "estimator": primary,
        "free_energy_difference": curve - curve[reference_bin],
        "standard_error": standard_error,
        "support": support,
        "reference_bin": int(reference_bin),
        "block_size": int(block_size),
        "blocks": int(blocks),
        "leave_one_block_out": leave_one_out,
    }


def quantum_kde_block_jackknife(
    beads: np.ndarray,
    raw_log_weights: np.ndarray,
    grids: Sequence[np.ndarray],
    bandwidth: Sequence[float],
    *,
    kbt: float,
    block_frames: int,
    reference_grid_index: Sequence[int],
    relative_density_support: float,
    primary_estimator: str = "probability_mean",
) -> dict[str, object]:
    """Estimate fixed-bandwidth primary-estimator errors in one or two CVs.

    Grid indices follow CV order; 2D arrays follow the existing (y, x) layout.
    Whole contiguous frame blocks are deleted. Support must survive every
    deletion. This estimates sampling error, not KDE smoothing bias.
    """
    values = np.asarray(beads, dtype=float)
    axes = [np.asarray(axis, dtype=float) for axis in grids]
    widths = np.asarray(bandwidth, dtype=float)
    dims = len(axes)
    require(dims in (1, 2), "jackknife requires one or two CV grids")
    require(
        values.ndim == 3
        and values.shape[2] == dims
        and values.shape[0] > 0
        and values.shape[1] > 0,
        "invalid bead shape",
    )
    require(np.isfinite(values).all(), "non-finite bead CV")
    require(
        all(
            axis.ndim == 1
            and len(axis) > 0
            and np.isfinite(axis).all()
            and np.all(np.diff(axis) > 0)
            for axis in axes
        ),
        "invalid grid",
    )
    require(
        widths.shape == (dims,) and np.isfinite(widths).all() and np.all(widths > 0),
        "invalid bandwidth",
    )
    require(np.isfinite(kbt) and kbt > 0, "kBT must be finite and positive")
    require(
        np.isfinite(relative_density_support) and 0 < relative_density_support <= 1,
        "invalid relative density support",
    )
    weights = np.asarray(raw_log_weights, dtype=float)
    require(weights.shape == (len(values),), "weight/frame shape mismatch")
    normalized_log_weights(weights)
    require(
        isinstance(block_frames, (int, np.integer))
        and not isinstance(block_frames, bool)
        and block_frames > 0,
        "block_frames must be a positive integer",
    )
    size = int(block_frames)
    require(len(values) % size == 0, "equal blocks must cover all frames")
    blocks = len(values) // size
    require(blocks >= 2, "jackknife requires at least two blocks")
    require(
        isinstance(reference_grid_index, (list, tuple, np.ndarray)), "invalid reference_grid_index"
    )
    index = tuple(reference_grid_index)
    require(
        len(index) == dims
        and all(
            isinstance(i, (int, np.integer)) and not isinstance(i, bool) and 0 <= i < len(axis)
            for i, axis in zip(index, axes)
        ),
        "invalid reference_grid_index",
    )
    array_index = index if dims == 1 else index[::-1]
    threshold = math.log(relative_density_support)
    primary = validate_primary_estimator(primary_estimator)

    def curve(keep):
        selected = values[keep]
        logw = normalized_log_weights(weights[keep])
        if dims == 1:
            logs = [
                weighted_log_kde_1d(selected[:, b, 0], logw, axes[0], widths[0])
                for b in range(values.shape[1])
            ]
        else:
            logs = [
                weighted_log_kde_2d(selected[:, b], logw, axes[0], axes[1], widths)
                for b in range(values.shape[1])
            ]
        bead_logs = np.asarray(logs)
        estimators = bead_density_estimators(bead_logs, kbt, primary_estimator=primary)
        logp = np.asarray(logsumexp(bead_logs, axis=0)) - math.log(values.shape[1])
        if primary == "probability_mean":
            support = logp - np.max(logp) >= threshold
        else:
            bead_axes = tuple(range(1, bead_logs.ndim))
            relative = bead_logs - np.max(bead_logs, axis=bead_axes, keepdims=True)
            support = np.all(relative >= threshold, axis=0)
        require(support[array_index], "reference grid point loses density support")
        curve_values = np.asarray(estimators[primary], dtype=float)
        return curve_values - curve_values[array_index], support

    full, support = curve(np.ones(len(values), dtype=bool))
    estimates = []
    for block in range(blocks):
        keep = np.ones(len(values), dtype=bool)
        keep[block * size : (block + 1) * size] = False
        current, current_support = curve(keep)
        estimates.append(current)
        support &= current_support
    deleted = np.asarray(estimates)
    error = np.sqrt(
        (blocks - 1) / blocks * np.sum((deleted - np.mean(deleted, axis=0)) ** 2, axis=0)
    )
    error[~support] = np.nan
    return {
        "free_energy_difference": full,
        "standard_error": error,
        "support": support,
        "blocks": blocks,
        "block_frames": size,
    }
