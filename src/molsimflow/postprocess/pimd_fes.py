"""Core estimators and data checks for bead-defined PIMD free energies."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


BIAS_MODES = {"centroid_coord", "bead_mean", "bead_density_shared"}
WEIGHT_KINDS = {"fixed_bias", "quasi_static_opes", "precomputed"}
FES_ESTIMATORS = {"probability_mean", "free_energy_mean"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalized_log_weights(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalize finite log weights without exponentiating their absolute scale."""

    raw = np.asarray(values, dtype=float)
    _require(raw.ndim == 1 and raw.size > 0, "log weights must be a nonempty vector")
    _require(np.isfinite(raw).all(), "log weights must be finite")
    maximum = float(np.max(raw))
    shifted = raw - maximum
    return shifted - math.log(float(np.sum(np.exp(shifted))))


def validate_bias_mode(value: str) -> str:
    """Validate the path-CV bias modes supported by this workflow."""

    mode = str(value)
    _require(mode in BIAS_MODES, f"unsupported PIMD bias mode: {mode}")
    return mode


def validate_primary_estimator(value: str) -> str:
    """Validate the public estimator names accepted as the primary result."""

    estimator = str(value)
    _require(estimator in FES_ESTIMATORS, f"unsupported primary estimator: {estimator}")
    return estimator


def total_bias_energy(
    bias_mode: str,
    *,
    sampling_bias_energy: Sequence[float] | np.ndarray | None = None,
    bead_bias_energies: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    """Return one complete-path bias energy per ring-polymer frame."""

    mode = validate_bias_mode(bias_mode)
    if mode == "bead_density_shared":
        _require(sampling_bias_energy is None, "bead-density bias uses bead-local energies")
        _require(bead_bias_energies is not None, "bead-density bias energies are required")
        values = np.asarray(bead_bias_energies, dtype=float)
        _require(values.ndim == 2 and min(values.shape) > 0, "invalid bead-density bias table")
        _require(np.isfinite(values).all(), "bead-density bias energies must be finite")
        return np.mean(values, axis=1)

    _require(bead_bias_energies is None, "centroid/bead-mean bias uses one sampling energy")
    _require(sampling_bias_energy is not None, "sampling bias energy is required")
    values = np.asarray(sampling_bias_energy, dtype=float)
    _require(values.ndim == 1 and values.size > 0, "sampling bias energy must be a vector")
    _require(np.isfinite(values).all(), "sampling bias energy must be finite")
    return values


def frame_log_weights(
    weight_kind: str,
    *,
    bias_energy: Sequence[float] | np.ndarray | None = None,
    kbt: float | None = None,
    precomputed: Sequence[float] | np.ndarray | None = None,
    quasi_static: bool | None = None,
) -> np.ndarray:
    """Build one log weight per complete ring-polymer frame.

    ``fixed_bias`` and ``quasi_static_opes`` both use ``+U_bias / kBT``.  The
    distinct names keep their sampling assumptions explicit.  In particular,
    this function never subtracts an OPES ``rct`` column and does not infer that
    an adaptive trajectory has reached the quasi-static regime.
    """

    kind = str(weight_kind)
    _require(kind in WEIGHT_KINDS, f"unsupported PIMD weight kind: {kind}")
    if kind == "precomputed":
        _require(precomputed is not None, "precomputed log weights are required")
        _require(bias_energy is None, "precomputed weights cannot also use bias energy")
        raw = np.asarray(precomputed, dtype=float)
    else:
        _require(bias_energy is not None, f"bias energy is required for {kind}")
        _require(precomputed is None, "bias-energy weights cannot also be precomputed")
        _require(kbt is not None and np.isfinite(kbt) and kbt > 0.0, "kBT must be positive")
        if kind == "quasi_static_opes":
            _require(quasi_static is True, "quasi-static OPES must be declared explicitly")
        raw = np.asarray(bias_energy, dtype=float) / float(kbt)
    _require(raw.ndim == 1 and raw.size > 0, "frame weights must be a nonempty vector")
    _require(np.isfinite(raw).all(), "frame weights must be finite")
    return raw


def restart_unique_indices(
    frame_ids: Sequence[float] | np.ndarray,
    *,
    policy: str = "keep_first",
) -> Tuple[np.ndarray, int]:
    """Return indices after removing adjacent restart-seam duplicates."""

    ids = np.asarray(frame_ids, dtype=float)
    _require(ids.ndim == 1 and ids.size > 0, "frame ids must be a nonempty vector")
    _require(np.isfinite(ids).all(), "frame ids must be finite")
    _require(np.all(np.diff(ids) >= 0.0), "frame ids decrease across a restart seam")
    _require(policy in {"error", "keep_first", "keep_last"}, "invalid restart policy")
    duplicate = np.r_[False, ids[1:] == ids[:-1]]
    count = int(np.count_nonzero(duplicate))
    if count == 0:
        return np.arange(ids.size), 0
    if policy == "error":
        raise ValueError("duplicate restart-seam frame ids")
    if policy == "keep_first":
        keep = ~duplicate
    else:
        keep = ~np.r_[ids[:-1] == ids[1:], False]
    return np.flatnonzero(keep), count


def assemble_bead_frames(
    frame_ids: Sequence[int] | np.ndarray,
    bead_ids: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    expected_beads: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble a long bead table into complete frame-by-bead arrays."""

    frames = np.asarray(frame_ids)
    beads = np.asarray(bead_ids)
    data = np.asarray(values, dtype=float)
    _require(frames.ndim == beads.ndim == 1, "frame and bead ids must be vectors")
    _require(data.ndim in {1, 2}, "invalid bead value table")
    _require(frames.size == beads.size == data.shape[0], "long-table row counts differ")
    _require(frames.size > 0, "invalid bead value table")
    _require(np.isfinite(data).all(), "bead values must be finite")
    _require(
        isinstance(expected_beads, (int, np.integer))
        and not isinstance(expected_beads, (bool, np.bool_))
        and expected_beads > 0,
        "expected_beads must be a positive integer",
    )

    ordered_frames = []
    groups: Dict[object, list[int]] = {}
    for row, frame in enumerate(frames.tolist()):
        if frame not in groups:
            ordered_frames.append(frame)
            groups[frame] = []
        groups[frame].append(row)

    first_rows = groups[ordered_frames[0]]
    bead_labels = np.sort(beads[first_rows])
    _require(len(bead_labels) == int(expected_beads), "first frame has missing beads")
    _require(len(np.unique(bead_labels)) == len(bead_labels), "duplicate bead in first frame")
    assembled = []
    for frame in ordered_frames:
        rows = groups[frame]
        labels = beads[rows]
        _require(len(rows) == int(expected_beads), f"frame {frame} has missing beads")
        _require(len(np.unique(labels)) == len(labels), f"frame {frame} has duplicate beads")
        _require(np.array_equal(np.sort(labels), bead_labels), f"frame {frame} bead ids differ")
        order = np.argsort(labels)
        assembled.append(data[np.asarray(rows)[order]])
    return np.asarray(ordered_frames), bead_labels, np.asarray(assembled)


def _histogram_log_mass(
    samples: np.ndarray, log_weights: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Accumulate histogram masses per bin without linear-space cancellation."""

    values = np.asarray(samples, dtype=float)
    logs = np.asarray(log_weights, dtype=float)
    _require(values.ndim == logs.ndim == 1 and values.shape == logs.shape,
             "histogram samples and log weights must be matching vectors")
    bins = np.searchsorted(edges, values, side="right") - 1
    bins[values == edges[-1]] = len(edges) - 2
    result = np.full(len(edges) - 1, -np.inf)
    for index in range(len(result)):
        selected = logs[bins == index]
        if selected.size:
            result[index] = np.logaddexp.reduce(selected)
    return result


def quantum_histogram_masses(
    bead_cv: Sequence[Sequence[float]] | np.ndarray,
    log_frame_weights: Sequence[float] | np.ndarray,
    bin_edges: Sequence[float] | np.ndarray,
    *,
    conditioning: Sequence[float] | np.ndarray | None = None,
    conditioning_edges: Sequence[float] | np.ndarray | None = None,
) -> Dict[str, np.ndarray | float]:
    """Compute direct and conditionally decomposed bead histogram masses.

    Every bead in a frame uses the same normalized frame weight.  The optional
    conditional route uses weighted conditional histograms and is therefore an
    exact finite-sample decomposition of the direct estimator, up to rounding.
    It is a regression oracle, not evidence that an adaptive bias converged.
    """

    values = np.asarray(bead_cv, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    _require(values.ndim == 2 and min(values.shape) > 0, "bead_cv must be frame x bead")
    _require(np.isfinite(values).all(), "bead_cv must be finite")
    _require(edges.ndim == 1 and edges.size >= 2, "bin edges must be a vector")
    _require(np.isfinite(edges).all() and np.all(np.diff(edges) > 0.0), "invalid bin edges")
    log_weights = normalized_log_weights(log_frame_weights)
    _require(len(log_weights) == values.shape[0], "frame weight count differs from bead frames")
    log_per_bead = np.asarray(
        [
            _histogram_log_mass(values[:, bead], log_weights, edges)
            for bead in range(values.shape[1])
        ]
    )
    log_direct = np.logaddexp.reduce(log_per_bead, axis=0) - math.log(values.shape[1])
    # Linear masses remain a compatibility representation. They can underflow
    # for extreme dynamic ranges; FES consumers must use the log fields below.
    per_bead = np.exp(log_per_bead)
    direct = np.exp(log_direct)

    if conditioning is None and conditioning_edges is None:
        log_conditional = log_direct.copy()
        conditional = direct.copy()
    else:
        _require(conditioning is not None and conditioning_edges is not None, "incomplete conditioning")
        condition = np.asarray(conditioning, dtype=float)
        condition_edges = np.asarray(conditioning_edges, dtype=float)
        _require(condition.shape == log_weights.shape, "conditioning must have one value per frame")
        _require(np.isfinite(condition).all(), "conditioning values must be finite")
        _require(
            condition_edges.ndim == 1
            and condition_edges.size >= 2
            and np.isfinite(condition_edges).all()
            and np.all(np.diff(condition_edges) > 0.0),
            "invalid conditioning edges",
        )
        _require(
            np.all((condition >= condition_edges[0]) & (condition <= condition_edges[-1])),
            "conditioning values fall outside the conditioning grid",
        )
        condition_bin = np.searchsorted(condition_edges, condition, side="right") - 1
        condition_bin[condition == condition_edges[-1]] = len(condition_edges) - 2
        log_conditional = np.full_like(log_direct, -np.inf)
        for index in range(len(condition_edges) - 1):
            mask = condition_bin == index
            if not np.any(mask):
                continue
            local = np.asarray(
                [
                    _histogram_log_mass(values[mask, bead], log_weights[mask], edges)
                    for bead in range(values.shape[1])
                ]
            )
            contribution = np.logaddexp.reduce(local, axis=0) - math.log(values.shape[1])
            log_conditional = np.logaddexp(log_conditional, contribution)
        conditional = np.exp(log_conditional)

    return {
        "direct": direct,
        "conditional": conditional,
        "per_bead": per_bead,
        "log_direct": log_direct,
        "log_conditional": log_conditional,
        "log_per_bead": log_per_bead,
        "in_range_mass": float(np.sum(direct)),
        "log_in_range_mass": float(np.logaddexp.reduce(log_direct)),
    }


def bead_density_estimators(
    log_bead_density: Sequence[Sequence[float]] | np.ndarray,
    kbt: float,
    *,
    primary_estimator: str = "probability_mean",
) -> Dict[str, np.ndarray | float]:
    """Combine bead densities using either documented estimator as primary.

    The input has shape ``(bead, *grid)``. Each bead density must use the same
    complete-frame weights, normalization, grid measure, and smoothing rule.
    Finite log densities and ``-inf`` (zero density) are accepted.

    ``probability_mean`` is ``-kBT * log(mean_bead(density))`` and
    ``free_energy_mean`` averages the bead free energies. At finite sampling
    they are generally different quantities. Both curves use the minimum of
    the explicitly selected primary estimator as their common zero. Raw values
    are also returned for callers that use another reference.

    Probability-mean support requires at least one bead with positive density;
    free-energy-mean support requires positive density for every bead. Their
    intersection is used only for comparisons between the two estimators.
    """

    logs = np.asarray(log_bead_density, dtype=float)
    _require(logs.ndim >= 2 and min(logs.shape) > 0, "log bead density must be bead x grid")
    _require(
        np.all(np.isfinite(logs) | np.isneginf(logs)),
        "log bead density must be finite or negative infinity",
    )
    _require(np.isfinite(kbt) and kbt > 0.0, "kBT must be positive")
    primary = validate_primary_estimator(primary_estimator)
    # logaddexp keeps narrow or remote KDE tails from underflowing to zero.
    log_probability_mean = np.logaddexp.reduce(logs, axis=0) - math.log(logs.shape[0])
    raw_probability_mean = -float(kbt) * log_probability_mean
    # Divide before summing to avoid overflow for large finite log densities.
    raw_free_energy_mean = -float(kbt) * np.sum(logs / logs.shape[0], axis=0)
    probability_support = np.isfinite(raw_probability_mean)
    common_support = probability_support & np.isfinite(raw_free_energy_mean)
    _require(np.any(probability_support), "no finite probability-mean support")
    raw_primary = (
        raw_probability_mean if primary == "probability_mean" else raw_free_energy_mean
    )
    primary_support = np.isfinite(raw_primary)
    _require(np.any(primary_support), f"no finite {primary} support")
    zero = float(np.min(raw_primary[primary_support]))
    return {
        "log_probability_mean": log_probability_mean,
        "raw_probability_mean": raw_probability_mean,
        "raw_free_energy_mean": raw_free_energy_mean,
        "raw_free_energy_mean_diagnostic": raw_free_energy_mean,
        "probability_mean": raw_probability_mean - zero,
        "free_energy_mean": raw_free_energy_mean - zero,
        "free_energy_mean_diagnostic": raw_free_energy_mean - zero,
        "probability_support": probability_support,
        "free_energy_mean_support": np.isfinite(raw_free_energy_mean),
        "common_support": common_support,
        "zero_reference": zero,
    }


def quantum_fes_1d(
    bead_cv: Sequence[Sequence[float]] | np.ndarray,
    log_frame_weights: Sequence[float] | np.ndarray,
    bin_edges: Sequence[float] | np.ndarray,
    *,
    kbt: float,
    primary_estimator: str = "probability_mean",
) -> Dict[str, np.ndarray | float]:
    """Return both maintained bead-density FES estimators.

    Histogram densities account for bin widths. Both estimators share the
    primary FES minimum as their zero. The legacy ``eq8``, ``eq10``,
    ``logmean_diagnostic``, and ``support`` keys are retained as aliases;
    new consumers should use the descriptive estimator and support keys.
    """

    edges = np.asarray(bin_edges, dtype=float)
    masses = quantum_histogram_masses(bead_cv, log_frame_weights, edges)
    log_bead_density = np.asarray(masses["log_per_bead"]) - np.log(np.diff(edges))[None, :]
    result = bead_density_estimators(
        log_bead_density, kbt, primary_estimator=primary_estimator
    )
    return {
        **result,
        # Compatibility aliases; equation numbers do not define the estimator.
        "eq8": result["probability_mean"],
        "eq10": result["free_energy_mean_diagnostic"],
        "logmean_diagnostic": result["free_energy_mean_diagnostic"],
        "support": result["common_support"],
        "centers": 0.5 * (edges[:-1] + edges[1:]),
    }


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
    full = quantum_fes_1d(
        values, weights, bin_edges, kbt=kbt, primary_estimator=primary
    )
    _require(
        isinstance(block_size, (int, np.integer)) and not isinstance(block_size, bool)
        and block_size > 0, "block_size must be a positive integer",
    )
    block_size = int(block_size)
    frames = len(values)
    _require(frames % block_size == 0, "equal blocks must cover all frames")
    blocks = frames // block_size
    _require(blocks >= 2, "jackknife requires at least two blocks")
    bins = len(full[primary])
    _require(
        isinstance(reference_bin, (int, np.integer)) and not isinstance(reference_bin, bool)
        and 0 <= reference_bin < bins, "invalid reference_bin",
    )
    support_key = (
        "probability_support"
        if primary == "probability_mean"
        else "free_energy_mean_support"
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
        (blocks - 1) / blocks
        * np.sum((supported - np.mean(supported, axis=0)) ** 2, axis=0)
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
