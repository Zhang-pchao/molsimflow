"""Bead-density estimators and their distinct support and gauge conventions."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

from .kde import weighted_log_kde_1d, weighted_log_kde_2d
from .types import (
    validate_primary_estimator,
)
from .units import EV_TO_KCAL_MOL
from .weights import normalized_log_weights


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def _histogram_log_mass(
    samples: np.ndarray, log_weights: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Accumulate histogram masses per bin without linear-space cancellation."""

    values = np.asarray(samples, dtype=float)
    logs = np.asarray(log_weights, dtype=float)
    _require(
        values.ndim == logs.ndim == 1 and values.shape == logs.shape,
        "histogram samples and log weights must be matching vectors",
    )
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
        _require(
            conditioning is not None and conditioning_edges is not None, "incomplete conditioning"
        )
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
    raw_primary = raw_probability_mean if primary == "probability_mean" else raw_free_energy_mean
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
    result = bead_density_estimators(log_bead_density, kbt, primary_estimator=primary_estimator)
    return {
        **result,
        # Compatibility aliases; equation numbers do not define the estimator.
        "eq8": result["probability_mean"],
        "eq10": result["free_energy_mean_diagnostic"],
        "logmean_diagnostic": result["free_energy_mean_diagnostic"],
        "support": result["common_support"],
        "centers": 0.5 * (edges[:-1] + edges[1:]),
    }


def alternate_estimator(primary_estimator: str) -> str:
    """Return the estimator reported alongside the selected primary result."""

    primary = validate_primary_estimator(primary_estimator)
    return "free_energy_mean" if primary == "probability_mean" else "probability_mean"


def estimator_support_masks(
    log_sampling_density: np.ndarray,
    log_probability_mean: np.ndarray,
    log_bead_density: np.ndarray,
    log_relative_threshold: float,
) -> Dict[str, np.ndarray]:
    """Build independent and pairwise support masks for all FES comparisons."""

    sampling = np.asarray(log_sampling_density, dtype=float)
    probability = np.asarray(log_probability_mean, dtype=float)
    beads = np.asarray(log_bead_density, dtype=float)
    require(
        beads.shape[1:] == sampling.shape == probability.shape, "support density shape mismatch"
    )
    require(
        np.isfinite(sampling).all() and np.isfinite(probability).all() and np.isfinite(beads).all(),
        "non-finite KDE density",
    )
    sampling_support = sampling - np.max(sampling) >= log_relative_threshold
    probability_support = probability - np.max(probability) >= log_relative_threshold
    bead_axes = tuple(range(1, beads.ndim))
    relative_beads = beads - np.max(beads, axis=bead_axes, keepdims=True)
    free_energy_support = np.all(relative_beads >= log_relative_threshold, axis=0)
    estimator_common = probability_support & free_energy_support
    return {
        "centroid": sampling_support,
        "probability_mean": probability_support,
        "free_energy_mean": free_energy_support,
        # Compatibility name now has the scientifically relevant estimator-pair meaning.
        "common": estimator_common,
        "estimator_common": estimator_common,
        "all_common": sampling_support & estimator_common,
        "sampling_probability_common": sampling_support & probability_support,
        "sampling_free_energy_common": sampling_support & free_energy_support,
    }


def compute_surfaces(
    beads: np.ndarray,
    centroid: np.ndarray,
    raw_log_weights: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth: Sequence[float],
    kbt_ev: float,
    primary_estimator: str = "probability_mean",
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
    log_centroid = weighted_log_kde_2d(centroid, log_weights, x_grid, y_grid, bandwidth)
    estimators = bead_density_estimators(log_beads, kbt_ev, primary_estimator=primary_estimator)
    raw_sampling = -float(kbt_ev) * log_centroid
    return {
        "log_weights": log_weights,
        "weights": np.exp(log_weights),
        "log_centroid": log_centroid,
        "log_beads": log_beads,
        "log_probability_mean": estimators["log_probability_mean"],
        "raw_centroid": raw_sampling,
        "raw_probability_mean": estimators["raw_probability_mean"],
        "raw_free_energy_mean": estimators["raw_free_energy_mean"],
        "centroid": raw_sampling - np.min(raw_sampling),
        "probability_mean": estimators["probability_mean"],
        "free_energy_mean": estimators["free_energy_mean"],
    }


def compute_marginals(
    beads: np.ndarray,
    centroid: np.ndarray,
    raw_log_weights: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
    component: int,
    kbt_ev: float,
    primary_estimator: str = "probability_mean",
) -> Dict[str, np.ndarray]:
    log_weights = normalized_log_weights(raw_log_weights)
    log_beads = np.asarray(
        [
            weighted_log_kde_1d(beads[:, bead, component], log_weights, grid, bandwidth)
            for bead in range(beads.shape[1])
        ]
    )
    log_centroid = weighted_log_kde_1d(centroid[:, component], log_weights, grid, bandwidth)
    estimators = bead_density_estimators(log_beads, kbt_ev, primary_estimator=primary_estimator)
    sampling_f = -float(kbt_ev) * log_centroid
    return {
        "centroid": sampling_f - np.min(sampling_f),
        "probability_mean": estimators["probability_mean"],
        "free_energy_mean": estimators["free_energy_mean"],
        "log_centroid": log_centroid,
        "log_probability_mean": estimators["log_probability_mean"],
        "log_beads": log_beads,
    }


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


def optional_surface_difference_metrics(
    reference: np.ndarray, current: np.ndarray, support: np.ndarray
) -> Tuple[int, float | None, float | None]:
    """Keep insufficient diagnostic overlap separate from a valid primary FES."""
    count = int(np.count_nonzero(support))
    if count < 2:
        return count, None, None
    return surface_difference_metrics(reference, current, support)


def diagnostic_gap_metrics(gap: np.ndarray, support: np.ndarray) -> Dict[str, float | None]:
    """Compare the diagnostic using the primary zero, without extrapolated tails."""
    selected = np.asarray(gap)[np.asarray(support, dtype=bool)]
    return {
        "probability_free_energy_mean_rmse_common_support_kcal_mol": (
            float(np.sqrt(np.mean(selected**2))) if selected.size else None
        ),
        "probability_free_energy_mean_max_abs_common_support_kcal_mol": (
            float(np.max(np.abs(selected))) if selected.size else None
        ),
    }


def histogram_edges_for_kde_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Choose unsmoothed bins from a KDE grid and its recorded bandwidth.

    Bin width is close to the selected KDE sigma, with no more bins than grid
    points. This gives a reproducible direct-count companion without a
    system-specific numerical scale or any kernel convolution.
    """
    values = np.asarray(grid, dtype=float)
    width = float(sigma)
    require(values.ndim == 1 and len(values) >= 2, "histogram grid needs two points")
    require(np.isfinite(values).all() and np.all(np.diff(values) > 0), "invalid histogram grid")
    require(np.isfinite(width) and width > 0, "histogram reference sigma must be positive")
    bins = max(2, min(len(values), math.ceil((values[-1] - values[0]) / width)))
    return np.linspace(values[0], values[-1], bins + 1)


def weighted_unsmoothed_path_histogram(
    beads: np.ndarray,
    raw_log_weights: np.ndarray,
    edges: Sequence[np.ndarray],
    kbt_ev: float,
) -> Dict[str, np.ndarray | float]:
    """Direct bead-marginal FES from full-frame weights and zero-count gaps.

    The input is frame x bead x CV. All beads in a frame share its normalized
    weight and contribute 1/P. No pseudocount, KDE, or interpolation is applied.
    Two-dimensional arrays are returned in (y, x) plotting order.
    """
    values = np.asarray(beads, dtype=float)
    bins = tuple(np.asarray(edge, dtype=float) for edge in edges)
    require(values.ndim == 3 and values.shape[2] in (1, 2), "histogram CV shape mismatch")
    require(len(bins) == values.shape[2], "histogram edge dimension mismatch")
    require(values.shape[0] > 0 and values.shape[1] > 0, "empty histogram input")
    require(np.isfinite(values).all(), "non-finite histogram CV")
    require(np.isfinite(kbt_ev) and kbt_ev > 0, "invalid histogram kBT")
    for edge in bins:
        require(edge.ndim == 1 and len(edge) >= 3, "histogram needs two bins")
        require(np.isfinite(edge).all() and np.all(np.diff(edge) > 0), "invalid histogram edges")
    log_weights = normalized_log_weights(raw_log_weights)
    require(len(log_weights) == values.shape[0], "histogram frame weight mismatch")
    shape = tuple(len(edge) - 1 for edge in bins)
    indices = []
    valid = np.ones(values.shape[:2], dtype=bool)
    for axis, edge in enumerate(bins):
        component = values[:, :, axis]
        index = np.searchsorted(edge, component, side="right") - 1
        index = np.where(component == edge[-1], len(edge) - 2, index)
        valid &= (index >= 0) & (index < len(edge) - 1)
        indices.append(index)
    cell_count = math.prod(shape)
    flat = np.ravel_multi_index(tuple(index[valid] for index in indices), shape)
    path_logs = np.broadcast_to(log_weights[:, None] - math.log(values.shape[1]), values.shape[:2])
    log_mass = np.full(cell_count, -np.inf, dtype=float)
    np.logaddexp.at(log_mass, flat, path_logs[valid])
    observed_beads = np.bincount(flat, minlength=cell_count)
    frame_ids = np.broadcast_to(np.arange(values.shape[0])[:, None], values.shape[:2])
    frame_cell_pairs = np.unique(frame_ids[valid] * cell_count + flat)
    observed_frames = np.bincount(frame_cell_pairs % cell_count, minlength=cell_count)
    bin_volume = np.ones(shape, dtype=float)
    for axis, edge in enumerate(bins):
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = len(edge) - 1
        bin_volume *= np.diff(edge).reshape(broadcast_shape)
    log_density = log_mass.reshape(shape) - np.log(bin_volume)
    support = np.isfinite(log_density)
    fes = np.full(shape, np.nan, dtype=float)
    if np.any(support):
        raw = -float(kbt_ev) * log_density[support]
        fes[support] = (raw - np.min(raw)) * EV_TO_KCAL_MOL
    result = {
        "fes_kcal_mol": fes,
        "support": support,
        "weight_mass": np.exp(log_mass).reshape(shape),
        "observed_beads": observed_beads.reshape(shape),
        "observed_complete_frames": observed_frames.reshape(shape),
        "out_of_grid_weight_mass": float(max(0.0, 1.0 - np.sum(np.exp(log_mass)))),
    }
    if len(shape) == 2:
        for key in (
            "fes_kcal_mol",
            "support",
            "weight_mass",
            "observed_beads",
            "observed_complete_frames",
        ):
            result[key] = np.asarray(result[key]).T
    return result
