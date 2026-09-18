"""Periodic geometry diagnostics for finite-droplet translation coordinates.

These helpers are intended for analysis, not for defining a differentiable
bias.  They distinguish a first-harmonic phase coordinate from a coherently
reconstructed arithmetic center, contact-footprint motion, and rigid density
translation under periodic boundary conditions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicArithmeticCenter:
    """Arithmetic center on the branch selected by the largest empty gap."""

    value: float
    offsets: np.ndarray
    largest_empty_gap: float
    occupied_arc: float


@dataclass(frozen=True)
class DensityRegistration:
    """Periodic displacement and residual after translating a reference profile."""

    displacement: float
    residual: float


def _period(lower: float, upper: float) -> float:
    period = float(upper) - float(lower)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("upper must be greater than lower")
    return period


def minimum_image(values: np.ndarray | float, period: float) -> np.ndarray:
    """Return minimum-image scalar displacements in ``[-period/2, period/2)``."""

    length = float(period)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("period must be positive and finite")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("values contain non-finite entries")
    return array - length * np.floor(array / length + 0.5)


def periodic_arithmetic_center(
    values: np.ndarray, lower: float, upper: float
) -> PeriodicArithmeticCenter:
    """Reconstruct and average a localized periodic coordinate selection.

    The branch cut is placed in the largest empty interval between consecutive
    coordinates.  ``largest_empty_gap`` and ``occupied_arc`` expose whether
    that branch is well defined; callers must not interpret the result as a
    finite-body center when the selection percolates along the same axis.
    """

    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("values contain non-finite entries")
    length = _period(lower, upper)
    wrapped = np.sort((coordinates - float(lower)) % length)
    following = np.concatenate((wrapped[1:], wrapped[:1] + length))
    gaps = following - wrapped
    gap_index = int(np.argmax(gaps))
    cut = float(following[gap_index] % length)
    branch = (coordinates - float(lower) - cut) % length + cut
    mean_branch = float(np.mean(branch))
    center = float(lower) + (mean_branch % length)
    offsets = minimum_image(coordinates - center, length)
    largest_gap = float(gaps[gap_index])
    return PeriodicArithmeticCenter(
        value=center,
        offsets=offsets,
        largest_empty_gap=largest_gap,
        occupied_arc=float(length - largest_gap),
    )


def smooth_contact_weights(
    z: np.ndarray,
    surface_z: float,
    *,
    midpoint_offset: float = 4.5,
    width: float = 0.5,
) -> np.ndarray:
    """Return logistic weights for water oxygen atoms near an upper surface."""

    heights = np.asarray(z, dtype=float)
    if heights.ndim != 1 or not np.all(np.isfinite(heights)):
        raise ValueError("z must be a finite one-dimensional array")
    if not math.isfinite(surface_z) or not math.isfinite(midpoint_offset):
        raise ValueError("surface_z and midpoint_offset must be finite")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("width must be positive and finite")
    scaled = np.clip(
        (heights - (float(surface_z) + float(midpoint_offset))) / float(width),
        -60.0,
        60.0,
    )
    return 1.0 / (1.0 + np.exp(scaled))


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    """Return deterministic weighted quantiles using midpoint cumulative weights."""

    data = np.asarray(values, dtype=float)
    fixed_weights = np.asarray(weights, dtype=float)
    requested = np.asarray(tuple(quantiles), dtype=float)
    if data.ndim != 1 or data.size == 0 or fixed_weights.shape != data.shape:
        raise ValueError("values and weights must be matching non-empty vectors")
    if (
        not np.all(np.isfinite(data))
        or not np.all(np.isfinite(fixed_weights))
        or np.any(fixed_weights < 0.0)
        or float(np.sum(fixed_weights)) <= 0.0
    ):
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    if requested.ndim != 1 or np.any((requested < 0.0) | (requested > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")
    order = np.argsort(data, kind="mergesort")
    ordered = data[order]
    ordered_weights = fixed_weights[order]
    positions = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / np.sum(
        ordered_weights
    )
    return np.interp(requested, positions, ordered, left=ordered[0], right=ordered[-1])


def periodic_density_profile(
    values: np.ndarray,
    lower: float,
    upper: float,
    *,
    weights: np.ndarray | None = None,
    bins: int = 512,
    sigma: float = 1.0,
) -> np.ndarray:
    """Build a normalized periodic Gaussian-smoothed one-dimensional density."""

    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if bins < 16:
        raise ValueError("bins must be at least 16")
    length = _period(lower, upper)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")
    if weights is None:
        fixed_weights = np.ones_like(coordinates)
    else:
        fixed_weights = np.asarray(weights, dtype=float)
        if fixed_weights.shape != coordinates.shape:
            raise ValueError("weights must match values")
    if (
        not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(fixed_weights))
        or np.any(fixed_weights < 0.0)
        or float(np.sum(fixed_weights)) <= 0.0
    ):
        raise ValueError("coordinates and weights must be finite with positive total weight")
    wrapped = (coordinates - float(lower)) % length
    profile, _ = np.histogram(
        wrapped, bins=bins, range=(0.0, length), weights=fixed_weights
    )
    frequencies = 2.0 * np.pi * np.fft.fftfreq(bins, d=length / bins)
    kernel = np.exp(-0.5 * (sigma * frequencies) ** 2)
    smoothed = np.fft.ifft(np.fft.fft(profile) * kernel).real
    smoothed = np.maximum(smoothed, 0.0)
    return smoothed / np.sum(smoothed)


def _shift_profile(profile: np.ndarray, shift_bins: float) -> np.ndarray:
    frequencies = np.fft.fftfreq(len(profile))
    phase = np.exp(-2.0j * np.pi * frequencies * float(shift_bins))
    return np.fft.ifft(np.fft.fft(profile) * phase).real


def register_periodic_density(
    reference: np.ndarray, observed: np.ndarray, period: float
) -> DensityRegistration:
    """Register ``observed`` against periodic ``reference`` with sub-bin refinement.

    The returned displacement is positive when the observed density is shifted
    in the positive coordinate direction relative to the reference.  Residual
    is the L2 mismatch divided by the L2 norm of the observed profile.
    """

    ref = np.asarray(reference, dtype=float)
    current = np.asarray(observed, dtype=float)
    if ref.ndim != 1 or ref.shape != current.shape or len(ref) < 3:
        raise ValueError("profiles must be matching one-dimensional arrays")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(current)):
        raise ValueError("profiles contain non-finite entries")
    length = float(period)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("period must be positive and finite")
    ref = ref / np.sum(ref)
    current = current / np.sum(current)
    correlation = np.fft.ifft(np.conj(np.fft.fft(ref)) * np.fft.fft(current)).real
    peak = int(np.argmax(correlation))
    left = float(correlation[(peak - 1) % len(ref)])
    center = float(correlation[peak])
    right = float(correlation[(peak + 1) % len(ref)])
    denominator = left - 2.0 * center + right
    fractional = 0.0 if abs(denominator) < 1.0e-30 else 0.5 * (left - right) / denominator
    fractional = float(np.clip(fractional, -0.5, 0.5))
    shift_bins = float(peak) + fractional
    if shift_bins >= len(ref) / 2.0:
        shift_bins -= len(ref)
    fitted = _shift_profile(ref, shift_bins)
    residual = float(np.linalg.norm(current - fitted) / max(np.linalg.norm(current), 1.0e-30))
    return DensityRegistration(
        displacement=float(shift_bins * length / len(ref)), residual=residual
    )


def periodic_connectivity_winding(
    coordinates: np.ndarray,
    bounds: np.ndarray,
    cutoff: float,
    *,
    periodic: Sequence[bool] = (True, True, False),
) -> tuple[bool, bool, bool]:
    """Detect graph winding of a distance-connected selection along x, y, and z."""

    from scipy.spatial import cKDTree

    points = np.asarray(coordinates, dtype=float)
    limits = np.asarray(bounds, dtype=float)
    periodic_mask = np.asarray(tuple(periodic), dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("coordinates must have shape (N, 3) with N > 0")
    if limits.shape != (3, 2) or periodic_mask.shape != (3,):
        raise ValueError("bounds and periodic must have shapes (3, 2) and (3,)")
    if cutoff <= 0.0 or not math.isfinite(cutoff):
        raise ValueError("cutoff must be positive and finite")
    lengths = limits[:, 1] - limits[:, 0]
    normalized = points.copy()
    normalized[:, periodic_mask] = (
        normalized[:, periodic_mask] - limits[periodic_mask, 0]
    ) % lengths[periodic_mask]
    pairs = cKDTree(
        normalized, boxsize=np.where(periodic_mask, lengths, 0.0)
    ).query_pairs(float(cutoff), output_type="ndarray")
    neighbors: list[list[tuple[int, np.ndarray]]] = [[] for _ in points]
    for raw_left, raw_right in pairs:
        left, right = int(raw_left), int(raw_right)
        delta = normalized[right] - normalized[left]
        images = np.zeros(3, dtype=int)
        images[periodic_mask] = np.rint(
            delta[periodic_mask] / lengths[periodic_mask]
        ).astype(int)
        neighbors[left].append((right, -images))
        neighbors[right].append((left, images))
    labels: list[np.ndarray | None] = [None] * len(points)
    winding = np.zeros(3, dtype=bool)
    for seed in range(len(points)):
        if labels[seed] is not None:
            continue
        labels[seed] = np.zeros(3, dtype=int)
        stack = [seed]
        while stack:
            left = stack.pop()
            assert labels[left] is not None
            for right, jump in neighbors[left]:
                candidate = labels[left] + jump
                if labels[right] is None:
                    labels[right] = candidate
                    stack.append(right)
                else:
                    winding |= labels[right] != candidate
    winding &= periodic_mask
    return tuple(bool(value) for value in winding)
