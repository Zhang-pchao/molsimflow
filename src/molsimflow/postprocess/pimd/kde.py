"""Fixed-bandwidth log-space kernels on paired complete-path observations."""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    if axis is None:
        return float(result.squeeze())
    return np.squeeze(result, axis=axis)



def _validate_kde_weights(samples: np.ndarray, log_weights: np.ndarray) -> None:
    require(samples.shape[0] > 0 and np.isfinite(samples).all(), "invalid KDE samples")
    require(log_weights.shape == (samples.shape[0],), "KDE frame weight count mismatch")
    require(np.isfinite(log_weights).all(), "non-finite KDE log weights")



def weighted_log_kde_1d(
    samples: np.ndarray, log_weights: np.ndarray, grid: np.ndarray, bandwidth: float
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float)
    require(samples.ndim == 1, "1D sample shape mismatch")
    _validate_kde_weights(samples, log_weights)
    require(np.isfinite(bandwidth) and float(bandwidth) > 0.0,
            "bandwidth must be finite and positive")
    exponent = log_weights[None, :] - 0.5 * (
        (grid[:, None] - samples[None, :]) / float(bandwidth)
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
    log_weights = np.asarray(log_weights, dtype=float)
    require(samples.ndim == 2 and samples.shape[1] == 2, "2D sample shape mismatch")
    _validate_kde_weights(samples, log_weights)
    hx, hy = (float(value) for value in bandwidth)
    require(np.isfinite([hx, hy]).all() and hx > 0.0 and hy > 0.0,
            "bandwidths must be finite and positive")
    require(isinstance(chunk_points, (int, np.integer))
            and not isinstance(chunk_points, bool) and chunk_points > 0,
            "chunk_points must be a positive integer")
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
