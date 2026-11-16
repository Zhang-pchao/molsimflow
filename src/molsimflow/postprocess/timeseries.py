"""Time-series diagnostics with explicit sampling and truncation conventions."""

from __future__ import annotations

import math
import numpy as np


def normalized_autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Return a mean-subtracted, overlap-normalized autocorrelation."""

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return np.array([math.nan])
    centered = finite - float(np.mean(finite))
    variance = float(np.dot(centered, centered) / len(centered))
    maximum = min(int(max_lag), len(centered) - 1)
    if variance <= np.finfo(float).eps:
        result = np.full(maximum + 1, math.nan)
        result[0] = 1.0
        return result
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    overlap = np.arange(len(centered), 0, -1, dtype=float)
    correlation = correlation / overlap / variance
    return correlation[: maximum + 1]


def positive_acf_time_ps(acf: np.ndarray, sample_ps: float) -> float:
    """Integrate the positive ACF lobe with a trapezoidal endpoint at lag zero."""

    values = np.asarray(acf, dtype=float)
    if not len(values) or not math.isfinite(float(values[0])):
        return math.nan
    positive: list[float] = []
    for value in values[1:]:
        if not math.isfinite(float(value)) or value <= 0.0:
            break
        positive.append(float(value))
    return sample_ps * (0.5 + float(np.sum(positive)))


def autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    """Overlap-normalized ACF of finite, equally spaced observations.

    Keep gaps and missing-data selection upstream. A constant series returns
    [1, 0, ...]. The legacy normalized_autocorrelation retains its own missing
    and constant-series conventions for existing callers.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite equally spaced values are required")
    centered = values - values.mean()
    variance = float(np.dot(centered, centered) / len(centered))
    if variance <= np.finfo(float).eps:
        result = np.zeros(len(values))
        result[0] = 1.0
        return result
    size = 1 << (2 * len(values) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size)
    covariance = np.fft.irfft(spectrum * spectrum.conjugate(), n=size)[: len(values)]
    covariance /= np.arange(len(values), 0, -1)
    return covariance / variance


def statistical_inefficiency(acf: np.ndarray, *, method: str = "initial_positive") -> float:
    """Return g >= 1 using positive lags or adjacent positive lag pairs.

    These truncations are different finite-sample estimators. g describes this
    recorded series and does not establish stationarity or replica independence.
    """
    values = np.asarray(acf, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("A finite one-dimensional ACF is required")
    if not np.isclose(values[0], 1.0):
        raise ValueError("ACF lag zero must equal one")
    if method == "initial_positive":
        terms = values[1:]
    elif method == "initial_positive_pair":
        count = (len(values) - 1) // 2
        terms = values[1 : 1 + 2 * count].reshape(-1, 2).sum(axis=1)
    else:
        raise ValueError("Unknown ACF truncation method")
    total = 1.0
    for term in terms:
        if term <= 0:
            break
        total += 2.0 * float(term)
    return max(1.0, total)
