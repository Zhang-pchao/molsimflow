"""Explicit block resampling, fixed-control residuals and FDR adjustment."""

from __future__ import annotations
import math
import numpy as np


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Fit finite rows to an intercept and supplied controls; preserve missing rows.

    Output has the original row order. At least parameters+2 finite rows are
    required. This operation does not infer a sampling or causal model.
    """
    y, x = np.asarray(values, float), np.asarray(controls, float)
    if y.ndim != 1 or x.ndim not in (1, 2) or len(x) != len(y):
        raise ValueError("Values and controls must have matching observation axes")
    matrix = np.column_stack([np.ones(len(y)), x])
    mask = np.isfinite(y) & np.all(np.isfinite(matrix), axis=1)
    result = np.full(len(y), np.nan)
    if mask.sum() >= matrix.shape[1] + 2:
        beta, *_ = np.linalg.lstsq(matrix[mask], y[mask], rcond=None)
        result[mask] = y[mask] - matrix[mask] @ beta
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Paired finite-row rank correlation with mean ranks for ties."""
    from scipy.stats import rankdata

    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Rank correlation requires matching one-dimensional values")
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return math.nan
    rx, ry = rankdata(x[valid]), rankdata(y[valid])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def block_bootstrap_partial_spearman(
    topology, outcome, controls, blocks, *, samples=2000, seed=None
):
    """Resample supplied blocks after fitting residuals once to the full data.

    This preserves the recorded fixed-residual bootstrap convention. All rows
    of a sampled block stay together and a repeated draw repeats the whole block.
    Caller-defined blocks must reflect the dependence/replica structure. A
    refitted-regression bootstrap is a distinct estimator.
    """
    x, y = np.asarray(topology, float), np.asarray(outcome, float)
    labels = np.asarray(blocks)
    if x.ndim != 1 or y.shape != x.shape or labels.shape != x.shape or not len(x):
        raise ValueError("Observations and block labels must have the same nonempty row axis")
    if labels.dtype.kind not in "iuUS" or (labels.dtype.kind in "US" and np.any(labels == "")):
        raise ValueError("Block IDs must be integer or nonempty string labels")
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 1:
        raise ValueError("Bootstrap samples must be a positive integer")
    left, right = residualize(x, controls), residualize(y, controls)
    unique = np.unique(labels)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(labels == block) for block in chosen])
        value = spearman(left[indices], right[indices])
        if math.isfinite(value):
            values.append(value)
    low, high = np.quantile(values, [0.025, 0.975]) if values else (math.nan, math.nan)
    return {
        "samples": len(x),
        "blocks": len(unique),
        "partial_spearman": spearman(left, right),
        "block_bootstrap_ci025": float(low),
        "block_bootstrap_ci975": float(high),
        "bootstrap_valid": len(values),
    }


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in original hypothesis order."""
    values = np.asarray(pvalues, float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must be a finite vector in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(1, adjusted)
    return result
