"""Finite-history CV coverage with complete paths as the counting units."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class GridCoverage:
    """Unweighted cell incidences; serial paths need not be independent."""

    cells: np.ndarray
    path_counts: np.ndarray
    block_counts: np.ndarray
    threshold_times: np.ndarray
    progress_times: np.ndarray
    growth_counts: np.ndarray
    cell_widths: np.ndarray
    origin_fraction: np.ndarray
    minimum_paths: int

    @property
    def supported_cells(self) -> np.ndarray:
        return self.cells[self.path_counts >= self.minimum_paths]


def path_grid_coverage(
    samples: np.ndarray,
    times: np.ndarray,
    cell_widths: Sequence[float],
    *,
    origin_fraction: Sequence[float] = (0.0, 0.0),
    minimum_paths: int = 10,
    block_width: float = 250.0,
    progress_times: Sequence[float] = (),
) -> GridCoverage:
    """Count unique (path, cell) hits on a fixed two-dimensional grid.

    ``samples`` is (paths, 2) or (paths, correlated observations, 2).
    Several beads in the same cell contribute only one hit for that path.
    Cells are floor(samples / cell_widths - origin_fraction). A supported
    cell has at least ``minimum_paths`` hits. Growth counts use the time of
    the threshold hit, without interpolation. Time blocks are right-closed,
    anchored at zero. No density smoothing or equilibrium weighting is used.
    """
    x = np.asarray(samples, dtype=float)
    t = np.asarray(times, dtype=float)
    widths = np.asarray(cell_widths, dtype=float)
    phase = np.asarray(origin_fraction, dtype=float)
    progress = np.asarray(progress_times, dtype=float)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] < 1:
        raise ValueError("samples must have shape (paths, [observations,] 2)")
    if t.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("samples and times must be finite with matching path counts")
    if np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError("times must be nonnegative and strictly increasing")
    if widths.shape != (2,) or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("two finite positive cell widths are required")
    if phase.shape != (2,) or not np.isfinite(phase).all():
        raise ValueError("two finite origin fractions are required")
    if not isinstance(minimum_paths, (int, np.integer)) or minimum_paths < 1:
        raise ValueError("minimum_paths must be a positive integer")
    if not np.isfinite(block_width) or block_width <= 0:
        raise ValueError("block_width must be finite and positive")
    if progress.ndim != 1 or not np.isfinite(progress).all() or np.any(np.diff(progress) < 0):
        raise ValueError("progress_times must be finite and sorted")
    cell = np.floor(x / widths - phase).astype(np.int64).reshape(-1, 2)
    cells, inverse = np.unique(cell, axis=0, return_inverse=True)
    incidence = np.unique(
        np.column_stack((np.repeat(np.arange(len(x)), x.shape[1]), inverse)), axis=0
    )
    frames, ids = incidence.T
    counts = np.bincount(ids, minlength=len(cells))
    block = np.maximum(0, np.floor(np.nextafter(t, -np.inf) / block_width)).astype(int)
    block_hits = np.unique(np.column_stack((block[frames], ids)), axis=0)
    nblocks = np.bincount(block_hits[:, 1], minlength=len(cells))
    threshold = np.full(len(cells), np.nan)
    if len(incidence):
        ordered = incidence[np.lexsort((frames, ids))]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered[:, 1])) + 1]
        supported = counts >= minimum_paths
        threshold[supported] = t[ordered[starts[supported] + minimum_paths - 1, 0]]
    growth = np.searchsorted(np.sort(threshold[np.isfinite(threshold)]), progress, side="right")
    return GridCoverage(cells, counts, nblocks, threshold, progress, growth, widths,
                        phase, minimum_paths)


def compare_grid_coverage(first: GridCoverage, second: GridCoverage) -> dict:
    """Return common/exclusive cell counts; empty denominators remain undefined."""
    if (not np.array_equal(first.cell_widths, second.cell_widths)
            or not np.array_equal(first.origin_fraction, second.origin_fraction)
            or first.minimum_paths != second.minimum_paths):
        raise ValueError("coverage comparisons require identical grids and thresholds")
    a = set(map(tuple, first.supported_cells))
    b = set(map(tuple, second.supported_cells))
    union = a | b
    return {
        "first": len(a), "second": len(b), "common": len(a & b),
        "only_first": len(a - b), "only_second": len(b - a), "union": len(union),
        "second_over_first": len(b) / len(a) if a else None,
        "jaccard": len(a & b) / len(union) if union else None,
    }
